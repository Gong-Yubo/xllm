/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kv_cache/transfer/yuanrong_store_backend.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <cstring>
#include <limits>
#include <utility>

namespace xllm {
namespace {

constexpr char kDefaultBridgeLibrary[] = "libxllm_yuanrong_store_bridge.so";

// DataSystem validates keys against ^[a-zA-Z0-9.\-_!@#%^*()+;:]*$ (and a 1024
// byte cap). KVCacheStore keys are "xllm-kv-v3:<...>" + 32 raw hash bytes
// (16 schema + 16 block content), so they would be rejected. Hex-encoding the
// whole key is injective, keeps the prefix and both hashes unambiguous, and is
// applied to put and get alike, so round trips are unaffected.
std::string encode_key(const std::string& key) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(key.size() * 2);
  for (unsigned char byte : key) {
    encoded.push_back(kHex[byte >> 4]);
    encoded.push_back(kHex[byte & 0x0F]);
  }
  return encoded;
}

}  // namespace

YuanRongStoreBackend::YuanRongStoreBackend(YuanRongStoreConfig config)
    : config_(std::move(config)) {}

YuanRongStoreBackend::~YuanRongStoreBackend() {
  if (bridge_client_ != nullptr) {
    bridge_destroy_(bridge_client_);
    bridge_client_ = nullptr;
  }
  if (bridge_handle_ != nullptr) {
    dlclose(bridge_handle_);
    bridge_handle_ = nullptr;
  }
}

bool YuanRongStoreBackend::initialize() { return ensure_initialized(); }

std::vector<uint8_t> YuanRongStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<StoreBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (keys.size() != buffers.size()) {
    LOG(ERROR) << "YuanRong store put requires one buffer per key, keys="
               << keys.size() << ", buffers=" << buffers.size();
    return statuses;
  }
  if (keys.empty() || !ensure_initialized()) {
    return statuses;
  }

  // The bridge takes arrays of ranges, so the converted size vectors must
  // outlive the call.
  std::vector<std::vector<uint64_t>> size_storage;
  size_storage.reserve(buffers.size());
  for (const StoreBuffer& buffer : buffers) {
    if (buffer.addresses.size() != buffer.sizes.size()) {
      LOG(ERROR) << "YuanRong store put buffer has mismatched range arrays";
      return statuses;
    }
    std::vector<uint64_t> sizes;
    sizes.reserve(buffer.sizes.size());
    for (size_t size : buffer.sizes) {
      sizes.emplace_back(static_cast<uint64_t>(size));
    }
    size_storage.emplace_back(std::move(sizes));
  }

  std::vector<BridgeBuffer> bridge_buffers;
  bridge_buffers.reserve(buffers.size());
  for (size_t index = 0; index < buffers.size(); ++index) {
    bridge_buffers.push_back(BridgeBuffer{buffers[index].addresses.data(),
                                          size_storage[index].data(),
                                          size_storage[index].size()});
  }

  std::vector<std::string> encoded_keys;
  encoded_keys.reserve(keys.size());
  for (const std::string& key : keys) {
    encoded_keys.emplace_back(encode_key(key));
  }
  std::vector<const char*> raw_keys;
  raw_keys.reserve(encoded_keys.size());
  for (const std::string& key : encoded_keys) {
    raw_keys.push_back(key.c_str());
  }

  if (!bridge_mset_(bridge_client_,
                    raw_keys.data(),
                    bridge_buffers.data(),
                    raw_keys.size())) {
    LOG(ERROR) << "YuanRong store MSet failed: " << last_error();
    return statuses;
  }
  return std::vector<uint8_t>(keys.size(), /*value=*/1);
}

bool YuanRongStoreBackend::get(const std::string& key,
                               const StoreBuffer& buffer) {
  if (buffer.addresses.size() != buffer.sizes.size()) {
    LOG(ERROR) << "YuanRong store get buffer has mismatched range arrays";
    return false;
  }
  if (!ensure_initialized()) {
    return false;
  }

  std::vector<uint64_t> sizes;
  sizes.reserve(buffer.sizes.size());
  for (size_t size : buffer.sizes) {
    sizes.emplace_back(static_cast<uint64_t>(size));
  }
  BridgeBuffer bridge_buffer{
      buffer.addresses.data(), sizes.data(), sizes.size()};

  const std::string encoded_key = encode_key(key);
  const char* raw_key = encoded_key.c_str();
  if (!bridge_mget_(
          bridge_client_, &raw_key, &bridge_buffer, 1, config_.timeout_ms)) {
    LOG(ERROR) << "YuanRong store MGet failed for key " << key << ": "
               << last_error();
    return false;
  }
  return true;
}

const char* YuanRongStoreBackend::last_error() const {
  if (bridge_last_error_ == nullptr) {
    return "bridge not loaded";
  }
  const char* message = bridge_last_error_(bridge_client_);
  return message == nullptr ? "" : message;
}

bool YuanRongStoreBackend::load_bridge() {
  if (bridge_handle_ != nullptr) {
    return true;
  }
  const std::string library = config_.bridge_path.empty()
                                  ? std::string(kDefaultBridgeLibrary)
                                  : config_.bridge_path;
  bridge_handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (bridge_handle_ == nullptr) {
    LOG(ERROR) << "Failed to load the YuanRong store bridge " << library
               << " (set store_yuanrong_bridge_path to override): "
               << dlerror();
    return false;
  }

  auto load_symbol = [this](const char* name) -> void* {
    void* symbol = dlsym(bridge_handle_, name);
    if (symbol == nullptr) {
      LOG(ERROR) << "YuanRong store bridge is missing symbol " << name;
    }
    return symbol;
  };

  bridge_create_ =
      reinterpret_cast<BridgeCreateFn>(load_symbol("xllm_yr_store_create"));
  bridge_destroy_ =
      reinterpret_cast<BridgeDestroyFn>(load_symbol("xllm_yr_store_destroy"));
  bridge_mset_ =
      reinterpret_cast<BridgeMSetFn>(load_symbol("xllm_yr_store_mset"));
  bridge_mget_ =
      reinterpret_cast<BridgeMGetFn>(load_symbol("xllm_yr_store_mget"));
  bridge_last_error_ = reinterpret_cast<BridgeLastErrorFn>(
      load_symbol("xllm_yr_store_last_error"));
  return bridge_create_ != nullptr && bridge_destroy_ != nullptr &&
         bridge_mset_ != nullptr && bridge_mget_ != nullptr &&
         bridge_last_error_ != nullptr;
}

bool YuanRongStoreBackend::ensure_initialized() {
  if (bridge_client_ != nullptr) {
    return true;
  }
  if (!load_bridge()) {
    return false;
  }
  CHECK_GE(config_.port, 0);
  CHECK_LE(config_.port, std::numeric_limits<uint16_t>::max());
  bridge_client_ =
      bridge_create_(config_.host.c_str(), config_.port, config_.timeout_ms);
  if (bridge_client_ == nullptr) {
    LOG(ERROR) << "YuanRong store client create failed for " << config_.host
               << ":" << config_.port << ": " << last_error();
    return false;
  }
  LOG(INFO) << "YuanRong store backend ready, worker=" << config_.host << ":"
            << config_.port;
  return true;
}

}  // namespace xllm
