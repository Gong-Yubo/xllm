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

#include <chrono>
#include <cstring>
#include <string>
#include <utility>

#include "core/kv_cache/transfer/store_config_validation.h"

namespace xllm {
namespace {

constexpr char kDefaultBridgeLibrary[] = "libxllm_yuanrong_store_bridge.so";

// DataSystem rejects non-ASCII object keys. The 0.9.3 SDK header documents
// letters, digits and ~!@#$%^&*.-_ with a 1024-byte cap (kv_client.h, Create),
// and the 9.9.9 worker applies a stricter charset again. KVCacheStore keys are
// "xllm-kv-v3:<...>" + 32 raw hash bytes (16 schema + 16 block content), so
// they would be rejected. Hex-encoding the whole key leaves only [0-9a-f], a
// subset of every documented charset; it is injective and is applied to put
// and get alike, so round trips are unaffected.
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

// Records one put/get at the backend boundary. The v4 logs had no way to tell
// a not-found from a transport error, or to see the in-process latency
// distribution; both are needed to interpret the next 32K re-test.
class ScopedStoreCall {
 public:
  ScopedStoreCall(StoreCallMetrics* metrics, const char* op, size_t keys)
      : metrics_(metrics),
        op_(op),
        keys_(keys),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopedStoreCall() {
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_)
            .count();
    const StoreFailureKind kind =
        success_ ? StoreFailureKind::kNone : classify_store_failure(error_);
    const bool report = metrics_->record(success_, elapsed_us, kind);
    VLOG(2) << "[KVCacheStore][YuanRong] op=" << op_ << " keys=" << keys_
            << " success=" << success_ << " elapsed_us=" << elapsed_us
            << " error=" << (success_ ? "" : error_);
    if (!report) {
      return;
    }
    const auto snapshot = metrics_->snapshot();
    VLOG(1) << "[KVCacheStore][YuanRong] metrics calls=" << snapshot.calls
            << " success=" << snapshot.successes
            << " failed=" << snapshot.failures << " p50_us=" << snapshot.p50_us
            << " p99_us=" << snapshot.p99_us << " max_us=" << snapshot.max_us
            << " samples=" << snapshot.samples << " not_found="
            << snapshot.failures_by_kind[static_cast<size_t>(
                   StoreFailureKind::kKeyNotFound)]
            << " timeout="
            << snapshot.failures_by_kind[static_cast<size_t>(
                   StoreFailureKind::kTimeout)]
            << " transport="
            << snapshot.failures_by_kind[static_cast<size_t>(
                   StoreFailureKind::kTransport)]
            << " sdk_other="
            << snapshot.failures_by_kind[static_cast<size_t>(
                   StoreFailureKind::kSdkOther)];
  }

  void succeed() { success_ = true; }
  void fail(const char* error) {
    error_ =
        error == nullptr || error[0] == '\0' ? "unknown store error" : error;
  }

 private:
  StoreCallMetrics* metrics_;
  const char* op_;
  size_t keys_;
  std::chrono::steady_clock::time_point start_;
  bool success_ = false;
  std::string error_ = "store call did not complete";
};

}  // namespace

YuanRongStoreBackend::YuanRongStoreBackend(YuanRongStoreConfig config)
    : config_(std::move(config)) {}

YuanRongStoreBackend::~YuanRongStoreBackend() {
  std::lock_guard<std::mutex> lock(bridge_mutex_);
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

bool YuanRongStoreBackend::marshal_buffers(
    const std::vector<StoreBuffer>& buffers,
    std::vector<std::vector<uint64_t>>* size_storage,
    std::vector<BridgeBuffer>* bridge_buffers) {
  // reserve() before emplace_back() keeps the inner vectors' data pointers
  // stable while bridge_buffers is being filled.
  size_storage->reserve(buffers.size());
  for (const StoreBuffer& buffer : buffers) {
    if (buffer.addresses.size() != buffer.sizes.size()) {
      return false;
    }
    std::vector<uint64_t> sizes;
    sizes.reserve(buffer.sizes.size());
    for (size_t size : buffer.sizes) {
      sizes.emplace_back(static_cast<uint64_t>(size));
    }
    size_storage->emplace_back(std::move(sizes));
  }
  bridge_buffers->reserve(buffers.size());
  for (size_t index = 0; index < buffers.size(); ++index) {
    bridge_buffers->push_back(BridgeBuffer{buffers[index].addresses.data(),
                                           (*size_storage)[index].data(),
                                           (*size_storage)[index].size()});
  }
  return true;
}

void YuanRongStoreBackend::encode_keys(const std::vector<std::string>& keys,
                                       std::vector<std::string>* encoded,
                                       std::vector<const char*>* raw) {
  encoded->reserve(keys.size());
  for (const std::string& key : keys) {
    encoded->emplace_back(encode_key(key));
  }
  raw->reserve(encoded->size());
  for (const std::string& key : *encoded) {
    raw->push_back(key.c_str());
  }
}

std::vector<uint8_t> YuanRongStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<StoreBuffer>& buffers) {
  ScopedStoreCall call(&metrics_, "MSet", keys.size());
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (keys.size() != buffers.size()) {
    LOG(ERROR) << "YuanRong store put requires one buffer per key, keys="
               << keys.size() << ", buffers=" << buffers.size();
    call.fail("put key/buffer count mismatch");
    return statuses;
  }
  if (keys.empty()) {
    call.succeed();
    return statuses;
  }
  if (!ensure_initialized()) {
    call.fail("bridge not loaded");
    return statuses;
  }

  std::vector<std::vector<uint64_t>> size_storage;
  std::vector<BridgeBuffer> bridge_buffers;
  if (!marshal_buffers(buffers, &size_storage, &bridge_buffers)) {
    LOG(ERROR) << "YuanRong store put buffer has mismatched range arrays";
    call.fail("put buffer has mismatched range arrays");
    return statuses;
  }

  std::vector<std::string> encoded_keys;
  std::vector<const char*> raw_keys;
  encode_keys(keys, &encoded_keys, &raw_keys);

  std::lock_guard<std::mutex> lock(bridge_mutex_);
  if (!bridge_mset_(bridge_client_,
                    raw_keys.data(),
                    bridge_buffers.data(),
                    raw_keys.size())) {
    const char* error = last_error_locked();
    LOG(ERROR) << "YuanRong store MSet failed: " << error;
    call.fail(error);
    return statuses;
  }
  call.succeed();
  return std::vector<uint8_t>(keys.size(), /*value=*/1);
}

std::vector<uint8_t> YuanRongStoreBackend::batch_get(
    const std::vector<std::string>& keys,
    const std::vector<StoreBuffer>& buffers) {
  ScopedStoreCall call(&metrics_, "MGet", keys.size());
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (keys.size() != buffers.size()) {
    LOG(ERROR) << "YuanRong store get requires one buffer per key, keys="
               << keys.size() << ", buffers=" << buffers.size();
    call.fail("get key/buffer count mismatch");
    return statuses;
  }
  if (keys.empty()) {
    call.succeed();
    return statuses;
  }
  if (!ensure_initialized()) {
    call.fail("bridge not loaded");
    return statuses;
  }

  std::vector<std::vector<uint64_t>> size_storage;
  std::vector<BridgeBuffer> bridge_buffers;
  if (!marshal_buffers(buffers, &size_storage, &bridge_buffers)) {
    LOG(ERROR) << "YuanRong store get buffer has mismatched range arrays";
    call.fail("get buffer has mismatched range arrays");
    return statuses;
  }

  std::vector<std::string> encoded_keys;
  std::vector<const char*> raw_keys;
  encode_keys(keys, &encoded_keys, &raw_keys);

  // One SDK Get for the whole prefetch. The bridge fills one status byte per
  // key: a partial hit is still a successful call, and a missing object only
  // clears its own byte.
  std::lock_guard<std::mutex> lock(bridge_mutex_);
  if (!bridge_mget_(bridge_client_,
                    raw_keys.data(),
                    bridge_buffers.data(),
                    statuses.data(),
                    raw_keys.size())) {
    const char* error = last_error_locked();
    LOG(ERROR) << "YuanRong store MGet failed for " << keys.size()
               << " keys: " << error;
    call.fail(error);
    return std::vector<uint8_t>(keys.size(), /*value=*/0);
  }
  call.succeed();
  return statuses;
}

bool YuanRongStoreBackend::get(const std::string& key,
                               const StoreBuffer& buffer) {
  const std::vector<uint8_t> statuses = batch_get({key}, {buffer});
  return !statuses.empty() && statuses[0] != 0;
}

const char* YuanRongStoreBackend::last_error_locked() const {
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
  // Lazy initialization is reachable from every prefetch thread, so the
  // dlopen, symbol resolution and client creation share the call mutex.
  std::lock_guard<std::mutex> lock(bridge_mutex_);
  if (bridge_client_ != nullptr) {
    return true;
  }
  if (!load_bridge()) {
    return false;
  }
  if (!is_valid_yuanrong_port(config_.port)) {
    LOG(ERROR) << "Invalid store_yuanrong_port=" << config_.port
               << "; expected 0..65535 (0 lets the SDK choose its default).";
    return false;
  }
  bridge_client_ =
      bridge_create_(config_.host.c_str(), config_.port, config_.timeout_ms);
  if (bridge_client_ == nullptr) {
    LOG(ERROR) << "YuanRong store client create failed for " << config_.host
               << ":" << config_.port << ": " << last_error_locked();
    return false;
  }
  LOG(INFO) << "YuanRong store backend ready, worker=" << config_.host << ":"
            << config_.port;
  return true;
}

}  // namespace xllm
