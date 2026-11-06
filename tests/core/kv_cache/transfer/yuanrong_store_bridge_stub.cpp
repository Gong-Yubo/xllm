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

// Test double for libxllm_yuanrong_store_bridge.so.
//
// Implements the same xllm_yr_store_* ABI over an in-process map, so
// YuanRongStoreBackend's range marshalling, key handling and status mapping are
// exercised for real without a YuanRong cluster. Only the service behind the
// ABI is replaced; the backend under test is the production code.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>

extern "C" {

// Mirrors XllmYrStoreBuffer in yuanrong_store_bridge.cpp.
struct XllmYrStoreBuffer {
  void* const* addresses;
  const uint64_t* sizes;
  size_t count;
};

}  // extern "C"

namespace {

struct StubStore {
  std::map<std::string, std::string> objects;
  std::string last_error;
};

// Creation has no store to report to, so it keeps its own error.
thread_local std::string g_last_error;

bool set_error(StubStore* store, const std::string& message) {
  if (store != nullptr) {
    store->last_error = message;
  }
  g_last_error = message;
  return false;
}

bool flatten(const XllmYrStoreBuffer& buffer, std::string* out) {
  size_t total = 0;
  for (size_t index = 0; index < buffer.count; ++index) {
    total += static_cast<size_t>(buffer.sizes[index]);
  }
  out->assign(total, '\0');
  size_t offset = 0;
  for (size_t index = 0; index < buffer.count; ++index) {
    const size_t size = static_cast<size_t>(buffer.sizes[index]);
    if (size == 0) {
      continue;
    }
    std::memcpy(out->data() + offset, buffer.addresses[index], size);
    offset += size;
  }
  return true;
}

// DataSystem (>= 9.9.9) rejects keys that contain bytes outside this set, so
// the stub enforces the same constraint: a regression in the backend's key
// encoding fails here instead of only on a real worker.
bool is_legal_key(const std::string& key) {
  for (unsigned char byte : key) {
    const bool legal =
        (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= '0' && byte <= '9') || byte == '.' || byte == '-' ||
        byte == '_' || byte == '!' || byte == '@' || byte == '#' ||
        byte == '%' || byte == '^' || byte == '*' || byte == '(' ||
        byte == ')' || byte == '+' || byte == '=' || byte == ':' || byte == ';';
    if (!legal) {
      return false;
    }
  }
  return true;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void* xllm_yr_store_create(
    const char* /*host*/,
    int32_t /*port*/,
    int32_t /*timeout_ms*/) {
  return new StubStore();
}

extern "C" __attribute__((visibility("default"))) void xllm_yr_store_destroy(
    void* opaque) {
  delete static_cast<StubStore*>(opaque);
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mset(
    void* opaque,
    const char* const* keys,
    const XllmYrStoreBuffer* buffers,
    size_t count) {
  auto* store = static_cast<StubStore*>(opaque);
  if (store == nullptr) {
    return set_error(store, "stub client is null");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!is_legal_key(keys[index])) {
      return set_error(store, "illegal char(s)");
    }
    std::string payload;
    flatten(buffers[index], &payload);
    // ExistenceOpt::NX, as the production bridge asks for: an existing key is
    // left untouched and the write still counts as success.
    store->objects.emplace(keys[index], std::move(payload));
  }
  store->last_error.clear();
  return true;
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mget(
    void* opaque,
    const char* const* keys,
    XllmYrStoreBuffer* buffers,
    size_t count,
    int32_t /*timeout_ms*/) {
  auto* store = static_cast<StubStore*>(opaque);
  if (store == nullptr) {
    return set_error(store, "stub client is null");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!is_legal_key(keys[index])) {
      return set_error(store, "illegal char(s)");
    }
    const auto found = store->objects.find(keys[index]);
    if (found == store->objects.end()) {
      return set_error(store, "K_NOT_FOUND");
    }
    const std::string& payload = found->second;
    size_t expected = 0;
    for (size_t range = 0; range < buffers[index].count; ++range) {
      expected += static_cast<size_t>(buffers[index].sizes[range]);
    }
    if (payload.size() != expected) {
      return set_error(store, "object size mismatch");
    }
    size_t offset = 0;
    for (size_t range = 0; range < buffers[index].count; ++range) {
      const size_t size = static_cast<size_t>(buffers[index].sizes[range]);
      if (size == 0) {
        continue;
      }
      std::memcpy(
          buffers[index].addresses[range], payload.data() + offset, size);
      offset += size;
    }
  }
  store->last_error.clear();
  return true;
}

extern "C" __attribute__((visibility("default"))) const char*
xllm_yr_store_last_error(void* opaque) {
  auto* store = static_cast<StubStore*>(opaque);
  if (store != nullptr && !store->last_error.empty()) {
    return store->last_error.c_str();
  }
  return g_last_error.c_str();
}
