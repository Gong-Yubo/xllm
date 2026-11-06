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

// Pure C ABI bridge over the YuanRong KV client. The SDK itself ships under the
// `datasystem` name, which is why the header, namespace and library keep it.
//
// Only this shared object links libdatasystem. xLLM loads it at runtime and
// talks to it through the xllm_yr_store_* symbols below, so the main binary
// keeps zero YuanRong/abseil/grpc symbol dependency.
//
// Everything here is host memory: KVCacheStore's offload cache lives on the
// CPU, and the object KV client copies host buffers into the store. The device
// oriented MSetD2H/MGetH2D pair is intentionally unused — it would interpret
// a host pointer as device memory.

#include <datasystem/kv_client.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

// One logical object's payload: non-contiguous host ranges concatenated in
// order.
struct XllmYrStoreBuffer {
  void* const* addresses;
  const uint64_t* sizes;
  size_t count;
};

}  // extern "C"

namespace {

struct StoreClient {
  std::unique_ptr<datasystem::KVClient> client;
  std::string last_error;
};

thread_local std::string g_last_error;

bool set_error(StoreClient* store, const std::string& error) {
  if (store != nullptr) {
    store->last_error = error;
  }
  g_last_error = error;
  return false;
}

// Copies one payload into a contiguous staging buffer. YuanRong's batch API
// takes a single value per key, while a KV block is split across per-role host
// tensors, so the ranges must be joined before the call.
std::string flatten(const XllmYrStoreBuffer& buffer) {
  uint64_t total = 0;
  for (size_t index = 0; index < buffer.count; ++index) {
    if (buffer.sizes[index] > UINT64_MAX - total) {
      throw std::length_error("YuanRong store buffer size overflow");
    }
    total += buffer.sizes[index];
  }
  std::string flat(static_cast<size_t>(total), '\0');
  size_t offset = 0;
  for (size_t index = 0; index < buffer.count; ++index) {
    const uint64_t size = buffer.sizes[index];
    if (size == 0) {
      continue;
    }
    if (buffer.addresses[index] == nullptr) {
      throw std::invalid_argument("YuanRong store buffer has a null range");
    }
    std::memcpy(flat.data() + offset, buffer.addresses[index], size);
    offset += static_cast<size_t>(size);
  }
  if (offset != flat.size()) {
    throw std::runtime_error("YuanRong store buffer flatten mismatch");
  }
  return flat;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void*
xllm_yr_store_create(const char* host, int32_t port, int32_t timeout_ms) {
  try {
    auto store = std::make_unique<StoreClient>();
    datasystem::ConnectOptions options;
    options.host = host == nullptr || host[0] == '\0' ? "127.0.0.1" : host;
    options.port = port;
    options.connectTimeoutMs = timeout_ms;
    options.requestTimeoutMs = timeout_ms;
    store->client = std::make_unique<datasystem::KVClient>(options);
    const auto status = store->client->Init();
    if (status.IsError()) {
      g_last_error = status.ToString();
      return nullptr;
    }
    return store.release();
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) void xllm_yr_store_destroy(
    void* opaque) {
  auto* store = reinterpret_cast<StoreClient*>(opaque);
  if (store == nullptr) {
    return;
  }
  if (store->client != nullptr) {
    store->client->ShutDown();
  }
  delete store;
}

// Writes one object per key. On success every key was stored; on failure the
// caller treats the whole batch as failed, which is what KVCacheStore expects
// (it aggregates per-key statuses itself from the per-call return).
extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mset(
    void* opaque,
    const char* const* keys,
    const XllmYrStoreBuffer* buffers,
    size_t count) {
  auto* store = reinterpret_cast<StoreClient*>(opaque);
  if (store == nullptr || store->client == nullptr) {
    return set_error(store, "YuanRong store client is null");
  }
  if (keys == nullptr || buffers == nullptr) {
    return set_error(store, "YuanRong store batch has null input");
  }

  try {
    std::vector<std::string> key_vec;
    std::vector<std::string> payload_vec;
    key_vec.reserve(count);
    payload_vec.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      if (keys[index] == nullptr) {
        return set_error(store, "YuanRong store batch has a null key");
      }
      key_vec.emplace_back(keys[index]);
      payload_vec.emplace_back(flatten(buffers[index]));
    }
    std::vector<datasystem::StringView> values;
    values.reserve(payload_vec.size());
    for (const std::string& payload : payload_vec) {
      values.emplace_back(payload);
    }

    datasystem::MSetParam param;
    // KV blocks are one-shot content; keep them out of the shared L2 cache so
    // they do not evict reusable objects.
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    // NX leaves an existing object alone and still reports success, so
    // re-publishing a block costs nothing. Keys are content hashes, so the
    // duplicate would carry the same bytes anyway.
    param.existence = datasystem::ExistenceOpt::NX;
    std::vector<std::string> failed_keys;
    const auto status =
        store->client->MSet(key_vec, values, failed_keys, param);
    if (status.IsError() || !failed_keys.empty()) {
      return set_error(
          store,
          status.ToString() + ", failed=" + std::to_string(failed_keys.size()));
    }
    store->last_error.clear();
    return true;
  } catch (const std::exception& error) {
    return set_error(store, error.what());
  }
}

// Reads one object per key into the caller's host ranges, in place.
extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mget(
    void* opaque,
    const char* const* keys,
    XllmYrStoreBuffer* buffers,
    size_t count,
    int32_t timeout_ms) {
  auto* store = reinterpret_cast<StoreClient*>(opaque);
  if (store == nullptr || store->client == nullptr) {
    return set_error(store, "YuanRong store client is null");
  }
  if (keys == nullptr || buffers == nullptr) {
    return set_error(store, "YuanRong store batch has null input");
  }

  try {
    std::vector<std::string> key_vec;
    key_vec.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      if (keys[index] == nullptr) {
        return set_error(store, "YuanRong store batch has a null key");
      }
      key_vec.emplace_back(keys[index]);
    }

    std::vector<std::string> payload_vec(count);
    const auto status = store->client->Get(key_vec, payload_vec, timeout_ms);
    if (status.IsError()) {
      return set_error(store, status.ToString());
    }

    for (size_t index = 0; index < count; ++index) {
      const std::string& payload = payload_vec[index];
      if (payload.empty()) {
        return set_error(store,
                         "YuanRong store returned an empty object for key " +
                             key_vec[index]);
      }
      const XllmYrStoreBuffer& buffer = buffers[index];
      uint64_t expected = 0;
      for (size_t range = 0; range < buffer.count; ++range) {
        expected += buffer.sizes[range];
      }
      if (payload.size() != static_cast<size_t>(expected)) {
        return set_error(
            store,
            "YuanRong store object size mismatch for key " + key_vec[index]);
      }
      size_t offset = 0;
      for (size_t range = 0; range < buffer.count; ++range) {
        const uint64_t size = buffer.sizes[range];
        if (size == 0) {
          continue;
        }
        if (buffer.addresses[range] == nullptr) {
          return set_error(store, "YuanRong store buffer has a null range");
        }
        std::memcpy(buffer.addresses[range], payload.data() + offset, size);
        offset += static_cast<size_t>(size);
      }
    }
    store->last_error.clear();
    return true;
  } catch (const std::exception& error) {
    return set_error(store, error.what());
  }
}

extern "C" __attribute__((visibility("default"))) const char*
xllm_yr_store_last_error(void* opaque) {
  auto* store = reinterpret_cast<StoreClient*>(opaque);
  if (store != nullptr && !store->last_error.empty()) {
    return store->last_error.c_str();
  }
  return g_last_error.c_str();
}
