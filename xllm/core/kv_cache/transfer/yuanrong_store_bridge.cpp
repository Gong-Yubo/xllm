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
//
// Threading contract: one client returned by xllm_yr_store_create owns a
// single KVClient plus a shared last-error string, and is NOT safe to call
// concurrently. YuanRongStoreBackend serializes every call with its own
// mutex; any other caller of this C ABI must do the same.

#include <datasystem/kv_client.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/kv_cache/transfer/store_config_validation.h"

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

// Establishing a connection is a one-off start-up cost, while a store read is
// on the inference critical path. The two must not share one timeout: a cache
// miss has to fail fast instead of holding a prefetch thread. SDK defaults are
// 9 s for connect and 0 (unbounded) for requests.
constexpr int32_t kConnectTimeoutMs = 10000;

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

// `request_timeout_ms` bounds a single store request. It is NOT a per-call
// "wait for the object to become ready" argument: DataSystem 9.9.9 dropped
// that third Get parameter, and the 0.9.3 default (0) means waiting without
// a deadline. A non-positive value would therefore reintroduce the
// minute-long stalls seen in the v4 test, and so would a very large positive
// one (the v3 deployment scripts used 300000). Clamp both ends to the shared
// [1, kMaxStoreRequestTimeoutMs] policy.
extern "C" __attribute__((visibility("default"))) void* xllm_yr_store_create(
    const char* host,
    int32_t port,
    int32_t request_timeout_ms) {
  try {
    auto store = std::make_unique<StoreClient>();
    datasystem::ConnectOptions options;
    options.host = host == nullptr || host[0] == '\0' ? "127.0.0.1" : host;
    options.port = port;
    options.connectTimeoutMs = kConnectTimeoutMs;
    options.requestTimeoutMs =
        xllm::normalize_store_request_timeout_ms(request_timeout_ms);
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

// Writes one object per key. A status error or a non-empty failed_keys list
// reports the whole batch as failed: MSet has no per-key status array, so the
// caller conservatively marks every key in this batch as not stored, including
// any keys the SDK wrote before the failure. That direction is safe (no false
// hit), but it is deliberately coarser than the per-key MGet result below.
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
    // NOTE: despite the original intent of "keep KV blocks out of the shared
    // L2 cache", NONE_L2_CACHE_EVICT is an *evictable* write in the 0.9.3
    // headers. It is a suspect for the all-miss behaviour of the v4 32K run.
    // Deliberately unchanged here: switching write modes needs a real-cluster
    // A/B before an assumption is baked into a CLI flag (design decision D5).
    // A 0.9.3 single-worker simulation narrowed that A/B: with the worker's
    // default L2_cache_type=none, WRITE_THROUGH_L2_CACHE, WRITE_BACK_L2_CACHE
    // and WRITE_BACK_L2_CACHE_EVICT are rejected with "Invalid parameter",
    // leaving only NONE_L2_CACHE and NONE_L2_CACHE_EVICT comparable unless the
    // deployment configures an L2 cache first. Do not read this comment as an
    // endorsement of the current mode either way.
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
//
// DataSystem's multi-key Get reports K_OK when at least one key is present:
// a missing key leaves its payload empty. A miss is therefore a per-key
// result, not a failed batch. `statuses` receives one byte per key (1 hit,
// 0 miss); returning false is reserved for a genuine client/SDK error.
extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mget(
    void* opaque,
    const char* const* keys,
    XllmYrStoreBuffer* buffers,
    uint8_t* statuses,
    size_t count) {
  auto* store = reinterpret_cast<StoreClient*>(opaque);
  if (store == nullptr || store->client == nullptr) {
    return set_error(store, "YuanRong store client is null");
  }
  if (keys == nullptr || buffers == nullptr || statuses == nullptr) {
    return set_error(store, "YuanRong store batch has null input");
  }
  std::memset(statuses, 0, count);

  try {
    std::vector<std::string> key_vec;
    key_vec.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      if (keys[index] == nullptr) {
        return set_error(store, "YuanRong store batch has a null key");
      }
      key_vec.emplace_back(keys[index]);
    }

    // Two-argument Get(keys, vals) compiles against both 0.9.3 (third
    // subTimeoutMs parameter has a default) and 9.9.9 (no third parameter).
    // The request deadline comes from ConnectOptions at create time.
    //
    // Pre-sizing is safe under both known SDK behaviours: the 0.9.3 header
    // documents assignment at the key's index, while 0.10.1 GetWithLatch
    // clears the vector and emplaces one value per key. 9.9.9 is still
    // unverified on a real cluster, so reject the append-without-clear case
    // loudly instead of reading the pre-sized empty slots as a full miss.
    std::vector<std::string> payload_vec(count);
    const auto status = store->client->Get(key_vec, payload_vec);
    if (status.IsError()) {
      return set_error(store, status.ToString());
    }
    if (payload_vec.size() > count) {
      return set_error(store,
                       "YuanRong Get returned " +
                           std::to_string(payload_vec.size()) + " values for " +
                           std::to_string(count) +
                           " keys (append semantics are unsupported)");
    }

    const size_t returned = std::min(payload_vec.size(), count);
    for (size_t index = 0; index < count; ++index) {
      // The SDK may return fewer entries than requested; absent entries are
      // misses. An empty payload is the SDK's encoding for a missing key.
      if (index >= returned || payload_vec[index].empty()) {
        continue;
      }
      const std::string& payload = payload_vec[index];
      const XllmYrStoreBuffer& buffer = buffers[index];
      uint64_t expected = 0;
      for (size_t range = 0; range < buffer.count; ++range) {
        expected += buffer.sizes[range];
      }
      if (payload.size() != static_cast<size_t>(expected)) {
        // A size mismatch means this object is not the block we asked for.
        // Treat it as a miss for this key and keep the other hits.
        continue;
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
      statuses[index] = 1;
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
