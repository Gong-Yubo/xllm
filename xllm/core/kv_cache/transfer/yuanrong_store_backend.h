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

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_store_backend.h"
#include "core/kv_cache/transfer/store_config_validation.h"
#include "core/kv_cache/transfer/store_metrics.h"

namespace xllm {

struct YuanRongStoreConfig {
  std::string host = "127.0.0.1";
  int32_t port = 0;
  // Upper bound on one store request, not a connection timeout and not a
  // "wait until the object is ready" deadline. A cache miss must fail fast:
  // the v4 test showed 60 s stalls when this value was also used as the
  // per-call Get sub-timeout. The bridge applies the same normalizer, which
  // also caps oversized values at kMaxStoreRequestTimeoutMs.
  int32_t timeout_ms = kDefaultStoreRequestTimeoutMs;
  // Path to libxllm_yuanrong_store_bridge.so. Empty means the default
  // library name, resolved through the loader path.
  std::string bridge_path;
};

// YuanRong implementation of KVCacheStoreBackend.
//
// KVCacheStore hands this backend host memory ranges (the offload cache lives
// on the CPU) and a key per object. YuanRong keeps those objects in a shared
// host store, so the transfer is a host-buffer put/get: no device pairing and
// no memory registration is involved. The device-side MSetD2H/MGetH2D pair is
// deliberately not used here, because it assumes device pointers and would
// misinterpret a host address.
//
// The SDK stays out of this binary: every call crosses the C ABI bridge
// (xllm_yr_store_*), which is loaded at runtime.
class YuanRongStoreBackend final : public KVCacheStoreBackend {
 public:
  explicit YuanRongStoreBackend(YuanRongStoreConfig config);

  ~YuanRongStoreBackend() override;

  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) override;

  bool get(const std::string& key, const StoreBuffer& buffer) override;

  // DataSystem has a native multi-key Get. Override the default per-key loop
  // so one prefetch costs one SDK call instead of one call per block.
  std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) override;

  // Open the client eagerly so a misconfigured deployment fails at startup
  // instead of on the first put.
  bool initialize();

  // Read-only view of the in-process call statistics, for tests and for
  // re-test harnesses that want p50/p99 without scraping log lines.
  StoreCallMetrics::Snapshot store_metrics() const {
    return metrics_.snapshot();
  }

 private:
  // Mirrors XllmYrStoreBuffer in yuanrong_store_bridge.cpp. Declared here so
  // the dlopen function-pointer types match the ABI exactly instead of being
  // reinterpret_cast through void*.
  struct BridgeBuffer {
    void* const* addresses;
    const uint64_t* sizes;
    size_t count;
  };

  using BridgeCreateFn = void* (*)(const char*, int32_t, int32_t);
  using BridgeDestroyFn = void (*)(void*);
  using BridgeMSetFn = bool (*)(void*,
                                const char* const*,
                                const BridgeBuffer*,
                                size_t);
  // `statuses` is a caller-owned array of `count` bytes: the bridge sets one
  // byte per key (1 hit, 0 miss). A batch with partial hits still returns
  // true, because DataSystem reports K_OK when at least one key is present.
  using BridgeMGetFn =
      bool (*)(void*, const char* const*, BridgeBuffer*, uint8_t*, size_t);
  using BridgeLastErrorFn = const char* (*)(void*);

  // Shared marshalling for put and get. `size_storage` must outlive the bridge
  // call because BridgeBuffer holds pointers into it.
  static bool marshal_buffers(const std::vector<StoreBuffer>& buffers,
                              std::vector<std::vector<uint64_t>>* size_storage,
                              std::vector<BridgeBuffer>* bridge_buffers);
  static void encode_keys(const std::vector<std::string>& keys,
                          std::vector<std::string>* encoded,
                          std::vector<const char*>* raw);

  bool ensure_initialized();
  // Callers must hold bridge_mutex_: the bridge keeps its last error in the
  // shared client, so reading it is part of the serialized call.
  const char* last_error_locked() const;
  bool load_bridge();

  YuanRongStoreConfig config_;

  // WorkerService prefetches on a thread pool, but one backend owns a single
  // DataSystem KVClient and that client is not documented as safe for
  // concurrent Get/MSet. Serialize every bridge call (and the lazy dlopen /
  // client creation) until the E1 concurrency experiment settles whether a
  // per-thread client pays off.
  std::mutex bridge_mutex_;
  mutable StoreCallMetrics metrics_;
  void* bridge_handle_ = nullptr;
  void* bridge_client_ = nullptr;

  BridgeCreateFn bridge_create_ = nullptr;
  BridgeDestroyFn bridge_destroy_ = nullptr;
  BridgeMSetFn bridge_mset_ = nullptr;
  BridgeMGetFn bridge_mget_ = nullptr;
  BridgeLastErrorFn bridge_last_error_ = nullptr;
};

}  // namespace xllm
