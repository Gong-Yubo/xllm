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

// Stand-in for the Mooncake backend. The batching tests inject their own
// KVCacheStoreBackend and never call init(), so this only has to satisfy the
// Mooncake branch that the production TU compiles unconditionally.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_store_backend.h"

namespace xllm {

struct MooncakeStoreBackendConfig {
  std::string localhost_name;
  std::string protocol;
  std::string rdma_devices;
  std::string metadata_server;
  std::string master_server_address;
  int32_t replica_num = 1;
  size_t global_segment_size = 0;
};

class MooncakeStoreBackend final : public KVCacheStoreBackend {
 public:
  ~MooncakeStoreBackend() override = default;

  bool init(const MooncakeStoreBackendConfig& /*config*/,
            const std::vector<StoreRange>& /*ranges*/) {
    return false;
  }

  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& /*buffers*/) override {
    return std::vector<uint8_t>(keys.size(), 0);
  }

  bool get(const std::string& /*key*/, const StoreBuffer& /*buffer*/) override {
    return false;
  }
};

}  // namespace xllm
