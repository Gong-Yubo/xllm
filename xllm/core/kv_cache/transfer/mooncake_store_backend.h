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

#pragma push_macro("BLOCK_SIZE")
#include <Mooncake/mooncake-store/include/real_client.h>
#pragma pop_macro("BLOCK_SIZE")

#include <cstdint>
#include <memory>
#include <optional>
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
};

class MooncakeStoreBackend final : public KVCacheStoreBackend {
 public:
  MooncakeStoreBackend() = default;
  ~MooncakeStoreBackend() override;

  bool init(const MooncakeStoreBackendConfig& config,
            const std::vector<StoreRange>& ranges);

  bool register_ranges(const std::vector<StoreRange>& ranges);

  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) override;

  bool get(const std::string& key, const StoreBuffer& buffer) override;

 private:
  friend class MooncakeStoreBackendTestPeer;

  static std::optional<std::vector<StoreRange>> unique_ranges(
      const std::vector<StoreRange>& ranges);
  static bool get_succeeded(int64_t expected_bytes,
                            const std::vector<int>& results);
  static bool put_succeeded(int result);
  void unregister_ranges();

 private:
  mooncake::ReplicateConfig rep_config_;
  std::shared_ptr<mooncake::RealClient> client_ptr_;
  std::vector<StoreRange> registered_ranges_;
};

}  // namespace xllm
