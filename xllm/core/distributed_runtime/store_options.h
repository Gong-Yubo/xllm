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

#include "core/common/options.h"
#include "core/runtime/options.h"

namespace xllm {

// Copies every option that configures the host KV cache store from the CLI
// options into the engine options handed to workers.
//
// Every master path (LLM, VLM, VLM speculative) must call this. The master
// options carry the CLI values, but a worker only sees runtime::Options, whose
// defaults are "mooncake"/off. A field omitted here fails silently: the engine
// starts with a different store than the one the user selected. Keeping the
// copy in one place is what makes the three call sites stay in sync when a new
// store flag is added.
//
// host_blocks_factor and the disaggregated-PD transfer options are deliberately
// not here: they also gate host offload eligibility and PD topology, which the
// VLM non-speculative path rejects (see the CHECK in vlm_master.cpp).
inline void apply_store_options(runtime::Options* engine_options,
                                const Options& options) {
  if (engine_options == nullptr) {
    return;
  }
  engine_options->enable_kvcache_store(options.enable_kvcache_store())
      .store_backend(options.store_backend())
      .store_protocol(options.store_protocol())
      .store_rdma_devices(options.store_rdma_devices())
      .store_master_server_address(options.store_master_server_address())
      .store_metadata_server(options.store_metadata_server())
      .store_local_hostname(options.store_local_hostname())
      .store_yuanrong_host(options.store_yuanrong_host())
      .store_yuanrong_port(options.store_yuanrong_port())
      .store_yuanrong_timeout_ms(options.store_yuanrong_timeout_ms())
      .store_yuanrong_bridge_path(options.store_yuanrong_bridge_path())
      .prefetch_batch_size(options.prefetch_batch_size())
      .prefetch_timeout(options.prefetch_timeout())
      .layers_wise_copy_batchs(options.layers_wise_copy_batchs())
      .kv_cache_dtype(options.kv_cache_dtype());
}

}  // namespace xllm
