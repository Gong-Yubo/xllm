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

#include "core/distributed_runtime/store_options.h"

#include <gtest/gtest.h>

#include <string>

namespace xllm {
namespace {

// A master that forgets one of these fields starts the worker with the
// runtime::Options default ("mooncake", store disabled) instead of what the
// CLI asked for. That failure is silent at startup, so guard the whole copy.
TEST(StoreOptionsTest, CopiesEveryStoreOptionToEngineOptions) {
  Options cli;
  cli.enable_kvcache_store(true)
      .store_backend("yuanrong")
      .store_protocol("rdma")
      .store_rdma_devices("mlx5_0")
      .store_master_server_address("10.0.0.1:31201")
      .store_metadata_server("etcd://10.0.0.2:2379")
      .store_local_hostname("10.0.0.3")
      .store_yuanrong_host("10.0.0.4")
      .store_yuanrong_port(31501)
      .store_yuanrong_timeout_ms(750)
      .store_yuanrong_bridge_path("/opt/xllm/lib/libbridge.so")
      .prefetch_batch_size(8)
      .prefetch_timeout(4321)
      .layers_wise_copy_batchs(6)
      .kv_cache_dtype("bfloat16");

  runtime::Options engine;
  apply_store_options(&engine, cli);

  EXPECT_TRUE(engine.enable_kvcache_store());
  EXPECT_EQ(engine.store_backend(), "yuanrong");
  EXPECT_EQ(engine.store_protocol(), "rdma");
  EXPECT_EQ(engine.store_rdma_devices(), "mlx5_0");
  EXPECT_EQ(engine.store_master_server_address(), "10.0.0.1:31201");
  EXPECT_EQ(engine.store_metadata_server(), "etcd://10.0.0.2:2379");
  EXPECT_EQ(engine.store_local_hostname(), "10.0.0.3");
  EXPECT_EQ(engine.store_yuanrong_host(), "10.0.0.4");
  EXPECT_EQ(engine.store_yuanrong_port(), 31501);
  EXPECT_EQ(engine.store_yuanrong_timeout_ms(), 750);
  EXPECT_EQ(engine.store_yuanrong_bridge_path(), "/opt/xllm/lib/libbridge.so");
  EXPECT_EQ(engine.prefetch_batch_size(), 8U);
  EXPECT_EQ(engine.prefetch_timeout(), 4321U);
  EXPECT_EQ(engine.layers_wise_copy_batchs(), 6U);
  EXPECT_EQ(engine.kv_cache_dtype(), "bfloat16");
}

TEST(StoreOptionsTest, KeepsTheCliDefaults) {
  const Options cli{};
  runtime::Options engine;
  apply_store_options(&engine, cli);

  EXPECT_EQ(engine.store_backend(), cli.store_backend());
  EXPECT_EQ(engine.store_yuanrong_host(), cli.store_yuanrong_host());
  EXPECT_EQ(engine.store_yuanrong_port(), cli.store_yuanrong_port());
  EXPECT_EQ(engine.store_yuanrong_timeout_ms(),
            cli.store_yuanrong_timeout_ms());
  EXPECT_EQ(engine.store_yuanrong_bridge_path(),
            cli.store_yuanrong_bridge_path());
  EXPECT_FALSE(engine.enable_kvcache_store());
}

TEST(StoreOptionsTest, IgnoresANullEngineOptions) {
  const Options cli{};
  // Guards the call shape used by masters that build engine options
  // conditionally; must not dereference the pointer.
  EXPECT_NO_FATAL_FAILURE(apply_store_options(nullptr, cli));
}

}  // namespace
}  // namespace xllm
