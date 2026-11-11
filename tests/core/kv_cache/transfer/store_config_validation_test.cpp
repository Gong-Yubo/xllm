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

#include "core/kv_cache/transfer/store_config_validation.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

TEST(StoreConfigValidationTest, AcceptsOnlyImplementedBackends) {
  EXPECT_TRUE(is_supported_store_backend("mooncake"));
  EXPECT_TRUE(is_supported_store_backend("yuanrong"));

  // A typo used to fall through to the Mooncake path and start the wrong
  // store, so near-misses are the interesting cases here.
  EXPECT_FALSE(is_supported_store_backend("yuanrongg"));
  EXPECT_FALSE(is_supported_store_backend("YuanRong"));
  EXPECT_FALSE(is_supported_store_backend("mooncake "));
  EXPECT_FALSE(is_supported_store_backend(""));
  EXPECT_FALSE(is_supported_store_backend("hdfss"));
}

TEST(StoreConfigValidationTest, AcceptsSdkDefaultPortButRejectsOutOfRange) {
  EXPECT_TRUE(is_valid_yuanrong_port(0));
  EXPECT_TRUE(is_valid_yuanrong_port(31501));
  EXPECT_TRUE(is_valid_yuanrong_port(65535));

  EXPECT_FALSE(is_valid_yuanrong_port(-1));
  EXPECT_FALSE(is_valid_yuanrong_port(65536));
  EXPECT_FALSE(is_valid_yuanrong_port(std::numeric_limits<int32_t>::min()));
  EXPECT_FALSE(is_valid_yuanrong_port(std::numeric_limits<int32_t>::max()));
}

TEST(StoreConfigValidationTest, ClampsNonPositiveRequestTimeouts) {
  static_assert(normalize_store_request_timeout_ms(0) ==
                kDefaultStoreRequestTimeoutMs);

  // 0 is the SDK's "no deadline" default. The v4 run stalled for a minute
  // when it leaked into the per-call Get timeout, so the bridge must clamp it
  // (and anything else non-positive) back to the bounded CLI default.
  EXPECT_EQ(normalize_store_request_timeout_ms(0),
            kDefaultStoreRequestTimeoutMs);
  EXPECT_EQ(normalize_store_request_timeout_ms(-1),
            kDefaultStoreRequestTimeoutMs);
  EXPECT_EQ(
      normalize_store_request_timeout_ms(std::numeric_limits<int32_t>::min()),
      kDefaultStoreRequestTimeoutMs);

  // Positive values are deliberate user choices and pass through untouched.
  EXPECT_EQ(normalize_store_request_timeout_ms(1), 1);
  EXPECT_EQ(normalize_store_request_timeout_ms(1234), 1234);
  EXPECT_EQ(
      normalize_store_request_timeout_ms(std::numeric_limits<int32_t>::max()),
      std::numeric_limits<int32_t>::max());
}

}  // namespace
}  // namespace xllm
