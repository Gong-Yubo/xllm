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

#include "core/kv_cache/transfer/store_ranges.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace xllm {
namespace {

TEST(StoreRangesTest, SegmentCoversEachRegisteredTensorOnce) {
  std::vector<uint8_t> first(64);
  std::vector<uint8_t> second(32);
  // The same host cache can appear under two block types. Summing the raw
  // ranges would ask Mooncake for 160 bytes while only 96 are registered.
  const std::vector<StoreRange> ranges{{first.data(), first.size()},
                                       {second.data(), second.size()},
                                       {first.data(), first.size()}};

  const std::optional<std::vector<StoreRange>> unique =
      unique_store_ranges(ranges);
  ASSERT_TRUE(unique.has_value());
  EXPECT_EQ(unique->size(), 2U);

  const std::optional<size_t> bytes = total_range_bytes(*unique);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ(*bytes, 96U);
}

TEST(StoreRangesTest, RejectsConflictingOrOverlappingRegistrations) {
  std::vector<uint8_t> buffer(128);

  const std::vector<StoreRange> mismatched{{buffer.data(), 64},
                                           {buffer.data(), 32}};
  EXPECT_FALSE(unique_store_ranges(mismatched).has_value());

  const std::vector<StoreRange> overlapping{{buffer.data(), 64},
                                            {buffer.data() + 16, 32}};
  EXPECT_FALSE(unique_store_ranges(overlapping).has_value());

  const std::vector<StoreRange> null_range{{nullptr, 64}};
  EXPECT_FALSE(unique_store_ranges(null_range).has_value());
}

TEST(StoreRangesTest, DetectsASegmentSizeOverflow) {
  const size_t half = std::numeric_limits<size_t>::max() / 2 + 1;
  const std::vector<StoreRange> oversized{
      {reinterpret_cast<void*>(0x1000), half},
      {reinterpret_cast<void*>(0x20000000), half}};
  EXPECT_FALSE(total_range_bytes(oversized).has_value());
  EXPECT_EQ(total_range_bytes({}), 0U);
}

}  // namespace
}  // namespace xllm
