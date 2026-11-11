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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_store_backend.h"

namespace xllm {

// A host cache can be registered under several block types, so collect_ranges()
// may emit the same tensor more than once. Registration deduplicates by base
// address; this is that exact rule, shared with the segment sizing code so the
// pool is never sized from bytes that are not actually registered.
inline std::optional<std::vector<StoreRange>> unique_store_ranges(
    const std::vector<StoreRange>& ranges) {
  std::vector<StoreRange> unique;
  unique.reserve(ranges.size());
  for (const StoreRange& range : ranges) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(range.address);
    if (range.address == nullptr || range.bytes == 0 ||
        range.bytes > std::numeric_limits<uintptr_t>::max() - address) {
      return std::nullopt;
    }
    const auto same_address = std::find_if(
        unique.begin(), unique.end(), [address](const StoreRange& item) {
          return reinterpret_cast<uintptr_t>(item.address) == address;
        });
    if (same_address == unique.end()) {
      unique.emplace_back(range);
    } else if (same_address->bytes != range.bytes) {
      return std::nullopt;
    }
  }

  std::vector<StoreRange> ordered = unique;
  std::sort(ordered.begin(),
            ordered.end(),
            [](const StoreRange& lhs, const StoreRange& rhs) {
              return reinterpret_cast<uintptr_t>(lhs.address) <
                     reinterpret_cast<uintptr_t>(rhs.address);
            });
  for (size_t range_index = 1; range_index < ordered.size(); ++range_index) {
    const StoreRange& previous = ordered[range_index - 1];
    const uintptr_t previous_end =
        reinterpret_cast<uintptr_t>(previous.address) + previous.bytes;
    if (reinterpret_cast<uintptr_t>(ordered[range_index].address) <
        previous_end) {
      return std::nullopt;
    }
  }
  return unique;
}

// Sums the ranges the backend will register. Callers pass the deduplicated
// ranges from unique_store_ranges(); the overflow check keeps a malformed
// registration from wrapping into a tiny segment.
inline std::optional<size_t> total_range_bytes(
    const std::vector<StoreRange>& ranges) {
  size_t total = 0;
  for (const StoreRange& range : ranges) {
    if (range.bytes > std::numeric_limits<size_t>::max() - total) {
      return std::nullopt;
    }
    total += range.bytes;
  }
  return total;
}

}  // namespace xllm
