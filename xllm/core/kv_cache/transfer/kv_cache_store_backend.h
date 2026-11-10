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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

// One host memory range a store backend may touch. Address is the base of a
// whole cache tensor; a block is addressed by offsetting into it, because a
// zero-copy backend must register the range covering every block a later
// put/get may reach.
struct StoreRange {
  void* address = nullptr;
  size_t bytes = 0;
};

// One logical object's payload: a list of non-contiguous host memory ranges,
// concatenated in order. KVCacheStore produces one per key from the host cache
// tensors, so the backend only has to move bytes.
struct StoreBuffer {
  std::vector<void*> addresses;
  std::vector<size_t> sizes;
};

// Storage backend for KVCacheStore.
//
// The only job of an implementation is to move a key's worth of bytes between
// host memory and a shared store. Everything above it (key construction,
// request grouping, per-block buffer assembly, result aggregation) lives in
// KVCacheStore and is backend independent.
class KVCacheStoreBackend {
 public:
  virtual ~KVCacheStoreBackend() = default;

  // Write one object per key. Returns one status per key, 1 for success and 0
  // for failure, in the same order as `keys`.
  virtual std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) = 0;

  // Read one object into `buffer`. False means the object is unavailable or
  // its size does not match, and the caller treats it as a cache miss.
  virtual bool get(const std::string& key, const StoreBuffer& buffer) = 0;

  // Read one object per key, returning one status per key (1 hit, 0 miss) in
  // the same order as `keys`. The default implementation preserves the
  // historical per-key behaviour, so a backend without a native batch read
  // needs no changes. Backends that can fetch several keys in one round trip
  // should override this: prefetch otherwise issues N sequential store calls
  // with a timeout each.
  virtual std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) {
    std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
    if (keys.size() != buffers.size()) {
      return statuses;
    }
    for (size_t index = 0; index < keys.size(); ++index) {
      statuses[index] = get(keys[index], buffers[index]) ? 1 : 0;
    }
    return statuses;
  }
};

}  // namespace xllm
