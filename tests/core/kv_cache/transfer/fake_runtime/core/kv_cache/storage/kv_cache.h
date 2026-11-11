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

// CPU-only stand-in for the host KV cache types. It keeps the same shape:
// each cache exposes several per-role tensors, and a block is a view into a
// tensor's leading dimension. That is all KVCacheStore needs to build a
// multi-range buffer and a content-addressed key.

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <cstring>
#include <map>

#include "util/hash_util.h"

namespace xllm {

enum class BlockType : int32_t {
  KV = 0,
  LINEAR = 1,
};

enum class TransferType : int32_t {
  H2D = 0,
  D2H = 1,
  G2H = 2,
  H2G = 3,
};

using BlockTypeTensorMap = std::map<int32_t, torch::Tensor>;

struct KVCacheTensors final {
  torch::Tensor k;
  torch::Tensor v;
};

class KVCache final {
 public:
  KVCache() = default;
  explicit KVCache(const KVCacheTensors& tensors) {
    tensors_[0] = tensors.k;
    tensors_[1] = tensors.v;
  }

  BlockTypeTensorMap get_block_type_tensors(BlockType /*type*/) const {
    return tensors_;
  }

 private:
  BlockTypeTensorMap tensors_;
};

class BlockTransferInfo final {
 public:
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  BlockType block_type = BlockType::KV;
  TransferType transfer_type = TransferType::G2H;

  BlockTransferInfo(int32_t src_id,
                    int32_t dst_id,
                    const uint8_t* key,
                    TransferType type,
                    BlockType btype = BlockType::KV)
      : src_block_id(src_id),
        dst_block_id(dst_id),
        block_type(btype),
        transfer_type(type) {
    std::memcpy(hash_key, key, XXH3_128BITS_HASH_VALUE_LEN);
  }
};

}  // namespace xllm
