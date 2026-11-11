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

// Deterministic stand-in for the xxHash-backed key helper. KVCacheStore only
// needs a stable 16-byte schema hash; the batching tests do not care which
// hash function produced it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace xllm {

constexpr uint32_t XXH3_128BITS_HASH_VALUE_LEN = 16;

struct XXH3Key {
  uint8_t data[XXH3_128BITS_HASH_VALUE_LEN] = {};
};

inline XXH3Key hash_string(const std::string& value) {
  XXH3Key key;
  for (size_t index = 0; index < value.size(); ++index) {
    const size_t slot = index % XXH3_128BITS_HASH_VALUE_LEN;
    key.data[slot] = static_cast<uint8_t>(
        key.data[slot] ^ static_cast<uint8_t>(value[index] * 31 + index));
  }
  return key;
}

}  // namespace xllm
