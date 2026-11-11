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

#include <cstdint>
#include <limits>
#include <string_view>

namespace xllm {

inline constexpr std::string_view kStoreBackendMooncake = "mooncake";
inline constexpr std::string_view kStoreBackendYuanRong = "yuanrong";

// KVCacheStore used to dispatch on `backend == "yuanrong"` alone, so a typo
// or a not-yet-implemented name quietly selected Mooncake. Validate the name
// at the boundary, before any store state is mutated.
inline bool is_supported_store_backend(std::string_view backend) {
  return backend == kStoreBackendMooncake || backend == kStoreBackendYuanRong;
}

// Port 0 is meaningful: it lets the YuanRong SDK pick its default endpoint.
// A negative or out-of-range value is a configuration error, not a reason to
// abort the process with CHECK.
inline bool is_valid_yuanrong_port(int32_t port) {
  return port >= 0 && port <= std::numeric_limits<uint16_t>::max();
}

// One store request's deadline, shared by the CLI defaults and the bridge's
// clamp. A non-positive value would restore the unbounded SDK default (0)
// that caused the minute-long stalls in the v4 test.
inline constexpr int32_t kDefaultStoreRequestTimeoutMs = 500;

constexpr int32_t normalize_store_request_timeout_ms(int32_t requested_ms) {
  return requested_ms > 0 ? requested_ms : kDefaultStoreRequestTimeoutMs;
}

}  // namespace xllm
