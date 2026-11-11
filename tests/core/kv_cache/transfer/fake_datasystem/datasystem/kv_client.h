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

// Minimal stand-in for <datasystem/kv_client.h>.
//
// The bridge under test is compiled unchanged; only the SDK behind its C ABI
// is replaced. That matters because the production bridge owns the error
// handling, payload flattening and per-key status mapping that the v4 test
// found broken, and the existing backend tests replace the *whole* bridge.
//
// The 0.9.3/0.10.1 SDK has several Get() fill behaviours. `GetFillMode` lets
// a test choose which one the SDK appears to implement, so the bridge's
// contract can be checked against all of them without a worker cluster.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace datasystem {

enum class WriteMode : int {
  NONE_L2_CACHE = 0,
  NONE_L2_CACHE_EVICT = 1,
  WRITE_THROUGH_L2_CACHE = 2,
};

enum class ExistenceOpt : int {
  NONE = 0,
  NX = 1,
};

enum class CacheType : int {
  MEMORY = 0,
};

struct MSetParam {
  WriteMode writeMode = WriteMode::NONE_L2_CACHE;
  uint32_t ttlSecond = 0;
  ExistenceOpt existence = ExistenceOpt::NONE;
  CacheType cacheType = CacheType::MEMORY;
};

struct ConnectOptions {
  std::string host;
  int32_t port = 0;
  int32_t connectTimeoutMs = 9 * 1000;
  int32_t requestTimeoutMs = 0;
};

class Status {
 public:
  Status() = default;
  static Status Error(std::string text) {
    Status status;
    status.is_error_ = true;
    status.text_ = std::move(text);
    return status;
  }
  bool IsError() const { return is_error_; }
  const std::string& ToString() const { return text_.empty() ? kOk : text_; }

 private:
  static inline const std::string kOk = "K_OK";
  bool is_error_ = false;
  std::string text_;
};

class StringView {
 public:
  StringView() = default;
  StringView(const std::string& value)  // NOLINT(runtime/explicit)
      : value_(value) {}

  const char* data() const { return value_.data(); }
  size_t size() const { return value_.size(); }
  std::string str() const { return value_; }

 private:
  // The real StringView is non-owning; owning here keeps the recorded payload
  // readable after the bridge's staging strings go out of scope.
  std::string value_;
};

namespace fake {

enum class GetFillMode {
  // 0.9.3 header: assign at the key's index, leaving missing keys empty.
  kAssignAtIndex,
  // 0.10.1 GetWithLatch: clear, then emplace one value per key.
  kClearAndAppend,
  // An SDK that appends without clearing: pre-sized slots survive at the
  // front and the result is longer than the key list. Unsupported.
  kAppendWithoutClear,
  // Truncated result: fewer entries than keys were returned.
  kTruncate,
};

struct State {
  bool connect_options_seen = false;
  ConnectOptions connect_options;
  bool init_error = false;
  std::string init_error_text;
  int32_t shutdown_calls = 0;

  bool mset_error = false;
  std::string mset_error_text;
  std::vector<std::string> mset_failed_keys;
  std::vector<std::string> last_mset_keys;
  std::vector<std::string> last_mset_payloads;
  WriteMode last_write_mode = WriteMode::NONE_L2_CACHE;
  ExistenceOpt last_existence = ExistenceOpt::NONE;
  int32_t mset_calls = 0;

  bool get_error = false;
  std::string get_error_text;
  GetFillMode get_fill_mode = GetFillMode::kAssignAtIndex;
  std::vector<std::string> get_payloads;
  size_t truncate_to = 0;
  std::vector<std::string> last_get_keys;
  size_t get_input_vals_size = 0;
  int32_t get_calls = 0;
};

inline State& state() {
  static State value;
  return value;
}

inline void reset() { state() = State{}; }

}  // namespace fake

class KVClient {
 public:
  explicit KVClient(const ConnectOptions& options = {}) {
    fake::state().connect_options = options;
    fake::state().connect_options_seen = true;
  }

  Status Init() {
    const auto& current = fake::state();
    return current.init_error ? Status::Error(current.init_error_text)
                              : Status();
  }

  Status ShutDown() {
    ++fake::state().shutdown_calls;
    return Status();
  }

  Status MSet(const std::vector<std::string>& keys,
              const std::vector<StringView>& vals,
              std::vector<std::string>& out_failed_keys,
              const MSetParam& param = {}) {
    auto& current = fake::state();
    ++current.mset_calls;
    current.last_mset_keys = keys;
    current.last_mset_payloads.clear();
    for (const StringView& value : vals) {
      current.last_mset_payloads.emplace_back(value.str());
    }
    current.last_write_mode = param.writeMode;
    current.last_existence = param.existence;
    out_failed_keys = current.mset_failed_keys;
    return current.mset_error ? Status::Error(current.mset_error_text)
                              : Status();
  }

  Status Get(const std::vector<std::string>& keys,
             std::vector<std::string>& vals,
             int32_t /*sub_timeout_ms*/ = 0) {
    auto& current = fake::state();
    ++current.get_calls;
    current.last_get_keys = keys;
    current.get_input_vals_size = vals.size();
    if (current.get_error) {
      return Status::Error(current.get_error_text);
    }

    switch (current.get_fill_mode) {
      case fake::GetFillMode::kAssignAtIndex: {
        const size_t assigned =
            std::min(current.get_payloads.size(), vals.size());
        for (size_t index = 0; index < assigned; ++index) {
          vals[index] = current.get_payloads[index];
        }
        break;
      }
      case fake::GetFillMode::kClearAndAppend:
        vals.clear();
        for (const std::string& payload : current.get_payloads) {
          vals.emplace_back(payload);
        }
        break;
      case fake::GetFillMode::kAppendWithoutClear:
        for (const std::string& payload : current.get_payloads) {
          vals.emplace_back(payload);
        }
        break;
      case fake::GetFillMode::kTruncate:
        vals.resize(std::min(current.truncate_to, vals.size()));
        for (size_t index = 0; index < vals.size(); ++index) {
          if (index < current.get_payloads.size()) {
            vals[index] = current.get_payloads[index];
          }
        }
        break;
    }
    return Status();
  }
};

}  // namespace datasystem
