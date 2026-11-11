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
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <vector>

namespace xllm {

// Coarse buckets for the failure text the C ABI bridge reports. The v4 logs
// could only tell "MGet failed" from the outside; a per-process breakdown of
// not-found vs timeout vs transport errors is what makes a re-test
// interpretable when the hit rate is low.
enum class StoreFailureKind {
  kNone = 0,
  kKeyNotFound,
  kTimeout,
  kTransport,
  kSdkOther,
  kCount,
};

inline constexpr size_t kStoreFailureKindCount =
    static_cast<size_t>(StoreFailureKind::kCount);

inline std::string_view store_failure_kind_name(StoreFailureKind kind) {
  switch (kind) {
    case StoreFailureKind::kNone:
      return "none";
    case StoreFailureKind::kKeyNotFound:
      return "not_found";
    case StoreFailureKind::kTimeout:
      return "timeout";
    case StoreFailureKind::kTransport:
      return "transport";
    case StoreFailureKind::kSdkOther:
      return "sdk_other";
    case StoreFailureKind::kCount:
      break;
  }
  return "unknown";
}

inline bool contains_ascii_case_insensitive(std::string_view text,
                                            std::string_view pattern) {
  if (pattern.empty()) {
    return true;
  }
  if (text.size() < pattern.size()) {
    return false;
  }
  const auto lower = [](char value) {
    return static_cast<char>(value >= 'A' && value <= 'Z' ? value - 'A' + 'a'
                                                          : value);
  };
  for (size_t start = 0; start + pattern.size() <= text.size(); ++start) {
    size_t offset = 0;
    while (offset < pattern.size() &&
           lower(text[start + offset]) == lower(pattern[offset])) {
      ++offset;
    }
    if (offset == pattern.size()) {
      return true;
    }
  }
  return false;
}

inline StoreFailureKind classify_store_failure(std::string_view error) {
  if (error.empty()) {
    return StoreFailureKind::kNone;
  }
  // Order matters: the SDK wraps "Key not found" in a longer message that
  // also mentions the worker RPC layer.
  if (contains_ascii_case_insensitive(error, "key not found") ||
      contains_ascii_case_insensitive(error, "cannot get objects")) {
    return StoreFailureKind::kKeyNotFound;
  }
  if (contains_ascii_case_insensitive(error, "timed out") ||
      contains_ascii_case_insensitive(error, "timeout") ||
      contains_ascii_case_insensitive(error, "deadline")) {
    return StoreFailureKind::kTimeout;
  }
  if (contains_ascii_case_insensitive(error, "rpc") ||
      contains_ascii_case_insensitive(error, "connection") ||
      contains_ascii_case_insensitive(error, "unreachable") ||
      contains_ascii_case_insensitive(error, "peer dead") ||
      contains_ascii_case_insensitive(error, "transport")) {
    return StoreFailureKind::kTransport;
  }
  return StoreFailureKind::kSdkOther;
}

// Lightweight in-process observability for store calls. Kept on the C++ side
// of the bridge on purpose: adding a metrics getter to the C ABI would widen
// the dlopen symbol surface that the build-time link check guards. A bounded
// ring of recent latencies gives p50/p99 without allocating per request.
class StoreCallMetrics {
 public:
  static constexpr size_t kSampleCapacity = 256;
  static constexpr uint64_t kReportInterval = 128;

  struct Snapshot {
    uint64_t calls = 0;
    uint64_t successes = 0;
    uint64_t failures = 0;
    int64_t p50_us = 0;
    int64_t p99_us = 0;
    int64_t max_us = 0;
    int64_t total_us = 0;
    size_t samples = 0;
    std::array<uint64_t, kStoreFailureKindCount> failures_by_kind{};
  };

  // Returns true once every kReportInterval calls, so the caller can emit one
  // summary line instead of logging every request on the critical path.
  bool record(bool success,
              int64_t elapsed_us,
              StoreFailureKind failure_kind = StoreFailureKind::kNone) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++calls_;
    total_us_ += elapsed_us;
    if (success) {
      ++successes_;
    } else {
      ++failures_;
      ++failures_by_kind_[kind_index(failure_kind)];
    }
    samples_[write_index_] = elapsed_us;
    write_index_ = (write_index_ + 1) % kSampleCapacity;
    sample_count_ = std::min(sample_count_ + 1, kSampleCapacity);
    return calls_ % kReportInterval == 0;
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot result;
    result.calls = calls_;
    result.successes = successes_;
    result.failures = failures_;
    result.total_us = total_us_;
    result.failures_by_kind = failures_by_kind_;
    result.samples = sample_count_;

    std::vector<int64_t> ordered(sample_count_);
    // The ring is filled from index 0 up to sample_count_ - 1; once full,
    // every slot holds a valid sample and their order is irrelevant to a
    // percentile.
    std::copy_n(samples_.begin(), sample_count_, ordered.begin());
    if (!ordered.empty()) {
      std::sort(ordered.begin(), ordered.end());
      result.p50_us = percentile(ordered, 50);
      result.p99_us = percentile(ordered, 99);
      result.max_us = ordered.back();
    }
    return result;
  }

 private:
  static size_t kind_index(StoreFailureKind kind) {
    const size_t index = static_cast<size_t>(kind);
    return index < kStoreFailureKindCount
               ? index
               : static_cast<size_t>(StoreFailureKind::kSdkOther);
  }

  static int64_t percentile(const std::vector<int64_t>& ordered, int percent) {
    const size_t rank =
        std::max<size_t>(1, (ordered.size() * percent + 99) / 100);
    return ordered[std::min(rank, ordered.size()) - 1];
  }

  mutable std::mutex mutex_;
  uint64_t calls_ = 0;
  uint64_t successes_ = 0;
  uint64_t failures_ = 0;
  int64_t total_us_ = 0;
  size_t write_index_ = 0;
  size_t sample_count_ = 0;
  std::array<int64_t, kSampleCapacity> samples_{};
  std::array<uint64_t, kStoreFailureKindCount> failures_by_kind_{};
};

}  // namespace xllm
