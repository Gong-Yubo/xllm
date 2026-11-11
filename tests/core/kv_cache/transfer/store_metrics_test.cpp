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

#include "core/kv_cache/transfer/store_metrics.h"

#include <gtest/gtest.h>

#include <string_view>

namespace xllm {
namespace {

TEST(StoreMetricsTest, EmptySnapshotIsZero) {
  const StoreCallMetrics metrics;
  const auto snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.calls, 0u);
  EXPECT_EQ(snapshot.samples, 0u);
  EXPECT_EQ(snapshot.p50_us, 0);
  EXPECT_EQ(snapshot.p99_us, 0);
  EXPECT_EQ(snapshot.total_us, 0);
}

TEST(StoreMetricsTest, ClassifiesSdkFailureText) {
  // "Key not found" is wrapped in a longer worker message: it must win over
  // the transport keywords, otherwise a plain miss looks like an outage.
  EXPECT_TRUE(classify_store_failure(
                  "[Key not found] Cannot get objects from worker via RPC") ==
              StoreFailureKind::kKeyNotFound);
  EXPECT_TRUE(classify_store_failure("KEY NOT FOUND") ==
              StoreFailureKind::kKeyNotFound);
  EXPECT_TRUE(classify_store_failure("Timed out waiting for request") ==
              StoreFailureKind::kTimeout);
  EXPECT_TRUE(classify_store_failure("deadline exceeded") ==
              StoreFailureKind::kTimeout);
  EXPECT_TRUE(classify_store_failure("RPC peer dead ... unreachable") ==
              StoreFailureKind::kTransport);
  EXPECT_TRUE(classify_store_failure("ucp_ep_create failed: connection") ==
              StoreFailureKind::kTransport);
  EXPECT_TRUE(classify_store_failure("YuanRong store buffer size overflow") ==
              StoreFailureKind::kSdkOther);
  EXPECT_TRUE(classify_store_failure("") == StoreFailureKind::kNone);
  EXPECT_TRUE(std::string_view("none") ==
              store_failure_kind_name(StoreFailureKind::kNone));
}

TEST(StoreMetricsTest, ComputesPercentilesFromUnorderedSamples) {
  StoreCallMetrics metrics;
  for (int64_t value = 100; value >= 1; --value) {
    metrics.record(true, value);
  }
  const auto snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.calls, 100u);
  EXPECT_EQ(snapshot.successes, 100u);
  EXPECT_EQ(snapshot.failures, 0u);
  EXPECT_EQ(snapshot.samples, 100u);
  EXPECT_EQ(snapshot.p50_us, 50);
  EXPECT_EQ(snapshot.p99_us, 99);
  EXPECT_EQ(snapshot.max_us, 100);
  EXPECT_EQ(snapshot.total_us, 5050);
}

TEST(StoreMetricsTest, RingBufferKeepsOnlyRecentSamples) {
  StoreCallMetrics metrics;
  for (size_t index = 0; index < StoreCallMetrics::kSampleCapacity; ++index) {
    metrics.record(true, 1);
  }
  for (size_t index = 0; index < StoreCallMetrics::kSampleCapacity; ++index) {
    metrics.record(true, 1000);
  }
  const auto snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.calls, 2 * StoreCallMetrics::kSampleCapacity);
  EXPECT_EQ(snapshot.samples, StoreCallMetrics::kSampleCapacity);
  EXPECT_EQ(snapshot.p50_us, 1000);
  EXPECT_EQ(snapshot.p99_us, 1000);
  EXPECT_EQ(snapshot.max_us, 1000);
}

TEST(StoreMetricsTest, CountsFailuresByKindAndReportsPeriodically) {
  StoreCallMetrics metrics;
  for (uint64_t index = 1; index < StoreCallMetrics::kReportInterval; ++index) {
    EXPECT_FALSE(metrics.record(true, 10));
  }
  EXPECT_TRUE(metrics.record(false, 20, StoreFailureKind::kKeyNotFound));
  EXPECT_FALSE(metrics.record(false, 30, StoreFailureKind::kTimeout));
  metrics.record(false, 40, StoreFailureKind::kTransport);
  metrics.record(false, 50, StoreFailureKind::kSdkOther);

  const auto snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.calls, StoreCallMetrics::kReportInterval + 3);
  EXPECT_EQ(snapshot.successes, StoreCallMetrics::kReportInterval - 1);
  EXPECT_EQ(snapshot.failures, 4u);
  EXPECT_EQ(snapshot.failures_by_kind[static_cast<size_t>(
                StoreFailureKind::kKeyNotFound)],
            1u);
  EXPECT_EQ(
      snapshot
          .failures_by_kind[static_cast<size_t>(StoreFailureKind::kTimeout)],
      1u);
  EXPECT_EQ(
      snapshot
          .failures_by_kind[static_cast<size_t>(StoreFailureKind::kTransport)],
      1u);
  EXPECT_EQ(
      snapshot
          .failures_by_kind[static_cast<size_t>(StoreFailureKind::kSdkOther)],
      1u);
  EXPECT_EQ(
      snapshot.failures_by_kind[static_cast<size_t>(StoreFailureKind::kNone)],
      0u);
}

}  // namespace
}  // namespace xllm
