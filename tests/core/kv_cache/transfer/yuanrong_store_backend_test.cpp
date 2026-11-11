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

#include "core/kv_cache/transfer/yuanrong_store_backend.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xllm {
namespace {

// The stub bridge is built next to this binary by the test CMakeLists; it
// implements the xllm_yr_store_* ABI over an in-process map, so the backend's
// range marshalling and status mapping run against real code.
YuanRongStoreConfig make_yuanrong_config() {
  YuanRongStoreConfig config;
  config.host = "127.0.0.1";
  config.port = 31501;
#if defined(XLLM_YUANRONG_BRIDGE_STUB_PATH)
  config.bridge_path = XLLM_YUANRONG_BRIDGE_STUB_PATH;
#else
  config.bridge_path = "/nonexistent/libxllm_yuanrong_store_bridge.so";
#endif
  return config;
}

// A KV block reaches the store as separate K and V ranges, so every case uses
// two buffers that are not contiguous in memory.
class BlockPayload {
 public:
  BlockPayload(uint8_t key_fill, uint8_t value_fill)
      : key_range_(64, key_fill), value_range_(32, value_fill) {}

  StoreBuffer buffer() {
    return StoreBuffer{{key_range_.data(), value_range_.data()},
                       {key_range_.size(), value_range_.size()}};
  }

  std::vector<void*> addresses() {
    return {key_range_.data(), value_range_.data()};
  }

  void wipe() {
    std::fill(key_range_.begin(), key_range_.end(), 0);
    std::fill(value_range_.begin(), value_range_.end(), 0);
  }

  const std::vector<uint8_t>& key_range() const { return key_range_; }
  const std::vector<uint8_t>& value_range() const { return value_range_; }

 private:
  std::vector<uint8_t> key_range_;
  std::vector<uint8_t> value_range_;
};

void expect_payload(const BlockPayload& payload,
                    uint8_t key_fill,
                    uint8_t value_fill) {
  EXPECT_EQ(payload.key_range(), std::vector<uint8_t>(64, key_fill));
  EXPECT_EQ(payload.value_range(), std::vector<uint8_t>(32, value_fill));
}

TEST(YuanRongStoreBackendTest, FailsWhenTheBridgeCannotBeLoaded) {
  YuanRongStoreConfig config;
  config.bridge_path = "/nonexistent/libxllm_yuanrong_store_bridge.so";
  YuanRongStoreBackend backend(config);

  // A missing bridge must surface as a start-up failure. The alternative --
  // reporting success and dropping the objects -- would silently disable the
  // offload tier.
  EXPECT_FALSE(backend.initialize());

  BlockPayload payload(0x5A, 0x5B);
  EXPECT_FALSE(backend.get("missing/key", payload.buffer()));
  const std::vector<uint8_t> statuses =
      backend.batch_put({"missing/key"}, {payload.buffer()});
  ASSERT_EQ(statuses.size(), 1U);
  EXPECT_EQ(statuses[0], 0);
}

TEST(YuanRongStoreBackendTest, RoundTripsABlockSplitAcrossRanges) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());
  // KVCacheStore may register a store more than once; a second call has to
  // reuse the open client instead of reopening the bridge.
  EXPECT_TRUE(backend.initialize());

  BlockPayload payload(0x11, 0x22);
  const std::vector<uint8_t> statuses =
      backend.batch_put({"model/blk0"}, {payload.buffer()});
  ASSERT_EQ(statuses.size(), 1U);
  EXPECT_EQ(statuses[0], 1);

  // Wipe the source so a read-back can only be satisfied from the store.
  payload.wipe();
  ASSERT_TRUE(backend.get("model/blk0", payload.buffer()));
  expect_payload(payload, 0x11, 0x22);
}

TEST(YuanRongStoreBackendTest, RoundTripsAKeyWithRawHashBytes) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  // KVCacheStore keys are "xllm-kv-v3:<...>" + 32 raw hash bytes (16 schema +
  // 16 block content). DataSystem rejects non-ASCII bytes, so the backend hex-
  // encodes before the bridge. The stub enforces the same charset, so this
  // test fails if the encoding regresses.
  std::string binary_key = "xllm-kv-v3:qwen3-32b:main:8:0:0:";
  binary_key.append("\x00\x01\x02\x03\xfe\xff", 6);

  BlockPayload payload(0x21, 0x43);
  ASSERT_EQ(backend.batch_put({binary_key}, {payload.buffer()})[0], 1);
  payload.wipe();
  ASSERT_TRUE(backend.get(binary_key, payload.buffer()));
  expect_payload(payload, 0x21, 0x43);
}

TEST(YuanRongStoreBackendTest, ReportsAStatusPerKeyInOneBatch) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload first(0x01, 0x02);
  BlockPayload second(0x03, 0x04);
  const std::vector<uint8_t> statuses = backend.batch_put(
      {"model/blk0", "model/blk1"}, {first.buffer(), second.buffer()});
  ASSERT_EQ(statuses.size(), 2U);
  EXPECT_EQ(statuses[0], 1);
  EXPECT_EQ(statuses[1], 1);

  first.wipe();
  second.wipe();
  ASSERT_TRUE(backend.get("model/blk0", first.buffer()));
  ASSERT_TRUE(backend.get("model/blk1", second.buffer()));
  expect_payload(first, 0x01, 0x02);
  expect_payload(second, 0x03, 0x04);
}

TEST(YuanRongStoreBackendTest, TreatsAMissingKeyAsAMiss) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload payload(0x00, 0x00);
  // A miss is the normal outcome when nothing published the prefix; it has to
  // be reported rather than turned into a zero-filled hit.
  EXPECT_FALSE(backend.get("model/never-published", payload.buffer()));
}

TEST(YuanRongStoreBackendTest, TreatsADuplicatePutAsSuccess) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload first(0x77, 0x88);
  ASSERT_EQ(backend.batch_put({"model/blk0"}, {first.buffer()})[0], 1);

  // The bridge asks for ExistenceOpt::NX, and the SDK counts an existing key as
  // success without touching the stored value. Re-publishing a block is a no-op
  // rather than an error; content-addressed keys make the two writes identical
  // in practice, so this only proves the second write is not an overwrite.
  BlockPayload second(0x99, 0xAA);
  const std::vector<uint8_t> duplicate =
      backend.batch_put({"model/blk0"}, {second.buffer()});
  ASSERT_EQ(duplicate.size(), 1U);
  EXPECT_EQ(duplicate[0], 1);

  first.wipe();
  ASSERT_TRUE(backend.get("model/blk0", first.buffer()));
  expect_payload(first, 0x77, 0x88);
}

TEST(YuanRongStoreBackendTest, RejectsMismatchedRangeArrays) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload payload(0xAA, 0xBB);
  StoreBuffer broken;
  broken.addresses = payload.addresses();
  broken.sizes = {payload.key_range().size()};

  const std::vector<uint8_t> statuses =
      backend.batch_put({"model/blk0"}, {broken});
  ASSERT_EQ(statuses.size(), 1U);
  EXPECT_EQ(statuses[0], 0);
  EXPECT_FALSE(backend.get("model/blk0", broken));

  // The rejected put wrote nothing.
  EXPECT_FALSE(backend.get("model/blk0", payload.buffer()));
}

TEST(YuanRongStoreBackendTest, BatchGetReturnsOneStatusPerKey) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload first(0x01, 0x02);
  BlockPayload second(0x03, 0x04);
  BlockPayload third(0x05, 0x06);
  const std::vector<uint8_t> puts =
      backend.batch_put({"model/blk0", "model/blk1", "model/blk2"},
                        {first.buffer(), second.buffer(), third.buffer()});
  ASSERT_EQ(puts.size(), 3U);
  EXPECT_EQ(puts[0], 1);
  EXPECT_EQ(puts[1], 1);
  EXPECT_EQ(puts[2], 1);

  first.wipe();
  second.wipe();
  third.wipe();
  const std::vector<uint8_t> statuses =
      backend.batch_get({"model/blk0", "model/blk1", "model/blk2"},
                        {first.buffer(), second.buffer(), third.buffer()});
  ASSERT_EQ(statuses.size(), 3U);
  EXPECT_EQ(statuses[0], 1);
  EXPECT_EQ(statuses[1], 1);
  EXPECT_EQ(statuses[2], 1);
  expect_payload(first, 0x01, 0x02);
  expect_payload(second, 0x03, 0x04);
  expect_payload(third, 0x05, 0x06);

  EXPECT_TRUE(backend.batch_get({}, {}).empty());
  const std::vector<uint8_t> mismatched = backend.batch_get({"model/blk0"}, {});
  ASSERT_EQ(mismatched.size(), 1U);
  EXPECT_EQ(mismatched[0], 0);
}

TEST(YuanRongStoreBackendTest, PartialHitIsNotABatchFailure) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  BlockPayload first(0x11, 0x12);
  BlockPayload second(0x13, 0x14);
  ASSERT_EQ(backend.batch_put({"model/blk0", "model/blk1"},
                              {first.buffer(), second.buffer()})[0],
            1);

  first.wipe();
  second.wipe();
  BlockPayload absent(0x00, 0x00);
  // DataSystem returns K_OK for a partially hit batch: the missing key must
  // not discard the two objects that were actually found.
  const std::vector<uint8_t> statuses =
      backend.batch_get({"model/blk0", "model/never-published", "model/blk1"},
                        {first.buffer(), absent.buffer(), second.buffer()});
  ASSERT_EQ(statuses.size(), 3U);
  EXPECT_EQ(statuses[0], 1);
  EXPECT_EQ(statuses[1], 0);
  EXPECT_EQ(statuses[2], 1);
  expect_payload(first, 0x11, 0x12);
  expect_payload(second, 0x13, 0x14);
}

TEST(YuanRongStoreBackendTest, EmptyPayloadIsAMissNotAnError) {
  YuanRongStoreBackend backend(make_yuanrong_config());
  ASSERT_TRUE(backend.initialize());

  // A zero-range object is stored as an empty payload. The SDK uses an empty
  // payload to encode a missing key, so reading it back must be a miss that
  // leaves the rest of the batch intact.
  StoreBuffer empty_object;
  ASSERT_EQ(backend.batch_put({"model/empty"}, {empty_object})[0], 1);

  BlockPayload good(0x31, 0x32);
  ASSERT_EQ(backend.batch_put({"model/good"}, {good.buffer()})[0], 1);

  good.wipe();
  BlockPayload empty_destination(0x00, 0x00);
  const std::vector<uint8_t> statuses =
      backend.batch_get({"model/empty", "model/good"},
                        {empty_destination.buffer(), good.buffer()});
  ASSERT_EQ(statuses.size(), 2U);
  EXPECT_EQ(statuses[0], 0);
  EXPECT_EQ(statuses[1], 1);
  expect_payload(good, 0x31, 0x32);
}

TEST(YuanRongStoreBackendTest, SerializesConcurrentPrefetchThreads) {
#if defined(XLLM_YUANRONG_BRIDGE_STUB_PATH)
  void* stub = dlopen(XLLM_YUANRONG_BRIDGE_STUB_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_TRUE(stub != nullptr);
  using ResetConcurrencyFn = void (*)();
  using PeakConcurrencyFn = int32_t (*)();
  auto reset_concurrency = reinterpret_cast<ResetConcurrencyFn>(
      dlsym(stub, "xllm_yr_store_stub_reset_concurrency"));
  auto peak_concurrency = reinterpret_cast<PeakConcurrencyFn>(
      dlsym(stub, "xllm_yr_store_stub_peak_concurrency"));
  ASSERT_TRUE(reset_concurrency != nullptr);
  ASSERT_TRUE(peak_concurrency != nullptr);
  reset_concurrency();

  constexpr int32_t kThreadCount = 8;
  constexpr int32_t kBlocksPerThread = 24;
  std::vector<int32_t> failures(kThreadCount, 0);
  {
    YuanRongStoreBackend backend(make_yuanrong_config());
    ASSERT_TRUE(backend.initialize());

    // WorkerService prefetches on a thread pool against one shared client.
    // Every thread owns distinct keys and buffers, so a failure here means a
    // race in the backend rather than conflicting test data.
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int32_t thread = 0; thread < kThreadCount; ++thread) {
      workers.emplace_back([&backend, &failures, thread]() {
        for (int32_t block = 0; block < kBlocksPerThread; ++block) {
          const std::string key = "model/t" + std::to_string(thread) + "/blk" +
                                  std::to_string(block);
          const uint8_t key_fill = static_cast<uint8_t>(thread + 1);
          const uint8_t value_fill = static_cast<uint8_t>(block + 1);
          BlockPayload payload(key_fill, value_fill);
          const std::vector<uint8_t> put =
              backend.batch_put({key}, {payload.buffer()});
          if (put.size() != 1U || put[0] != 1) {
            ++failures[thread];
            continue;
          }
          payload.wipe();
          if (!backend.get(key, payload.buffer())) {
            ++failures[thread];
            continue;
          }
          if (payload.key_range() != std::vector<uint8_t>(64, key_fill) ||
              payload.value_range() != std::vector<uint8_t>(32, value_fill)) {
            ++failures[thread];
          }
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  for (int32_t failed : failures) {
    EXPECT_EQ(failed, 0);
  }
  // A peak above one would mean two prefetch threads reached the single
  // KVClient (and its shared last_error string) at the same time.
  EXPECT_EQ(peak_concurrency(), 1);
  dlclose(stub);
#endif
}

TEST(YuanRongStoreBackendTest, DefaultRequestTimeoutIsBounded) {
  // A cache miss must fail fast. The old 60000 ms default let a prefetch hold
  // a thread for a minute; guard against reintroducing it here as well.
  const YuanRongStoreConfig config{};
  EXPECT_TRUE(config.timeout_ms > 0);
  EXPECT_TRUE(config.timeout_ms <= 1000);
}

}  // namespace
}  // namespace xllm
