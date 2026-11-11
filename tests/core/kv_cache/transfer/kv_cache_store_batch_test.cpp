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

// Exercises KVCacheStore's real prefetch batching against an injected backend.
// The production kv_cache_store.cpp is compiled unchanged; only its device
// dependencies are replaced by CPU-only stand-ins (see fake_runtime/). The
// existing kv_cache_store_test.cpp covers key construction and result
// aggregation, but never wires up a backend, and is gated behind NPU/MLU.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_store.h"

namespace xllm {

class KVCacheStoreTestPeer final {
 public:
  static void initialize_index(KVCacheStore* store,
                               const KVCacheStoreInitConfig& config,
                               HostCacheStoreIndex store_index) {
    store->config_ = config;
    store->initialize_store_index(std::move(store_index));
  }

  static void mark_initialized(KVCacheStore* store) {
    store->is_initialized_ = true;
  }

  static void set_backend(KVCacheStore* store,
                          std::unique_ptr<KVCacheStoreBackend> backend) {
    store->backend_ = std::move(backend);
  }
};

namespace {

// Records the call crossing the backend seam so a test can assert that one
// prefetch produces exactly one batch request with the right keys and buffers.
class RecordingBackend final : public KVCacheStoreBackend {
 public:
  std::vector<std::string> keys;
  std::vector<size_t> range_counts;
  std::vector<void*> first_addresses;
  std::vector<uint8_t> reply;
  int32_t batch_get_calls = 0;

  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& /*keys*/,
      const std::vector<StoreBuffer>& /*buffers*/) override {
    return {};
  }

  bool get(const std::string& /*key*/, const StoreBuffer& /*buffer*/) override {
    return false;
  }

  std::vector<uint8_t> batch_get(
      const std::vector<std::string>& request_keys,
      const std::vector<StoreBuffer>& buffers) override {
    ++batch_get_calls;
    keys = request_keys;
    range_counts.clear();
    first_addresses.clear();
    for (const StoreBuffer& buffer : buffers) {
      range_counts.emplace_back(buffer.addresses.size());
      first_addresses.emplace_back(
          buffer.addresses.empty() ? nullptr : buffer.addresses.front());
    }
    return reply;
  }
};

BlockTransferInfo make_block(uint8_t hash_value, int32_t dst_block_id) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash_key{};
  hash_key.fill(hash_value);
  return BlockTransferInfo(/*src_id=*/0,
                           dst_block_id,
                           hash_key.data(),
                           TransferType::G2H,
                           BlockType::KV);
}

KVCacheStoreInitConfig make_config() {
  KVCacheStoreInitConfig config;
  config.model_id = "target-model";
  config.tp_size = 1;
  config.tp_rank = 0;
  return config;
}

class KVCacheStoreBatchTest : public ::testing::Test {
 protected:
  KVCacheStoreBatchTest()
      : first_cache_(
            KVCacheTensors{torch::zeros({4, 2, 3}), torch::zeros({4, 2, 3})}),
        second_cache_(
            KVCacheTensors{torch::zeros({4, 2, 5}), torch::zeros({4, 2, 5})}) {}

  void SetUp() override {
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0, "main", &first_cache_});
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/1, "draft", &second_cache_});
    KVCacheStoreTestPeer::initialize_index(
        &store_, make_config(), std::move(index));
    KVCacheStoreTestPeer::mark_initialized(&store_);
    auto backend = std::make_unique<RecordingBackend>();
    backend_ = backend.get();
    KVCacheStoreTestPeer::set_backend(&store_, std::move(backend));
  }

  std::vector<uint8_t> get_statuses(std::vector<BlockTransferInfo>& blocks) {
    Slice<BlockTransferInfo> slice(blocks);
    return store_.batch_get_with_status(slice);
  }

  void* first_tensor_address(int32_t block_id) {
    const BlockTypeTensorMap tensors =
        first_cache_.get_block_type_tensors(BlockType::KV);
    return tensors.begin()->second[block_id].data_ptr();
  }

  KVCache first_cache_;
  KVCache second_cache_;
  KVCacheStore store_;
  RecordingBackend* backend_ = nullptr;  // owned by store_
};

TEST_F(KVCacheStoreBatchTest, IssuesOneBatchCallForAllPhysicalRequests) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0),
                                           make_block(2, /*dst=*/1)};
  backend_->reply = {1, 1, 1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({1, 1}));
  EXPECT_EQ(backend_->batch_get_calls, 1);
  EXPECT_EQ(backend_->keys.size(), 4U);
  // Each store entry carries two per-role host tensors, so every key's
  // payload is a two-range buffer.
  EXPECT_EQ(backend_->range_counts, std::vector<size_t>({2U, 2U, 2U, 2U}));
}

TEST_F(KVCacheStoreBatchTest, PartialHitNeedsEveryPhysicalRequestToSucceed) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0),
                                           make_block(2, /*dst=*/1)};
  backend_->reply = {1, 1, 0, 1};

  // Logical block 0 owns requests 0 and 1; block 1 owns 2 and 3. One missing
  // physical result must fail the whole logical block, not report a hit.
  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({1, 0}));
  EXPECT_EQ(backend_->batch_get_calls, 1);
}

TEST_F(KVCacheStoreBatchTest, MismatchedStatusCountFailsEveryBlock) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0),
                                           make_block(2, /*dst=*/1)};
  backend_->reply = {1, 1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({0, 0}));
  EXPECT_EQ(backend_->batch_get_calls, 1);
}

TEST_F(KVCacheStoreBatchTest, SkippedBufferKeepsRemainingKeysAligned) {
  // An invalid host block id makes build_multi_buffer fail for that logical
  // block. Its two physical requests are skipped, so the statuses returned
  // for the remaining keys must map back to requests 2 and 3, not 0 and 1.
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/-1),
                                           make_block(2, /*dst=*/1)};
  backend_->reply = {1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({0, 1}));
  EXPECT_EQ(backend_->batch_get_calls, 1);
  ASSERT_EQ(backend_->keys.size(), 2U);
}

TEST_F(KVCacheStoreBatchTest, SkippedTrailingBufferKeepsLeadingKeysAligned) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0),
                                           make_block(2, /*dst=*/-1)};
  backend_->reply = {1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({1, 0}));
  EXPECT_EQ(backend_->batch_get_calls, 1);
  ASSERT_EQ(backend_->keys.size(), 2U);
}

TEST_F(KVCacheStoreBatchTest, BatchGetCountMatchesTheStatusVector) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0),
                                           make_block(2, /*dst=*/1)};
  backend_->reply = {1, 1, 0, 1};

  EXPECT_EQ(store_.batch_get(blocks), 1U);
  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({1, 0}));
  EXPECT_EQ(backend_->batch_get_calls, 2);
}

TEST_F(KVCacheStoreBatchTest, BuffersPointAtTheRequestedHostBlock) {
  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/2)};
  backend_->reply = {1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({1}));
  ASSERT_EQ(backend_->first_addresses.size(), 2U);
  EXPECT_EQ(backend_->first_addresses[0], first_tensor_address(2));
}

TEST_F(KVCacheStoreBatchTest, EmptyPrefetchDoesNotReachTheBackend) {
  std::vector<BlockTransferInfo> blocks;
  EXPECT_TRUE(get_statuses(blocks).empty());
  EXPECT_EQ(backend_->batch_get_calls, 0);
}

TEST_F(KVCacheStoreBatchTest, DuplicateKeysAreRejectedBeforeTheBackend) {
  // Same content hash on two logical blocks produces the same key per store
  // entry. Batching those would confuse the per-key status mapping.
  std::vector<BlockTransferInfo> blocks = {make_block(7, /*dst=*/0),
                                           make_block(7, /*dst=*/1)};
  backend_->reply = {1, 1, 1, 1};

  EXPECT_EQ(get_statuses(blocks), std::vector<uint8_t>({0, 0}));
  EXPECT_EQ(backend_->batch_get_calls, 0);
}

TEST(KVCacheStoreBatchStandaloneTest, UninitializedStoreSkipsTheBackend) {
  KVCache cache(
      KVCacheTensors{torch::zeros({4, 2, 3}), torch::zeros({4, 2, 3})});
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &cache});

  KVCacheStore store;
  KVCacheStoreTestPeer::initialize_index(
      &store, make_config(), std::move(index));
  auto backend = std::make_unique<RecordingBackend>();
  RecordingBackend* observer = backend.get();
  KVCacheStoreTestPeer::set_backend(&store, std::move(backend));
  // Deliberately no mark_initialized().
  observer->reply = {1, 1};

  std::vector<BlockTransferInfo> blocks = {make_block(1, /*dst=*/0)};
  Slice<BlockTransferInfo> slice(blocks);
  EXPECT_EQ(store.batch_get_with_status(slice), std::vector<uint8_t>({0}));
  EXPECT_EQ(observer->batch_get_calls, 0);
}

}  // namespace
}  // namespace xllm
