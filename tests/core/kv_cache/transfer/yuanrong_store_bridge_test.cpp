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

// Exercises the production bridge itself, not a replacement for it.
// `fake_datasystem/datasystem/kv_client.h` stands in for the SDK header so
// yuanrong_store_bridge.cpp compiles unchanged; the tests drive the five C ABI
// entry points and inspect the SDK calls through the fake's state.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/kv_cache/transfer/store_config_validation.h"
#include "fake_datasystem/datasystem/kv_client.h"

extern "C" {

// Mirrors XllmYrStoreBuffer in yuanrong_store_bridge.cpp.
struct XllmYrStoreBuffer {
  void* const* addresses;
  const uint64_t* sizes;
  size_t count;
};

void* xllm_yr_store_create(const char* host,
                           int32_t port,
                           int32_t request_timeout_ms);
void xllm_yr_store_destroy(void* opaque);
bool xllm_yr_store_mset(void* opaque,
                        const char* const* keys,
                        const XllmYrStoreBuffer* buffers,
                        size_t count);
bool xllm_yr_store_mget(void* opaque,
                        const char* const* keys,
                        XllmYrStoreBuffer* buffers,
                        uint8_t* statuses,
                        size_t count);
const char* xllm_yr_store_last_error(void* opaque);

}  // extern "C"

namespace xllm {
namespace {

using datasystem::fake::GetFillMode;
using datasystem::fake::state;

// A block is several non-contiguous host ranges. The bridge must join them
// for MSet and split a returned payload back into exactly those ranges.
class Ranges final {
 public:
  explicit Ranges(std::vector<std::string> parts) : parts_(std::move(parts)) {
    addresses_.reserve(parts_.size());
    sizes_.reserve(parts_.size());
    for (std::string& part : parts_) {
      addresses_.emplace_back(part.data());
      sizes_.emplace_back(part.size());
    }
  }

  XllmYrStoreBuffer abi() {
    return XllmYrStoreBuffer{addresses_.data(), sizes_.data(), sizes_.size()};
  }

  std::string joined() const {
    std::string result;
    for (const std::string& part : parts_) {
      result.append(part);
    }
    return result;
  }

  const std::vector<std::string>& parts() const { return parts_; }

  void replace_address(size_t index, void* address) {
    addresses_[index] = address;
  }

 private:
  std::vector<std::string> parts_;
  std::vector<void*> addresses_;
  std::vector<uint64_t> sizes_;
};

bool error_contains(void* client, const std::string& fragment) {
  return std::string(xllm_yr_store_last_error(client)).find(fragment) !=
         std::string::npos;
}

class YuanRongStoreBridgeTest : public ::testing::Test {
 protected:
  void SetUp() override { datasystem::fake::reset(); }

  void TearDown() override {
    if (client_ != nullptr) {
      xllm_yr_store_destroy(client_);
      client_ = nullptr;
    }
  }

  void create(const char* host = "127.0.0.1",
              int32_t port = 31501,
              int32_t timeout_ms = 500) {
    client_ = xllm_yr_store_create(host, port, timeout_ms);
  }

  void* client_ = nullptr;
};

TEST_F(YuanRongStoreBridgeTest, SeparatesConnectAndRequestTimeouts) {
  create("10.0.0.4", 31501, 750);
  ASSERT_NE(client_, nullptr);
  ASSERT_TRUE(state().connect_options_seen);
  EXPECT_EQ(state().connect_options.host, "10.0.0.4");
  EXPECT_EQ(state().connect_options.port, 31501);
  // A one-off start-up cost, deliberately not the request deadline.
  EXPECT_EQ(state().connect_options.connectTimeoutMs, 10000);
  EXPECT_EQ(state().connect_options.requestTimeoutMs, 750);
}

TEST_F(YuanRongStoreBridgeTest, EmptyHostFallsBackToLoopback) {
  create("", 0, 500);
  ASSERT_NE(client_, nullptr);
  EXPECT_EQ(state().connect_options.host, "127.0.0.1");

  xllm_yr_store_destroy(client_);
  client_ = nullptr;
  datasystem::fake::reset();
  create(nullptr, 0, 500);
  ASSERT_NE(client_, nullptr);
  EXPECT_EQ(state().connect_options.host, "127.0.0.1");
}

TEST_F(YuanRongStoreBridgeTest, ClampsNonPositiveRequestTimeout) {
  for (const int32_t requested : {0, -1}) {
    datasystem::fake::reset();
    xllm_yr_store_destroy(client_);
    client_ = nullptr;
    create("127.0.0.1", 31501, requested);
    ASSERT_NE(client_, nullptr);
    EXPECT_EQ(state().connect_options.requestTimeoutMs,
              kDefaultStoreRequestTimeoutMs);
  }
}

TEST_F(YuanRongStoreBridgeTest, InitFailureIsReportedThroughLastError) {
  state().init_error = true;
  state().init_error_text = "worker unreachable";
  create();
  EXPECT_EQ(client_, nullptr);
  EXPECT_TRUE(error_contains(nullptr, "worker unreachable"));
}

TEST_F(YuanRongStoreBridgeTest, MSetFlattensRangesInOrder) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges first({"abc", "def"});
  Ranges second({"gh"});
  std::vector<const char*> keys = {"blk0", "blk1"};
  std::vector<XllmYrStoreBuffer> buffers = {first.abi(), second.abi()};

  ASSERT_TRUE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 2U));
  EXPECT_EQ(state().mset_calls, 1);
  ASSERT_EQ(state().last_mset_keys.size(), 2U);
  EXPECT_EQ(state().last_mset_keys[0], "blk0");
  ASSERT_EQ(state().last_mset_payloads.size(), 2U);
  EXPECT_EQ(state().last_mset_payloads[0], "abcdef");
  EXPECT_EQ(state().last_mset_payloads[1], "gh");
  EXPECT_EQ(state().last_write_mode,
            datasystem::WriteMode::NONE_L2_CACHE_EVICT);
  EXPECT_EQ(state().last_existence, datasystem::ExistenceOpt::NX);
}

TEST_F(YuanRongStoreBridgeTest, MSetSkipsZeroSizedRanges) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"ab", "", "cd"});
  std::vector<const char*> keys = {"blk0"};
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};

  ASSERT_TRUE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 1U));
  ASSERT_EQ(state().last_mset_payloads.size(), 1U);
  EXPECT_EQ(state().last_mset_payloads[0], "abcd");
}

TEST_F(YuanRongStoreBridgeTest, MSetRejectsNullInputs) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"ab"});
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};
  std::vector<const char*> keys = {"blk0"};

  EXPECT_FALSE(xllm_yr_store_mset(nullptr, keys.data(), buffers.data(), 1U));
  EXPECT_FALSE(xllm_yr_store_mset(client_, nullptr, buffers.data(), 1U));
  EXPECT_FALSE(xllm_yr_store_mset(client_, keys.data(), nullptr, 1U));
  EXPECT_TRUE(error_contains(client_, "null"));

  std::vector<const char*> null_key = {nullptr};
  EXPECT_FALSE(
      xllm_yr_store_mset(client_, null_key.data(), buffers.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "null key"));
}

TEST_F(YuanRongStoreBridgeTest, MSetCatchesFlattenOverflow) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"a", "b"});
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};
  std::vector<uint64_t> overflowing = {std::numeric_limits<uint64_t>::max(), 1};
  buffers[0].sizes = overflowing.data();
  std::vector<const char*> keys = {"blk0"};

  EXPECT_FALSE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "overflow"));
  EXPECT_EQ(state().mset_calls, 0);
}

TEST_F(YuanRongStoreBridgeTest, MSetCatchesNullRange) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"ab", "cd"});
  payload.replace_address(1, nullptr);  // deliberate bad input
  XllmYrStoreBuffer buffer = payload.abi();
  std::vector<XllmYrStoreBuffer> buffers = {buffer};
  std::vector<const char*> keys = {"blk0"};

  EXPECT_FALSE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "null range"));
}

TEST_F(YuanRongStoreBridgeTest, MSetReportsSdkErrorAndFailedKeys) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"ab"});
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};
  std::vector<const char*> keys = {"blk0"};

  state().mset_error = true;
  state().mset_error_text = "metadata unavailable";
  EXPECT_FALSE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "metadata unavailable"));

  state().mset_error = false;
  state().mset_failed_keys = {"blk0"};
  EXPECT_FALSE(xllm_yr_store_mset(client_, keys.data(), buffers.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "failed=1"));
}

TEST_F(YuanRongStoreBridgeTest, IndexAssignmentFillsOnlyHitKeys) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kAssignAtIndex;
  state().get_payloads = {"abcd", "", "efgh"};

  Ranges first({"aa", "aa"});
  Ranges missing({"bb", "bb"});
  Ranges third({"cc", "cc"});
  std::vector<const char*> keys = {"blk0", "blk1", "blk2"};
  std::vector<XllmYrStoreBuffer> buffers = {
      first.abi(), missing.abi(), third.abi()};
  std::vector<uint8_t> statuses(3U, 9U);

  ASSERT_TRUE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 3U));
  EXPECT_EQ(state().get_input_vals_size, 3U);
  EXPECT_EQ(statuses, std::vector<uint8_t>({1U, 0U, 1U}));
  EXPECT_EQ(first.joined(), "abcd");
  EXPECT_EQ(third.joined(), "efgh");
  EXPECT_EQ(missing.joined(), "bbbb");  // a miss must not write anything
}

TEST_F(YuanRongStoreBridgeTest, ClearAndAppendFillsOnlyHitKeys) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kClearAndAppend;
  state().get_payloads = {"abcd", "", "efgh"};

  Ranges first({"aa", "aa"});
  Ranges missing({"bb", "bb"});
  Ranges third({"cc", "cc"});
  std::vector<const char*> keys = {"blk0", "blk1", "blk2"};
  std::vector<XllmYrStoreBuffer> buffers = {
      first.abi(), missing.abi(), third.abi()};
  std::vector<uint8_t> statuses(3U, 9U);

  ASSERT_TRUE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 3U));
  EXPECT_EQ(statuses, std::vector<uint8_t>({1U, 0U, 1U}));
  EXPECT_EQ(first.joined(), "abcd");
  EXPECT_EQ(third.joined(), "efgh");
}

TEST_F(YuanRongStoreBridgeTest, RejectsAppendWithoutClearInsteadOfAllMiss) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kAppendWithoutClear;
  state().get_payloads = {"abcd", "efgh"};

  Ranges first({"aa", "aa"});
  Ranges second({"bb", "bb"});
  std::vector<const char*> keys = {"blk0", "blk1"};
  std::vector<XllmYrStoreBuffer> buffers = {first.abi(), second.abi()};
  std::vector<uint8_t> statuses(2U, 9U);

  // Pre-sized empty slots would otherwise be read as a silent full miss.
  EXPECT_FALSE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 2U));
  EXPECT_EQ(statuses, std::vector<uint8_t>({0U, 0U}));
  EXPECT_TRUE(error_contains(client_, "append semantics are unsupported"));
  EXPECT_EQ(first.joined(), "aaaa");
}

TEST_F(YuanRongStoreBridgeTest, ShortResultTreatsTailKeysAsMisses) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kTruncate;
  state().get_payloads = {"abcd"};
  state().truncate_to = 1;

  Ranges first({"aa", "aa"});
  Ranges second({"bb", "bb"});
  std::vector<const char*> keys = {"blk0", "blk1"};
  std::vector<XllmYrStoreBuffer> buffers = {first.abi(), second.abi()};
  std::vector<uint8_t> statuses(2U, 9U);

  ASSERT_TRUE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 2U));
  EXPECT_EQ(statuses, std::vector<uint8_t>({1U, 0U}));
  EXPECT_EQ(first.joined(), "abcd");
  EXPECT_EQ(second.joined(), "bbbb");
}

TEST_F(YuanRongStoreBridgeTest, SizeMismatchIsAPerKeyMiss) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kAssignAtIndex;
  state().get_payloads = {"abc", "efgh"};

  Ranges wrong_size({"aa", "aa"});  // expects 4 bytes, got 3
  Ranges good({"cc", "cc"});
  std::vector<const char*> keys = {"blk0", "blk1"};
  std::vector<XllmYrStoreBuffer> buffers = {wrong_size.abi(), good.abi()};
  std::vector<uint8_t> statuses(2U, 9U);

  ASSERT_TRUE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 2U));
  EXPECT_EQ(statuses, std::vector<uint8_t>({0U, 1U}));
  EXPECT_EQ(wrong_size.joined(), "aaaa");
  EXPECT_EQ(good.joined(), "efgh");
}

TEST_F(YuanRongStoreBridgeTest, MGetSdkErrorFailsTheWholeBatch) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_error = true;
  state().get_error_text = "worker lost";

  Ranges payload({"aa", "aa"});
  std::vector<const char*> keys = {"blk0"};
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};
  std::vector<uint8_t> statuses(1U, 9U);

  EXPECT_FALSE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 1U));
  EXPECT_EQ(statuses, std::vector<uint8_t>({0U}));
  EXPECT_TRUE(error_contains(client_, "worker lost"));
}

TEST_F(YuanRongStoreBridgeTest, MGetRejectsNullInputs) {
  create();
  ASSERT_NE(client_, nullptr);
  Ranges payload({"aa", "aa"});
  std::vector<XllmYrStoreBuffer> buffers = {payload.abi()};
  std::vector<const char*> keys = {"blk0"};
  std::vector<uint8_t> statuses(1U, 9U);

  EXPECT_FALSE(xllm_yr_store_mget(
      nullptr, keys.data(), buffers.data(), statuses.data(), 1U));
  EXPECT_FALSE(xllm_yr_store_mget(
      client_, nullptr, buffers.data(), statuses.data(), 1U));
  EXPECT_FALSE(
      xllm_yr_store_mget(client_, keys.data(), nullptr, statuses.data(), 1U));
  EXPECT_FALSE(
      xllm_yr_store_mget(client_, keys.data(), buffers.data(), nullptr, 1U));
  EXPECT_TRUE(error_contains(client_, "null"));

  std::vector<const char*> null_key = {nullptr};
  EXPECT_FALSE(xllm_yr_store_mget(
      client_, null_key.data(), buffers.data(), statuses.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "null key"));
}

TEST_F(YuanRongStoreBridgeTest, MGetCatchesNullDestinationRange) {
  create();
  ASSERT_NE(client_, nullptr);
  state().get_fill_mode = GetFillMode::kAssignAtIndex;
  state().get_payloads = {"abcd"};

  Ranges payload({"aa", "aa"});
  payload.replace_address(1, nullptr);  // deliberate bad input
  XllmYrStoreBuffer buffer = payload.abi();
  std::vector<XllmYrStoreBuffer> buffers = {buffer};
  std::vector<const char*> keys = {"blk0"};
  std::vector<uint8_t> statuses(1U, 9U);

  EXPECT_FALSE(xllm_yr_store_mget(
      client_, keys.data(), buffers.data(), statuses.data(), 1U));
  EXPECT_TRUE(error_contains(client_, "null range"));
}

TEST_F(YuanRongStoreBridgeTest, DestroyShutsDownTheClient) {
  create();
  ASSERT_NE(client_, nullptr);
  xllm_yr_store_destroy(client_);
  client_ = nullptr;
  EXPECT_EQ(state().shutdown_calls, 1);
}

}  // namespace
}  // namespace xllm
