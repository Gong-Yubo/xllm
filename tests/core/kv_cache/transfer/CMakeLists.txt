include(cc_test)

# Pure configuration guards: run in every build, independent of the store
# backend that was compiled in.
cc_test(
  NAME
    store_config_validation_test
  CPU_ONLY
  SRCS
    store_config_validation_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    store_ranges_test
  CPU_ONLY
  SRCS
    store_ranges_test.cpp
  DEPS
    GTest::gtest_main
)

# The production bridge is compiled unchanged; only the SDK header behind its
# C ABI is replaced. The backend tests swap out the whole bridge, so without
# this target the flattening, error handling and per-key status mapping are
# never exercised on a CPU build or without a YuanRong worker.
cc_test(
  NAME
    yuanrong_store_bridge_test
  CPU_ONLY
  SRCS
    yuanrong_store_bridge_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/kv_cache/transfer/yuanrong_store_bridge.cpp
  DEPS
    GTest::gtest_main
)
target_include_directories(yuanrong_store_bridge_test BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/fake_datasystem
)

cc_test(
  NAME
    host_kv_transfer_test
  CPU_ONLY
  SRCS
    host_transfer/transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    push_route_test
  CPU_ONLY
  SRCS
    push_route_test.cpp
  DEPS
    :push_route
    GTest::gtest_main
)

cc_test(
  NAME
    pd_topology_guard_test
  CPU_ONLY
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

cc_test(
  NAME
    kv_transfer_completion_test
  CPU_ONLY
  SRCS
    kv_transfer_completion_test.cpp
  DEPS
    :kv_transfer_completion
    GTest::gtest_main
)

cc_test(
  NAME
    reshard_planner_test
  CPU_ONLY
  SRCS
    reshard_planner_test.cpp
  DEPS
    :reshard_planner
    GTest::gtest_main
)

if(USE_MLU)
cc_test(
  NAME
    compact_load_test
  SRCS
    host_transfer/compact_load_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    compact_offload_test
  SRCS
    host_transfer/compact_offload_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    kv_cache_store_test
  CPU_ONLY
  SRCS
    kv_cache_store_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

if(USE_YUANRONG)
  # The backend only needs the C ABI, so this test compiles it directly instead
  # of dragging in the whole transfer library. A stub bridge stands in for a
  # YuanRong cluster: only the service behind the ABI is replaced, the backend
  # under test is production code.
  cc_test(
    NAME
      yuanrong_store_backend_test
    CPU_ONLY
    SRCS
      yuanrong_store_backend_test.cpp
      ${PROJECT_SOURCE_DIR}/xllm/core/kv_cache/transfer/yuanrong_store_backend.cpp
    DEPS
      glog::glog
      GTest::gtest_main
    LINKOPTS
      ${CMAKE_DL_LIBS}
  )
  add_library(xllm_yuanrong_store_bridge_stub SHARED
    yuanrong_store_bridge_stub.cpp
  )
  set_target_properties(xllm_yuanrong_store_bridge_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  add_dependencies(yuanrong_store_backend_test xllm_yuanrong_store_bridge_stub)
  target_compile_definitions(yuanrong_store_backend_test PRIVATE
    "XLLM_YUANRONG_BRIDGE_STUB_PATH=\"$<TARGET_FILE:xllm_yuanrong_store_bridge_stub>\""
  )
endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    basic_host_kv_transfer_test
  SRCS
    host_transfer/basic_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    hierarchy_kv_cache_transfer_test
  SRCS
    hierarchy_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)
endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    mooncake_kv_cache_transfer_test
  SRCS
    mooncake_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)
if(USE_NPU)
  # The controller and peer must inherit port isolation before runtime startup.
  set_tests_properties(
    MooncakeKVCacheTransferDefaultTest.NpuLinearIndexerScalePushAndPullRoundTrip
    PROPERTIES RUN_SERIAL TRUE
               ENVIRONMENT "HCCL_NPU_SOCKET_PORT_RANGE=auto"
  )
endif()
endif()
