include(cc_library)

cc_library(
  NAME
    cache_layout
  HDRS
    cache_layout.h
  SRCS
    cache_layout.cpp
  DEPS
    :common
    glog::glog
    proto::xllm_proto
)

cc_library(
  NAME
    reshard_planner
  HDRS
    reshard_planner.h
  SRCS
    reshard_planner.cpp
  DEPS
    :cache_layout
    :common
    :transfer_types
    glog::glog
)

cc_library(
  NAME
    push_route
  HDRS
    push_route.h
  SRCS
    push_route.cpp
)

cc_library(
  NAME
    pd_topology_guard
  HDRS
    pd_topology_guard.h
  SRCS
    pd_topology_guard.cpp
  DEPS
    :cache_layout
    :common
    glog::glog
)

cc_library(
  NAME
    kv_transfer_completion
  HDRS
    kv_transfer_completion.h
  SRCS
    kv_transfer_completion.cpp
  DEPS
    glog::glog
    Folly::folly
)

cc_library(
  NAME
    kv_cache_transfer
  HDRS
    kv_cache_transfer.h
    kv_cache_transfer_coordinator_base.h
    virtual_kv_cache_transfer_memory_provider.h
    kv_cache_store.h
    mooncake_store_backend.h
    host_transfer/layout.h
    host_transfer/transfer.h
    host_transfer/basic_transfer.h
    host_transfer/compact_transfer.h
    prefetch_result.h
    kv_transport_codec.h
    hierarchy_kv_cache_transfer.h
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer.h>
    mooncake_kv_cache_transfer_engine.h
    kv_cache_store_backend.h
    $<$<BOOL:${USE_YUANRONG}>:yuanrong_store_backend.h>
  SRCS
    kv_cache_transfer.cpp
    kv_transport_codec.cpp
    kv_cache_store.cpp
    mooncake_store_backend.cpp
    $<$<BOOL:${USE_YUANRONG}>:yuanrong_store_backend.cpp>
    host_transfer/layout.cpp
    host_transfer/transfer.cpp
    host_transfer/transfer_utils.cpp
    host_transfer/basic_transfer.cpp
    host_transfer/compact_transfer.cpp
    host_transfer/compact_load_executor.cpp
    host_transfer/compact_offload_executor.cpp
    hierarchy_kv_cache_transfer.cpp
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer.cpp>
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer_engine.cpp>
  DEPS
    :cache_layout
    :common
    :kv_cache
    :mooncake_transfer_engine
    :push_route
    :reshard_planner
    :util
    :platform
    $<$<BOOL:${USE_NPU}>:graph>
    glog::glog
    $<$<BOOL:${USE_NPU}>:c_sec>
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    mooncake_store
    proto::xllm_proto
    $<$<BOOL:${USE_NPU}>:platform_npu>
    $<$<BOOL:${USE_DCU}>:hip::host>
    $<$<BOOL:${USE_MLU}>:mlu_kernels>
    $<$<BOOL:${USE_YUANRONG}>:${CMAKE_DL_LIBS}>
)

# The YuanRong SDK is never linked into the main binary: YuanRongStoreBackend
# talks to it through a pure C ABI bridge, and only this shim links
# libdatasystem.
if(USE_YUANRONG)
  file(WRITE "${CMAKE_BINARY_DIR}/xllm_yuanrong_bridge_symbols.map"
       "{\n  global:\n    xllm_yr_store_*;\n  local:\n    *;\n};\n")
  add_library(xllm_yuanrong_store_bridge SHARED
    yuanrong_store_bridge.cpp
  )
  target_include_directories(xllm_yuanrong_store_bridge
    PRIVATE
      ${COMMON_INCLUDE_DIRS}
      "${YUANRONG_INCLUDE_DIR}"
  )
  target_link_libraries(xllm_yuanrong_store_bridge
    PRIVATE
      "${YUANRONG_LIBRARY}"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(xllm_yuanrong_store_bridge
      PRIVATE
        # The top-level link_libraries(cust_opapi) and --export-dynamic used by
        # the main binary would otherwise leave unused Ascend/NPU DT_NEEDED
        # entries on this shim. dlopen would then pull that runtime into the
        # process. --as-needed drops them; the version script hides symbols.
        "LINKER:--as-needed"
        "LINKER:--version-script=${CMAKE_BINARY_DIR}/xllm_yuanrong_bridge_symbols.map")
  endif()
  option(XLLM_YUANRONG_LINK_CHECK
         "Assert the YuanRong bridge's DT_NEEDED/exported-symbol contract after linking"
         ON)
  if(XLLM_YUANRONG_LINK_CHECK AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_custom_command(TARGET xllm_yuanrong_store_bridge POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        "-DBRIDGE_LIBRARY=$<TARGET_FILE:xllm_yuanrong_store_bridge>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_yuanrong_bridge.cmake"
      COMMENT "Checking the YuanRong bridge link contract"
      VERBATIM)
  endif()
  set_target_properties(xllm_yuanrong_store_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xllm/core/server"
    BUILD_RPATH "$ORIGIN;${YUANRONG_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN;${YUANRONG_LIB_DIR}"
  )
  # dlopen resolves a bare library name through the caller's RUNPATH, which is
  # $ORIGIN, so the shim has to sit next to the xllm artifact.
  if(GENERATE_SO)
    set(XLLM_YUANRONG_BRIDGE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
  else()
    set(XLLM_YUANRONG_BRIDGE_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
  endif()
  install(TARGETS xllm_yuanrong_store_bridge
    LIBRARY DESTINATION ${XLLM_YUANRONG_BRIDGE_INSTALL_DIR})
endif()
