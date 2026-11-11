# Post-build contract check for libxllm_yuanrong_store_bridge.so.
#
# The main binary is linked with --export-dynamic and a global
# link_libraries(cust_opapi) (top-level CMakeLists.txt), so a shim built with
# the same link line can inherit Ascend/NPU DT_NEEDED entries. dlopen'ing that
# shim then pulls the whole NPU runtime into the process, which was one of the
# failures seen during the v4 deployment. LINKER:--as-needed removes the
# unused entries; the version script hides every symbol except xllm_yr_store_*.
#
# This script turns those two manual patches into a build-time assertion. It is
# invoked only on Linux + USE_YUANRONG builds, and can be disabled with
# -DXLLM_YUANRONG_LINK_CHECK=OFF if a toolchain lacks readelf/nm.
#
# Scope: native Linux/ELF builds whose .so the host's readelf/nm can parse and
# whose linker honours --as-needed/--version-script. Cross-compiled targets
# (host readelf against a foreign ELF), non-glibc or non-GNU/LLVM linkers and
# non-ELF outputs should disable the check; the dependency blacklist below is
# deliberately tied to this project's Ascend/NPU library names.

# Script mode (-P) has no project policy baseline, so opt in explicitly.
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

if(NOT DEFINED BRIDGE_LIBRARY OR NOT EXISTS "${BRIDGE_LIBRARY}")
  message(FATAL_ERROR
    "check_yuanrong_bridge: BRIDGE_LIBRARY is not set or does not exist")
endif()

find_program(XLLM_READELF NAMES readelf llvm-readelf)
find_program(XLLM_NM NAMES nm llvm-nm)
if(NOT XLLM_READELF OR NOT XLLM_NM)
  message(WARNING
    "check_yuanrong_bridge: readelf/nm unavailable; link contract NOT checked. "
    "Set -DXLLM_YUANRONG_LINK_CHECK=OFF to silence this warning.")
  return()
endif()

# --- 1. DT_NEEDED must not drag in the host binary's NPU/torch dependencies ---
execute_process(
  COMMAND "${XLLM_READELF}" -d "${BRIDGE_LIBRARY}"
  RESULT_VARIABLE readelf_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE readelf_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT readelf_result EQUAL 0)
  message(FATAL_ERROR
    "check_yuanrong_bridge: readelf -d failed: ${readelf_error}")
endif()

string(REGEX MATCHALL "\\(NEEDED\\)[^\r\n]*\\[([^]]+)\\]" needed_entries
       "${dynamic_section}")
set(needed_names "")
foreach(entry IN LISTS needed_entries)
  string(REGEX MATCH "\\[([^]]+)\\]" _match "${entry}")
  list(APPEND needed_names "${CMAKE_MATCH_1}")
endforeach()
message(STATUS "check_yuanrong_bridge: DT_NEEDED = ${needed_names}")

set(forbidden_patterns
    "cust_opapi"
    "ascend"
    "libtorch"
    "libgraph"
    "libge_"
    "libtbe"
    "libxllm")
foreach(name IN LISTS needed_names)
  string(TOLOWER "${name}" lower_name)
  foreach(pattern IN LISTS forbidden_patterns)
    if(lower_name MATCHES "${pattern}")
      message(FATAL_ERROR
        "check_yuanrong_bridge: '${name}' must not be a DT_NEEDED entry of the "
        "bridge. The shim must only reach the DataSystem SDK, not the main "
        "binary's NPU/torch dependencies. Check that --as-needed is active.")
    endif()
  endforeach()
endforeach()

# --- 2. Only the C ABI symbols may be exported ---
execute_process(
  COMMAND "${XLLM_NM}" -D --defined-only "${BRIDGE_LIBRARY}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE exported_symbols
  ERROR_VARIABLE nm_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "check_yuanrong_bridge: nm -D failed: ${nm_error}")
endif()

# Linkers add a handful of section markers even with a version script.
set(allowed_linker_symbols
    "_init" "_fini" "_edata" "_end" "__bss_start" "__bss_start__"
    "__end__" "__bss_end__" "_DYNAMIC")
string(REGEX REPLACE "\r?\n" ";" symbol_lines "${exported_symbols}")
foreach(line IN LISTS symbol_lines)
  string(STRIP "${line}" line)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[A-Za-z_][A-Za-z0-9_@.]*$" symbol "${line}")
  if(NOT symbol)
    continue()
  endif()
  if(symbol MATCHES "^xllm_yr_store_" OR symbol IN_LIST allowed_linker_symbols)
    continue()
  endif()
  message(FATAL_ERROR
    "check_yuanrong_bridge: '${symbol}' is exported from the bridge. Only "
    "xllm_yr_store_* may be global; check the version script.")
endforeach()

message(STATUS "check_yuanrong_bridge: link contract OK (${BRIDGE_LIBRARY})")
