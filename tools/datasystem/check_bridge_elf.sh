#!/usr/bin/env bash
# Pre-package gate for libxllm_yuanrong_store_bridge.so.
#
# The v4 deployment regressed when a newly built package overwrote a local
# patch: the bridge inherited the main binary's NPU DT_NEEDED entries, and the
# loader pulled the whole runtime into the process. This script re-checks the
# shipped artifact (not just the build tree) before it goes to the test nodes.
#
# Usage: tools/datasystem/check_bridge_elf.sh /path/to/libxllm_yuanrong_store_bridge.so

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <libxllm_yuanrong_store_bridge.so>" >&2
  exit 2
fi

LIB="$(realpath "$1")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMAKE_CHECK="$REPO_ROOT/xllm/core/kv_cache/transfer/check_yuanrong_bridge.cmake"

for tool in readelf nm strings cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "FAIL: $tool is required" >&2
    exit 2
  fi
done
if [[ ! -f "$LIB" ]]; then
  echo "FAIL: $LIB does not exist" >&2
  exit 2
fi

failures=0
echo "artifact: $LIB"

needed="$(readelf -d "$LIB" | grep '(NEEDED)' || true)"
echo "DT_NEEDED:"
echo "$needed" | sed 's/^/  /'
for forbidden in cust_opapi ascend libtorch libgraph libge_ libtbe libxllm; do
  if grep -qi "$forbidden" <<<"$needed"; then
    echo "  FAIL: forbidden DT_NEEDED entry matching '$forbidden'"
    failures=$((failures + 1))
  fi
done

exported="$(nm -D --defined-only "$LIB" | awk '{print $NF}' | grep -v '^$' || true)"
echo "exported symbols:"
echo "$exported" | sed 's/^/  /'
while IFS= read -r symbol; do
  case "$symbol" in
    xllm_yr_store_*|_init|_fini|_edata|_end|__bss_start*|__end__|__bss_end__|_DYNAMIC) ;;
    *)
      echo "  FAIL: unexpected exported symbol '$symbol'"
      failures=$((failures + 1))
      ;;
  esac
done <<<"$exported"
if ! grep -q '^xllm_yr_store_create$' <<<"$exported"; then
  echo "  FAIL: xllm_yr_store_create is missing"
  failures=$((failures + 1))
fi

# dlmopen was a local experiment (namespace isolation) that was deliberately
# not merged. A packaged binary must still use ordinary dlopen.
if nm -D "$LIB" | grep -q 'dlmopen'; then
  echo "  FAIL: bridge references dlmopen (unmerged local experiment)"
  failures=$((failures + 1))
fi
if strings "$LIB" | grep -q 'dlmopen'; then
  echo "  WARN: 'dlmopen' string present in the artifact; inspect it manually"
fi

echo "build-time link contract:"
if cmake -DBRIDGE_LIBRARY="$LIB" -P "$CMAKE_CHECK"; then
  echo "  OK"
else
  echo "  FAIL: build-time link contract rejected the artifact"
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  echo "RESULT: FAIL ($failures finding(s))"
  exit 1
fi
echo "RESULT: PASS"
