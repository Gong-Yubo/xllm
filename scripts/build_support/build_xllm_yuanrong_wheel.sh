#!/usr/bin/env bash
# Build an xllm wheel that ships the YuanRong KV cache store backend.
#
#   ./build_xllm_yuanrong_wheel.sh --device npu
#   ./build_xllm_yuanrong_wheel.sh --device cuda
#
# Environment:
#   YUANRONG_ROOT   YuanRong SDK root (the dir holding include/ and lib/).
#                   Auto-detected from the installed openyuanrong_datasystem
#                   wheel when unset.
#   CMAKE_ARGS      extra cmake arguments, appended to the ones set here.
#
# The xllm binary cannot be built without a device toolchain: the CUDA path
# needs a CUDA-enabled torch (NCCL symbols), the NPU path needs CANN. This
# script fails fast with a clear message instead of letting cmake fail deep in
# a third-party target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="npu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[build] %s\n' "$*"; }
die() { printf '[build] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- SDK --------
if [[ -z "${YUANRONG_ROOT:-}" ]]; then
  YUANRONG_ROOT="$(python3 - <<'PY' 2>/dev/null || true
import os
import yr.datasystem
print(os.path.dirname(yr.datasystem.__file__))
PY
)"
fi
[[ -n "${YUANRONG_ROOT}" ]] || die "YuanRong SDK not found. Install openyuanrong_datasystem-*.whl or set YUANRONG_ROOT."
[[ -f "${YUANRONG_ROOT}/include/datasystem/kv_client.h" ]] || die "${YUANRONG_ROOT} has no include/datasystem/kv_client.h"
if [[ -f "${YUANRONG_ROOT}/lib/libdatasystem.so" ]]; then
  YUANRONG_LIB_DIR="${YUANRONG_ROOT}/lib"
elif [[ -f "${YUANRONG_ROOT}/lib64/libdatasystem.so" ]]; then
  YUANRONG_LIB_DIR="${YUANRONG_ROOT}/lib64"
else
  die "${YUANRONG_ROOT} has no lib/libdatasystem.so"
fi
log "YuanRong SDK: ${YUANRONG_ROOT}"

# ----------------------------------------------------------- toolchain ------
case "${DEVICE}" in
  npu)
    CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
    [[ -d "${CANN}" ]] || die "CANN not found at ${CANN}; export ASCEND_HOME_PATH or source set_env.sh"
    python3 -c 'import torch_npu' 2>/dev/null || die "torch_npu is not importable; the NPU build needs it"
    ;;
  cuda)
    python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
      || die "torch has no CUDA (torch.version.cuda is empty); install a CUDA build of torch"
    ;;
  mlu|ilu|musa|dcu|maca)
    log "device ${DEVICE}: relying on its own SDK being on the include/link path"
    ;;
  *) die "unsupported device: ${DEVICE}" ;;
esac
log "device: ${DEVICE}"

# --------------------------------------------------------------- build ------
cd "${ROOT}"
log "cmake args: -DUSE_YUANRONG=ON -DYUANRONG_ROOT=${YUANRONG_ROOT} ${CMAKE_ARGS:-}"
CMAKE_ARGS="-DUSE_YUANRONG=ON -DYUANRONG_ROOT=${YUANRONG_ROOT} ${CMAKE_ARGS:-}" \
  python3 setup.py --device "${DEVICE}" bdist_wheel

WHEEL="$(ls -t "${ROOT}"/dist/*.whl 2>/dev/null | head -1 || true)"
[[ -n "${WHEEL}" ]] || die "no wheel produced under dist/"
log "wheel: ${WHEEL}"

# --------------------------------------------------------------- verify -----
# A wheel without the bridge would silently disable the backend at runtime, so
# the build is only considered successful when the shim is actually inside.
python3 - "${WHEEL}" "${ROOT}" <<'PY'
import os
import sys
import zipfile

wheel_path, root = sys.argv[1], sys.argv[2]
bridge_name = "libxllm_yuanrong_store_bridge.so"

with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()

hits = [n for n in names if n.endswith(bridge_name)]
if not hits:
    # The bridge is built into the cmake output dir. If setup.py did not pick it
    # up, drop it next to the xllm binary so $ORIGIN still resolves it.
    candidates = [n for n in names if n.endswith("core/server/xllm")]
    if not candidates:
        sys.exit("wheel has no xllm binary either; nothing to attach the bridge to")
    target = candidates[0][: -len("xllm")] + bridge_name
    source = os.path.join(root, "xllm", "core", "server", bridge_name)
    if not os.path.isfile(source):
        sys.exit(f"bridge not found at {source}; run the build again")
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.write(source, target)
    print(f"[verify] injected {bridge_name} -> {target}")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
    hits = [n for n in names if n.endswith(bridge_name)]

if not hits:
    sys.exit("wheel still has no bridge after injection")
print(f"[verify] {bridge_name} present at {hits[0]}")
PY

log "done"
log "install:  pip install ${WHEEL}"
log "runtime:  export LD_LIBRARY_PATH=${YUANRONG_LIB_DIR}:\$LD_LIBRARY_PATH"
log "check:    xllm --store_backend=yuanrong --store_yuanrong_host=<worker> --store_yuanrong_port=<port>"
