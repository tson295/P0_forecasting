#!/usr/bin/env bash
# Bootstrap môi trường trên Vast.ai (Ubuntu + CUDA). Chạy trong tmux:  bash scripts/vast_bootstrap.sh
#
# FAIL-FAST: mọi blocker đều exit non-zero (không print "DỪNG" rồi chạy tiếp).
# Training chỉ GPU (plan §0) — nhưng KHÔNG hard-code một backend: script tự RESOLVE backend GPU hợp lệ cho từng
# thư viện rồi GHI vào config của run ($CFG, mặc định configs/p0_full.json) để mọi đường dùng nhất quán:
#   torch/LSTM/TimesFM : CUDA
#   XGBoost            : device=cuda
#   CatBoost           : task_type=GPU
#   jax (xreg TimesFM) : cuda12 (XLA_PYTHON_CLIENT_PREALLOCATE=false)
#   LightGBM           : device_type = gpu (build OpenCL) HOẶC cuda (build CUDA) — cái nào FIT THẬT được trên máy này;
#                        backend đó được truyền sang cả AutoTS-WR probe và mọi template LightGBM của bake-off.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/p0_full.json}
mkdir -p experiments
ENV_FILE=experiments/env.txt
: > "$ENV_FILE"
log() { echo "$@" | tee -a "$ENV_FILE"; }
die() { echo "BOOTSTRAP FAIL: $*" | tee -a "$ENV_FILE" >&2; exit 1; }

log "== GPU =="
command -v nvidia-smi >/dev/null 2>&1 || die "không có nvidia-smi → không có GPU (cấm training CPU)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$ENV_FILE" \
  || die "nvidia-smi lỗi"

log "== apt: OpenCL + boost (cho LightGBM build GPU) =="
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  apt-get update -qq || die "apt-get update lỗi"
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      ocl-icd-opencl-dev opencl-headers clinfo libboost-dev libboost-system-dev libboost-filesystem-dev \
      cmake build-essential >/dev/null || die "apt-get install lỗi"
  mkdir -p /etc/OpenCL/vendors
  [ -f /etc/OpenCL/vendors/nvidia.icd ] || echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
else
  # Container không chạy bằng root (không có passwordless sudo) → không apt được. Không đổi methodology:
  # toolchain build phải CÓ SẴN, nếu thiếu thì die. Boost (chỉ cần cho build OpenCL) có thể thiếu → build
  # OpenCL sẽ fail và script tự fallback sang USE_CUDA; backend cuối cùng vẫn do bước RESOLVE (fit thật) quyết định.
  log "apt: bỏ qua (uid=$(id -u), không phải root) — dùng toolchain có sẵn"
  command -v cmake >/dev/null 2>&1 || die "thiếu cmake và không có quyền apt"
  command -v g++ >/dev/null 2>&1   || die "thiếu g++ và không có quyền apt"
  [ -e /usr/include/CL/cl.h ]      || die "thiếu OpenCL headers và không có quyền apt"
  ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1 || log "không thấy OpenCL ICD"
  [ -e /usr/include/boost/version.hpp ] || log "thiếu Boost → build LightGBM OpenCL sẽ fail, fallback USE_CUDA"
fi
clinfo -l 2>/dev/null | tee -a "$ENV_FILE" || log "clinfo: không thấy platform OpenCL (sẽ thử build CUDA cho LightGBM)"

log "== pip =="
python -m pip install -q -U pip || die "pip upgrade lỗi"
python -m pip install -q -r requirements.txt || die "pip install requirements lỗi"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "torch CUDA chưa dùng được → cài wheel cu128"
  python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 || die "cài torch CUDA lỗi"
fi
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch.cuda.is_available() = False (LSTM/TimesFM bắt buộc CUDA)"

log "== package cho TimesFM + AutoTS (§2.2 #4/#6) =="
python -m pip install -q "timesfm[torch]==2.0.2" || die "cài timesfm 2.0.2 lỗi"
python -m pip install -q "autots==1.0.4" statsmodels || die "cài autots 1.0.4 + statsmodels lỗi"
python -m pip install -q "jax[cuda12]==0.11.1" || die "cài jax[cuda12]==0.11.1 lỗi (xreg của TimesFM trên GPU, quyết định 2026-09-01)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # jax không chiếm trước VRAM (torch/LoRA dùng chung GPU)
python -c "import jax, sys; d = jax.devices(); print('jax devices:', d); sys.exit(0 if any(k in str(d).lower() for k in ('cuda', 'gpu')) else 1)" | tee -a "$ENV_FILE" || die "jax không thấy GPU (xreg phải chạy GPU)"

log "== LightGBM: build GPU rồi RESOLVE backend thật sự fit được =="
if ! python -c "import lightgbm" 2>/dev/null; then LGB_NEED_BUILD=1; else LGB_NEED_BUILD=0; fi
build_lgb() {  # $1 = USE_GPU | USE_CUDA
  local extra=()
  if [ "$1" = "USE_CUDA" ]; then
    # Image này có NCCL 2.31 (CUDA 13, device ABI 8) trong khi nvcc là 12.8 (ABI 7) → nvlink từ chối
    # libnccl_static.a ("ABI version 8 is incompatible with target ABI version 7"). Link NCCL dạng SHARED thì
    # LightGBM không device-link object của NCCL nữa. Chỉ build đúng arch của GPU đang có (rút ngắn build).
    # Đây là fix môi trường build, KHÔNG đổi methodology: backend cuối vẫn do bước RESOLVE (fit thật) quyết định.
    extra+=(--config-settings=cmake.define.BUILD_WITH_SHARED_NCCL=ON)
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    [ -n "$cc" ] && extra+=(--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="$cc")
  fi
  python -m pip install -q --force-reinstall --no-binary lightgbm --config-settings=cmake.define."$1"=ON "${extra[@]}" lightgbm
}
if [ "${FORCE_LGB_BUILD:-1}" = "1" ] || [ "$LGB_NEED_BUILD" = "1" ]; then
  build_lgb USE_GPU || { log "build LightGBM OpenCL thất bại → thử build CUDA"; build_lgb USE_CUDA || die "không build được LightGBM GPU (cả OpenCL lẫn CUDA)"; }
fi

# Resolve: thử FIT THẬT với từng device_type; cái nào chạy được thì ghi vào config (không đoán theo build flag).
PYTHONPATH=src:. python - "$CFG" <<'PY' | tee -a "$ENV_FILE" || die "LightGBM không chạy được trên GPU với cả device_type=gpu lẫn cuda"
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
cfg_path = sys.argv[1]
import lightgbm as lgb
print("lightgbm", lgb.__version__)
x = np.random.default_rng(0).normal(size=(512, 6)).astype(np.float32)
y = (x[:, 0] * 0.3 + np.random.default_rng(1).normal(0, .1, 512)).astype(np.float32)
ok = None
for dev in ("gpu", "cuda"):
    try:
        lgb.LGBMRegressor(n_estimators=8, num_leaves=7, max_bin=63, device_type=dev, gpu_use_dp=False,
                          verbosity=-1).fit(x, y)
        print(f"LightGBM fit OK với device_type={dev}")
        ok = dev
        break
    except Exception as e:
        print(f"LightGBM device_type={dev} FAIL: {type(e).__name__}: {str(e)[:160]}")
if ok is None:
    sys.exit(1)
cfg = json.loads(open(cfg_path, encoding="utf-8").read())
cfg.setdefault("models", {}).setdefault("lgbm", {})["device_type"] = ok
open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print(f"→ đã ghi models.lgbm.device_type={ok} vào {cfg_path} (AutoTS-WR + template bake-off dùng chung backend này)")
PY

log "== preflight GPU: torch / XGBoost / CatBoost / LightGBM(B0 assert) =="
PYTHONPATH=src:. python - "$CFG" <<'PY' | tee -a "$ENV_FILE" || die "preflight GPU thất bại"
import importlib, json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
import importlib.metadata as _md
_DIST = {"sklearn": "scikit-learn"}  # tên module != tên distribution
for m in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "torch", "matplotlib",
          "timesfm", "autots", "statsmodels", "jax"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", None)
        if v is None:  # timesfm 2.0.2 không export __version__ → lấy từ metadata của distribution
            v = _md.version(_DIST.get(m, m))
        print(m, v)
    except Exception as e:
        print(m, "MISSING", e); sys.exit(1)
import torch
print("torch.cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
import xgboost as xgb
xgb.XGBRegressor(n_estimators=3, device="cuda", tree_method="hist").fit(x, x[:, 0]); print("XGBoost device=cuda: OK")
from catboost import CatBoostRegressor
CatBoostRegressor(iterations=3, task_type="GPU", verbose=False, allow_writing_files=False).fit(x, x[:, 0]); print("CatBoost task_type=GPU: OK")
from Baseline_LGBM import LGBMConfig, assert_p100_lightgbm
dev = cfg["models"]["lgbm"]["device_type"]
assert_p100_lightgbm(LGBMConfig(require_p100=False, device_type=dev)); print(f"LightGBM assert_p100 (device_type={dev}): OK")
PY

log "== git lfs =="
command -v git-lfs >/dev/null 2>&1 && git lfs install --local | tee -a "$ENV_FILE" || log "git-lfs không có: artifact .npz/.pt dưới experiments/ sẽ không push được qua LFS — cài git-lfs trước khi push"

log "== unit test (CPU, không training) =="
PYTHONPATH=src:. python -m pytest -q -x || die "unit test FAIL — không được training"

log "bootstrap OK — backend đã resolve và ghi vào $CFG; xem $ENV_FILE"
log "bước tiếp: PYTHONPATH=src:. python scripts/vast_canary.py --config $CFG"
