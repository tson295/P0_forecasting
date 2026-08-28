#!/usr/bin/env bash
# Bootstrap môi trường trên Vast.ai (Ubuntu + CUDA). Chạy trong tmux: bash scripts/vast_bootstrap.sh
# Training chỉ GPU (plan §0): LightGBM phải là BUILD GPU (OpenCL); XGBoost device=cuda; CatBoost GPU; torch CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments
ENV_FILE=experiments/env.txt

echo "== GPU ==" | tee "$ENV_FILE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$ENV_FILE"

echo "== apt: OpenCL + boost (cho LightGBM GPU) =="
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ocl-icd-opencl-dev opencl-headers clinfo libboost-dev libboost-system-dev libboost-filesystem-dev cmake build-essential >/dev/null
  # ICD của NVIDIA trong container đôi khi thiếu file vendor
  mkdir -p /etc/OpenCL/vendors
  [ -f /etc/OpenCL/vendors/nvidia.icd ] || echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
fi
clinfo -l 2>/dev/null | tee -a "$ENV_FILE" || echo "clinfo: không thấy platform OpenCL (sẽ thử device_type=cuda)" | tee -a "$ENV_FILE"

echo "== pip =="
python -m pip install -q -U pip
python -m pip install -q -r requirements.txt
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
echo "== lightgbm GPU build =="
python -m pip install -q --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm || {
  echo "Build OpenCL thất bại → thử build CUDA (device_type=cuda trong configs/*.json → models.lgbm.device_type)";
  python -m pip install -q --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm;
}

echo "== preflight GPU (B0 assert_p100_lightgbm với require_p100=False) ==" | tee -a "$ENV_FILE"
PYTHONPATH=src:. python - <<'EOF' 2>&1 | tee -a experiments/env.txt
import importlib, torch
for m in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "torch", "matplotlib"]:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "MISSING", e)
print("torch.cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
from Baseline_LGBM import LGBMConfig, assert_p100_lightgbm
try:
    assert_p100_lightgbm(LGBMConfig(require_p100=False))
    print("LightGBM GPU preflight: OK (device_type=gpu)")
except Exception as e:
    print("LightGBM GPU preflight FAILED:", e)
    try:
        assert_p100_lightgbm(LGBMConfig(require_p100=False, device_type="cuda"))
        print("LightGBM CUDA preflight: OK (đặt models.lgbm.device_type=cuda trong config)")
    except Exception as e2:
        print("LightGBM CUDA preflight FAILED:", e2, "→ DỪNG: cấm training CPU")
import numpy as np, xgboost as xgb
x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
xgb.XGBRegressor(n_estimators=3, device="cuda", tree_method="hist").fit(x, x[:, 0]); print("XGBoost cuda: OK")
from catboost import CatBoostRegressor
CatBoostRegressor(iterations=3, task_type="GPU", verbose=False, allow_writing_files=False).fit(x, x[:, 0]); print("CatBoost GPU: OK")
EOF
echo "== unit test (CPU, không training) =="
PYTHONPATH=src:. python -m pytest -q -x || { echo "TEST FAIL — không chạy training"; exit 1; }
echo "bootstrap xong; xem $ENV_FILE"
