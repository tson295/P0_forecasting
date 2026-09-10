#!/usr/bin/env bash
# Environment setup actually used for the OB run on the Vast instance.
# Installs only; no probe fit, smoke run, benchmark or test. Log: ../logs/env_install.log
# Versions follow the stack previously recorded in experiments/15d/env.txt (torch cu128, LightGBM CUDA build).
set -euo pipefail
cd "$(dirname "$0")/../../.."
VENV=/home/ubuntu/venv-ob
uv venv --python /usr/bin/python3.12 "$VENV"
source "$VENV/bin/activate"
# Offline prepare dependencies first (data preparation can start before the GPU stack is ready).
uv pip install -r src_OB/requirements-data.txt
# CUDA-enabled PyTorch matching the CUDA 12.8 image.
uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
# Model stack (requirements-vast.txt) with the previously used versions pinned.
uv pip install -r src_OB/requirements-vast.txt "xgboost==3.4.1" "catboost==1.2.10" "pandas==3.0.5" \
    "scikit-learn==1.9.0" "statsmodels==0.15.0"
# LightGBM from source with the CUDA tree learner (config tree.lightgbm_device = "cuda").
# Link the shared libnccl: the first attempt (see ../logs/env_install.log) failed at cmake_device_link because the
# image's /usr/lib/x86_64-linux-gnu/libnccl_static.a (NCCL 2.31.2) carries nvlink ABI 8, newer than CUDA 12.8 nvcc (ABI 7).
uv pip install "lightgbm==4.7.0" --no-binary lightgbm -C cmake.define.USE_CUDA=ON \
    -C cmake.define.BUILD_WITH_SHARED_NCCL=ON --verbose
uv pip freeze > experiments/orderbook_hf/run_meta/pip_freeze.txt
echo "ENV_INSTALL_DONE"
