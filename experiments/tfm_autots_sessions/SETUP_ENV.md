# Vast setup provenance — tfm_autots (2026-09-11)

Metadata/build records only. No test, smoke, probe fit, warmup or benchmark was run during setup.
Evidence of actual GPU fits comes only from the real phase logs in `run_*/training.log`.

## Machine

- GPU: 1× NVIDIA GeForce RTX 3090, 24576 MiB, compute capability 8.6; driver 580.159.03 (reports CUDA 13.0).
- CPU 80 threads, RAM 188 GiB, overlay disk 46 GB. Ubuntu 24.04.4. No sudo (password required).
- System toolchain already in the image: nvcc 12.8.93 (`/usr/local/cuda-12.8`), gcc 13.3.0, cmake 3.28.3, ninja,
  cuDNN 9.8 (cuda 12), `libnccl2`/`libnccl-dev` 2.31.2-1+cuda13.3.

## Data (Git LFS)

`git lfs pull --include="data/BTC_1m_2y.csv,data/BTC_5m_2y.csv" --exclude=""` then `sha256sum`:

- `data/BTC_1m_2y.csv` 101766374 bytes, `559ce040efd737d38f6d541b26e1533f4afc4682b5af2a94ef5cf842e31f8097`
- `data/BTC_5m_2y.csv` 21146273 bytes, `0e5fb9ad20478dd4cc8b26c3669453ff35b3ceff1c72b81d20a6337540f52fef`

Both match `data/data_checksums_2y.json` and `data/BTC_5m_2y.derivation.json`. First/last 1m rows:
2024-09-03 16:29:00+00:00 and 2026-09-03 16:29:00+00:00. No Order Book data downloaded.

## Python environment `.venv`

- Python 3.11.16 (uv-managed CPython at `/.uv/python_install/cpython-3.11-linux-x86_64-gnu`), `python3.11 -m venv .venv`, pip 26.2.1.
- `pip install torch --index-url https://download.pytorch.org/whl/cu128` → torch 2.11.0+cu128
  (nvidia-*-cu12 12.8 runtime wheels, nvidia-nccl-cu12 2.28.9, cuDNN 9.19).
- `pip install -r requirements-tfm-autots.txt` → numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.1,
  matplotlib 3.11.1, joblib 1.6.0, xgboost 3.2.0, cupy-cuda12x 14.2.0, autots 1.0.4, timesfm 2.0.2,
  statsmodels 0.15.0, huggingface_hub 1.31.0, safetensors 0.8.0. No JAX / XReg extra installed.
- LightGBM 4.7.0 built from sdist (no CPU wheel was ever installed):
  `pip wheel --no-binary=lightgbm --no-deps --no-cache-dir --config-settings=cmake.define.USE_CUDA=ON
  --config-settings=cmake.define.BUILD_WITH_SHARED_NCCL=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=86 lightgbm==4.7.0`
  then installed the wheel. CMake found CUDAToolkit 12.8.93 and shared NCCL at `/usr/lib/x86_64-linux-gnu/libnccl.so`
  (system libnccl 2.31.2+cuda13.3). `lib_lightgbm.so` ELF: NEEDED `libnccl.so.2`, `libgomp.so.1`; embedded cubin `sm_86`;
  cudart linked statically. Build had only nvcc warnings (fmt constexpr, host/device) and exited 0.
- `pip check`: no broken requirements.

## Deviation from the reference runbook commands

- `HF_HOME` in the image is `/workspace/.hf_home`, owned `root:root 775`, not writable by user `ubuntu`.
  The tmux launch passes `HF_HOME=/home/ubuntu/.cache/huggingface` so the pinned TimesFM checkpoint downloads on the
  real model load outside `experiments/`. This changes only the cache location, not code/config/data.
- `CMAKE_CUDA_ARCHITECTURES=86` restricts the LightGBM build to the actual GPU (runbook allows this).

## Pretrained checkpoint and NCCL binding

- TimesFM checkpoint downloaded on the first real model load (loop:tfm, 19:03 UTC) into
  `/home/ubuntu/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3`,
  identical to `REVISION` pinned in `src/p0/models_tfm.py:32` (`REPO_ID = google/timesfm-2.5-200m-pytorch`). Not committed.
- `lib_lightgbm.so` has no RPATH: at runtime `libnccl.so.2` binds to whichever copy is loaded first in the process
  (torch's nvidia-nccl-cu12 2.28.9 if torch is imported first, else system 2.31.2+cuda13.3). The actual binding is
  read from `/proc/<pid>/maps` during the AutoTS stages; single-GPU LightGBM training does not initialise NCCL.

## Logs

Raw setup logs copied from the session scratchpad to `setup_logs/` (`pip_torch.log`, `pip_req.log`, `lgbm_build.log`).

## Launch

`tmux new-session -d -s p0_tfm_autots -c <repo> -e HF_HOME=/home/ubuntu/.cache/huggingface 'bash scripts/vast_tfm_autots_run.sh'`
with `remain-on-exit on` for the window. Launcher records config/commit/patch/gpu/environment in `run_*/`.
