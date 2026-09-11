"""Record code/config/GPU/package provenance of the real Vast environment (metadata reads only, no fit)."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def run(*cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


packages = ["torch", "lightgbm", "xgboost", "catboost", "cupy-cuda12x", "timesfm", "autots", "statsmodels",
            "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "duckdb", "sortedcontainers", "joblib",
            "safetensors", "huggingface-hub", "jax"]
record = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "git": {"head": run("git", "rev-parse", "HEAD"), "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("git", "status", "--porcelain").splitlines()},
    "config_sha256": hashlib.sha256((ROOT / "configs/orderbook_zenodo.json").read_bytes()).hexdigest(),
    "host": {"hostname": platform.node(), "vast_container": os.environ.get("CONTAINER_ID"),
             "cpu_count": os.cpu_count(), "kernel": platform.release()},
    "gpu": run("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"),
    "cuda_toolkit": run("nvcc", "--version").splitlines()[-1:],
    "image_cuda_version": os.environ.get("CUDA_VERSION"),
    "python": {"executable": sys.executable, "version": sys.version},
    "packages": {name: version(name) for name in packages},
}
import torch  # noqa: E402  (device enumeration only)
import xgboost  # noqa: E402

record["torch_cuda"] = {"torch_version_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "cuda_available": torch.cuda.is_available(),
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
record["xgboost_build_USE_CUDA"] = xgboost.build_info().get("USE_CUDA")
lib = next(Path(md.distribution("lightgbm").locate_file("lightgbm")).glob("**/lib_lightgbm.so"), None)
record["lightgbm_lib_cuda_links"] = [] if lib is None else [
    line.split()[0] for line in run("ldd", str(lib)).splitlines() if "cuda" in line or "nccl" in line]
from catboost.utils import get_gpu_device_count  # noqa: E402

record["catboost_gpu_device_count"] = get_gpu_device_count()
(OUT / "environment.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
print(json.dumps(record, indent=2, default=str))
