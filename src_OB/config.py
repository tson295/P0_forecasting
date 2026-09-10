from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "2024-09-10"
END = "2026-09-10"
START_US = int(datetime(2024, 9, 10, tzinfo=timezone.utc).timestamp()) * 1_000_000
END_US = int(datetime(2026, 9, 10, tzinfo=timezone.utc).timestamp()) * 1_000_000


def load(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg["download_start"], cfg["download_end"]) != (START, END):
        raise ValueError("Historical dataset đã freeze: [2024-09-10, 2026-09-10) UTC.")
    for key in ("raw_dir", "prepared_dir", "output_dir"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if cfg["levels"] != 10 or cfg["gap_days"] <= 5:
        raise ValueError("L2 phải có 10 level và gap_days phải > 5.")
    if cfg["context"] < 10 or any(h <= 0 for h in cfg["horizons_seconds"]):
        raise ValueError("context >= 10; horizon tính bằng số giây dương.")
    if cfg["tree"]["lightgbm_device"] not in ("gpu", "cuda"):
        raise ValueError("LightGBM training phải dùng gpu/cuda.")
    if any(cfg["inference"][k] < 1 for k in ("batch_size", "single_origin_samples")):
        raise ValueError("Inference batch size và single_origin_samples phải >= 1.")
    return cfg


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)
