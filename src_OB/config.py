from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    cfg = json.loads(Path(path).read_text())
    if re.fullmatch(r"[0-9a-f]{40}", cfg["dataset_revision"]) is None:
        raise ValueError("dataset_revision phải là full commit SHA để freeze public archive.")
    if cfg["exchange"] != "binance" or cfg["symbol"] != "BTCUSDT":
        raise ValueError("Pipeline này dùng BTCUSDT Binance Spot, không gộp market.")
    for key in ("raw_dir", "prepared_dir", "output_dir"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if cfg["levels"] != 10 or cfg["gap_days"] <= 5:
        raise ValueError("L2 phải có 10 level và gap_days phải > 5.")
    if cfg["book_max_depth"] < cfg["levels"] or cfg.get("include_distances", False):
        raise ValueError("Book cache >= 10 levels; baseline chỉ dùng OF/OFI và timing.")
    if any(cfg[k] <= 0 for k in ("train_days", "val_days", "step_days", "n_folds",
                                 "max_feed_gap_seconds", "max_price_age_seconds", "chunk_rows")):
        raise ValueError("Fold durations, gap limits, counts and chunk_rows must be positive.")
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
