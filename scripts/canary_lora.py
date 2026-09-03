"""Canary LoRA TimesFM trên data THẬT: 1 fold × 1 epoch × vài origin — chỉ để đo THỜI GIAN/VRAM trước khi cam kết ETA.

Không phải một bước của plan: không ghi artifact vào experiments/ (adapter vào thư mục tạm), không ghi log.csv/wins/.
Vẫn là training thật trên GPU → đi qua `cli.gate` (TRAINING phải UNLOCKED, GPU preflight) và checksum data.

    PYTHONPATH=src:. python scripts/canary_lora.py --config configs/p0_full.json [--fold 0] [--origins 64] [--epochs 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from p0 import cli  # noqa: E402
from p0.config import RunConfig  # noqa: E402
from p0.harness import ColSet, _standardize_fit  # noqa: E402
from p0.models import SeriesBatch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p0_full.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--origins", type=int, default=64, help="số origin VAL để đo forecast native + covariate (1 cột)")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    cfg = RunConfig.load(args.config)
    cli.gate(cfg, Namespace(smoke=False, allow_cpu=False), ["tfm"])
    store, folds, _, _ = cli.load_store(cfg)
    fold = folds[args.fold]
    import torch

    tmp = tempfile.mkdtemp(prefix="p0_canary_lora_")
    params = {**cli._params_for(cfg, "tfm"), "adapter_dir": tmp}
    params["lora"] = {**params.get("lora", {}), "max_epochs": int(args.epochs), "patience": 1}
    model = cli.make_model("tfm", params, allow_cpu=False)
    idx_fit, idx_es, idx_val = (p.origins(store.ts, store.eligible) for p in (fold.fit, fold.es, fold.val))
    idx_val = idx_val[: args.origins]
    X_fit, X_es, X_val = (SeriesBatch(store.ts, store.r1, i, None, ()) for i in (idx_fit, idx_es, idx_val))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = model.fit_predict(X_fit, None, X_es, None, X_val, None, cfg.calib_seed)  # ES bật, ≤ --epochs epoch
    t_train = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    from p0 import models_tfm

    meta = next(iter(models_tfm._ADAPTER_META.values()))
    n_fit = meta["n_windows_fit"]
    say = cli.say
    say(f"fold {fold.name}: {n_fit} cửa sổ FIT, {meta['n_windows_es']} cửa sổ ES, {len(meta['curve'])} epoch")
    say(f"train LoRA + forecast {len(idx_val)} origin native: {t_train:.1f} s | VRAM đỉnh {peak:.2f} GiB | curve {json.dumps(meta['curve'])}")
    per_epoch = t_train / max(1, len(meta["curve"]))
    # covariate path: 1 cột ext, 1 origin/lời gọi
    col = "ret_2"
    cov = _standardize_fit(store.grid_matrix(ColSet((), (col,))), idx_fit)
    Xc = SeriesBatch(store.ts, store.r1, idx_val, cov, (col,))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    yc = res.predictors[0](Xc)
    t_cov = time.perf_counter() - t1
    say(f"forecast +XReg({col}) {len(idx_val)} origin: {t_cov:.1f} s → {1000 * t_cov / len(idx_val):.0f} ms/origin; hữu hạn: {bool(np.isfinite(yc).all())}")
    n_val_all = len(fold.val.origins(store.ts, store.eligible))
    say(f"ƯỚC LƯỢNG: 1 epoch ≈ {per_epoch / 60:.1f} phút; 7 adapter/fold × 5 fold × ~{max(1, len(meta['curve']))} epoch ≈ "
        f"{35 * per_epoch * max(1, len(meta['curve'])) / 3600:.1f} h; 1 pass XReg 5 fold ({5 * n_val_all} origin) ≈ {5 * n_val_all * t_cov / len(idx_val) / 3600:.2f} h "
        f"(tuần tự; chia cho số worker fold-parallel)")
    say(f"adapter tạm ở {tmp} (không phải artifact) — xoá tay nếu muốn")


if __name__ == "__main__":
    main()
