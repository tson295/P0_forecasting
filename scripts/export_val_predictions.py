"""Xuất prediction VAL đã lưu (`experiments/wins/<model>_seed<k>.npz`) ra CSV.

Không chạy lại model, không train: chỉ đọc npz (idx origin + ŷ log-return 3 horizon), dựng lại
lưới thời gian/giá đóng cửa từ snapshot data để gắn timestamp, C_t, giá dự báo và giá thực tế.

    python scripts/export_val_predictions.py [--exp experiments] [--out experiments/predictions]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from p0.config import HORIZONS  # noqa: E402
from p0.data import read_ohlcv_csv, to_b0_frame  # noqa: E402
from p0.metrics import price_from_logret  # noqa: E402
from Baseline_LGBM import build_ohlcv_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(ROOT / "experiments" / "15d"), help="thư mục run (mặc định: vòng 15 ngày đã xong)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hf", default=str(ROOT / "data" / "BTC_hf_1min.csv"))
    args = ap.parse_args()
    exp, win_dir = Path(args.exp), Path(args.exp) / "wins"
    out = Path(args.out) if args.out else exp / "predictions"
    out.mkdir(parents=True, exist_ok=True)

    fd = build_ohlcv_features(to_b0_frame(read_ohlcv_csv(args.hf)))
    ts = fd.frame["timestamp"].to_numpy(np.int64)
    close = fd.frame["Close"].to_numpy(float)
    dt = pd.to_datetime(ts, unit="s", utc=True)

    manifest = []
    for meta_path in sorted(win_dir.glob("*.json")):
        model = meta_path.stem
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seeds = meta.get("eval_seeds", [])
        folds = meta.get("folds", [])
        rows = []
        for k, seed in enumerate(seeds):
            p = win_dir / f"{model}_seed{k}.npz"
            if not p.exists():
                print(f"[{model}] thiếu {p.name} — bỏ qua")
                continue
            z = np.load(p)
            n_fold = len([c for c in z.files if c.startswith("idx_")])
            for i in range(n_fold):
                # yhat GIỮ NGUYÊN float32 như lúc chạy: price_from_logret dùng đúng exp(float32)
                # của pipeline → CSV tái tạo RMSE khớp bit với wins/<model>.json
                idx, yhat = z[f"idx_{i}"], z[f"yhat_{i}"]
                p_hat = price_from_logret(close[idx], yhat)
                d = pd.DataFrame({
                    "model": model, "seed": int(seed), "seed_slot": k,
                    "fold": folds[i] if i < len(folds) else f"fold{i + 1}",
                    "t_idx": idx, "t_utc": dt[idx], "close_t": close[idx],
                })
                for j, h in enumerate(HORIZONS):
                    d[f"yhat_h{h}"] = yhat[:, j]
                    d[f"pred_close_h{h}"] = p_hat[:, j]
                    d[f"actual_close_h{h}"] = close[idx + h]
                rows.append(d)
        if not rows:
            continue
        df = pd.concat(rows, ignore_index=True)
        f = out / f"val_pred_{model}.csv"
        df.to_csv(f, index=False)  # không round: CSV phải tái tạo ĐÚNG RMSE của wins/<model>.json
        manifest.append({"model": model, "file": f.name, "rows": len(df), "seeds": len(seeds),
                         "folds": len(folds), "which": meta.get("which", ""),
                         "n_ext": len(meta.get("colset", {}).get("ext", [])),
                         "median_gain_vs_e0": meta.get("median_gain_vs_e0", "")})
        print(f"[{model}] {len(df)} dòng → {f}")
    pd.DataFrame(manifest).to_csv(out / "index.csv", index=False)
    print(f"\nindex → {out / 'index.csv'}")


if __name__ == "__main__":
    main()
