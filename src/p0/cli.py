"""CLI: python run.py <step> --config configs/p0_full.json [--model lgbm] [--smoke] [--allow-cpu]

Bước (plan §8, vòng expanded-data 2026-09-03/04): check-data → lock-s0 (S0_m khoá toàn bộ + overlap audit per model + Candidate_m) →
loop --model m (calibrate riêng trên S0_m, ε_m mới, add-one Candidate_m, prune PI chỉ cột mới, confirmation 3 seed → win_m,
champion) → tfm-final (so HAI HỆ THỐNG HOÀN CHỈNH: A = TimesFM-LoRA baseline feature-free vs B = cùng adapter + XReg(F_win)) → autots-search (framework AutoTS trên F_WR_best / F_MR_best)
→ ensemble → final (TEST một lần) → visualize (hậu kỳ, không train). `calibrate`/`filter-b0` giữ cho lọc B0 §1.4 (đã xong ở
vòng 15 ngày; smoke synthetic vẫn dùng). `smoke-e2e` chạy toàn bộ trên data tổng hợp CPU (chỉ debug).
KHÔNG vẽ figure trong bất kỳ bước training/search nào — mọi artifact cần cho figure được lưu, `visualize` dựng lại sau.
Ba vai trò seed tách bạch (§1.3): `calib_seed` CHỈ cho run ES tìm số vòng/epoch cố định; `eval_seeds` đo ε và confirmation
3 seed; `selection_seed` là MỘT seed cố định cho mọi bước selection.

Gate: training chỉ khi `.claude/MEMORY.md` ghi `TRAINING: UNLOCKED`; GPU preflight bắt buộc. `--smoke` / `--allow-cpu` CHỈ được
chấp nhận khi dataset_label bắt đầu bằng "synthetic" (data tổng hợp) — với data thật CLI từ chối (plan §0: cấm training CPU).
Data thật phải khớp file checksum của config (§6.1) ở mọi bước sau check-data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import fold_parallel, gpu, scheduler
from .checker_log import hard_fail
from .checker_log import record as ck_record
from .config import HORIZONS, RunConfig
from .data import check_ohlcv, derive_lf_5min, file_sha256, read_ohlcv_csv, verify_checksums, write_checksums, write_lf_csv
from .features_ext import ALL_EXT_COLUMNS, CANDIDATE_BY_NAME, CANDIDATES
from .features_short import SHORT_COLUMNS
from .filter_b0 import FilterTable, median_over_folds, mutual_info, permutation_importance, standalone_gain, verify_sets
from .harness import ColSet, Store, calibrate, rounds_from, run_at_seed, run_config, seed_noise
from .logs import load_preds, log_champion, log_latency, log_run, new_exp_id, save_run
from .loop import (add_one_loop, compare, confirm, decide_win, ensemble_rmse, inverse_mse_weights, load_champion, prune_pi,
                   save_champion)
from .metrics import cell_metrics, e0_rmse, gain_pp, mean_rmse_over_seeds, seed_noise_cells, seed_noise_eps, summarize
from .models import make_model
from .s0 import S0_MODELS, collision_audit, load_lock, prev_dropped, s0_for, save_lock
from .split import Fold, RollingSpec, check_fold, make_final, make_folds, make_rolling_from_end, make_rolling_spread, utc_ts


# ----------------------------------------------------------------------------- helpers
_SAY_PREFIX = ""


def set_say_prefix(prefix: str) -> None:
    """Worker GPU đặt tiền tố (`gpu0`/`gpu1`) để log của các process không lẫn nhau."""
    global _SAY_PREFIX
    _SAY_PREFIX = f" {prefix}" if prefix else ""


def say(msg: str) -> None:
    print(f"[p0 {time.strftime('%H:%M:%S')}{_SAY_PREFIX}] {msg}", flush=True)


def training_state(root: Path) -> str:
    mem = root / ".claude" / "MEMORY.md"
    if not mem.exists():
        return "UNKNOWN"
    for line in mem.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
        if line.startswith("TRAINING:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


def _is_synthetic(cfg: RunConfig) -> bool:
    return str(cfg.dataset_label).startswith("synthetic")


def gate(cfg: RunConfig, args, model_names: list[str]) -> None:
    """Khóa training (MEMORY) + GPU preflight. --smoke / --allow-cpu CHỈ hợp lệ với dataset tổng hợp (dataset_label 'synthetic*')."""
    smoke, allow_cpu = bool(getattr(args, "smoke", False)), bool(getattr(args, "allow_cpu", False))
    if (smoke or allow_cpu) and not _is_synthetic(cfg):
        hard_fail(cfg.exp_dir, "gate", "CPU_ON_REAL_DATA", f"--smoke/--allow-cpu bị từ chối với dataset '{cfg.dataset_label}': chỉ cho data "
                  "tổng hợp (plan §0: cấm training CPU; không bỏ khóa training/GPU gate trên data thật).")
    if smoke:
        return
    st = training_state(Path(cfg.root))
    if st != "UNLOCKED":
        hard_fail(cfg.exp_dir, "gate", "TRAINING_LOCKED", f"TRAINING_LOCKED (MEMORY.md: {st}) — cần user unlock rõ ràng trước khi chạy training.")
    if cfg.require_gpu and not allow_cpu:
        for m in model_names:  # GPU không có / backend không phải CUDA → dừng NGAY, không hỏi, không CPU fallback
            try:
                gpu_preflight(m, cfg)
            except SystemExit as e:
                hard_fail(cfg.exp_dir, "gpu_preflight", "GPU_UNAVAILABLE", str(e), model=m)
            except Exception as e:  # ImportError / lỗi thư viện khi thử fit GPU
                hard_fail(cfg.exp_dir, "gpu_preflight", "GPU_UNAVAILABLE", f"{type(e).__name__}: {e}", model=m)
        if model_names:
            ck_record(cfg.exp_dir, "gpu_preflight", "PASS", "GPU_PREFLIGHT", f"GPU preflight OK: {list(model_names)}")


def gpu_preflight(model: str, cfg: RunConfig) -> None:
    if model == "lgbm":
        from Baseline_LGBM import LGBMConfig, assert_p100_lightgbm

        assert_p100_lightgbm(LGBMConfig(require_p100=False, **cfg.model_params("lgbm")))
        say("GPU preflight LightGBM: OK")
    elif model in ("xgb", "xgbrf"):
        import xgboost as xgb

        info = xgb.build_info()
        if not bool(info.get("USE_CUDA", False)):
            sys.exit(f"GPU preflight XGBoost: wheel không build CUDA (USE_CUDA={info.get('USE_CUDA')}) — cấm CPU fallback.")
        try:
            import torch

            if not torch.cuda.is_available():
                sys.exit("GPU preflight XGBoost: không có CUDA device trên máy (torch.cuda.is_available() = False).")
        except ImportError:
            pass
        x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
        m = xgb.XGBRegressor(n_estimators=3, device="cuda", tree_method="hist")
        m.fit(x, x[:, 0])
        conf = json.loads(m.get_booster().save_config())
        dev = str(conf.get("learner", {}).get("generic_param", {}).get("device", ""))
        if not dev.startswith("cuda"):  # không chỉ YÊU CẦU device=cuda: booster phải THỰC SỰ ở cuda
            sys.exit(f"GPU preflight XGBoost: booster báo device={dev!r} ≠ cuda — CPU fallback bị cấm.")
        say(f"GPU preflight XGBoost: OK (build CUDA, booster device {dev})")
    elif model == "cat":
        from catboost import CatBoostRegressor

        x = np.random.default_rng(0).normal(size=(256, 4))
        CatBoostRegressor(iterations=3, task_type="GPU", verbose=False, allow_writing_files=False).fit(x, x[:, 0])
        say("GPU preflight CatBoost: OK")
    elif model in ("lstm", "tfm"):
        import torch

        if not torch.cuda.is_available():
            sys.exit(f"GPU preflight {model}: CUDA không có — cấm training/inference CPU.")
        say(f"GPU preflight torch ({model}): {torch.cuda.get_device_name(0)}")
        if model == "tfm":
            import timesfm  # noqa: F401  — chỉ kiểm tra đã cài đúng version (pin trong requirements.txt)
    elif model in ("autots_wr", "autots_mr", "autots"):
        import autots  # noqa: F401

        for m in (("lgbm",) if model == "autots_wr" else ("xgb",) if model == "autots_mr" else ("lgbm", "xgb")):
            gpu_preflight(m, cfg)  # regression_model bên trong chạy GPU (AutoTS-final: bake-off có cả LightGBM lẫn xgboost)
        say(f"GPU preflight AutoTS ({autots.__version__}): OK")


def checksum_path(cfg: RunConfig) -> Path:
    return cfg.path(cfg.checksums)


def make_partitions(store: Store, cfg: RunConfig) -> tuple[list[Fold], Fold]:
    """Fold + final: `split` rolling_from_end (data đầy đủ, neo vào cuối data thật) hoặc `val_days`/`test_start` (15 ngày)."""
    if cfg.split:
        spec = RollingSpec.from_dict(cfg.split)
        if spec.mode == "rolling_spread":  # data 2 năm (2026-09-04): 5 VAL rải đều trên lịch sử trước TEST, FIT 120 ngày rolling
            return make_rolling_spread(store.first_origin_ts, store.last_ts, spec)
        return make_rolling_from_end(store.first_origin_ts, store.last_ts, spec)
    folds = make_folds(store.first_origin_ts, cfg.val_days, cfg.purge_minutes, cfg.es_hours)
    test_end = utc_ts(cfg.test_end) if cfg.test_end else store.last_ts + 60
    return folds, make_final(store.first_origin_ts, cfg.test_start, test_end, cfg.purge_minutes)


def load_store(cfg: RunConfig, need_lf: bool = True, verify: bool = True):
    """Đọc data + kiểm tra §1.1; verify=True → sha256 phải khớp file checksum của config (§6.1, trừ lúc check-data ghi file đó)."""
    if verify:
        ck = checksum_path(cfg)
        if not ck.exists():
            hard_fail(cfg.exp_dir, "load_store", "CHECKSUM_MISSING", f"Thiếu {ck} — chạy `python run.py check-data --config <cfg> --write-checksums` (§6.1) trước.")
        ok, problems = verify_checksums(ck, Path(cfg.root), label=cfg.dataset_label)
        if not ok:
            hard_fail(cfg.exp_dir, "load_store", "CHECKSUM_MISMATCH", "Checksum data không khớp §6.1 — dừng: " + "; ".join(problems))
    hf_path = cfg.path(cfg.hf_csv)
    raw_hf = read_ohlcv_csv(hf_path)
    rep = check_ohlcv(raw_hf)
    if not rep["ok"]:
        hard_fail(cfg.exp_dir, "load_store", "DATA_QUALITY", f"Data HF không đạt §1.1: {rep}")
    raw_lf = None
    lf_path = cfg.path(cfg.lf_csv) if cfg.lf_csv else None
    if need_lf and lf_path and not lf_path.exists() and not _is_synthetic(cfg):  # data thật: LF khai báo mà thiếu → cột 5' của S0 sẽ NaN âm thầm → cấm
        hard_fail(cfg.exp_dir, "load_store", "LF_MISSING", f"Thiếu file LF 5' {lf_path} (config khai báo lf_csv) — cung cấp LF phủ toàn bộ HF.")
    if need_lf and lf_path and lf_path.exists():
        raw_lf = read_ohlcv_csv(lf_path)
        # LF phải PHỦ toàn bộ HF: cột r5_* / log_c5_* trong S0 của nhiều model là as-of join; thiếu LF → NaN âm thầm → cấm
        lf_ts = raw_lf["timestamp"].to_numpy(np.int64)
        hf_ts = raw_hf["timestamp"].to_numpy(np.int64)
        if lf_ts.min() > hf_ts.min() + 5 * 60 * 288 or lf_ts.max() < hf_ts.max() - 5 * 60:
            hard_fail(cfg.exp_dir, "load_store", "LF_COVERAGE",
                      f"LF {lf_path} không phủ HF: LF {pd.Timestamp(lf_ts.min(), unit='s', tz='UTC')} → "
                      f"{pd.Timestamp(lf_ts.max(), unit='s', tz='UTC')} vs HF {pd.Timestamp(hf_ts.min(), unit='s', tz='UTC')} → "
                      f"{pd.Timestamp(hf_ts.max(), unit='s', tz='UTC')} — cung cấp LF 5' đủ phủ (feature 5' của S0 cần).")
    store = Store(raw_hf, raw_lf)
    folds, final = make_partitions(store, cfg)
    return store, folds, final, rep


def _params_for(cfg: RunConfig, name: str) -> dict:
    return dict(cfg.model_params(name))


def model_for(cfg: RunConfig, name: str, allow_cpu: bool):
    if allow_cpu and not _is_synthetic(cfg):
        sys.exit(f"allow_cpu với dataset '{cfg.dataset_label}' bị cấm (plan §0: training chỉ GPU).")
    params = _params_for(cfg, name)
    if allow_cpu:  # smoke/unit: ép CPU rõ ràng
        params = {k: v for k, v in params.items() if k not in ("device_type", "device", "task_type")}
        params.update({"lgbm": {"device_type": "cpu"}, "cat": {"task_type": "CPU"}}.get(name, {"device": "cpu"}))
    elif name in ("autots_wr", "autots_mr") and "regression_model" not in params:
        # Backend GPU đã RESOLVE (models.lgbm.device_type / models.xgb.device, do vast_bootstrap.sh ghi sau khi thử fit thật)
        # PHẢI chảy vào regression_model bên trong AutoTS (nếu không AutoTSModel rơi về hằng số device_type="gpu").
        reg = autots_regressors(cfg)
        key = "LightGBM" if name == "autots_wr" else "xgboost"
        params["regression_model"] = {"model": key, "model_params": reg[key]}
    if name == "tfm" and "adapter_dir" not in params:
        params["adapter_dir"] = str(cfg.exp_dir / "lora")  # adapter LoRA đã freeze: artifact versioned (LFS)
    m = make_model(name, params, allow_cpu=allow_cpu)
    # đánh dấu: model này dựng lại được Y HỆT trong worker GPU từ (cfg, name, allow_cpu) → được phép đi qua scheduler.
    # Model mang state riêng (AutoTS frozen template, stub trong test) KHÔNG có dấu này và luôn chạy trong process gọi.
    setattr(m, fold_parallel.POOL_MARK, name)
    return m


PROBE_MODELS = ("autots_wr", "autots_mr", "tfm")  # chạy đủ §2.1 (add-one → prune PI → confirmation) nhưng KHÔNG so champion
# ở `loop`; bước "final" tương ứng gộp kết quả thành model đại diện rồi mới so champion / ensemble / Final:
FINAL_STEP = {"autots_wr": "autots-search", "autots_mr": "autots-search", "tfm": "tfm-final"}

# --------------------------------------------------------------------- TimesFM: hai HỆ THỐNG HOÀN CHỈNH (2026-09-04c)
# A = TimesFM-LoRA baseline: LoRA fine-tune xong, KHÔNG feature, KHÔNG B0*, KHÔNG covariate XReg.
# B = CÙNG adapter LoRA đã freeze + XReg(F_win) — F_win là bộ ĐÃ thắng confirmation F_raw vs F_pruned.
# `tfm-final` so A với B (không phải "XReg vs LoRA": XReg không phải model độc lập) → TimesFM-final = wins/tfm.json.
TFM_BASELINE_WIN = "tfm_lora_baseline"      # hệ thống A
TFM_XREG_WIN = "tfm_lora_xreg"              # hệ thống B
TFM_BASELINE_LEGACY = "tfm_lora_native"     # tên cũ (2026-09-03/04) — vẫn ĐỌC được, không ghi mới
# artifact wins/ KHÔNG phải thành viên ensemble / cấu hình Final: probe AutoTS và hai cấu hình nội bộ của TimesFM
NON_MEMBER_WINS = ("autots_wr", "autots_mr", TFM_BASELINE_WIN, TFM_BASELINE_LEGACY, TFM_XREG_WIN)
# Không bao giờ được so champion: probe/cấu hình nội bộ. Chỉ ĐẠI DIỆN (tfm = TFM-final, autots = AutoTS-final) mới đủ tư cách.
CHAMPION_INELIGIBLE = NON_MEMBER_WINS
# §3: thứ tự so champion là METHODOLOGY, cố định — KHÔNG phải thứ tự chạy xong (2026-09-04c: replay sau khi mọi đại diện có đủ)
CHAMPION_ORDER = ("lgbm", "xgb", "cat", "tfm", "xgbrf", "autots", "lstm")
REPRESENTATIVE_OF = {"lgbm": "loop lgbm", "xgb": "loop xgb", "cat": "loop cat", "tfm": "tfm-final", "xgbrf": "loop xgbrf",
                     "autots": "autots-search", "lstm": "loop lstm"}


def champion_deferred(cfg: RunConfig) -> bool:
    """Hoãn so champion (§14): mỗi branch chỉ SINH artifact đại diện; `champion-replay` so lại theo THỨ TỰ CỐ ĐỊNH."""
    env = os.environ.get("P0_DEFER_CHAMPION")
    if env is not None:
        return env.strip() not in ("", "0", "false", "False")
    return bool(getattr(cfg, "defer_champion", False))


def colset_from_arg(store: Store, cfg: RunConfig, arg: str) -> ColSet:
    if arg == "b0306":
        return store.all_b0()
    if arg == "b0star":
        p = cfg.exp_dir / "b0_star.json"
        if not p.exists():
            sys.exit("Chưa có experiments/b0_star.json — chạy filter-b0 trước (§1.4).")
        return ColSet.load(p)
    return ColSet.load(Path(arg))


def _log(cfg: RunConfig, **row) -> None:
    row.setdefault("dataset_label", cfg.dataset_label)
    row.setdefault("config_hash", cfg.hash())
    log_run(cfg.exp_dir, row)


def _summ_row(run, base_rmse, base_name: str) -> dict:
    g = run.gain_vs(base_rmse)
    s = summarize(g)
    return {"base": base_name, **{k: round(s[k], 4) for k in ("MedianGain", "WinRate", "P10Gain", "WorstGain")},
            "rmse_cells": json.dumps(np.round(run.rmse, 3).tolist()), "mae_cells": json.dumps(np.round(run.mae, 3).tolist()),
            "e0_cells": json.dumps(np.round(run.e0, 3).tolist()), "gain_cells": json.dumps(np.round(g, 4).tolist()),
            "n_cols": len(run.colset.names)}


def _cells(a) -> str:
    return json.dumps(np.round(np.asarray(a, float), 4).tolist())


# ----------------------------------------------------------------------------- steps
def cmd_check_data(cfg: RunConfig, args) -> None:
    hf_path = cfg.path(cfg.hf_csv)
    raw = read_ohlcv_csv(hf_path)
    rep = check_ohlcv(raw)
    say(f"HF {hf_path}: {json.dumps(rep, ensure_ascii=False)}")
    reports = {"hf": rep}
    files = {"hf": hf_path}
    lf_path = cfg.path(cfg.lf_csv) if cfg.lf_csv else None
    if lf_path and lf_path.exists():
        lf = read_ohlcv_csv(lf_path)
        reports["lf"] = check_ohlcv(lf, step=300)
        files["lf"] = lf_path
        say(f"LF {lf_path}: {json.dumps(reports['lf'], ensure_ascii=False)}")
    if lf_path and lf_path.exists():  # LF dẫn xuất: sidecar phải trỏ đúng HF hiện tại (không dùng LF của nguồn khác)
        side = lf_path.with_suffix(".derivation.json")
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
            hf_sha = file_sha256(hf_path)
            if meta.get("source_sha256") != hf_sha:
                hard_fail(cfg.exp_dir, "check-data", "LF_DERIVATION_MISMATCH",
                          f"{side}: LF dẫn xuất từ HF sha {meta.get('source_sha256')} ≠ HF hiện tại {hf_sha} — chạy lại `derive-lf`.")
            say(f"LF dẫn xuất từ đúng HF hiện tại (sha {hf_sha[:12]}…): {meta.get('rows_lf')} bar 5', bỏ {meta.get('dropped_incomplete_buckets')} nhóm thiếu")
    store, folds, final, _ = load_store(cfg, verify=False)
    say(f"B0-eligible origins: {int(store.eligible.sum())} | first {pd.Timestamp(store.first_origin_ts, unit='s', tz='UTC')} | last {pd.Timestamp(store.last_ts, unit='s', tz='UTC')}")
    if cfg.split:
        spec = RollingSpec.from_dict(cfg.split)
        say(f"split {spec.mode}: {spec.n_folds} fold × VAL {spec.val_days} ngày"
            + (" RẢI ĐỀU trên lịch sử trước TEST" if spec.mode == "rolling_spread" else " liên tiếp trước TEST")
            + f", train region rolling FIT {spec.fit_days} + ES {spec.es_days} ngày, TEST {spec.test_days} ngày cuối, purge {spec.purge_minutes}' "
            f"— cần ≥ {spec.days_needed} ngày data (suy ra từ data thật, không hard-code ngày)")
    for f in folds + [final]:
        chk = check_fold(f, store.ts, store.eligible, cfg.purge_minutes)
        say(f"{f.name}: FIT {f.fit.label()} n={chk['n_fit']} | ES {f.es.label()} n={chk['n_es']} | VAL {f.val.label()} n={chk['n_val']} "
            f"| cuối=T_end−4' {chk['last_val_origin_is_Tend_minus_4min']} | {'OK' if chk['ok'] else chk['problems']}")
        if not chk["ok"]:
            hard_fail(cfg.exp_dir, "check-data", "LEAKAGE_BOUNDARY", f"{f.name}: fold không đạt biên/purge: {chk['problems']}")
    out = checksum_path(cfg)
    if args.write_checksums:
        write_checksums(cfg.dataset_label, files, reports, out, root=Path(cfg.root))
        say(f"checksum (sha256, path tương đối root) → {out}")
    elif out.exists():
        ok, problems = verify_checksums(out, Path(cfg.root), label=cfg.dataset_label)
        say(f"verify {out}: {'OK — khớp snapshot đã ghi' if ok else 'KHÔNG KHỚP: ' + '; '.join(problems)}")
        if not ok:
            hard_fail(cfg.exp_dir, "check-data", "CHECKSUM_MISMATCH", "Data khác snapshot đã ghi (§6.1) — dừng: " + "; ".join(problems))
    else:
        say(f"chưa có {out} — chạy lại với --write-checksums để ghi anchor §6.1 (bắt buộc trước mọi bước training)")


def cmd_derive_lf(cfg: RunConfig, args) -> None:
    """Dẫn xuất LF 5' ĐÃ ĐÓNG từ HF 1' của config (data 2 năm chỉ có 1'): tất định, causal, bỏ nhóm thiếu bar; ghi `lf_csv`
    + sidecar `<lf>.derivation.json` (sha nguồn, số bar, phương pháp). Không phải training. Data raw không bị sửa."""
    hf_path, lf_path = cfg.path(cfg.hf_csv), cfg.path(cfg.lf_csv)
    if lf_path is None:
        sys.exit("config không khai báo lf_csv")
    if lf_path.exists() and not getattr(args, "force", False):
        sys.exit(f"{lf_path} đã tồn tại — dùng --force để dẫn xuất lại (tất định, cùng byte nếu cùng HF)")
    raw = read_ohlcv_csv(hf_path)
    rep_hf = check_ohlcv(raw)
    if not rep_hf["ok"]:
        hard_fail(cfg.exp_dir, "derive-lf", "DATA_QUALITY", f"HF không đạt §1.1, không dẫn xuất LF: {rep_hf}")
    lf, meta = derive_lf_5min(raw)
    sha_lf = write_lf_csv(lf, lf_path)
    side = {**meta, "source": cfg.hf_csv, "source_sha256": file_sha256(hf_path), "lf": cfg.lf_csv, "lf_sha256": sha_lf,
            "dataset_label": cfg.dataset_label}
    lf_path.with_suffix(".derivation.json").write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"LF 5' → {lf_path}: {meta['rows_lf']} bar ({meta['lf_start']} → {meta['lf_end']}), bỏ {meta['dropped_incomplete_buckets']} nhóm thiếu; "
        f"sha256 {sha_lf[:12]}… (sidecar {lf_path.with_suffix('.derivation.json').name})")


def cmd_lock_s0(cfg: RunConfig, args) -> None:
    """S0_m khoá từ artifact vòng trước + collision audit (bằng số, trên data) + Candidate_m → experiments/<run>/s0/.

    Không phải training, không phải stage nghiên cứu: chỉ dựng lại và ghi lại không gian tìm kiếm để tái tạo được.
    `--data-config`: dùng data của config khác cho phần kiểm tra bằng số (định nghĩa trùng nhau thì trùng trên mọi data)."""
    prev = cfg.path(cfg.prev_run_dir) if cfg.prev_run_dir else None
    exp = cfg.exp_dir
    if prev is not None and not prev.exists():
        sys.exit(f"prev_run_dir không tồn tại: {prev}")
    star = None
    if prev is None:
        p = exp / "b0_star.json"
        if not p.exists():
            sys.exit("Không có prev_run_dir và chưa có experiments/<run>/b0_star.json (filter-b0) — không dựng được S0.")
        star = ColSet.load(p)
    models = [m for m in cfg.model_order if m in S0_MODELS]
    s0 = {m: s0_for(m, prev, star) for m in models}
    for m, cs in s0.items():
        say(f"S0_{m}: locked_b0 = {len(cs.locked_b0)} cột B0* + locked_ext = {len(cs.locked_ext)} cột ext thắng cũ (toàn bộ S0 khoá)"
            + (" — TimesFM: S0 = ∅ (baseline = TimesFM-LoRA native, không B0*)" if m == "tfm" else ""))
    c_short = tuple(cfg.short_candidates) if cfg.short_candidates else SHORT_COLUMNS
    bad = [c for c in c_short if c not in SHORT_COLUMNS]
    if bad:
        hard_fail(exp, "lock-s0", "S0_ARTIFACT", f"short_candidates ngoài C_short: {bad}")
    if set(c_short) & set(ALL_EXT_COLUMNS):  # C_short = định nghĩa MỚI; candidate cũ §2.3 (KEEP/DROP) không quay lại (§6)
        hard_fail(exp, "lock-s0", "S0_ARTIFACT", f"C_short chứa định nghĩa candidate cũ §2.3: {sorted(set(c_short) & set(ALL_EXT_COLUMNS))[:5]}")
    if prev is not None:
        for m in models:
            if m == "tfm":
                continue
            drop = set(prev_dropped(prev, m))
            if drop & set(c_short):
                hard_fail(exp, "lock-s0", "S0_ARTIFACT", f"{m}: cột DROP cũ xuất hiện trong C_short: {sorted(drop & set(c_short))[:5]}")
    data_cfg = RunConfig.load(args.data_config) if getattr(args, "data_config", None) else cfg
    store, _, _, _ = load_store(data_cfg)
    say(f"overlap audit bằng số trên data '{data_cfg.dataset_label}': Candidate_m = C_short ({len(c_short)} cột) \\ overlap(C_short, S0_m) — "
        "chỉ trừ cột đã có trong S0 của CHÍNH model (trùng tên / giá trị giống hệt cùng timestamp); tương quan cao chỉ báo cáo")
    rep = collision_audit(store, s0, c_short, max_rows=int(getattr(args, "max_rows", 60_000) or 60_000), dataset_label=data_cfg.dataset_label)
    save_lock(exp, s0, rep)
    n_intra = len(rep["intra_short_identical"])
    say(f"C_short {len(rep['c_short'])} cột | trùng giá trị nội bộ C_short (chỉ báo, không bỏ): {n_intra} | near nội bộ |ρ| ≥ {rep['corr_threshold']}: {len(rep['intra_short_near'])}")
    if n_intra:
        ck_record(exp, "lock-s0", "WARN", "C_SHORT_INTRA_IDENTICAL", f"{n_intra} cặp C_short trùng giá trị (chỉ báo, không tự bỏ): {rep['intra_short_identical'][:5]}")
    for m in models:
        pm = rep["per_model"][m]
        say(f"Candidate_{m}: {pm['n_candidates']} = {len(rep['c_short'])} − {len(pm['removed_by_overlap'])} overlap S0_{m} | near vs S0 (chỉ báo): {len(pm['near_vs_s0'])}")
        ck_record(exp, "lock-s0", "INFO", "CANDIDATE_M",
                  f"S0: locked_b0={pm['n_locked_b0']}, locked_ext={pm['n_locked_ext']}; Candidate_m={pm['n_candidates']}; "
                  f"overlap={[r['col'] for r in pm['removed_by_overlap']]}; near_vs_s0={len(pm['near_vs_s0'])} (diagnostic)", model=m)
    ck_record(exp, "lock-s0", "PASS", "S0_LOCK", f"S0/Candidate_m ghi cho {models} trên '{data_cfg.dataset_label}'")
    say(f"→ {exp / 's0'}: <model>.json (locked_b0/locked_ext), candidates_<model>.json, collisions.json, short_pool.json")


def cmd_calibrate(cfg: RunConfig, args) -> dict:
    """§1.3: run ES → 15fixed_m; 3 seed số vòng cố định → ε_m. Lưu experiments/calib/<model>_<tag>.json."""
    gate(cfg, args, [args.model])
    store, folds, _, _ = load_store(cfg)
    colset = colset_from_arg(store, cfg, args.colset)
    model = model_for(cfg, args.model, args.allow_cpu)
    nw = fold_parallel.configure(cfg, model, args.model, args.allow_cpu)
    say(f"calibrate {args.model} trên {args.colset} ({len(colset.names)} cột) — ES một lần (calib_seed = {cfg.calib_seed}); fold-parallel {nw}")
    cal = calibrate(store, model, colset, folds, seed=cfg.calib_seed, keep_states=False)
    rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
    say(f"số vòng cố định: {rounds}")
    say(f"ε: {len(cfg.eval_seeds)} evaluation seed {list(cfg.eval_seeds)} với số vòng cố định (không seed nào làm mốc)")
    eps, noise, runs = seed_noise(store, model, colset, folds, rounds, cfg.eval_seeds, cfg.eps_floor_pp)
    base = run_at_seed(runs, cfg.sel_seed)  # bảng RMSE mốc của selection (cùng seed với mọi bước selection)
    if base is None:
        base = run_config(store, model, colset, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=False)
    out = {"model": args.model, "tag": args.colset, "colset": colset.to_dict(), "rounds": rounds, "eps": eps,
           "noise_cells": np.round(noise, 5).tolist(), "calib_seed": cfg.calib_seed, "eval_seeds": list(cfg.eval_seeds),
           "selection_seed": cfg.sel_seed, "rmse": base.rmse.tolist(), "e0": base.e0.tolist(),
           "best_iters_es": cal.best_iters.tolist(), "folds": cal.fold_names,
           "seed_rmse": [r.rmse.tolist() for r in runs], "config_hash": cfg.hash()}
    path = cfg.exp_dir / "calib" / f"{args.model}_{args.colset.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    exp_id = new_exp_id("calibrate", args.model, args.colset)
    _log(cfg, exp_id=exp_id, step="calibrate", model=args.model, seed=cfg.sel_seed, colset=args.colset, rounds=json.dumps(rounds),
         **_summ_row(base, base.e0, "E0"), decision=f"eps={eps:.4f}", train_device=getattr(model, "train_device", ""),
         note=f"calib_seed={cfg.calib_seed} eval_seeds={list(cfg.eval_seeds)} noise_rms={eps:.4f}")
    save_run(cfg.exp_dir, exp_id, {**base.to_dict(), "step": "calibrate", "tag": args.colset, "eps": eps, "rounds_fixed": rounds,
                                   "best_iters_es": cal.best_iters.tolist(), "seed_rmse": out["seed_rmse"], "config_hash": cfg.hash()})
    say(f"ε_{args.model} = {eps:.4f} pp (RMS nhiễu 15 ô; min {noise.min():.4f}, max {noise.max():.4f}) → {path}")
    return out


def _load_calib(cfg: RunConfig, model: str, tag: str) -> dict:
    p = cfg.exp_dir / "calib" / f"{model}_{tag}.json"
    if not p.exists():
        sys.exit(f"Thiếu {p} — chạy: python run.py calibrate --model {model} --colset {tag}")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_filter_b0(cfg: RunConfig, args) -> None:
    """§1.4: PI + SA + MI trên B0-306 → cờ ≥ 2/3 → R1–R4 → 4 run kiểm chứng (15fixed_306) → B0*."""
    gate(cfg, args, ["lgbm"])
    store, folds, _, _ = load_store(cfg)
    cal = _load_calib(cfg, "lgbm", "b0306")
    rounds = {k: tuple(v) for k, v in cal["rounds"].items()}
    eps = float(cal["eps"])
    model = model_for(cfg, "lgbm", args.allow_cpu)
    names = list(store.b0_names)
    if args.max_cols:
        names = names[: args.max_cols]
    colset = ColSet(tuple(names))
    say(f"(a) PI: run baseline B0-306 (15fixed_306, selection_seed {cfg.sel_seed}) giữ state rồi xáo {len(names)} cột × 3 lần trong VAL")
    base_run = run_config(store, model, colset, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=True)
    base_rmse = base_run.rmse  # cùng một run làm mốc cho PI và cho 4 run kiểm chứng R1–R4
    if len(names) == len(store.b0_names) and "rmse" in cal:  # chỉ so được khi không cắt cột (--max-cols của smoke)
        d_cal = float(np.max(np.abs(base_rmse / np.asarray(cal["rmse"]) - 1)))
        say(f"   |RMSE base − RMSE calibrate| tối đa {d_cal * 100:.4f}% (cùng config + selection_seed → phải ≈ 0)")
    pi = median_over_folds(permutation_importance(store, base_run, list(range(len(names))), repeats=3, seed=cfg.sel_seed))
    say(f"(b) SA: {len(names)} model 1 cột (ES, selection_seed) — vs E0 và vs B0-306")
    sa_e0, sa_b0 = standalone_gain(store, model, folds, names, cfg.sel_seed, base_rmse,
                                   progress=lambda k, n, nm: say(f"  SA {k}/{n} {nm}") if k % 25 == 0 else None)
    say("(c) MI trên FIT (null xáo trộn)")
    mi = mutual_info(store, folds, colset, seed=cfg.sel_seed)
    table = FilterTable(names, pi, sa_e0, sa_b0, mi)
    df = table.to_frame()
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.exp_dir / "b0_filter.csv", index=False)
    red = df[(df[[f"SA_gain_b0306_h{h}" for h in (1, 2, 3)]] > eps).sum(axis=1) >= 2]
    if len(red):
        say(f"CỜ ĐỎ: {len(red)} cột đơn thắng B0-306 (≥ 2/3 horizon) — B0 bị nhiễu chi phối; R3/R4 sẽ tự thắng ở kiểm chứng")
    sets = table.sets()
    say("kiểm chứng R1–R4 (15fixed_306): " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    vdf, chosen, runs = verify_sets(store, model, sets, folds, rounds, base_rmse, eps, seed=cfg.sel_seed)
    vdf.to_csv(cfg.exp_dir / "b0_sets.csv", index=False)
    star = ColSet(tuple(sets[chosen])) if chosen in sets else colset
    star.save(cfg.exp_dir / "b0_star.json")
    for _, r in vdf.iterrows():
        exp_id = new_exp_id("filter_b0", "lgbm", str(r["set"]))
        extra = _summ_row(runs[r["set"]], base_rmse, "B0-306") if r["set"] in runs else {"base": "B0-306"}
        _log(cfg, exp_id=exp_id, step="filter_b0", model="lgbm", seed=cfg.sel_seed, colset=str(r["set"]), rounds="15fixed_306", **extra,
             decision="B0*" if r["chosen"] else "", note=f"n_cols={r['n_cols']}", train_device=getattr(model, "train_device", ""))
        if r["set"] in runs:
            save_run(cfg.exp_dir, exp_id, {**runs[r["set"]].to_dict(), "step": "filter_b0", "set": str(r["set"]), "base": "B0-306", "eps": eps})
    say(f"B0* = {chosen} ({len(star.names)} cột) → experiments/b0_star.json")


def _standalone_factory(store, folds, allow_cpu, cfg):
    lgbm = model_for(cfg, "lgbm", allow_cpu)

    def fn(cand) -> float:
        cs = ColSet((), tuple(cand.columns))
        run = run_config(store, lgbm, cs, folds, rounds=None, seed=cfg.sel_seed, keep_states=False)
        return float(np.median(run.gain_vs(run.e0)))

    return fn


def _resume_loop_state(cfg: RunConfig, mname: str, base: ColSet, cands: list, calib: dict, standalone_fn):
    """Khôi phục S_m + lịch sử KEEP/DROP đã ghi để TIẾP TỤC add-one (không chạy lại candidate cũ).

    Nguồn sự thật: `experiments/calib/<m>_base.json` (base_rmse/e0/eps/rounds — KHÔNG tính lại) + các dòng `step=loop,
    model=<m>` trong log.csv. Kiểm tra chặt, thà DỪNG còn hơn đoán."""
    log_path = cfg.exp_dir / "log.csv"
    if not log_path.exists():
        sys.exit(f"--resume: thiếu {log_path} — không có lịch sử để tiếp tục.")
    log = pd.read_csv(log_path)
    done = log[(log["step"] == "loop") & (log["model"] == mname)].drop_duplicates(subset=["colset"], keep="last")
    by_cols = {str(r["colset"]): r for _, r in done.iterrows()}
    base_rmse, e0 = np.asarray(calib["rmse"]), np.asarray(calib["e0"])
    eps = float(calib["eps"])
    S, S_rmse = base, base_rmse
    rows, kept, dropped = [], [], []
    n_done = 0
    for i, cand in enumerate(cands, start=1):
        key = "|".join(cand.columns)
        if key not in by_cols:
            break
        r = by_cols[key]
        cs = S.with_ext(cand.columns)
        if int(r["n_cols"]) != len(cs.names):
            sys.exit(f"--resume: candidate {cand.name} ghi n_cols={r['n_cols']} nhưng S hiện tại cho {len(cs.names)} "
                     "→ lịch sử không khớp S_m, DỪNG (không chạy lại từ đầu).")
        rmse = np.asarray(json.loads(r["rmse_cells"]))
        decision = str(r["decision"])
        if decision not in ("KEEP", "DROP"):
            sys.exit(f"--resume: decision lạ {decision!r} ở {cand.name} — DỪNG.")
        size_after = len(cs.names) if decision == "KEEP" else len(S.names)
        rows.append({"order": i, "candidate": cand.name, "columns": key, "group": cand.group,
                     "MedianGain_vs_S": float(r["MedianGain"]), "WinRate": float(r["WinRate"]),
                     "P10Gain": float(r["P10Gain"]), "WorstGain": float(r["WorstGain"]),
                     "MedianGain_vs_base": float(np.median(gain_pp(rmse, base_rmse))),
                     "MedianGain_vs_E0": float(np.median(gain_pp(rmse, e0))),
                     "gain_standalone_E0": float(standalone_fn(cand)) if standalone_fn else np.nan,
                     "decision": decision, "eps": eps, "size_S_after": size_after,
                     "rmse_cells": r["rmse_cells"], "gain_cells_vs_S": r["gain_cells"], "exp_id": r["exp_id"]})
        if decision == "KEEP":
            S, S_rmse = cs, rmse
            kept.append(cand.name)
        else:
            dropped.append(cand.name)
        n_done += 1
    later = [c.name for c in cands[n_done:] if "|".join(c.columns) in by_cols]
    if later:
        sys.exit(f"--resume: có candidate đã chạy nằm SAU điểm dừng ({later[:5]}) → lịch sử không liên tục, DỪNG.")
    return {"S": S, "S_rmse": S_rmse, "rows": rows, "kept": kept, "dropped": dropped, "start": n_done + 1}, n_done


def _adapter_identity(conf) -> list[dict] | None:
    """Danh tính adapter LoRA ĐÃ FREEZE thực sự dùng trong confirmation (TimesFM). Hệ thống A và B phải trùng danh sách này."""
    items: dict[str, dict] = {}
    for r in conf.runs:
        for a in (r.aux or []):
            if a:
                items[str(a["key"])] = {k: a.get(k) for k in ("key", "sha256", "best_epoch", "mode")}
    return [items[k] for k in sorted(items)] or None


def _write_json(path: Path, payload: dict) -> None:
    """Ghi JSON atomic (tmp + replace): nhiều branch chạy song song không bao giờ đọc phải file dở (§19)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _update_win(exp: Path, name: str, patch: dict) -> dict:
    p = exp / "wins" / f"{name}.json"
    payload = {**json.loads(p.read_text(encoding="utf-8")), **patch}
    _write_json(p, payload)
    return payload


def _save_win(exp: Path, name: str, conf, eps: float, which: str, folds: list[Fold], extra: dict | None = None) -> dict:
    """Artifact win: wins/<name>.json (colset, RMSE̅, E0, ε, best_iters, seed thật) + wins/<name>_seed<k>.npz (idx origin, ŷ)."""
    win_dir = exp / "wins"
    win_dir.mkdir(parents=True, exist_ok=True)
    adapters = _adapter_identity(conf)
    payload = {"model": name, "colset": conf.colset.to_dict(), "rmse_mean": conf.rmse_mean.tolist(), "e0": conf.e0.tolist(), "eps": eps,
               "best_iters_by_seed": [b.tolist() for b in conf.best_iters],
               "eval_seeds": [int(r.seed) for r in conf.runs], "which": which, "folds": [f.name for f in folds],
               "seed_rmse": [r.rmse.tolist() for r in conf.runs],
               "median_gain_vs_e0": float(np.median(gain_pp(conf.rmse_mean, conf.e0))),
               **({"lora_adapters": adapters} if adapters else {}), **(extra or {})}
    _write_json(win_dir / f"{name}.json", payload)
    for k, r in enumerate(conf.runs):
        np.savez_compressed(win_dir / f"{name}_seed{k}.npz", **{f"idx_{i}": p[0] for i, p in enumerate(r.preds())},
                            **{f"yhat_{i}": p[1] for i, p in enumerate(r.preds())})
    return payload


def _log_confirm(cfg: RunConfig, mname: str, tag: str, conf, eps: float, model) -> None:
    for k, r in enumerate(conf.runs):
        eid = new_exp_id("confirm", mname, f"{tag}_seed{k}")
        _log(cfg, exp_id=eid, step="confirm", model=mname, seed=r.seed, colset=tag, rounds="ES", **_summ_row(r, r.e0, "E0"),
             decision="", note=f"n_ext={len(conf.colset.ext)} n_new={len(conf.colset.new_ext)}", train_device=getattr(model, "train_device", ""))
        save_run(cfg.exp_dir, eid, {**r.to_dict(), "step": "confirm", "configuration": tag, "eps": eps}, r.preds())


def cmd_loop(cfg: RunConfig, args) -> None:
    """§2.1 cho model m từ S0_m KHOÁ: calibrate riêng (số vòng/epoch + ε mới trên data mới) → add-one Candidate_m → prune PI
    (chỉ cột mới) → confirmation 3 seed (raw vs pruned) → win_m → latency → §3 champion (probe: không so champion).
    Không vẽ figure (hậu kỳ: `visualize`)."""
    gate(cfg, args, [args.model])
    store, folds, _, _ = load_store(cfg)
    mname = args.model
    exp = cfg.exp_dir
    is_probe = mname in PROBE_MODELS
    try:
        base, cands = load_lock(exp, mname, dataset_label=cfg.dataset_label)  # overlap audit phải trên đúng dataset đang chạy
    except (FileNotFoundError, ValueError, KeyError) as e:
        hard_fail(exp, "loop", "S0_ARTIFACT", str(e), model=mname)
    if args.max_candidates:
        cands = cands[: args.max_candidates]
    model = model_for(cfg, args.model, args.allow_cpu)
    nw = fold_parallel.configure(cfg, model, mname, args.allow_cpu)
    if nw > 1:
        devs, slots, _ = gpu.worker_slots(cfg)
        say(f"[{mname}] scheduler GPU: {nw} worker đối xứng trên GPU {devs} ({slots} task nặng/GPU) — fold rải ĐỘNG, "
            "ghép theo đúng thứ tự fold; candidate vẫn TUẦN TỰ")
    deferred = champion_deferred(cfg)
    if not deferred and load_champion(exp / "champion.json") is None and mname != "lgbm":
        sys.exit("§3: champion ban đầu phải là LightGBM code gốc — chạy `loop --model lgbm` trước.")
    if deferred:  # §14: branch chỉ SINH đại diện; so champion để dành cho `champion-replay` theo thứ tự cố định
        say(f"[{mname}] champion HOÃN (defer_champion): branch chỉ ghi artifact đại diện; so champion ở `champion-replay`")
    base_label = ("∅ — hệ thống A = TimesFM-LoRA baseline (LoRA fine-tune trên r1, 0 covariate, KHÔNG B0*)" if mname == "tfm"
                  else f"S0_{mname} = {len(base.locked_b0)} B0* khoá + {len(base.locked_ext)} ext khoá")
    say(f"[{mname}] calibrate trên {base_label} — ES với calib_seed {cfg.calib_seed}; {len(cands)} candidate (Candidate_{mname})")
    if mname == "tfm":
        say("[tfm] calibrate = LoRA FIT + ES chọn epoch (calib_seed) → fixed_epoch_TFM → adapter cho eval_seeds (ε) → adapter selection_seed "
            "FREEZE cho toàn bộ add-one/prune (thêm candidate = fit lại XReg, không train lại LoRA)")
    calib_path = exp / "calib" / f"{mname}_base.json"
    standalone_fn = None if args.no_standalone else _standalone_factory(store, folds, args.allow_cpu, cfg)
    resume_state = None
    if getattr(args, "resume", False):
        if not calib_path.exists():
            sys.exit(f"--resume: thiếu {calib_path} — không có base đã ghi để tiếp tục.")
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        if ColSet.from_dict(calib["colset"]) != base:
            sys.exit("--resume: colset base trong calib khác S0_m hiện tại → DỪNG.")
        rounds, eps = calib["rounds"], float(calib["eps"])
        if isinstance(rounds, dict):
            rounds = {k: tuple(v) for k, v in rounds.items()}
        base_rmse, base_e0 = np.asarray(calib["rmse"]), np.asarray(calib["e0"])
        resume_state, n_done = _resume_loop_state(cfg, mname, base, cands, calib, standalone_fn)
        cands = cands[n_done:]
        say(f"[{mname}] RESUME: {n_done} candidate đã xong — |S_m| = {len(resume_state['S'].names)} "
            f"({len(resume_state['kept'])} KEEP / {len(resume_state['dropped'])} DROP); tiếp tục từ #{resume_state['start']}")
        say(f"[{mname}] rounds={rounds} ε={eps:.4f} pp (đọc lại từ calib, không tính lại)")
    else:
        cal = calibrate(store, model, base, folds, seed=cfg.calib_seed, keep_states=False)
        rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
        eps, noise, runs = seed_noise(store, model, base, folds, rounds, cfg.eval_seeds, cfg.eps_floor_pp)
        base_run = run_at_seed(runs, cfg.sel_seed)  # mốc của vòng lặp: cùng selection_seed với mọi candidate
        if base_run is None:
            base_run = run_config(store, model, base, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=False)
        base_rmse, base_e0 = base_run.rmse, base_run.e0
        say(f"[{mname}] rounds={rounds} ε={eps:.4f} pp; base MedianGain vs E0 = {np.median(base_run.gain_vs(base_run.e0)):+.4f}")
        (exp / "calib").mkdir(parents=True, exist_ok=True)
        calib_path.write_text(json.dumps(
            {"model": mname, "rounds": rounds, "eps": eps, "noise_cells": np.round(noise, 5).tolist(), "calib_seed": cfg.calib_seed,
             "eval_seeds": list(cfg.eval_seeds), "selection_seed": cfg.sel_seed, "rmse": base_run.rmse.tolist(),
             "e0": base_run.e0.tolist(), "seed_rmse": [r.rmse.tolist() for r in runs], "colset": base.to_dict(),
             "best_iters_es": cal.best_iters.tolist(), "config_hash": cfg.hash()}, indent=1), encoding="utf-8")
        _log(cfg, exp_id=new_exp_id("calibrate", mname, "base"), step="calibrate", model=mname, seed=cfg.sel_seed, colset="S0",
             rounds=json.dumps(rounds), **_summ_row(base_run, base_run.e0, "E0"), decision=f"eps={eps:.4f}",
             train_device=getattr(model, "train_device", ""),
             note=(f"S0: locked_b0={len(base.locked_b0)} locked_ext={len(base.locked_ext)}" if mname != "tfm"
                   else "TimesFM-LoRA baseline: calibrate = LoRA FIT + ES chọn epoch; S0 = ∅"))
    kd_path = exp / f"keepdrop_{mname}.csv"

    def on_row(row, run):
        exp_id = new_exp_id("loop", mname, row["candidate"])
        row["exp_id"] = exp_id  # vào keepdrop_<m>.csv (§7.2)
        say(f"[{mname}] {row['order']:03d} {row['candidate']:<28} Median {row['MedianGain_vs_S']:+.4f} → {row['decision']} (|S|={row['size_S_after']})")
        _log(cfg, exp_id=exp_id, step="loop", model=mname, seed=cfg.sel_seed, colset=row["columns"], n_cols=len(run.colset.names),
             rounds=json.dumps(rounds), base="S_m", MedianGain=row["MedianGain_vs_S"], WinRate=row["WinRate"], P10Gain=row["P10Gain"],
             WorstGain=row["WorstGain"], rmse_cells=row["rmse_cells"], mae_cells=_cells(run.mae), e0_cells=_cells(run.e0),
             gain_cells=row["gain_cells_vs_S"], decision=row["decision"], train_device=getattr(model, "train_device", ""))
        save_run(exp, exp_id, {**run.to_dict(), "step": "loop", "candidate": row["candidate"], "decision": row["decision"], "eps": eps,
                               "MedianGain_vs_S": row["MedianGain_vs_S"]})

    lr = add_one_loop(store, model, base, base_rmse, cands, folds, rounds, eps, cfg.sel_seed, base_e0, standalone_fn,
                      on_row, resume=resume_state)
    lr.table.to_csv(kd_path, index=False)
    for _, r in lr.table.iterrows():  # tư vấn (WARN, không dừng): Gain > ~1 pp so với S là nghi leakage/bug theo §6.8
        if float(r["MedianGain_vs_S"]) > 1.0:
            ck_record(exp, "loop", "WARN", "UNUSUAL_GAIN", f"{r['candidate']}: MedianGain vs S = {float(r['MedianGain_vs_S']):+.3f} pp > 1 pp — kiểm tra leakage", model=mname)
    say(f"[{mname}] F*_raw: {len(lr.kept)} KEEP / {len(lr.dropped)} DROP → {kd_path}")
    g_fin = summarize(gain_pp(lr.final_rmse, base_rmse))
    _log(cfg, exp_id=new_exp_id("loop_final", mname), step="loop_final", model=mname, seed=cfg.sel_seed,
         colset="|".join(lr.final.new_ext), n_cols=len(lr.final.names), rounds=json.dumps(rounds), base="baseline_model",
         MedianGain=round(g_fin["MedianGain"], 4), WinRate=round(g_fin["WinRate"], 4), P10Gain=round(g_fin["P10Gain"], 4),
         WorstGain=round(g_fin["WorstGain"], 4), rmse_cells=_cells(lr.final_rmse), decision=f"F*_{mname}_raw",
         note=f"{len(lr.kept)} KEEP / {len(lr.dropped)} DROP; locked_b0={len(lr.final.locked_b0)} locked_ext={len(lr.final.locked_ext)} giữ nguyên",
         train_device=getattr(model, "train_device", ""))
    say(f"[{mname}] F*_raw vs baseline S0: MedianGain = {g_fin['MedianGain']:+.4f} pp")
    # prune PI — CHỈ cột ext mới (S0 khoá không bị xét)
    pruned, pi_df = prune_pi(store, model, lr.final, folds, rounds, cfg.sel_seed)
    pi_df.to_csv(exp / f"prune_pi_{mname}.csv", index=False)
    say(f"[{mname}] prune PI (chỉ cột mới): giữ {len(pruned.new_ext)}/{len(lr.final.new_ext)} → F_pruned (+{len(pruned.locked_ext)} ext khoá, {len(pruned.locked_b0)} B0 khoá)")
    # confirmation 3 seed (ES bật) → win; latency §7.4 đo cho cả hai cấu hình (predictor sống chỉ tồn tại trong lúc chạy)
    unp = confirm(store, model, lr.final, folds, cfg.eval_seeds, keep_states=True, latency_origins=args.latency_origins, measure_latency=True)
    prn = confirm(store, model, pruned, folds, cfg.eval_seeds, keep_states=True, latency_origins=args.latency_origins, measure_latency=True) \
        if pruned.ext != lr.final.ext else unp
    which, g, s = decide_win(unp, prn, eps)
    win = prn if which == "prune" else unp
    _log_confirm(cfg, mname, "unprune", unp, eps, model)
    if prn is not unp:
        _log_confirm(cfg, mname, "prune", prn, eps, model)
    pd.DataFrame([
        {"configuration": "unprune", "n_ext": len(lr.final.ext), "n_new": len(lr.final.new_ext), "rmse_mean": json.dumps(np.round(unp.rmse_mean, 4).tolist())},
        {"configuration": "prune", "n_ext": len(pruned.ext), "n_new": len(pruned.new_ext), "rmse_mean": json.dumps(np.round(prn.rmse_mean, 4).tolist()),
         "MedianGain_prune_vs_unprune": s["MedianGain"], "WinRate": s["WinRate"], "P10Gain": s["P10Gain"], "WorstGain": s["WorstGain"],
         "eps": eps, "win": which},
    ]).to_csv(exp / f"prune_{mname}.csv", index=False)
    say(f"[{mname}] F_win = {which} (confirmation F_raw vs F_pruned: MedianGain {s['MedianGain']:+.4f}, ε={eps:.4f}) "
        f"→ {len(win.colset.new_ext)} cột mới")
    # F_win = bộ ĐÃ THẮNG confirmation raw-vs-pruned ở TRÊN. Mọi bước sau (kể cả so với baseline TimesFM-LoRA) dùng ĐÚNG bộ này.
    confirmation_meta = {"stage": "confirmation F_raw vs F_pruned (3 eval seed, ES bật)", "which": which,
                         "MedianGain_pruned_vs_raw": round(float(s["MedianGain"]), 6), "eps": eps,
                         "n_new_raw": len(lr.final.new_ext), "n_new_pruned": len(pruned.new_ext), "n_new_win": len(win.colset.new_ext)}
    win_name = TFM_XREG_WIN if mname == "tfm" else mname
    meta = ({"role": "hệ thống B = TimesFM-LoRA (adapter freeze) + XReg(F_win)", "configuration": TFM_XREG_WIN, "system": "B",
             "feature_set": "F_win", "feature_set_source": confirmation_meta,
             **model.artifact_meta(win.colset.ext, native=not win.colset.ext)} if mname == "tfm"
            else {"confirmation": confirmation_meta})
    payload = _save_win(exp, win_name, win, eps, which, folds, meta)
    if payload["median_gain_vs_e0"] > 1.0:
        ck_record(exp, "confirm", "WARN", "UNUSUAL_GAIN", f"win_{mname}: MedianGain vs E0 = {payload['median_gain_vs_e0']:+.3f} pp > 1 pp — kiểm tra leakage", model=mname)
    lat = None
    if win.latency:
        lat = pd.DataFrame(win.latency)
        lat.to_csv(exp / f"latency_{win_name}.csv", index=False)
        log_latency(exp, lat, split="VAL")
        say(f"[{mname}] latency p95 (ms) per h: {lat['p95_ms'].round(3).tolist()} (predict device {lat['predict_device'].iloc[0]})")
    if mname == "tfm":
        # HỆ THỐNG A — TimesFM-LoRA baseline: LoRA fine-tune xong, 0 feature, 0 B0*, 0 covariate XReg.
        # CÙNG adapter đã freeze như hệ thống B; chỉ được dựng SAU khi F_win đã có (raw-vs-pruned xong ở trên).
        with scheduler.stage("confirmation", configuration=TFM_BASELINE_WIN):
            baseline = unp if not lr.final.ext else confirm(store, model, ColSet((), ()), folds, cfg.eval_seeds, keep_states=True,
                                                            latency_origins=args.latency_origins, measure_latency=True)
        if baseline is not unp:
            _log_confirm(cfg, mname, "baseline", baseline, eps, model)
        base_payload = _save_win(exp, TFM_BASELINE_WIN, baseline, eps, "baseline", folds,
                                 {"role": "hệ thống A = TimesFM-LoRA baseline (LoRA fine-tune, 0 feature, 0 B0*, 0 XReg)",
                                  "configuration": TFM_BASELINE_WIN, "system": "A", "feature_set": "∅",
                                  "built_after": "confirmation F_raw vs F_pruned → F_win", **model.artifact_meta((), native=True)})
        if baseline.latency:
            pd.DataFrame(baseline.latency).to_csv(exp / f"latency_{TFM_BASELINE_WIN}.csv", index=False)
            log_latency(exp, pd.DataFrame(baseline.latency), split="VAL")
        if base_payload.get("lora_adapters") and payload.get("lora_adapters") and base_payload["lora_adapters"] != payload["lora_adapters"]:
            hard_fail(exp, "loop", "TFM_ADAPTER_IDENTITY", "TimesFM: hệ thống A (baseline) và B (+XReg) KHÔNG dùng cùng adapter LoRA đã freeze "
                      f"— A={[a['key'] for a in base_payload['lora_adapters']]} vs B={[a['key'] for a in payload['lora_adapters']]}", model="tfm")
        ck_record(exp, "loop", "PASS", "TFM_FLOW_ORDER", "TimesFM: add-one → F_raw → prune PI → F_pruned → confirmation → F_win → "
                  f"hệ thống B = LoRA + XReg(F_win) ({len(win.colset.ext)} cột); hệ thống A = LoRA baseline (0 covariate) — "
                  "so hai hệ thống ở `tfm-final`", model="tfm")
        say(f"[tfm] A = TimesFM-LoRA baseline (feature-free): MedianGain vs E0 = {np.median(gain_pp(baseline.rmse_mean, baseline.e0)):+.4f} pp; "
            f"B = TimesFM-LoRA + XReg(F_win, {len(win.colset.ext)} cột) = {payload['median_gain_vs_e0']:+.4f} pp → chạy `tfm-final` (so A vs B)")
    if is_probe:
        log_champion(exp, {"exp_id": new_exp_id("probe", mname), "model": mname, "win": which, "n_ext": len(win.colset.ext),
                           "ext_cols": "|".join(win.colset.ext), "MedianGain_vs_E0": round(payload["median_gain_vs_e0"], 4),
                           "rmse_mean_win": _cells(win.rmse_mean), "decision": "probe — không so champion",
                           "champion_after": (load_champion(exp / "champion.json") or {}).get("model", ""),
                           "train_device": getattr(model, "train_device", "")})
        say(f"[{mname}] feature-search xong (F_best: {len(win.colset.new_ext)} cột mới + {len(win.colset.locked_ext)} ext khoá) — không so champion; "
            f"chạy `{FINAL_STEP[mname]}` để có model đại diện")
        return
    champ_extra = {"win": which, **{f"mae_h{h}": round(float(np.mean([r.mae for r in win.runs], axis=0)[:, h - 1].mean()), 4) for h in HORIZONS},
                   **({f"latency_{k}_ms": json.dumps(lat[f"{k}_ms"].round(3).tolist()) for k in ("p95", "p99", "max")} if lat is not None else {}),
                   "train_device": getattr(model, "train_device", ""), "predict_device": getattr(model, "predict_device", "")}
    # đại diện của model đã sẵn sàng (artifact đóng băng) — champion đọc lại từ đây, không cần model sống
    _update_win(exp, win_name, {"champion_extra": champ_extra, "representative": mname})
    if deferred:
        ck_record(exp, "loop", "INFO", "CHAMPION_DEFERRED", f"đại diện {mname} sẵn sàng (wins/{win_name}.json); so champion ở `champion-replay` "
                  f"theo thứ tự cố định {list(CHAMPION_ORDER)}", model=mname)
        say(f"[{mname}] xong — đại diện wins/{win_name}.json; champion sẽ so ở `champion-replay` (thứ tự cố định, không theo thứ tự chạy xong)")
        return
    champion_step(cfg, mname, win.colset, win.rmse_mean, win.e0, eps, champ_extra)


DEFAULT_AUTOTS_TEMPLATES = [  # bake-off phương án A: mọi dòng GPU; nhóm theo shift regressor (wr:<window> / mr)
    {"model": "wr", "window_size": 60, "regressor": "LightGBM"},
    {"model": "wr", "window_size": 60, "regressor": "xgboost"},
    {"model": "mr", "regressor": "xgboost"},
    {"model": "mr", "regressor": "LightGBM"},
]


def _autots_group(spec: dict) -> str:
    """Các dòng template dùng CHUNG một `future_regressor` nên phải cùng phép dịch: MR = f(s−1), WR = f(s+W−1)."""
    kind = str(spec.get("model", "wr")).lower()
    kind = {"windowregression": "wr", "multivariateregression": "mr"}.get(kind, kind)
    return "mr" if kind == "mr" else f"wr:{int(spec.get('window_size', 60))}"


def autots_regressors(cfg: RunConfig) -> dict:
    """Backend GPU cho regressor BÊN TRONG AutoTS, lấy đúng backend đã resolve ở config — không hard-code."""
    from .models_autots import MR_PARAMS, WR_PARAMS

    lgb_dev = str(_params_for(cfg, "lgbm").get("device_type", WR_PARAMS["model_params"]["device_type"]))
    xgb_dev = str(_params_for(cfg, "xgb").get("device", MR_PARAMS["model_params"]["device"]))
    return {"LightGBM": {**WR_PARAMS["model_params"], "device_type": lgb_dev},
            "xgboost": {**MR_PARAMS["model_params"], "device": xgb_dev}}


def _autots_probe_model(cfg: RunConfig, group: str, allow_cpu: bool, frozen=None):
    from .models_autots import AutoTSModel

    params = _params_for(cfg, "autots_wr" if group.startswith("wr") else "autots_mr")
    params.pop("window_size", None)
    kw = dict(kind="mr" if group == "mr" else "wr", allow_cpu=allow_cpu, frozen=frozen, **params)
    if group.startswith("wr:"):
        kw["window_size"] = int(group.split(":")[1])
    if allow_cpu:
        kw["device"] = "cpu"
    elif "regression_model" not in kw:  # backend đã resolve, không dùng hằng số hard-code trong models_autots
        reg = autots_regressors(cfg)
        name = "LightGBM" if kw["kind"] == "wr" else "xgboost"
        kw["regression_model"] = {"model": name, "model_params": reg[name]}
    return AutoTSModel(**kw)


def autots_bakeoff_fold(cfg: RunConfig, store: Store, fold, colset: ColSet, group: str, specs: list[dict],
                        nv: int, allow_cpu: bool, cov_all=None) -> tuple[str, dict, object]:
    """Bake-off template GPU trên TRAINING-SIDE của một fold (FIT+ES, dừng trước purge) → (tên model, params, bảng candidate)."""
    from .autots_search import search_best_template, template_frame
    from .harness import _standardize_fit
    from .models import SeriesBatch

    idx_fit = fold.fit.origins(store.ts, store.eligible)
    idx_es = fold.es.origins(store.ts, store.eligible)
    cov = _standardize_fit(store.grid_matrix(colset) if cov_all is None else cov_all, idx_fit)
    seq = SeriesBatch(store.ts, store.r1, np.concatenate([idx_fit, idx_es]), cov, tuple(colset.names))
    lo, hi = int(idx_fit.min()), int(idx_es.max()) + 1
    probe = _autots_probe_model(cfg, group, allow_cpu)
    df_tr, R_tr = probe.frames(seq, lo, hi)
    say(f"[{fold.name}|{group}] search trên {len(df_tr)} bar (đến {df_tr.index[-1]}), {len(specs)} template × {nv} validation")
    reg = None if allow_cpu else autots_regressors(cfg)  # backend GPU đã resolve chảy vào MỌI dòng template
    tmpl = template_frame(specs, seed=cfg.sel_seed, regressors=reg)
    name, params, all_t = search_best_template(df_tr, R_tr, tmpl, nv, cfg.sel_seed, regressors=reg)
    return name, params, all_t


def _autots_bakeoff_all(cfg: RunConfig, store: Store, folds, colset: ColSet, group: str, gspecs: list[dict], nv: int,
                        allow_cpu: bool, cov_all=None) -> list[tuple]:
    """Bake-off template cho TỪNG fold. Scheduler bật → mỗi fold là một task GPU (rải động lên GPU rảnh);
    tắt → tuần tự y như cũ. Kết quả luôn trả theo ĐÚNG thứ tự fold (tất định)."""
    if fold_parallel.active():
        tasks = [scheduler.Task(kind="autots_bakeoff", model="autots", fold=f.name, stage="autots_bakeoff",
                                payload={"colset": colset.to_dict(), "fold": f.name, "group": group, "specs": gspecs, "nv": int(nv)})
                 for f in folds]
        out = fold_parallel.submit(tasks)
        return [(r["model"], r["params"], pd.DataFrame(r["table"]) if r["table"] is not None else pd.DataFrame()) for r in out]
    return [autots_bakeoff_fold(cfg, store, f, colset, group, gspecs, nv, allow_cpu, cov_all) for f in folds]


def _autots_score_all(cfg: RunConfig, store: Store, folds, colset: ColSet, group: str, frozen_by_fold: dict, seed: int,
                      allow_cpu: bool, want_preds: bool = False) -> list[tuple]:
    """Chấm AutoTS đã freeze template trên từng fold (song song trên GPU khi scheduler bật). Thứ tự = thứ tự fold."""
    if fold_parallel.active():
        tasks = [scheduler.Task(kind="autots_score", model="autots", fold=f.name, seed=int(seed), stage="autots_score",
                                payload={"colset": colset.to_dict(), "fold": f.name, "group": group, "seed": int(seed),
                                         "frozen": list(frozen_by_fold[f.name]), "want_preds": bool(want_preds)})
                 for f in folds]
        out = fold_parallel.submit(tasks)
        return [(np.asarray(r["rmse"]), np.asarray(r["e0"]), r["preds"]) for r in out]
    res = []
    for f in folds:
        m = _autots_probe_model(cfg, group, allow_cpu, frozen=frozen_by_fold[f.name])
        r = run_config(store, m, colset, [f], rounds=None, seed=seed, keep_states=want_preds)
        res.append((r.rmse[0], r.e0[0], (r.preds()[0] if want_preds else None)))
    return res


def autots_search_cfg(cfg: RunConfig) -> tuple[dict, int]:
    c = cfg.model_params("autots_search")
    specs = c.get("templates") or DEFAULT_AUTOTS_TEMPLATES
    groups = {}
    for sp in specs:
        groups.setdefault(_autots_group(sp), []).append(sp)
    return groups, int(c.get("num_validations", 10))


def cmd_autots_search(cfg: RunConfig, args) -> None:
    """§2.2 #6 (iii) — framework AutoTS trên HAI feature set đã freeze (F_WR_best, F_MR_best) → AutoTS-final. Không vẽ figure."""
    gate(cfg, args, ["autots_wr", "autots_mr"])
    exp = cfg.exp_dir
    nw = fold_parallel.configure(cfg, None, "autots", args.allow_cpu)
    if nw > 1:
        say(f"[autots-search] scheduler GPU: {nw} worker đối xứng — bake-off/chấm điểm từng fold rải động, kết quả ghép theo thứ tự fold")
    frozen_sets = {}
    for m in ("autots_wr", "autots_mr"):
        p = exp / "wins" / f"{m}.json"
        if not p.exists():
            sys.exit(f"Thiếu {p} — phải chạy `loop --model {m}` (add-one → prune → confirmation) trước.")
        w = json.loads(p.read_text(encoding="utf-8"))
        frozen_sets[f"F_{m.split('_')[1].upper()}_best"] = ColSet.from_dict(w["colset"])
    store, folds, _, _ = load_store(cfg)
    names = list(frozen_sets)
    if frozen_sets[names[0]].names == frozen_sets[names[1]].names:
        say(f"{names[0]} và {names[1]} trùng nhau → dedup, chỉ chạy framework một lần")
        frozen_sets = {f"{names[0]}={names[1]}": frozen_sets[names[0]]}
    groups, nv = autots_search_cfg(cfg)
    say(f"bake-off: {sum(len(v) for v in groups.values())} template / {len(groups)} nhóm shift {list(groups)}, "
        f"num_validations={nv}, {len(frozen_sets)} frozen set × {len(folds)} fold")
    tmpl_dir = exp / "autots_templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    rows, cands = [], {}
    for set_name, colset in frozen_sets.items():
        cov_all = None if fold_parallel.active() else store.grid_matrix(colset)  # scheduler bật: worker tự dựng regressor
        for group, gspecs in groups.items():
            frozen_by_fold = {}
            for f, (name, params, all_t) in zip(folds, _autots_bakeoff_all(cfg, store, folds, colset, group, gspecs, nv,
                                                                           args.allow_cpu, cov_all)):
                frozen_by_fold[f.name] = (name, params)
                tag = f"{set_name}_{group.replace(':', '')}_{f.name}".replace("=", "_")
                (tmpl_dir / f"best_{tag}.json").write_text(json.dumps({"set": set_name, "group": group, "fold": f.name,
                                                                      "model": name, "params": params}, indent=1), encoding="utf-8")
                try:
                    all_t.to_json(tmpl_dir / f"all_{tag}.json", orient="records", indent=1)
                except Exception:
                    pass
                say(f"[{set_name}|{group}|{f.name}] template thắng: {name}")
            fold_rmse, e0_rows = [], []
            for rmse_f, e0_f, _ in _autots_score_all(cfg, store, folds, colset, group, frozen_by_fold, cfg.sel_seed,
                                                     args.allow_cpu):  # CHỌN candidate: outer VAL ở ĐÚNG selection_seed
                fold_rmse.append(rmse_f)
                e0_rows.append(e0_f)
            rmse_sel, e0_tab = np.array(fold_rmse), np.array(e0_rows)
            key = f"{set_name}|{group}"
            cands[key] = {"set": set_name, "group": group, "colset": colset, "rmse_sel": rmse_sel, "e0": e0_tab, "templates": frozen_by_fold}
            g = float(np.median(gain_pp(rmse_sel, e0_tab)))
            rows.append({"candidate": key, "set": set_name, "group": group, "n_ext": len(colset.ext),
                         "MedianGain_vs_E0_sel": round(g, 4), "rmse_selection_seed": _cells(rmse_sel),
                         "templates": "|".join(sorted({v[0] for v in frozen_by_fold.values()}))})
            say(f"[{key}] outer VAL @ selection_seed {cfg.sel_seed}: MedianGain vs E0 = {g:+.4f} pp")
    final_key = max(cands, key=lambda k: float(np.median(gain_pp(cands[k]["rmse_sel"], cands[k]["e0"]))))
    fin = cands[final_key]
    say(f"AutoTS-final = {final_key} ({len(fin['colset'].ext)} cột ext) — chọn ở selection_seed {cfg.sel_seed}")
    tables, preds_by_seed = [], []
    for sd in cfg.eval_seeds:  # CONFIRMATION: winner đã FREEZE → 3 evaluation seed
        fold_rmse, preds = [], []
        for rmse_f, _e0, pr in _autots_score_all(cfg, store, folds, fin["colset"], fin["group"], fin["templates"], sd,
                                                 args.allow_cpu, want_preds=True):
            fold_rmse.append(rmse_f)
            preds.append(pr)
        tables.append(np.array(fold_rmse))
        preds_by_seed.append(preds)
    rmse_mean = mean_rmse_over_seeds(tables)
    noise = seed_noise_cells(tables)
    eps = seed_noise_eps(tables, cfg.eps_floor_pp)  # ε của CHÍNH AutoTS-final
    g_fin = float(np.median(gain_pp(rmse_mean, fin["e0"])))
    say(f"AutoTS-final confirmation {list(cfg.eval_seeds)}: MedianGain vs E0 = {g_fin:+.4f} pp, ε = {eps:.4f} pp")
    for r_ in rows:
        if r_["candidate"] == final_key:
            r_.update({"is_final": True, "MedianGain_vs_E0_confirm": round(g_fin, 4), "eps_autots_final": round(eps, 5)})
    pd.DataFrame(rows).to_csv(exp / "autots_search.csv", index=False)
    win_dir = exp / "wins"
    win_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": "autots", "role": "AutoTS-final", "source": final_key, "group": fin["group"],
               "colset": fin["colset"].to_dict(), "rmse_mean": rmse_mean.tolist(), "e0": fin["e0"].tolist(), "eps": eps,
               "noise_cells": np.round(noise, 5).tolist(), "seed_rmse": [t.tolist() for t in tables],
               "rmse_selection_seed": fin["rmse_sel"].tolist(), "selection_seed": cfg.sel_seed,
               "eval_seeds": [int(sd) for sd in cfg.eval_seeds], "folds": [f.name for f in folds],
               "templates_per_fold": {k: v[0] for k, v in fin["templates"].items()}, "median_gain_vs_e0": g_fin}
    payload["representative"] = "autots"
    payload["champion_extra"] = {"win": "autots_final", "source": final_key, "group": fin["group"]}
    _write_json(win_dir / "autots.json", payload)
    for k, preds in enumerate(preds_by_seed):
        np.savez_compressed(win_dir / f"autots_seed{k}.npz", **{f"idx_{i}": p[0] for i, p in enumerate(preds)},
                            **{f"yhat_{i}": p[1] for i, p in enumerate(preds)})
    _log(cfg, exp_id=new_exp_id("autots_search", "autots"), step="autots_search", model="autots", seed=cfg.sel_seed,
         colset="|".join(fin["colset"].ext), n_cols=len(fin["colset"].names), rounds="bake-off template",
         base="E0", MedianGain=round(g_fin, 4), rmse_cells=_cells(rmse_mean), e0_cells=_cells(fin["e0"]),
         decision=f"AutoTS-final={final_key}", note=f"chọn @seed {cfg.sel_seed}; confirmation {list(cfg.eval_seeds)}; ε={eps:.4f}")
    if champion_deferred(cfg):
        ck_record(exp, "autots-search", "INFO", "CHAMPION_DEFERRED", "đại diện AutoTS (AutoTS-final) sẵn sàng — so champion ở `champion-replay`",
                  model="autots")
        say("AutoTS-final đã lưu (wins/autots.json) — champion so ở `champion-replay` theo thứ tự cố định")
        return
    champion_step(cfg, "autots", fin["colset"], rmse_mean, fin["e0"], eps, {"win": "autots_final"})


def champion_step(cfg: RunConfig, mname: str, colset: ColSet, rmse_mean: np.ndarray, e0: np.ndarray, eps: float,
                  extra: dict | None = None) -> str:
    """§3: so bảng RMSE̅ của win_m với champion → đổi/giữ, ghi champion_log.csv. Trả tên champion sau. Không vẽ."""
    exp = cfg.exp_dir
    champ_path = exp / "champion.json"
    if mname in CHAMPION_INELIGIBLE:  # §3 + yêu cầu 2026-09-04c: probe/cấu hình nội bộ KHÔNG BAO GIỜ đụng champion
        hard_fail(exp, "champion", "CHAMPION_INELIGIBLE",
                  f"'{mname}' là probe/cấu hình nội bộ, không phải model đại diện — chỉ {list(CHAMPION_ORDER)} mới được so champion "
                  f"(TimesFM: chỉ TFM-final; AutoTS: chỉ AutoTS-final).", model=mname)
    champ = load_champion(champ_path)
    if champ is None and mname != "lgbm":  # §3: champion ban đầu phải là LightGBM code gốc
        sys.exit(f"§3: chưa có champion — chạy `loop --model lgbm` trước khi để '{mname}' so champion.")
    state = {"model": mname, "colset": colset.to_dict(), "rmse_mean": rmse_mean.tolist(), "eps": eps, "e0": e0.tolist()}
    row = {"exp_id": new_exp_id("champion", mname), "model": mname, "n_ext": len(colset.ext), "ext_cols": "|".join(colset.ext),
           "MedianGain_vs_E0": round(float(np.median(gain_pp(rmse_mean, e0))), 4), "rmse_mean_win": _cells(rmse_mean),
           **{f"rmse_h{h}": round(float(rmse_mean[:, h - 1].mean()), 4) for h in HORIZONS}, **(extra or {})}
    if champ is None:
        save_champion(champ_path, state)
        row.update({"champion_before": "", "decision": "champion ban đầu", "champion_after": mname})
    else:
        change, gc, sc = compare(rmse_mean, np.asarray(champ["rmse_mean"]), float(champ["eps"]))
        row.update({"champion_before": champ["model"], "MedianGain_vs_champion": round(sc["MedianGain"], 4),
                    "WinRate": round(sc["WinRate"], 4), "P10Gain": round(sc["P10Gain"], 4), "WorstGain": round(sc["WorstGain"], 4),
                    "eps_champion": champ["eps"], "decision": "đổi" if change else "giữ", "gain_cells": _cells(gc),
                    "rmse_mean_champion": _cells(np.asarray(champ["rmse_mean"])),
                    **{f"champ_rmse_h{h}": round(float(np.asarray(champ["rmse_mean"])[:, h - 1].mean()), 4) for h in HORIZONS},
                    "champion_after": mname if change else champ["model"]})
        if change:
            save_champion(champ_path, state)
    log_champion(exp, row)
    say(f"[{mname}] champion: {row['decision']} (champion sau = {row['champion_after']})")
    return str(row["champion_after"])


def _load_tfm_systems(exp: Path) -> tuple[dict, dict, str]:
    """Đọc HAI HỆ THỐNG HOÀN CHỈNH của TimesFM và kiểm tra vai trò của từng cái (2026-09-04c).

    A = `wins/tfm_lora_baseline.json` (tên cũ `tfm_lora_native.json` vẫn đọc được): TimesFM-LoRA đã fine-tune,
        0 feature / 0 B0* / 0 covariate XReg.
    B = `wins/tfm_lora_xreg.json`: CÙNG adapter LoRA đã freeze + XReg(F_win), với F_win là bộ ĐÃ THẮNG
        confirmation F_raw vs F_pruned (artifact phải chứng minh bằng `feature_set_source`/`which`).
    """
    xreg_path = exp / "wins" / f"{TFM_XREG_WIN}.json"
    if not xreg_path.exists():
        hard_fail(exp, "tfm-final", "S0_ARTIFACT", f"Thiếu {xreg_path} — phải chạy `loop --model tfm` (LoRA → freeze → XReg add-one → "
                  "prune PI → confirmation raw vs pruned → F_win) trước.", model="tfm")
    base_name = TFM_BASELINE_WIN if (exp / "wins" / f"{TFM_BASELINE_WIN}.json").exists() else TFM_BASELINE_LEGACY
    base_path = exp / "wins" / f"{base_name}.json"
    if not base_path.exists():
        hard_fail(exp, "tfm-final", "S0_ARTIFACT", f"Thiếu {exp / 'wins' / (TFM_BASELINE_WIN + '.json')} (hệ thống A = TimesFM-LoRA baseline "
                  "feature-free) — phải chạy `loop --model tfm` trước.", model="tfm")
    A = json.loads(base_path.read_text(encoding="utf-8"))
    B = json.loads(xreg_path.read_text(encoding="utf-8"))
    if base_name == TFM_BASELINE_LEGACY:
        ck_record(exp, "tfm-final", "INFO", "TFM_BASELINE_LEGACY_NAME",
                  f"đọc baseline từ tên cũ wins/{TFM_BASELINE_LEGACY}.json (= {TFM_BASELINE_WIN}); ngữ nghĩa không đổi", model="tfm")
    for tag, w in ((base_name, A), (TFM_XREG_WIN, B)):
        if w.get("finetune_method") != "LoRA" or "native" not in w or w["colset"]["b0"]:
            hard_fail(exp, "tfm-final", "S0_ARTIFACT", f"artifact {tag} thiếu metadata LoRA/native hoặc chứa B0* — không đúng vai trò", model="tfm")
    if not A.get("native") or A["colset"]["ext"]:
        hard_fail(exp, "tfm-final", "S0_ARTIFACT", f"{base_name} phải là TimesFM-LoRA baseline feature-free (0 covariate, 0 B0*)", model="tfm")
    src = B.get("feature_set_source") or {}
    if str(src.get("which", B.get("which", ""))) not in ("prune", "unprune"):
        hard_fail(exp, "tfm-final", "TFM_FLOW_ORDER", f"{TFM_XREG_WIN} không ghi kết quả confirmation F_raw vs F_pruned → không chứng minh được "
                  "feature set là F_win; `tfm-final` chỉ so hai hệ thống SAU khi raw-vs-pruned đã xong.", model="tfm")
    if A.get("lora_adapters") and B.get("lora_adapters") and A["lora_adapters"] != B["lora_adapters"]:
        hard_fail(exp, "tfm-final", "TFM_ADAPTER_IDENTITY", "hệ thống A và B không dùng cùng adapter LoRA đã freeze "
                  f"(A={[a['key'] for a in A['lora_adapters']]} vs B={[a['key'] for a in B['lora_adapters']]})", model="tfm")
    return A, B, base_name


def cmd_tfm_final(cfg: RunConfig, args) -> None:
    """§2.2 #4 — TFM-final: so HAI HỆ THỐNG HOÀN CHỈNH (KHÔNG phải "XReg vs LoRA": XReg không phải model độc lập):

        A: TimesFM-LoRA baseline      — LoRA fine-tune, KHÔNG feature, KHÔNG B0*, KHÔNG covariate XReg
        B: TimesFM-LoRA + XReg(F_win) — CÙNG adapter LoRA đã freeze, cộng XReg trên F_win

    F_win đã thắng confirmation F_raw vs F_pruned ở `loop --model tfm` TRƯỚC bước này. Luật project: B thay A khi
    MedianGain(B vs A) > +ε_TFM (ε đo trên chính baseline lúc calibrate), ngược lại TFM-final = A.
    Không train/inference lại: dùng bảng RMSE̅ 3 seed của hai hệ thống đã lưu. Chỉ TFM-final mới đủ tư cách champion."""
    exp = cfg.exp_dir
    A, B, base_name = _load_tfm_systems(exp)
    gate(cfg, args, [])
    eps = float(A["eps"])
    change, gc, sc = compare(np.asarray(B["rmse_mean"]), np.asarray(A["rmse_mean"]), eps)
    rows = []
    for cfg_name, w, sysname, role in ((base_name, A, "A", "TimesFM-LoRA baseline (feature-free)"),
                                       (TFM_XREG_WIN, B, "B", "TimesFM-LoRA + XReg(F_win)")):
        rows.append({"configuration": cfg_name, "system": sysname, "role": role, "n_ext": len(w["colset"]["ext"]),
                     "ext_cols": "|".join(w["colset"]["ext"]),
                     "MedianGain_vs_E0": round(float(np.median(gain_pp(np.asarray(w["rmse_mean"]), np.asarray(w["e0"])))), 4),
                     "rmse_mean": _cells(np.asarray(w["rmse_mean"]))})
    rows[1].update({"MedianGain_vs_baseline": round(sc["MedianGain"], 4), "WinRate": round(sc["WinRate"], 4),
                    "P10Gain": round(sc["P10Gain"], 4), "WorstGain": round(sc["WorstGain"], 4), "eps_tfm": eps, "gain_cells": _cells(gc)})
    best_cfg, best_sys = (TFM_XREG_WIN, "B") if change else (base_name, "A")
    for r in rows:
        r["is_final"] = r["configuration"] == best_cfg
    pd.DataFrame(rows).to_csv(exp / "tfm_final.csv", index=False)
    w = B if change else A
    say(f"TimesFM-final = hệ thống {best_sys} ({best_cfg}): B {{TimesFM-LoRA + XReg(F_win)}} vs A {{TimesFM-LoRA baseline, feature-free}} "
        f"MedianGain {sc['MedianGain']:+.4f} pp (ε_TFM {eps:.4f}) → {'B thắng' if change else 'không đủ → giữ A'}")
    ck_record(exp, "tfm-final", "PASS", "TFM_FINAL", f"TimesFM-final = {best_cfg} (hệ thống {best_sys}); B vs A MedianGain {sc['MedianGain']:+.4f}, "
              f"ε {eps:.4f}; F_win từ confirmation {(B.get('feature_set_source') or {}).get('which', B.get('which'))}", model="tfm")
    payload = {**w, "model": "tfm", "role": "TimesFM-final", "representative": "tfm", "configuration": best_cfg, "system": best_sys,
               "covariate_scope": "ext", "baseline_artifact": base_name, "xreg_artifact": TFM_XREG_WIN,
               "compare_systems": {"A": base_name, "B": TFM_XREG_WIN, "MedianGain_B_vs_A": sc["MedianGain"], "WinRate": sc["WinRate"],
                                   "P10Gain": sc["P10Gain"], "WorstGain": sc["WorstGain"], "eps": eps, "decision": best_cfg},
               "champion_extra": {"win": f"tfm_final={best_cfg}", "system": best_sys}}
    payload["compare_xreg_vs_native"] = payload["compare_systems"]  # tên cũ (tương thích công cụ đọc artifact 2026-09-04)
    _write_json(exp / "wins" / "tfm.json", payload)
    for k in range(len(w.get("eval_seeds", cfg.eval_seeds))):
        src = exp / "wins" / f"{best_cfg}_seed{k}.npz"
        if src.exists():
            (exp / "wins" / f"tfm_seed{k}.npz").write_bytes(src.read_bytes())
    _log(cfg, exp_id=new_exp_id("tfm_final", "tfm"), step="tfm_final", model="tfm", seed=cfg.sel_seed,
         colset="|".join(w["colset"]["ext"]), n_cols=len(w["colset"]["b0"]) + len(w["colset"]["ext"]), rounds="LoRA",
         base=base_name, MedianGain=round(sc["MedianGain"], 4), WinRate=round(sc["WinRate"], 4), P10Gain=round(sc["P10Gain"], 4),
         WorstGain=round(sc["WorstGain"], 4), rmse_cells=_cells(np.asarray(w["rmse_mean"])), e0_cells=_cells(np.asarray(w["e0"])),
         gain_cells=_cells(gc), decision=f"TimesFM-final={best_cfg}",
         note="hệ thống B {TimesFM-LoRA + XReg(F_win)} vs hệ thống A {TimesFM-LoRA baseline feature-free}, cùng adapter LoRA đã freeze")
    if champion_deferred(cfg):
        ck_record(exp, "tfm-final", "INFO", "CHAMPION_DEFERRED", "đại diện TimesFM (TFM-final) sẵn sàng — so champion ở `champion-replay`", model="tfm")
        say("TimesFM-final đã lưu (wins/tfm.json) — champion so ở `champion-replay` theo thứ tự cố định")
        return
    champion_step(cfg, "tfm", ColSet.from_dict(w["colset"]), np.asarray(w["rmse_mean"]), np.asarray(w["e0"]), float(w["eps"]),
                  {"win": f"tfm_final={best_cfg}"})


def representatives_expected(cfg: RunConfig) -> list[str]:
    """Các model ĐẠI DIỆN cần có trước khi replay champion, theo THỨ TỰ CỐ ĐỊNH của §3 (không phải thứ tự chạy xong)."""
    want = set()
    for m in (cfg.model_order or list(CHAMPION_ORDER)):
        want.add("autots" if str(m).startswith("autots") else str(m))
    return [m for m in CHAMPION_ORDER if m in want]


def cmd_champion_replay(cfg: RunConfig, args) -> None:
    """§3 + §14 (2026-09-04c): so champion trên các ARTIFACT ĐẠI DIỆN đã đóng băng, theo THỨ TỰ CỐ ĐỊNH
    lgbm → xgb → cat → tfm(TFM-final) → xgbrf → autots(AutoTS-final) → lstm.

    Vì các branch model có thể chạy xong theo thứ tự bất kỳ trên 2 GPU, thứ tự HOÀN THÀNH không được phép quyết định
    champion: replay đọc `wins/<m>.json` (RMSE̅ từng ô, ε, metadata, champion_extra) và áp đúng luật so sánh cũ
    (`compare`, MedianGain > +ε_champion). KHÔNG train, KHÔNG inference, không cần data/GPU."""
    exp = cfg.exp_dir
    order = representatives_expected(cfg)
    have = [m for m in order if (exp / "wins" / f"{m}.json").exists()]
    missing = [m for m in order if m not in have]
    if missing and not getattr(args, "allow_partial", False):
        hard_fail(exp, "champion-replay", "REPRESENTATIVE_MISSING",
                  f"thiếu đại diện {missing} (cần: {[REPRESENTATIVE_OF.get(m, m) for m in missing]}) — champion replay chỉ chạy khi mọi "
                  f"đại diện đã có; dùng --allow-partial nếu cố ý replay một phần.")
    if missing:
        ck_record(exp, "champion-replay", "WARN", "REPRESENTATIVE_MISSING", f"replay MỘT PHẦN: thiếu {missing} (--allow-partial)")
    if not have or have[0] != "lgbm":
        hard_fail(exp, "champion-replay", "CHAMPION_ORDER", f"§3: champion ban đầu phải là LightGBM — đại diện có: {have}")
    champ_path = exp / "champion.json"
    if champ_path.exists():
        if not getattr(args, "force_replay", False):
            hard_fail(exp, "champion-replay", "CHAMPION_EXISTS", f"{champ_path} đã tồn tại — replay là bước DUY NHẤT ghi champion state. "
                      "Dùng --force-replay để dựng lại (bản cũ được archive).")
        arch = exp / f"champion_prereplay_{time.strftime('%Y%m%d_%H%M%S')}.json"
        champ_path.replace(arch)
        ck_record(exp, "champion-replay", "WARN", "CHAMPION_REPLAY_FORCED", f"champion.json cũ → {arch.name}, dựng lại từ artifact đại diện")
    say(f"champion replay (không train): thứ tự cố định {have}")
    rows, champion = [], ""
    for m in have:
        w = json.loads((exp / "wins" / f"{m}.json").read_text(encoding="utf-8"))
        before = (load_champion(champ_path) or {}).get("model", "")
        champion = champion_step(cfg, m, ColSet.from_dict(w["colset"]), np.asarray(w["rmse_mean"]), np.asarray(w["e0"]),
                                 float(w["eps"]), {**(w.get("champion_extra") or {}), "replay": True})
        rows.append({"order": len(rows) + 1, "model": m, "artifact": f"wins/{m}.json", "champion_before": before,
                     "champion_after": champion, "MedianGain_vs_E0": round(float(w.get("median_gain_vs_e0", np.nan)), 4)})
    pd.DataFrame(rows).to_csv(exp / "champion_replay.csv", index=False)
    _write_json(exp / "champion_replay.json", {"order": have, "missing": missing, "champion": champion, "rows": rows,
                                               "rule": "MedianGain > +eps_champion (§3), thứ tự cố định — KHÔNG theo thứ tự chạy xong",
                                               "config_hash": cfg.hash(), "replayed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    ck_record(exp, "champion-replay", "PASS", "CHAMPION_REPLAY", f"replay {len(have)} đại diện theo thứ tự {have} → champion = {champion}; "
              "không train/inference")
    say(f"champion sau replay = {champion} → champion_replay.csv")


def cmd_gpu_probe(cfg: RunConfig, args) -> None:
    """Kiểm tra scheduler + định tuyến GPU THẬT trước khi chạy (không training, không đọc data).

    Mỗi worker báo GPU vật lý mình được giao (CUDA_VISIBLE_DEVICES → torch device name/uuid); chạy vài task giả
    có thời lượng khác nhau để chứng minh cả hai GPU đều nhận việc và không GPU nào bị bỏ trống."""
    from .scheduler import GpuScheduler, Task

    devices, slots, n = gpu.worker_slots(cfg)
    say(f"gpu-probe: devices={devices} slots/device={slots} → {n} worker đối xứng (không có affinity theo model family)")
    sch = GpuScheduler(cfg, allow_cpu=bool(getattr(args, "allow_cpu", False)), exp_dir=cfg.exp_dir, light=True).start()
    try:
        reports = sch.reports()
        for wid, rep in sorted(reports.items()):
            say(f"  worker {wid} → GPU vật lý {rep.get('gpu_physical_id')} | CUDA_VISIBLE_DEVICES={rep.get('cuda_visible_devices')} | "
                f"{rep.get('device_name')} | uuid={rep.get('device_uuid') or '?'} | torch device_count={rep.get('torch_device_count')}")
        uuids = [str(r.get("device_uuid") or "") for r in reports.values() if r.get("device_uuid")]
        if len(devices) > 1 and uuids:  # bằng chứng THẬT là uuid khác nhau, không chỉ CUDA_VISIBLE_DEVICES khác nhau
            if len(set(uuids)) < len(uuids):
                hard_fail(cfg.exp_dir, "gpu-probe", "GPU_UUID_COLLISION",
                          f"nhiều worker báo CÙNG một GPU (uuid {uuids}) — backend không tôn trọng CUDA_VISIBLE_DEVICES; "
                          "không được training trên cấu hình này (chạy 1 GPU: P0_GPU_DEVICES=0).")
            ck_record(cfg.exp_dir, "gpu-probe", "PASS", "GPU_UUID_DISTINCT", f"{len(set(uuids))} GPU vật lý phân biệt theo uuid: {sorted(set(uuids))}")
        elif len(devices) > 1:
            ck_record(cfg.exp_dir, "gpu-probe", "WARN", "GPU_UUID_UNKNOWN",
                      "không đọc được device uuid (torch thiếu hoặc không có CUDA trong worker) — không xác nhận được hai worker "
                      "nằm trên hai GPU vật lý khác nhau; kiểm bằng nvidia-smi lúc chạy thật.")
        ms = [400, 150, 150, 400, 150, 150]
        t0 = time.time()
        out = sch.submit([Task(kind="probe", stage="gpu_probe", payload={"sleep_ms": m, "tag": f"p{i}"}) for i, m in enumerate(ms)])
        used = sorted({int(r["gpu_physical_id"]) for r in out})
        say(f"  {len(ms)} task giả ({sum(ms)} ms tuần tự) xong sau {time.time() - t0:.2f}s trên GPU {used}")
        if len(devices) > 1 and len(used) < len(devices):
            ck_record(cfg.exp_dir, "gpu-probe", "WARN", "GPU_IDLE", f"chỉ GPU {used} nhận task trong probe (cấu hình {devices})")
        else:
            ck_record(cfg.exp_dir, "gpu-probe", "PASS", "GPU_SYMMETRIC", f"{n} worker, GPU {used} đều nhận task; định tuyến qua CUDA_VISIBLE_DEVICES")
    finally:
        sch.shutdown()


def cmd_ensemble(cfg: RunConfig, args) -> None:
    """§3 ensemble: thành viên = champion + mọi win_m có MedianGain vs E0 > 0; (a) đều, (b) 1/MSE; so với champion."""
    store, folds, _, _ = load_store(cfg)
    exp = cfg.exp_dir
    champ = load_champion(exp / "champion.json")
    if champ is None:
        sys.exit("Chưa có champion.")
    wins = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (exp / "wins").glob("*.json")
            if p.stem not in NON_MEMBER_WINS}  # probe/nhánh không phải thành viên; đại diện = autots (AutoTS-final), tfm (TimesFM-final)
    members = [m for m, w in wins.items() if w["median_gain_vs_e0"] > 0 or m == champ["model"]]
    if len(members) < 2:
        say("Không đủ 2 thành viên — không ensemble.")
        return
    preds = {m: [load_preds(exp / "wins" / f"{m}_seed{k}.npz") for k in range(len(wins[m]["eval_seeds"]))] for m in members}
    eq = ensemble_rmse(store, preds, folds)
    inv = ensemble_rmse(store, preds, folds, inverse_mse_weights({m: np.asarray(wins[m]["rmse_mean"]) for m in members}))
    champ_tab = np.asarray(champ["rmse_mean"])
    res = {}
    for name, tab in (("equal", eq), ("inv_mse", inv)):
        change, g, s = compare(tab, champ_tab, float(champ["eps"]))
        res[name] = (change, s, tab)
        say(f"ensemble {name}: MedianGain vs champion {s['MedianGain']:+.4f} → {'đổi' if change else 'giữ'}")
    best = max(res, key=lambda k: res[k][1]["MedianGain"])
    change, s, tab = res[best]
    g_best = gain_pp(tab, champ_tab)
    row = {"exp_id": new_exp_id("ensemble", best), "model": f"ensemble_{best}", "members": "|".join(members), "weighting": best,
           "MedianGain_vs_E0": round(float(np.median(gain_pp(tab, np.asarray(champ["e0"])))), 4), "champion_before": champ["model"],
           "MedianGain_vs_champion": round(s["MedianGain"], 4), "WinRate": round(s["WinRate"], 4), "P10Gain": round(s["P10Gain"], 4),
           "WorstGain": round(s["WorstGain"], 4), "eps_champion": champ["eps"], "decision": "đổi" if change else "giữ",
           "champion_after": f"ensemble_{best}" if change else champ["model"], "rmse_mean_win": _cells(tab), "rmse_mean_champion": _cells(champ_tab),
           "gain_cells": _cells(g_best), **{f"rmse_h{h}": round(float(tab[:, h - 1].mean()), 4) for h in HORIZONS},
           **{f"champ_rmse_h{h}": round(float(champ_tab[:, h - 1].mean()), 4) for h in HORIZONS}}
    log_champion(exp, row)
    (exp / "ensemble.json").write_text(json.dumps({"members": members, "weighting": best, "rmse_mean": tab.tolist(), "is_champion": change}, indent=1), encoding="utf-8")
    if change:
        save_champion(exp / "champion.json", {"model": f"ensemble_{best}", "members": members, "weighting": best, "rmse_mean": tab.tolist(),
                                              "eps": champ["eps"], "e0": champ["e0"]})


def cmd_final(cfg: RunConfig, args) -> None:
    """§4 Final một lần: refit B0-306, B0*, mọi win_m (+ ensemble) trên fold final → TEST; all_models_test.csv; lưu prediction
    TEST (final/<key>.npz + final/index.json) cho `visualize` — KHÔNG vẽ ở đây."""
    exp = cfg.exp_dir
    sentinel = exp / "final" / "TEST_SENTINEL.json"
    if sentinel.exists():  # §4/§15: TEST đúng MỘT lần — lần hai dừng ngay, không hỏi; chỉ --force-test-rerun (recovery) mới vượt
        prev = json.loads(sentinel.read_text(encoding="utf-8"))
        if not getattr(args, "force_test_rerun", False):
            hard_fail(exp, "final", "TEST_ALREADY_RUN",
                      f"TEST đã được chạm lúc {prev.get('started_at')} (status={prev.get('status')}, config_hash={prev.get('config_hash')}) — "
                      f"§4 TEST đúng MỘT lần; sentinel {sentinel}. Chạy lại chỉ với --force-test-rerun (recovery, ghi lý do vào checker_log).")
        ck_record(exp, "final", "WARN", "TEST_RERUN_FORCED", f"--force-test-rerun: chạy lại TEST (lần trước {prev.get('started_at')}, {prev.get('status')})")
    gate(cfg, args, [p.stem for p in (exp / "wins").glob("*.json") if p.stem not in NON_MEMBER_WINS] or ["lgbm"])
    if champion_deferred(cfg) and not (exp / "champion.json").exists():
        # chế độ hoãn champion (§14): champion CHỈ được quyết ở `champion-replay`. Chạm TEST khi chưa replay thì
        # cấu hình champion/ensemble chưa tồn tại → dừng ngay (TEST chỉ có một lần, không được phí).
        hard_fail(exp, "final", "CHAMPION_MISSING", "defer_champion đang bật nhưng chưa có champion.json — chạy "
                  "`python run.py champion-replay` (thứ tự cố định) và `ensemble` TRƯỚC khi chạm TEST.")
    store, folds, final, _ = load_store(cfg)
    (exp / "final").mkdir(parents=True, exist_ok=True)
    ck_path = checksum_path(cfg)
    sentinel_payload = {"status": "started", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_label": cfg.dataset_label,
                        "config_hash": cfg.hash(), "data_checksums": json.loads(ck_path.read_text(encoding="utf-8")) if ck_path.exists() else None,
                        "champion": (load_champion(exp / "champion.json") or {}).get("model"),
                        "wins_sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((exp / "wins").glob("*.json"))},
                        "test_partition": {"start": int(final.val.start), "end": int(final.val.end)}}
    sentinel.write_text(json.dumps(sentinel_payload, indent=1, ensure_ascii=False), encoding="utf-8")  # ghi TRƯỚC khi chạm TEST
    configs: dict[str, tuple[str, ColSet]] = {"b0_306": ("lgbm", store.all_b0())}
    star_path = exp / "b0_star.json"
    if not star_path.exists() and cfg.prev_run_dir and (cfg.path(cfg.prev_run_dir) / "b0_star.json").exists():
        star_path = cfg.path(cfg.prev_run_dir) / "b0_star.json"
    if star_path.exists():
        configs["b0_star"] = ("lgbm", ColSet.load(star_path))
    for p in sorted((exp / "wins").glob("*.json")):
        if p.stem in NON_MEMBER_WINS:  # probe / nhánh: đại diện là autots (AutoTS-final) / tfm (TimesFM-final)
            continue
        w = json.loads(p.read_text(encoding="utf-8"))
        configs[w["model"]] = (w["model"], ColSet.from_dict(w["colset"]))
    idx_test = final.val.origins(store.ts, store.eligible)
    c_t, c_future, _ = store.targets(idx_test)
    rows, ref_rmse, yhat_by_model = [], {}, {}
    (exp / "summary").mkdir(parents=True, exist_ok=True)
    (exp / "final").mkdir(parents=True, exist_ok=True)
    e0 = e0_rmse(c_t, c_future)
    rows.append({"model": "e0", **{f"rmse_h{h}": e0[h - 1] for h in HORIZONS}})
    index = {"test": {"start": int(final.val.start), "end": int(final.val.end), "n_origins": int(len(idx_test))},
             "fit": {"start": int(final.fit.start), "end": int(final.fit.end)}, "es": {"start": int(final.es.start), "end": int(final.es.end)},
             "keys": [], "models": {}, "dataset_label": cfg.dataset_label, "config_hash": cfg.hash()}

    def add_row(key, m, yhat, extra):
        ref_rmse[key] = m["rmse"]
        rows.append({"model": key, **{f"rmse_h{h}": m["rmse"][h - 1] for h in HORIZONS}, **{f"mae_h{h}": m["mae"][h - 1] for h in HORIZONS},
                     **{f"r_h{h}": m["r"][h - 1] for h in HORIZONS}, **{f"diracc_h{h}": m["dir_acc"][h - 1] for h in HORIZONS},
                     **{f"gain_e0_h{h}": gain_pp(m["rmse"], e0)[h - 1] for h in HORIZONS},
                     **{f"gain_b0306_h{h}": gain_pp(m["rmse"], ref_rmse["b0_306"])[h - 1] for h in HORIZONS if "b0_306" in ref_rmse},
                     **{f"gain_b0star_h{h}": gain_pp(m["rmse"], ref_rmse["b0_star"])[h - 1] for h in HORIZONS if "b0_star" in ref_rmse}, **extra})
        np.savez_compressed(exp / "final" / f"{key}.npz", idx_0=idx_test, yhat_0=np.asarray(yhat, np.float32))
        index["keys"].append(key)

    for key, (mname, cs) in configs.items():
        if mname == "autots":  # AutoTS-final: bake-off lại trên training-side của fold final rồi freeze template
            w = json.loads((exp / "wins" / "autots.json").read_text(encoding="utf-8"))
            groups, nv = autots_search_cfg(cfg)
            grp = w.get("group", next(iter(groups)))
            name, params, _ = autots_bakeoff_fold(cfg, store, final, cs, grp, groups[grp], nv, args.allow_cpu)
            (exp / "autots_templates").mkdir(parents=True, exist_ok=True)
            (exp / "autots_templates" / "best_FINAL.json").write_text(json.dumps(
                {"fold": final.name, "group": grp, "model": name, "params": params}, indent=1), encoding="utf-8")
            model = _autots_probe_model(cfg, grp, args.allow_cpu, frozen=(name, params))
        else:
            model = model_for(cfg, mname, args.allow_cpu)
        run = run_config(store, model, cs, [final], rounds=None, seed=cfg.sel_seed, keep_states=True)
        yhat = run.states[0].yhat
        yhat_by_model[key] = yhat
        m = cell_metrics(c_t, c_future, yhat)
        extra = {"best_iters": json.dumps(run.best_iters.tolist()), "n_ext": len(cs.ext), "n_locked": len(cs.locked),
                 "train_device": getattr(model, "train_device", ""), "predict_device": getattr(model, "predict_device", "")}
        try:
            from .latency import measure_tabular

            lat = measure_tabular(run, warmup=50, max_origins=args.latency_origins, model=model)
            lat["model"] = key
            lat.to_csv(exp / "summary" / f"latency_final_{key}.csv", index=False)
            log_latency(exp, lat, split="TEST")
            for _, lr in lat.iterrows():
                extra.update({f"lat_{k}_h{int(lr['h'])}": round(float(lr[f"{k}_ms"]), 3) for k in ("p95", "p99", "max")})
        except Exception as e:
            say(f"latency {key} bỏ qua: {e}")
        add_row(key, m, yhat, extra)
        index["models"][key] = {"model": mname, "colset": cs.to_dict(), "best_iters": run.best_iters.tolist()}
        exp_id = new_exp_id("final", key)
        _log(cfg, exp_id=exp_id, step="final", model=mname, seed=cfg.sel_seed, colset=key, rounds="ES", **_summ_row(run, run.e0, "E0"),
             train_device=getattr(model, "train_device", ""), decision="TEST", note="final_TEST")
        save_run(exp, exp_id, {**run.to_dict(), "step": "final", "key": key, "test_metrics": {k: np.asarray(v).tolist() for k, v in m.items()}},
                 [(idx_test, yhat)], pred_name="pred_test.npz")
        say(f"final {key}: RMSE {np.round(m['rmse'], 2).tolist()} | Gain vs E0 {np.round(gain_pp(m['rmse'], e0), 4).tolist()}")
    ens_path = exp / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text(encoding="utf-8"))
        mem = [m for m in ens["members"] if m in yhat_by_model]
        if len(mem) >= 2:
            w = {m: np.ones(3) for m in mem} if ens["weighting"] == "equal" else inverse_mse_weights(
                {m: np.asarray(json.loads((exp / "wins" / f"{m}.json").read_text())["rmse_mean"]) for m in mem})
            acc = sum(yhat_by_model[m] * w[m][None, :] for m in mem)
            yhat = acc / sum(w[m] for m in mem)[None, :]
            yhat_by_model["ensemble"] = yhat
            add_row("ensemble", cell_metrics(c_t, c_future, yhat), yhat, {"members": "|".join(mem), "weighting": ens["weighting"]})
            index["models"]["ensemble"] = {"members": mem, "weighting": ens["weighting"]}
    champ = load_champion(exp / "champion.json")
    champ_key = None if champ is None else ("ensemble" if str(champ["model"]).startswith("ensemble") else str(champ["model"]))
    if champ_key in ref_rmse:
        for r in rows:
            if r["model"] in ref_rmse:
                r.update({f"gain_champion_h{h}": gain_pp(ref_rmse[r["model"]], ref_rmse[champ_key])[h - 1] for h in HORIZONS})
        say(f"champion trên TEST = {champ_key}")
    index["champion"] = champ_key
    pd.DataFrame(rows).to_csv(exp / "summary" / "all_models_test.csv", index=False)
    (exp / "final" / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    sentinel_payload.update({"status": "completed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "keys": index["keys"]})
    sentinel.write_text(json.dumps(sentinel_payload, indent=1, ensure_ascii=False), encoding="utf-8")
    ck_record(exp, "final", "PASS", "TEST_ONCE", f"TEST chạm một lần: {len(index['keys'])} cấu hình, champion {champ_key}")
    say(f"final → {exp / 'summary' / 'all_models_test.csv'}; prediction TEST → {exp / 'final'} (figure: `python run.py visualize`)")


def cmd_visualize(cfg: RunConfig, args) -> None:
    """§7.3 hậu kỳ: dựng lại mọi figure từ artifact đã lưu — không train, không inference, không cần GPU."""
    from .visualize import regenerate_all

    store, folds, final, _ = load_store(cfg)
    out = Path(args.out) if getattr(args, "out", None) else cfg.exp_dir / "summary"
    made = regenerate_all(store, folds, final, cfg.exp_dir, out)
    for f in made:
        say(f"figure → {f}")
    say(f"{len(made)} figure từ artifact (wins/, champion_log.csv, final/)")


def cmd_smoke_e2e(cfg_unused, args) -> None:
    """Toàn bộ pipeline trên data tổng hợp, CPU (chỉ debug). Tạo cfg tạm trong --out."""
    from .synthetic import make_hf, make_lf

    out = Path(args.out).resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    hf = make_hf(n_days=args.days, seed=1)
    lf = make_lf(hf)
    hf.to_csv(out / "data" / "hf.csv", index=False)
    lf.to_csv(out / "data" / "lf.csv", index=False)
    start = pd.Timestamp(hf["timestamp"].iloc[0], unit="s", tz="UTC")
    days = [(start + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, int(args.days) - 1)]
    val_days = days[-3:]
    test_day = (start + pd.Timedelta(days=int(args.days) - 1)).strftime("%Y-%m-%d 00:00:00")
    cfg = RunConfig(dataset_label="synthetic_smoke", hf_csv="data/hf.csv", lf_csv="data/lf.csv", val_days=val_days, test_start=test_day,
                    calib_seed=1, eval_seeds=(2, 3, 4), selection_seed=2, experiments_dir="experiments", require_gpu=False,
                    root=str(out), candidates=[c.name for c in CANDIDATES])
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / "smoke.json").write_text(json.dumps({k: v for k, v in cfg.to_dict().items() if k != "root"}, indent=1), encoding="utf-8")
    (out / ".claude").mkdir(exist_ok=True)
    (out / ".claude" / "MEMORY.md").write_text("TRAINING: UNLOCKED\n", encoding="utf-8")
    ns = argparse.Namespace(smoke=True, allow_cpu=True, write_checksums=True, model="lgbm", colset="b0306", max_cols=6, max_candidates=3,
                            no_standalone=False, latency_origins=100, data_config=None, max_rows=None, out=None, resume=False,
                            force_test_rerun=False)
    cmd_check_data(cfg, ns)
    cmd_calibrate(cfg, ns)
    cmd_filter_b0(cfg, ns)
    cmd_lock_s0(cfg, ns)  # không có vòng trước → S0 = B0*; collision audit trên data tổng hợp
    cmd_loop(cfg, ns)
    ns.model = "xgb"
    cmd_loop(cfg, ns)
    cmd_ensemble(cfg, ns)
    cmd_final(cfg, ns)
    cmd_visualize(cfg, ns)
    say(f"smoke-e2e OK → {out / 'experiments'} (checker_log: {out / 'experiments' / 'checker_log.jsonl'})")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    def common(top: bool) -> argparse.ArgumentParser:
        c = argparse.ArgumentParser(add_help=False)
        sup = argparse.SUPPRESS
        c.add_argument("--config", default="configs/p0_full.json" if top else sup)
        c.add_argument("--smoke", action="store_true", default=False if top else sup,
                       help="bỏ qua khóa training + GPU gate — CHỈ chấp nhận với dataset_label 'synthetic*' (data tổng hợp / debug)")
        c.add_argument("--allow-cpu", action="store_true", default=False if top else sup,
                       help="ép model chạy CPU — CHỈ chấp nhận với dataset_label 'synthetic*' (unit/smoke test)")
        return c

    p = argparse.ArgumentParser(prog="p0", description="P0_forecasting harness (docs/RESEARCH_PLAN.md)", parents=[common(True)])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check-data", parents=[common(False)]); s.add_argument("--write-checksums", action="store_true")
    s = sub.add_parser("derive-lf", parents=[common(False)]); s.add_argument("--force", action="store_true", help="ghi đè LF đã có (tất định)")
    s = sub.add_parser("lock-s0", parents=[common(False)])
    s.add_argument("--data-config", default=None, help="config khác để lấy data cho collision audit bằng số (định nghĩa trùng thì trùng trên mọi data)")
    s.add_argument("--max-rows", type=int, default=None)
    s = sub.add_parser("calibrate", parents=[common(False)]); s.add_argument("--model", default="lgbm"); s.add_argument("--colset", default="b0306")
    s = sub.add_parser("filter-b0", parents=[common(False)]); s.add_argument("--max-cols", type=int, default=None)
    s = sub.add_parser("loop", parents=[common(False)]); s.add_argument("--model", required=True); s.add_argument("--max-candidates", type=int, default=None)
    s.add_argument("--resume", action="store_true", help="tiếp tục add-one từ candidate chưa chạy, dùng base/ε/S_m đã ghi")
    s.add_argument("--no-standalone", action="store_true"); s.add_argument("--latency-origins", type=int, default=None)
    sub.add_parser("tfm-final", parents=[common(False)])
    sub.add_parser("autots-search", parents=[common(False)])
    s = sub.add_parser("champion-replay", parents=[common(False)])
    s.add_argument("--allow-partial", action="store_true", help="replay khi CHƯA đủ đại diện (ghi WARN) — mặc định phải đủ")
    s.add_argument("--force-replay", action="store_true", help="dựng lại champion.json đã có (archive bản cũ)")
    s = sub.add_parser("orchestrate", parents=[common(False)])
    s.add_argument("--models", default=None, help="danh sách branch, mặc định model_order của config (vd: lgbm,xgb,cat,tfm,xgbrf,autots_wr,autots_mr,lstm)")
    s.add_argument("--max-candidates", type=int, default=None); s.add_argument("--no-standalone", action="store_true")
    s.add_argument("--latency-origins", type=int, default=None); s.add_argument("--resume", action="store_true")
    s.add_argument("--max-branches", type=int, default=None, help="số branch chạy đồng thời (mặc định = số worker GPU)")
    s.add_argument("--skip-ensemble", action="store_true"); s.add_argument("--allow-partial", action="store_true")
    s.add_argument("--force-replay", action="store_true"); s.add_argument("--dry-run", action="store_true", help="chỉ in DAG rồi thoát")
    sub.add_parser("gpu-probe", parents=[common(False)])
    sub.add_parser("ensemble", parents=[common(False)])
    s = sub.add_parser("final", parents=[common(False)]); s.add_argument("--latency-origins", type=int, default=None)
    s.add_argument("--force-test-rerun", action="store_true",
                   help="RECOVERY ONLY: vượt sentinel TEST-một-lần (final/TEST_SENTINEL.json); không bao giờ dùng tự động, phải ghi lý do")
    s = sub.add_parser("visualize", parents=[common(False)]); s.add_argument("--out", default=None)
    s = sub.add_parser("smoke-e2e", parents=[common(False)]); s.add_argument("--out", default="tmp_smoke"); s.add_argument("--days", type=float, default=6)
    args = p.parse_args(argv)
    if args.cmd == "smoke-e2e":
        cmd_smoke_e2e(None, args)
        return
    cfg = RunConfig.load(args.config)
    from .orchestrate import cmd_orchestrate

    {"check-data": cmd_check_data, "derive-lf": cmd_derive_lf, "lock-s0": cmd_lock_s0, "calibrate": cmd_calibrate, "filter-b0": cmd_filter_b0,
     "loop": cmd_loop, "tfm-final": cmd_tfm_final, "autots-search": cmd_autots_search, "champion-replay": cmd_champion_replay,
     "orchestrate": cmd_orchestrate, "gpu-probe": cmd_gpu_probe, "ensemble": cmd_ensemble,
     "final": cmd_final, "visualize": cmd_visualize}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
