"""
smoke_visualize.py — SINH SỐ GIẢ để xem trước layout các bảng/figure mà plan sẽ tạo.

FAKE / SMOKE: mọi con số trong output đều là số giả sinh bằng RNG cố định (seed 8586),
KHÔNG đọc data thật, KHÔNG train gì, KHÔNG phải kết quả experiment. Chỉ để thống nhất
layout của: bảng ε/lịch calibrate (§1.3), b0_filter (§1.4), keepdrop (§2.1), prune PI + win 3 seed (§2.1),
champion_log (§3), all_models VAL/TEST (§4, §7.2), figure win-vs-champion và Final (§7.3), latency (§7.4)
trong docs/RESEARCH_PLAN.md.

Chạy:  python reports/smoke_visualize.py
Output: reports/smoke/*.png và reports/smoke_visualize.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAKE = "FAKE / SMOKE — số giả để xem layout, không phải kết quả"
HERE = Path(__file__).resolve().parent
OUT = HERE / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
MD_PATH = HERE / "smoke_visualize.md"

rng = np.random.default_rng(8586)

H = [1, 2, 3]
FOLD_DAYS = ["01-27", "01-28", "01-29", "01-30", "01-31"]  # VAL 1 ngày/fold (§1.2)
TEST_DAYS = ["02-01", "02-02"]
SEEDS = [8586, 8587, 8588]
PRICE = 80_000.0  # mức giá BTC giả định (USD)
SIG1 = 7.65e-4  # std log-return 1 phút (đo trên snapshot, chỉ để scale số giả)

# "Skill" giả của từng model: Gain vs E0 (pp) theo horizon, dùng để sinh RMSE giả.
SKILL = {
    "E0": (0.00, 0.00, 0.00),
    "B0-306": (0.10, 0.07, 0.03),
    "B0*": (0.12, 0.08, 0.04),
    "LightGBM(F*)": (0.18, 0.12, 0.05),
    "XGBoost(F*)": (0.16, 0.11, 0.05),
    "CatBoost(F*)": (0.17, 0.11, 0.04),
    "TFM-POINT": (-0.05, -0.04, -0.03),
    "XGB-RF(F*)": (0.12, 0.09, 0.04),
    "AutoTS-WR(F*)": (0.10, 0.07, 0.03),
    "AutoTS-MR(F*)": (0.06, 0.05, 0.02),
    "LSTM(F*)": (0.09, 0.06, 0.03),
    "Ensemble": (0.21, 0.14, 0.06),
}
MODELS = list(SKILL)
CHAMPION = "LightGBM(F*)"  # champion hiện tại (§3)
WIN = "XGBoost(F*)"  # ví dụ: model vừa xong vòng lặp → win_m của nó so với champion

# Màu/marker cố định cho từng model ở MỌI figure nhiều model. Palette categorical đã validate
# (dataviz reference palette, thứ tự slot cố định, không xoay vòng); reference xám; giá thật = đen.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STYLE = {  # model -> (color, marker, linestyle)
    "B0-306": ("#898781", "s", "--"),
    "B0*": ("#52514e", "D", "--"),
    "LightGBM(F*)": (PALETTE[0], "o", "-"),
    "XGBoost(F*)": (PALETTE[1], "^", "-"),
    "CatBoost(F*)": (PALETTE[2], "v", "-"),
    "XGB-RF(F*)": (PALETTE[3], "X", "-"),
    "AutoTS-WR(F*)": (PALETTE[4], "<", "-"),
    "LSTM(F*)": (PALETTE[5], "*", "-"),
    "TFM-POINT": (PALETTE[6], "P", "-"),
    "Ensemble": (PALETTE[7], "h", "-"),
    "AutoTS-MR(F*)": (PALETTE[1], ">", ":"),  # model thứ 9 ngoài 8 slot: dùng lại orange (chỉ vẽ ở nhóm B, không cùng panel với XGBoost); đen dành riêng cho actual
}
# Ảnh so sánh win vs champion: màu theo VAI TRÒ (cặp xa nhau nhất: blue ↔ red), marker khác nhau; actual đen.
WIN_STYLE = ("#2a78d6", "^")
CHAMP_STYLE = ("#e34948", "o")
H_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ramp một màu (blue 250/450/650) cho h=1,2,3 hoặc p95/p99/max
INK, MUTED = "#0b0b0b", "#898781"
GROUP_A = ["LightGBM(F*)", "XGBoost(F*)", "CatBoost(F*)", "XGB-RF(F*)", "Ensemble"]  # ≤ 8 màu / panel
GROUP_B = ["TFM-POINT", "AutoTS-WR(F*)", "AutoTS-MR(F*)", "LSTM(F*)", "B0-306", "B0*"]

# Latency giả (ms) (p95, p99, max) tại h=1; shared = một lần gọi ra cả 3 bước (§7.4).
# Training luôn GPU; "device" ở đây là device của lời gọi predict (LightGBM/CatBoost predict trên CPU là đặc tính thư viện).
LAT = {
    "B0-306": ((0.60, 1.20, 3.1), False, "CPU (LightGBM predict)"),
    "B0*": ((0.50, 1.00, 2.6), False, "CPU (LightGBM predict)"),
    "LightGBM(F*)": ((0.70, 1.40, 3.5), False, "CPU (LightGBM predict)"),
    "XGBoost(F*)": ((1.10, 2.20, 5.0), False, "GPU"),
    "CatBoost(F*)": ((0.50, 1.00, 2.4), False, "CPU (CatBoost predict mặc định)"),
    "TFM-POINT": ((45.0, 90.0, 210.0), True, "GPU"),
    "XGB-RF(F*)": ((2.8, 6.0, 14.0), False, "GPU"),
    "AutoTS-WR(F*)": ((320.0, 650.0, 1500.0), True, "CPU pipeline + GPU regression_model"),
    "AutoTS-MR(F*)": ((420.0, 800.0, 1900.0), True, "CPU pipeline + GPU regression_model"),
    "LSTM(F*)": ((4.1, 8.5, 20.0), True, "GPU"),
}


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def rmse_e0(h: int, n: int, scale: float = 1.0) -> np.ndarray:
    base = PRICE * SIG1 * np.sqrt(h) * scale
    return base * (1.0 + 0.15 * rng.standard_normal(n))


def gain_pp(rmse_c: np.ndarray, rmse_b: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - rmse_c / rmse_b)


def summarize(g: np.ndarray) -> dict:
    return {
        "MedianGain": float(np.median(g)),
        "WinRate": float(np.mean(g > 0)),
        "P10Gain": float(np.percentile(g, 10)),
        "WorstGain": float(np.min(g)),
    }


def fmt_sum(s: dict) -> str:
    return f"Median {s['MedianGain']:+.3f} · Win {s['WinRate']:.2f} · P10 {s['P10Gain']:+.3f} · Worst {s['WorstGain']:+.3f}"


# ----------------------------------------------------------------------------- fake VAL cells (3 seed → median từng ô)
e0_val = {h: rmse_e0(h, len(FOLD_DAYS)) for h in H}  # E0 không có seed
rmse_seed: dict[str, dict[int, np.ndarray]] = {}  # model -> h -> array (3 seed, 5 fold)
for m, sk in SKILL.items():
    rmse_seed[m] = {}
    for j, h in enumerate(H):
        cell = sk[j] + (0.0 if m == "E0" else 0.02 * rng.standard_normal(len(FOLD_DAYS)))
        tabs = []
        for _ in SEEDS:
            g = cell + (0.0 if m == "E0" else 0.015 * rng.standard_normal(len(FOLD_DAYS)))
            tabs.append(e0_val[h] * (1.0 - g / 100.0))
        rmse_seed[m][h] = np.array(tabs)


def gain_table_median(m: str, base: str | None) -> np.ndarray:
    """Bảng Gain 15 ô (5 fold × 3 h) của model m so với base: tính bảng Gain cho TỪNG seed
    (base = bảng RMSE median 3 seed của base; E0 không có seed), chồng 3 bảng lên nhau, lấy median từng ô."""
    out = np.zeros((len(FOLD_DAYS), len(H)))
    for j, h in enumerate(H):
        b = e0_val[h] if base is None else np.median(rmse_seed[base][h], axis=0)
        per_seed = np.stack([gain_pp(rmse_seed[m][h][s], b) for s in range(len(SEEDS))])  # (3 seed, 5 fold)
        out[:, j] = np.median(per_seed, axis=0)
    return out


rmse_med = {m: {h: np.median(rmse_seed[m][h], axis=0) for h in H} for m in MODELS}

# Thành viên ensemble theo luật §3: champion + mọi model có MedianGain vs E0 > 0 (B0-306/B0* là reference)
ENSEMBLE_MEMBERS = [
    m for m in MODELS
    if m not in ("E0", "B0-306", "B0*", "Ensemble") and float(np.median(gain_table_median(m, None))) > 0
]


def lat_of(m: str) -> tuple[tuple[float, float, float], bool, str]:
    if m == "Ensemble":
        p = tuple(float(x) for x in np.sum([LAT[k][0] for k in ENSEMBLE_MEMBERS], axis=0))
        return p, False, "tổng các thành viên (CPU+GPU)"
    return LAT[m]


# TEST (2 ngày): một block per h + 8 khối 6 giờ × h cho heatmap Final
e0_test = {h: float(rmse_e0(h, 1, scale=1.08)[0]) for h in H}
rmse_test = {m: {h: e0_test[h] * (1.0 - (sk[j] + 0.03 * rng.standard_normal()) / 100.0) for j, h in enumerate(H)}
             for m, sk in SKILL.items()}
for h in H:
    rmse_test["E0"][h] = e0_test[h]
TEST_BLOCKS = ["02-01 00–06", "02-01 06–12", "02-01 12–18", "02-01 18–24", "02-02 00–06", "02-02 06–12", "02-02 12–18", "02-02 18–21"]
test_block_gain = {m: np.array([[sk[j] + 0.05 * rng.standard_normal() for j, _ in enumerate(H)] for _ in TEST_BLOCKS])
                   for m, sk in SKILL.items()}


def secondary(rmse: float, gain_vs_e0_pp: float) -> tuple[float, float, float]:
    mae = 0.72 * rmse * (1.0 + 0.03 * rng.standard_normal())
    r = float(np.sqrt(max(2.0 * gain_vs_e0_pp / 100.0, 0.0)) + 0.004 * rng.standard_normal())
    dacc = 0.5 + 0.4 * r + 0.003 * rng.standard_normal()
    return mae, r, dacc


# ----------------------------------------------------------------------------- §1.3 ε, lịch calibrate, số vòng
eps_rows = []
for m, e in [("LightGBM", 0.021), ("XGBoost", 0.024), ("CatBoost", 0.019), ("XGB-RF", 0.015),
             ("AutoTS-WR", 0.031), ("AutoTS-MR", 0.034), ("LSTM", 0.058)]:
    eps_rows.append({"model": m, "std_seed (pp)": f"{e:.3f}", "ε_m = max(0.005, std) (pp)": f"{max(0.005, e):.3f}"})
eps_df = pd.DataFrame(eps_rows)

rounds = pd.DataFrame(
    rng.integers(150, 450, size=(len(FOLD_DAYS), 3)),
    index=[f"fold {i + 1} (VAL {d})" for i, d in enumerate(FOLD_DAYS)],
    columns=[f"h={h}" for h in H],
).reset_index().rename(columns={"index": "fold"})
calib_df = pd.DataFrame([
    {"phase": "A. Lọc B0", "feature set": "B0-306", "model": "LightGBM", "run ES": "1 (seed 8586)", "kết quả": "15fixed_306 + ε_LGBM(B0-306)", "dùng cho": "4 run kiểm chứng R1–R4 → B0*"},
    {"phase": "B. Feature search", "feature set": "B0* (chung)", "model": "LightGBM", "run ES": "1", "kết quả": "15fixed_LGBM + ε_LGBM", "dùng cho": "39 candidate + prune PI của LightGBM"},
    {"phase": "B. Feature search", "feature set": "B0* (chung)", "model": "XGBoost", "run ES": "1", "kết quả": "15fixed_XGB + ε_XGB", "dùng cho": "39 candidate + prune PI của XGBoost"},
    {"phase": "B. Feature search", "feature set": "B0* (chung)", "model": "CatBoost", "run ES": "1", "kết quả": "15fixed_Cat + ε_Cat", "dùng cho": "39 candidate + prune PI của CatBoost"},
    {"phase": "B. Feature search", "feature set": "B0* (chung)", "model": "LSTM", "run ES": "1 (ES theo epoch)", "kết quả": "fixed_epoch_LSTM + ε_LSTM", "dùng cho": "39 candidate + prune PI của LSTM"},
    {"phase": "B. Feature search", "feature set": "B0* (chung)", "model": "XGB-RF / AutoTS / TimesFM", "run ES": "— (cơ chế riêng)", "kết quả": "chỉ ε_m (XGB-RF 1 vòng cố định; TimesFM zero-shot; AutoTS config cố định)", "dùng cho": "vòng lặp riêng của model đó"},
    {"phase": "C. Prune PI + win", "feature set": "F*_m và F*_m^prune", "model": "từng model", "run ES": "3 seed, ES bật", "kết quả": "bảng median 3 seed → win_m (+ số vòng/epoch cho Final)", "dùng cho": "so với champion (§3), figure §7.3"},
])

# ----------------------------------------------------------------------------- §1.4 b0_filter
b0_cols = [
    ("fine:t:return1", "return1", 0), ("fine:t-1m:return1", "return1", -1),
    ("fine:t:close_position", "close_position", 0), ("coarse:t:rv64", "rv64", 0),
    ("coarse:t-504m:time_of_day_sin", "time_of_day_sin", -504), ("fine:t-63m:minute_mod5_cos", "minute_mod5_cos", -63),
    ("coarse:t-256m:sign_flip_rate32", "sign_flip_rate32", -256), ("origin:rv60", "rv60", 0),
]
filt_rows = []
for name, base, lag in b0_cols:
    strong = base in ("return1", "close_position", "rv64", "rv60") and lag >= -1
    pi = [(0.9 if strong else 0.0) + 0.3 * rng.standard_normal() for _ in H]
    st = [(0.05 if strong else -0.01) + 0.02 * rng.standard_normal() for _ in H]
    mi = [(0.004 if strong else 0.0) + 0.001 * rng.standard_normal() for _ in H]
    # Cờ "+" của một tiêu chí = > 0 ở ÍT NHẤT 2 TRONG 3 horizon (luật user, rev 8)
    pi_p = sum(p > 0 for p in pi) >= 2
    st_p = sum(v > 0 for v in st) >= 2
    mi_p = sum(x > 0 for x in mi) >= 2
    keep = {"R1": pi_p or st_p or mi_p, "R2": pi_p or (st_p and mi_p), "R3": pi_p, "R4": st_p}
    filt_rows.append({
        "cột": name, "base": base, "lag": lag,
        "PI h1/h2/h3 (USD)": "/".join(f"{p:+.2f}" for p in pi),
        "SA Gain vs E0 h1/h2/h3 (pp)": "/".join(f"{s:+.3f}" for s in st),
        "SA Gain vs B0-306 (pp, median)": f"{-0.12 + 0.03 * rng.standard_normal():+.3f}",
        "MI − null h1/h2/h3": "/".join(f"{x:+.4f}" for x in mi),
        "PI+ / SA+ / MI+ (≥2/3 h)": f"{int(pi_p)} / {int(st_p)} / {int(mi_p)}",
        **{k: ("giữ" if v else "bỏ") for k, v in keep.items()},
    })
filt_df = pd.DataFrame(filt_rows)
rsets_df = pd.DataFrame([
    {"bộ": "B0-306", "luật giữ cột": "—", "số cột": 306, "MedianGain vs B0-306 (pp)": "0.000", "WinRate": "—", "quyết định": "reference"},
    {"bộ": "R1", "luật giữ cột": "PI+ hoặc SA+ hoặc MI+", "số cột": 245, "MedianGain vs B0-306 (pp)": "+0.020", "WinRate": "0.60", "quyết định": "không tệ hơn"},
    {"bộ": "R2", "luật giữ cột": "PI+ hoặc (SA+ và MI+)", "số cột": 197, "MedianGain vs B0-306 (pp)": "+0.031", "WinRate": "0.67", "quyết định": "**B0\\*** (không tệ hơn, cao nhất)"},
    {"bộ": "R3", "luật giữ cột": "PI+", "số cột": 143, "MedianGain vs B0-306 (pp)": "−0.044", "WinRate": "0.33", "quyết định": "tệ hơn −ε_LGBM (−0.021) → loại"},
    {"bộ": "R4", "luật giữ cột": "SA+", "số cột": 88, "MedianGain vs B0-306 (pp)": "−0.090", "WinRate": "0.20", "quyết định": "loại"},
])

# ----------------------------------------------------------------------------- §2.1 keepdrop_LightGBM + prune PI + win 3 seed
EPS_LGBM = 0.021
cands = [("vwap_amt_gap_1", 0.041), ("vwap_amt_gap_15", 0.012), ("vwap_amt_gap_60", -0.009), ("vwap_amt_gap_240", -0.033),
         ("ret_60", 0.002), ("ret_240", -0.018), ("ret_1440", -0.052), ("log_rv15_rv240", 0.024)]
kd_rows, size = [], 197
for i, (c, mg) in enumerate(cands, start=1):
    cell = mg + 0.05 * rng.standard_normal(15)
    s = summarize(cell)
    keep = mg >= -EPS_LGBM
    size += int(keep)
    kd_rows.append({
        "#": i, "cột": c,
        "MedianGain vs S_m (pp)": f"{mg:+.3f}", "WinRate": f"{s['WinRate']:.2f}",
        "P10Gain": f"{s['P10Gain']:+.3f}", "WorstGain": f"{s['WorstGain']:+.3f}",
        "Gain vs B0* (pp)": f"{mg + 0.03 * i / len(cands):+.3f}", "Gain vs E0 (pp)": f"{0.12 + mg + 0.03 * i / len(cands):+.3f}",
        "gain_standalone vs E0 (pp)": f"{max(mg, 0) * 0.6 + 0.01 * rng.standard_normal():+.3f}",
        "decision": "KEEP" if keep else "DROP", "size S_m sau": size, "exp_id": f"lgbm_c{i:03d}",
    })
kd_df = pd.DataFrame(kd_rows)

prune_df = pd.DataFrame([
    {"bộ": "F*_LGBM (sau vòng lặp)", "số cột ext": 14, "3 seed": "8586/8587/8588, ES bật", "MedianGain vs F*_LGBM (median từng ô)": "0.000", "WinRate": "—", "quyết định": "—"},
    {"bộ": "F*_LGBM^prune (bỏ 5 cột ext có PI ≤ 0)", "số cột ext": 9, "3 seed": "8586/8587/8588, ES bật", "MedianGain vs F*_LGBM (median từng ô)": "+0.006", "WinRate": "0.60", "quyết định": "≥ −ε_LGBM → **win_LGBM = F*_LGBM^prune**"},
])
# minh họa chồng 3 seed → median từng ô (Gain vs E0 của win, h=1)
win_seed_h1 = np.stack([gain_pp(rmse_seed[WIN][1][s], e0_val[1]) for s in range(3)])
stack_df = pd.DataFrame({
    "fold": [f"f{i + 1} {d}" for i, d in enumerate(FOLD_DAYS)],
    "seed 8586 (h=1)": [f"{v:+.3f}" for v in win_seed_h1[0]],
    "seed 8587 (h=1)": [f"{v:+.3f}" for v in win_seed_h1[1]],
    "seed 8588 (h=1)": [f"{v:+.3f}" for v in win_seed_h1[2]],
    "median từng ô (h=1)": [f"{v:+.3f}" for v in np.median(win_seed_h1, axis=0)],
})

# ----------------------------------------------------------------------------- §3 champion_log
champ_rows, champ = [], CHAMPION
first_tab = gain_table_median(CHAMPION, None)
champ_rows.append({
    "model (win_m)": CHAMPION, "F*_m (cột ext sau prune)": "9", "champion trước": "—",
    "MedianGain vs champion (pp, median 3 seed từng ô)": "—", "WinRate": "—", "P10Gain": "—", "WorstGain": "—", "ε_champion": f"{EPS_LGBM:.3f}",
    "decision": "champion ban đầu (§3)", "MedianGain vs E0": f"{np.median(first_tab):+.3f}",
    "latency p95 h1 (ms)": f"{LAT[CHAMPION][0][0]:.2f}", "champion sau": CHAMPION,
})
order = ["XGBoost(F*)", "CatBoost(F*)", "TFM-POINT", "XGB-RF(F*)", "AutoTS-WR(F*)", "AutoTS-MR(F*)", "LSTM(F*)", "Ensemble"]
for m in order:
    tab = gain_table_median(m, champ)
    s = summarize(tab.ravel())
    change = s["MedianGain"] > EPS_LGBM
    fstar = "—" if m == "TFM-POINT" else (f"{len(ENSEMBLE_MEMBERS)} thành viên, equal" if m == "Ensemble" else str(int(rng.integers(6, 20))))
    champ_rows.append({
        "model (win_m)": m, "F*_m (cột ext sau prune)": fstar, "champion trước": champ,
        "MedianGain vs champion (pp, median 3 seed từng ô)": f"{s['MedianGain']:+.3f}", "WinRate": f"{s['WinRate']:.2f}",
        "P10Gain": f"{s['P10Gain']:+.3f}", "WorstGain": f"{s['WorstGain']:+.3f}", "ε_champion": f"{EPS_LGBM:.3f}",
        "decision": "**đổi**" if change else "giữ", "MedianGain vs E0": f"{np.median(gain_table_median(m, None)):+.3f}",
        "latency p95 h1 (ms)": f"{lat_of(m)[0][0]:.2f}",
    })
    if change:
        champ = m
    champ_rows[-1]["champion sau"] = champ
champ_df = pd.DataFrame(champ_rows)

# ----------------------------------------------------------------------------- §7.2 all_models (VAL) + TEST
val_rows = []
for m in MODELS:
    row = {"model": m}
    g_b0 = gain_table_median(m, "B0-306").ravel()
    g_ch = gain_table_median(m, CHAMPION).ravel()
    for j, h in enumerate(H):
        r_mean = float(np.mean(rmse_med[m][h]))
        ge0 = float(np.median(gain_table_median(m, None)[:, j]))
        mae, r, dacc = secondary(r_mean, ge0)
        row[f"RMSE h{h} (USD)"] = f"{r_mean:.1f}"
        row[f"MAE h{h}"] = f"{mae:.1f}"
        row[f"r h{h}"] = f"{r:.3f}"
        row[f"dir-acc h{h}"] = f"{dacc:.3f}"
        row[f"Gain vs E0 h{h} (pp)"] = f"{ge0:+.3f}"
    s_b0, s_ch = summarize(g_b0), summarize(g_ch)
    row["MedianGain vs B0-306 (pp)"] = f"{s_b0['MedianGain']:+.3f}"
    row["WinRate vs B0-306"] = f"{s_b0['WinRate']:.2f}"
    row["MedianGain vs champion (pp)"] = f"{s_ch['MedianGain']:+.3f}"
    row["P10 vs champion"] = f"{s_ch['P10Gain']:+.3f}"
    row["Worst vs champion"] = f"{s_ch['WorstGain']:+.3f}"
    val_rows.append(row)
val_df = pd.DataFrame(val_rows)

test_rows = []
for m in MODELS:
    row = {"model": m}
    for j, h in enumerate(H):
        r_ = rmse_test[m][h]
        ge0 = 100.0 * (1.0 - r_ / e0_test[h])
        gb0 = 100.0 * (1.0 - r_ / rmse_test["B0-306"][h])
        mae, r, dacc = secondary(r_, ge0)
        row[f"RMSE h{h} (USD)"] = f"{r_:.1f}"
        row[f"MAE h{h}"] = f"{mae:.1f}"
        row[f"r h{h}"] = f"{r:.3f}"
        row[f"dir-acc h{h}"] = f"{dacc:.3f}"
        row[f"Gain vs B0-306 h{h} (pp)"] = f"{gb0:+.3f}"
        row[f"Gain vs E0 h{h} (pp)"] = f"{ge0:+.3f}"
    test_rows.append(row)
test_df = pd.DataFrame(test_rows)

# ----------------------------------------------------------------------------- §7.4 latency
lat_rows = []
for m in MODELS:
    if m == "E0":
        continue
    p, shared, dev = lat_of(m)
    for j, h in enumerate(H):
        k = 1.0 if shared else (1.0 + 0.08 * j)
        p95, p99, pmax = (x * k for x in p)
        lat_rows.append({"model": m, "h": h, "p95 (ms)": f"{p95:.2f}", "p99 (ms)": f"{p99:.2f}", "max (ms)": f"{pmax:.2f}",
                         "shared": str(shared).lower(), "train device": "GPU", "predict device": dev})
lat_df = pd.DataFrame(lat_rows)


# ----------------------------------------------------------------------------- figure helpers
def fake_window(n_origin: int = 60, hmax: int = 3, level: float = PRICE) -> np.ndarray:
    """Chuỗi giá thật giả: n_origin origin + hmax bar sau đó."""
    r = rng.normal(0.0, SIG1, n_origin + hmax)
    return level * np.exp(np.cumsum(r))


def fake_pred(price: np.ndarray, h: int, strength: float) -> np.ndarray:
    """P̂_{t+h} cho mọi origin t: ŷ = strength·r_thật + noise (số giả, strength cao hơn thực tế để nhìn rõ layout)."""
    n = len(price) - 3
    c_t = price[:n]
    r_true = np.log(price[h:h + n] / c_t)
    yhat = strength * r_true + rng.normal(0.0, 0.35 * SIG1 * np.sqrt(h), n)
    return c_t * np.exp(yhat)


def plot_h_windows(path: Path, h: int, windows: list[tuple[str, np.ndarray]], series: list[tuple[str, np.ndarray, str, str, str]],
                   title: str) -> None:
    """Một ảnh cho horizon h: 3 panel = 3 cửa sổ origin; chấm đen = giá thật C_{t+h}; marker màu = P̂_{t+h} theo origin.
    series: (label, preds cho từng cửa sổ nối dài? -> ta truyền list theo cửa sổ riêng), xử lý bởi caller."""
    fig, axes = plt.subplots(1, len(windows), figsize=(6.2 * len(windows), 4.4))
    for k, (label, price) in enumerate(windows):
        ax = axes[k]
        n = len(price) - 3
        x = np.arange(n)
        ax.plot(x, price[h:h + n], color=INK, lw=1.0, marker="o", ms=3.5, label=f"giá thật C_(t+{h})")
        ax.plot(x, price[:n], color=MUTED, ls="--", lw=0.9, label="E0 (P̂ = C_t)")
        for name, preds_by_window, color, marker, ls in series:
            ax.plot(x, preds_by_window[k], color=color, ls="none", marker=marker, ms=6, alpha=0.9, label=name)
        ax.set_title(f"{label} — h={h}", fontsize=9)
        ax.set_xlabel("origin t (phút trong cửa sổ)")
        if k == 0:
            ax.set_ylabel("USD")
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{title}  [{FAKE}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmap(ax, mat: np.ndarray, row_labels: list[str], title: str, vmax: float = 0.3) -> None:
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"h={h}" for h in H], fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


# ----------------------------------------------------------------------------- Fig H_h + HM: win vs champion (sau mỗi model)
VAL_WINDOWS = ["01-29 00:00–01:00", "01-29 08:00–09:00", "01-29 16:00–17:00"]  # ngày VAL có RMSE E0 ở giữa (giả định)
win_prices = [fake_window(level=PRICE * (1 + 0.004 * k)) for k in range(3)]
for h in H:
    win_preds = [fake_pred(p, h, 0.32) for p in win_prices]
    ch_preds = [fake_pred(p, h, 0.28) for p in win_prices]
    plot_h_windows(
        OUT / f"fig_H{h}_win_vs_champion.png", h, list(zip(VAL_WINDOWS, win_prices)),
        [(f"win = {WIN}", win_preds, WIN_STYLE[0], WIN_STYLE[1], "none"), (f"champion = {CHAMPION}", ch_preds, CHAMP_STYLE[0], CHAMP_STYLE[1], "none")],
        f"Fig H{h} — win vs champion, 3 cửa sổ origin, P̂_(t+{h}) theo origin t",
    )

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
row_labels = [f"f{i + 1} {d}" for i, d in enumerate(FOLD_DAYS)]
tab_w, tab_c = gain_table_median(WIN, None), gain_table_median(CHAMPION, None)
heatmap(axes[0], tab_w, row_labels, f"win = {WIN}: Gain vs E0 (pp)\n{fmt_sum(summarize(tab_w.ravel()))}")
im = heatmap(axes[1], tab_c, row_labels, f"champion = {CHAMPION}: Gain vs E0 (pp)\n{fmt_sum(summarize(tab_c.ravel()))}")
fig.subplots_adjust(left=0.09, right=0.86, top=0.80, bottom=0.16, wspace=0.45)
cax = fig.add_axes([0.89, 0.16, 0.02, 0.64])
fig.colorbar(im, cax=cax, label="Gain vs E0 (pp), median 3 seed từng ô")
s_wc = summarize(gain_table_median(WIN, CHAMPION).ravel())
fig.suptitle(f"Fig HM — 2 heatmap 15 ô (fold × horizon), cùng thang màu  [{FAKE}]", fontsize=9)
fig.text(0.5, 0.04, f"win vs champion (base = bảng RMSE median 3 seed của champion): {fmt_sum(s_wc)}  →  "
         f"{'ĐỔI champion' if s_wc['MedianGain'] > EPS_LGBM else 'GIỮ champion'} (ε = {EPS_LGBM:.3f})", ha="center", fontsize=8)
fig.savefig(OUT / "fig_HM_win_vs_champion.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- Final: heatmap mọi model + Fig H_h mọi model
hm_models = [m for m in MODELS if m != "E0"]
fig, axes = plt.subplots(3, 4, figsize=(17, 12))
for idx, (ax, m) in enumerate(zip(axes.ravel(), hm_models)):
    mat = test_block_gain[m]
    im = heatmap(ax, mat, TEST_BLOCKS if idx % 4 == 0 else [""] * len(TEST_BLOCKS), f"{m}\nGain vs E0 (pp), TEST", vmax=0.3)
for ax in axes.ravel()[len(hm_models):]:
    ax.axis("off")
fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.05, wspace=0.25, hspace=0.35)
cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
fig.colorbar(im, cax=cax, label="Gain vs E0 (pp)")
fig.suptitle(f"Final — heatmap TEST của mọi model: ô = khối 6 giờ × horizon (2 ngày ≈ 8 khối); cùng thang màu  [{FAKE}]", fontsize=10)
fig.savefig(OUT / "fig_final_heatmaps.png", dpi=120)
plt.close(fig)

TEST_WINDOWS = ["02-01 08:00–09:00", "02-01 16:00–17:00", "02-02 08:00–09:00"]
test_prices = [fake_window(level=PRICE * (0.98 + 0.004 * k)) for k in range(3)]
STRENGTH = {"B0-306": 0.20, "B0*": 0.22, "LightGBM(F*)": 0.28, "XGBoost(F*)": 0.27, "CatBoost(F*)": 0.27, "TFM-POINT": 0.05,
            "XGB-RF(F*)": 0.22, "AutoTS-WR(F*)": 0.20, "AutoTS-MR(F*)": 0.15, "LSTM(F*)": 0.18, "Ensemble": 0.32}
for h in H:
    preds = {m: [fake_pred(p, h, STRENGTH[m]) for p in test_prices] for m in STRENGTH}
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 8.6))
    for gi, (gname, group) in enumerate([("nhóm A: tree + ensemble", GROUP_A), ("nhóm B: TimesFM / AutoTS / LSTM + reference", GROUP_B)]):
        for k, (label, price) in enumerate(zip(TEST_WINDOWS, test_prices)):
            ax = axes[gi, k]
            n = len(price) - 3
            x = np.arange(n)
            ax.plot(x, price[h:h + n], color=INK, lw=1.0, marker="o", ms=3.5, label=f"giá thật C_(t+{h})")
            ax.plot(x, price[:n], color=MUTED, ls="--", lw=0.9, label="E0 (P̂ = C_t)")
            for m in group:
                col, mk, _ = STYLE[m]
                ax.plot(x, preds[m][k], color=col, ls="none", marker=mk, ms=5.5, alpha=0.9, label=m)
            ax.set_title(f"{gname} — {label} — h={h}", fontsize=8)
            ax.set_xlabel("origin t (phút trong cửa sổ)")
            if k == 0:
                ax.set_ylabel("USD")
                ax.legend(fontsize=6.5, loc="best")
    fig.suptitle(f"Final — Fig H{h} của mọi model trên TEST: P̂_(t+{h}) theo origin t; actual đen; ≤ 8 màu mỗi panel  [{FAKE}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_final_H{h}_all_models.png", dpi=120)
    plt.close(fig)

# ----------------------------------------------------------------------------- Fig D: latency (chỉ theo dõi)
fig, ax = plt.subplots(figsize=(12, 4.2))
lat_models = [m for m in MODELS if m not in ("E0",)]
xpos = np.arange(len(lat_models))
for k, (q, col) in enumerate(zip(["p95 (ms)", "p99 (ms)", "max (ms)"], H_RAMP)):
    vals = [float(lat_df[(lat_df.model == m) & (lat_df.h == 1)][q].iloc[0]) for m in lat_models]
    ax.bar(xpos + (k - 1) * 0.27, vals, width=0.27, color=col, label=q)
ax.set_yscale("log")
ax.set_xticks(xpos)
ax.set_xticklabels(lat_models, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("ms (log), predict 1 origin, h=1")
ax.set_title(f"Fig D — inference latency per model (chỉ theo dõi)  [{FAKE}]", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_D_latency.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- markdown report
md = []
A = md.append
A("# SMOKE VISUALIZE — layout bảng/figure theo plan (SỐ GIẢ)\n")
A(f"> **{FAKE}.** Sinh bởi `reports/smoke_visualize.py` (seed 8586), không đọc data thật, không train gì. "
  "Mục đích: thống nhất *hình dạng* output của từng bước trong `docs/RESEARCH_PLAN.md` trước khi code. "
  "Mọi con số dưới đây sẽ bị thay bằng kết quả thật khi chạy; không được trích dẫn như finding.\n")
A("Quy ước chung: prediction là log-return `ŷ_h`, metric tính trên **giá** `P̂ = C_t·exp(ŷ_h)` (USD); "
  "`Gain = 1 − RMSE_cand/RMSE_base` tính bằng **pp** (0.100 pp = RMSE thấp hơn base 0.1%); "
  "15 ô = 5 fold × 3 horizon; E0 = dự báo giá không đổi (`P̂ = C_t`). "
  "**Chỉ MedianGain (so với ε) là tiêu chí quyết định** ở mọi chỗ — KEEP/DROP, chọn B0\\*, win_m, đổi champion, thành viên ensemble; "
  "WinRate/P10Gain/WorstGain chỉ báo cáo. PI/MI/standalone chỉ dùng để lập các bộ R1–R4 khi lọc B0 và prune PI cuối vòng lặp. "
  "Bảng 3 seed: chồng 3 bảng Gain 15 ô lên nhau, **mỗi ô lấy median của 3 ô cùng vị trí**, rồi MedianGain = median 15 ô. "
  "Training chỉ trên GPU; cột device trong bảng latency là device của lời gọi predict.\n")

A("\n## 1. §1.3 — Nhiễu seed ε_m, lịch calibrate, số vòng cố định\n")
A(md_table(eps_df))
A("\n**Giải thích.** Mỗi model chạy 3 seed trên feature set của phase (xem lịch calibrate bên dưới); `std_seed` là độ lệch chuẩn của Gain giữa các seed "
  "trên 15 ô; `ε_m` là ngưỡng \"tệ hơn\" dùng cho KEEP/DROP, prune và champion của model đó. LSTM nhiễu seed lớn hơn tree, "
  "nên ngưỡng của nó rộng hơn — tự động, không chỉnh tay.\n")
A("\nLịch calibrate số vòng/epoch cố định (mỗi (phase, model) một run ES trên đúng feature set; không dùng chéo):\n")
A(md_table(calib_df))
A("\nVí dụ `15fixed_LGBM` (best_iteration mà ES dừng ở run calibrate của LightGBM trên B0*, per fold × horizon; dùng cho cả 39 candidate của LightGBM):\n")
A(md_table(rounds))
A("\n**Giải thích.** \"Số vòng cố định\" = chính best_iteration mà early stopping dừng ở run calibrate (không phải ước lượng thống kê). "
  "ES trên 1.377 dòng nhiễu, nên chỉ chạy ES một lần cho mỗi (phase, model) rồi cố định cho mọi run của phase đó ⇒ candidate và base cùng số vòng, "
  "chênh lệch Gain chỉ do feature. B0* là điểm xuất phát chung: mỗi model có early stopping (LightGBM, XGBoost, CatBoost theo số vòng; LSTM theo số epoch → fixed_epoch_LSTM) "
  "tự calibrate một run ES trên B0* → 15fixed_m riêng, rồi tự feature search bằng chính model đó → F*_LGBM, F*_XGB, F*_Cat có thể khác nhau; "
  "không model nào kế thừa F* của model khác. 15fixed_306 chỉ dùng cho lọc B0; không dùng số vòng của LightGBM cho model khác.\n")

A("\n## 2. §1.4 — Lọc 306 feature B0 → B0\\* (`experiments/b0_filter.csv`)\n")
A("Mẫu 8 dòng (thật sẽ có 306 dòng); mỗi cột có giữ/bỏ riêng cho từng bộ R1–R4:\n")
A(md_table(filt_df))
A("\nKiểm chứng 4 bộ so với B0-306 (mỗi bộ 1 run LightGBM gốc, số vòng cố định, seed 8586):\n")
A(md_table(rsets_df))
A("\n**Giải thích.** Ba điểm số per horizon (median 5 fold): PI = RMSE tăng thêm (USD) khi xáo cột đó trong VAL; "
  "SA = standalone, LightGBM chỉ trên một cột, Gain so với E0 và so với B0-306; MI − null = mutual information với z-target trên FIT trừ MI với target xáo trộn. "
  "Cờ **PI+ / SA+ / MI+** = điểm số > 0 ở **ít nhất 2 trong 3 horizon** "
  "(ví dụ PI > 0 ở h1, h2 nhưng < 0 ở h3 → PI+; chỉ h1 > 0 → không +). Không có tier: bốn bộ định nghĩa thẳng bằng cờ — "
  "R1 giữ nếu PI+ hoặc SA+ hoặc MI+ (bỏ cột âm cả ba); R2 giữ nếu PI+ hoặc (SA+ và MI+); R3 giữ nếu PI+; R4 giữ nếu SA+. "
  "Chọn B0\\* = trong các bộ có MedianGain ≥ −ε_LGBM so với B0-306, lấy bộ MedianGain cao nhất (chênh < ε → bộ nhỏ hơn); không bộ nào đạt → B0\\* = B0-306. "
  "Nếu một cột đơn lẻ thắng B0-306 (SA Gain vs B0-306 > +ε) thì đó là cờ đỏ B0 bị nhiễu chi phối — không cần luật riêng: R3/R4 sẽ tự thắng ở bước kiểm chứng. "
  "Bảng nhóm 38 base feature (gộp 8 lag) đi kèm để đọc, không dùng để quyết định.\n")

A("\n## 3. §2.1 — Vòng lặp feature của một model (`experiments/keepdrop_LightGBM.csv`)\n")
A("Mẫu 8 candidate đầu (thật: 39 dòng/model, mỗi model một file):\n")
A(md_table(kd_df))
A(f"\n**Giải thích.** Mỗi dòng = một candidate thêm vào bộ hiện tại `S_m` của model (xuất phát chung là B0*); base của Gain là chính model trên `S_m`; số vòng = 15fixed_m của model đó (calibrate trên B0*). "
  f"Luật: `MedianGain ≥ −ε_m` → KEEP (kể cả gần như không đổi), `< −ε_m` → DROP (ε_LGBM giả = {EPS_LGBM:.3f} pp). "
  "Mỗi model có file riêng (keepdrop_XGBoost.csv, keepdrop_CatBoost.csv, …) với cùng cấu trúc; các F*_m có thể khác nhau. "
  "`gain_standalone` là diagnostic (LightGBM chỉ trên cột đó vs E0): standalone > 0 nhưng vs S_m ≈ 0 ⇒ có tín hiệu nhưng trùng base. "
  "`size S_m sau` (số cột của S_m sau quyết định) cho thấy bộ feature lớn dần. Không còn safety-net; sau vòng lặp chỉ có prune PI (§3b).\n")

A("\n### 3b. §2.1 — Prune PI + confirmation 3 seed → win_m\n")
A(md_table(prune_df))
A("\nMinh họa \"chồng 3 seed lấy median từng ô\" (Gain vs E0 của win, chỉ h=1; thật sẽ là 5 fold × 3 horizon):\n")
A(md_table(stack_df))
A("\n**Giải thích.** Sau vòng lặp: (a) tính permutation importance trên VAL cho các cột ext của F*_m, bỏ đồng thời mọi cột có PI ≤ 0 → F*_m^prune; "
  "(b) chạy cả F*_m và F*_m^prune với 3 seed (ES bật); với mỗi bộ, 3 bảng Gain 15 ô (một bảng mỗi seed, cùng base) chồng lên nhau, "
  "**mỗi ô lấy median của 3 ô cùng vị trí** → một bảng 15 ô; MedianGain = median của 15 ô này (WinRate/P10/Worst tính trên cùng bảng, chỉ báo cáo). "
  "win_m = F*_m^prune nếu MedianGain(F*_m^prune vs F*_m) ≥ −ε_m, ngược lại = F*_m. Bảng median 3 seed của win_m là bảng dùng cho champion log và figure.\n")

A("\n## 4. §3 — Champion log (`experiments/champion_log.csv`)\n")
A(md_table(champ_df))
A("\n**Giải thích.** Champion ban đầu = LightGBM code gốc trên win_LGBM (dòng đầu, không so sánh). Sau khi mỗi model có win_m, "
  "tính bảng Gain 15 ô của win_m so với champion hiện tại (base = bảng RMSE median 3 seed của champion), chồng 3 seed lấy median từng ô → MedianGain; "
  "`MedianGain > +ε_champion` → đổi champion, ngược lại giữ — cả hai trường hợp đều ghi một dòng và vẽ figure §7.3 (win vs champion). "
  "Ensemble xét cuối cùng, cùng luật (ở mẫu này Ensemble thắng ⇒ champion cuối = Ensemble). "
  f"Thành viên ensemble theo luật §3 = champion + mọi model có MedianGain vs E0 > 0: {', '.join(ENSEMBLE_MEMBERS)} "
  "(TFM-POINT bị loại vì < 0; B0-306/B0\\* là reference). Trọng số: (a) đều, (b) 1/MSE_VAL per horizon — với chênh lệch RMSE ~0.1% thì (b) ≈ (a). "
  "Cột latency chỉ là thông tin (§7.4), không phải tiêu chí.\n")

A("\n## 5. §7.2 — Bảng tổng hợp mọi model (`experiments/summary/all_models.csv`)\n")
A("### 5.1 VAL (5 fold; RMSE/MAE = trung bình fold của bảng median 3 seed; Gain 15 ô median 3 seed)\n")
A(md_table(val_df))
A("\n### 5.2 TEST 2 ngày (§4, một block; refit trên FIT → 01-30, ES 01-31)\n")
A(md_table(test_df))
A("\n**Giải thích.** RMSE/MAE tính trên giá (USD) — với BTC ~80k và std return 1 phút ~0.077%, E0 ở h=1 cỡ 60 USD, h=3 cỡ 105 USD. "
  "Tín hiệu 1 phút rất nhỏ nên Gain thật chỉ cỡ 0.05–0.3 pp; Gain > ~1 pp là dấu hiệu leakage/bug. "
  "`r` và `dir-acc` tính trên thay đổi giá `P̂ − C_t` vs `C_{t+h} − C_t` (dir-acc bỏ bar giá không đổi). "
  "TFM-POINT zero-shot ở mẫu này thua E0 (Gain âm) ⇒ theo plan sẽ không chạy LoRA. TEST chỉ xem một lần, không sửa gì sau đó.\n")

A("\n## 6. §7.3 — Figure\n")
A("Màu: **actual luôn đen**; ảnh so sánh win vs champion dùng màu theo vai trò — win = blue `#2a78d6` ▲, champion = red `#e34948` ● (cặp xa nhau nhất), "
  "E0 = xám nét đứt. Ảnh nhiều model dùng màu/marker cố định cho từng model (palette categorical đã validate bằng validator của skill dataviz, "
  "thứ tự slot cố định, không xoay vòng; tối đa 8 màu mỗi panel — vượt thì tách nhóm): "
  + "; ".join(f"{m} = {STYLE[m][0]} `{STYLE[m][1]}`" for m in GROUP_A + GROUP_B)
  + ". Heatmap diverging xanh↔đỏ, cùng thang màu khi so sánh.\n")
A("### 6.1 Sau mỗi model — win_m vs champion hiện tại: mỗi horizon 3 ảnh (3 cửa sổ origin) + 2 heatmap\n")
for h in H:
    A(f"![Fig H{h}](smoke/fig_H{h}_win_vs_champion.png)\n")
A("![Fig HM](smoke/fig_HM_win_vs_champion.png)\n")
A("**Giải thích.** Fig H_h: mỗi ảnh một horizon; 3 panel = 3 cửa sổ 60 origin liên tiếp cố định (00:00–01:00, 08:00–09:00, 16:00–17:00 UTC của ngày VAL có RMSE E0 ở giữa 5 ngày). "
  "Trục x = origin t; chấm đen nối mảnh = giá thật `C_(t+h)` (chuỗi thật); marker màu = `P̂_(t+h)` của win và champion, **không nối** — mỗi prediction gắn với origin của nó, không vẽ chuỗi dự báo liên tục; "
  "đường xám đứt = `C_t` (E0). Fig HM: 2 heatmap 15 ô (fold × horizon) của win và champion, giá trị = Gain vs E0 median 3 seed từng ô, cùng thang màu; "
  "tiêu đề ghi MedianGain/WinRate/P10/Worst và kết quả win vs champion. Ở mẫu này prediction được vẽ với biên độ lớn hơn thực tế để nhìn rõ layout — "
  "với tín hiệu thật (~0.1–0.2 pp) các marker sẽ nằm rất sát `C_t`; đó là bình thường.\n")
A("### 6.2 Final (TEST) — heatmap của mọi model + Fig H_h của mọi model\n")
A("![Final heatmaps](smoke/fig_final_heatmaps.png)\n")
for h in H:
    A(f"![Final H{h}](smoke/fig_final_H{h}_all_models.png)\n")
A("**Giải thích.** Heatmap TEST: ô = khối 6 giờ × horizon (2 ngày ≈ 8 khối), giá trị Gain vs E0, một panel mỗi model (B0-306, B0*, mọi win_m, ensemble), cùng thang màu. "
  "Fig H_h Final: cùng định nghĩa Fig H_h nhưng vẽ prediction của tất cả model trên 3 cửa sổ TEST (02-01 08:00, 02-01 16:00, 02-02 08:00); "
  "tách 2 hàng (nhóm A tree + ensemble; nhóm B TimesFM/AutoTS/LSTM + reference) để mỗi panel ≤ 8 màu; actual đen ở mọi panel.\n")

A("\n## 7. §7.4 — Inference latency (chỉ theo dõi) (`experiments/summary/latency_summary.csv`)\n")
A(md_table(lat_df))
A("\n![Fig D](smoke/fig_D_latency.png)\n")
A("**Giải thích.** Thời gian gọi `predict` cho **một origin** (batch 1), đo ở pass riêng sau khi train (win và Final), "
  "bỏ 50 lần đầu warm-up, GPU có `cuda.synchronize`; báo cáo p95/p99/max (p50 không cần). Tree đo riêng từng h (3 model); `shared = true` nghĩa là một lần gọi ra cả 3 bước "
  "(LSTM/TimesFM/AutoTS) nên h=1,2,3 cùng giá trị. `train device` luôn GPU (cấm training CPU); `predict device` là device thực tế của lời gọi predict: "
  "LightGBM và CatBoost predict trên CPU là đặc tính thư viện (GPU chỉ dùng khi train), XGBoost/LSTM/TimesFM predict trên GPU, "
  "AutoTS chạy pipeline CPU quanh regression_model GPU. Chưa gồm thời gian tính feature. Không ảnh hưởng training/loss/quyết định.\n")

A("\n## 8. Cách sinh số giả (để không nhầm với kết quả)\n")
A("- RMSE E0 per (fold, h) = 80.000 × 0.000765 × √h × (1 ± 15% nhiễu); RMSE model per seed = E0 × (1 − skill/100) với skill giả gán sẵn "
  "(LightGBM 0.18/0.12/0.05 pp, TFM-POINT âm, Ensemble cao nhất) + nhiễu ô 0.02 pp + nhiễu seed 0.015 pp; 3 seed → median từng ô.\n"
  "- MAE = 0.72·RMSE; r ≈ √(2·Gain_vs_E0); dir-acc ≈ 0.5 + 0.4·r; latency, PI, MI, standalone, prune đều là hằng số + nhiễu.\n"
  "- Giá trong Fig H: random walk 63 phút; prediction = C_t·exp(strength·r_thật + noise) với strength 0.05–0.32 (cao hơn thực tế nhiều lần, chỉ để nhìn layout).\n"
  "- Seed 8586; chạy lại cho cùng số. Khi có pipeline thật, script này bị thay bằng `src/plots.py` + log thật.\n")

MD_PATH.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {MD_PATH} and {len(list(OUT.glob('*.png')))} figures in {OUT}")
