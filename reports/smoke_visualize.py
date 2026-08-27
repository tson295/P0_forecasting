"""
smoke_visualize.py — SINH SỐ GIẢ để xem trước layout các bảng/figure mà plan sẽ tạo.

FAKE / SMOKE: mọi con số trong output đều là số giả sinh bằng RNG cố định (seed 8586),
KHÔNG đọc data thật, KHÔNG train gì, KHÔNG phải kết quả experiment. Chỉ để thống nhất
layout của: bảng ε/số vòng (§1.3), b0_filter (§1.4), keepdrop (§2.1), champion_log (§3),
all_models VAL/TEST (§4, §7.2), Fig A/B/C (§7.3) và latency (§7.4) trong docs/RESEARCH_PLAN.md.

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
    "ExtraTrees(F*)": (0.12, 0.09, 0.04),
    "AutoTS-WR(F*)": (0.10, 0.07, 0.03),
    "AutoTS-MR(F*)": (0.06, 0.05, 0.02),
    "LSTM(F*)": (0.09, 0.06, 0.03),
    "Ensemble": (0.21, 0.14, 0.06),
}
MODELS = list(SKILL)
CHAMPION = "LightGBM(F*)"  # champion trước khi xét ensemble (§3)

# Latency giả (ms) p50/p95/p99 tại h=1; shared = một lần gọi ra cả 3 bước (§7.4)
LAT = {
    "B0-306": ((0.30, 0.60, 1.20), False, "CPU"),
    "B0*": ((0.24, 0.50, 1.00), False, "CPU"),
    "LightGBM(F*)": ((0.35, 0.70, 1.40), False, "CPU"),
    "XGBoost(F*)": ((0.60, 1.10, 2.20), False, "GPU"),
    "CatBoost(F*)": ((0.25, 0.50, 1.00), False, "CPU"),
    "TFM-POINT": ((28.0, 45.0, 90.0), True, "GPU"),
    "ExtraTrees(F*)": ((9.0, 14.0, 24.0), False, "CPU"),
    "AutoTS-WR(F*)": ((180.0, 320.0, 650.0), True, "CPU"),
    "AutoTS-MR(F*)": ((260.0, 420.0, 800.0), True, "CPU"),
    "LSTM(F*)": ((2.4, 4.1, 8.5), True, "GPU"),
}
ENSEMBLE_MEMBERS = ["LightGBM(F*)", "XGBoost(F*)", "CatBoost(F*)", "LSTM(F*)"]


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def f3(x: float) -> str:
    return f"{x:+.3f}" if x < 0 or x > 0 else "0.000"


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


# ----------------------------------------------------------------------------- fake VAL cells
# RMSE giá (USD) per model × fold × h. E0 chung cho mọi model trong cùng ô.
e0_val = {h: rmse_e0(h, len(FOLD_DAYS)) for h in H}
rmse_val: dict[str, dict[int, np.ndarray]] = {}
for m, sk in SKILL.items():
    rmse_val[m] = {}
    for j, h in enumerate(H):
        g = sk[j] + (0.0 if m == "E0" else 0.02 * rng.standard_normal(len(FOLD_DAYS)))
        rmse_val[m][h] = e0_val[h] * (1.0 - g / 100.0)

# TEST (2 ngày, một block) per h
e0_test = {h: float(rmse_e0(h, 1, scale=1.08)[0]) for h in H}
rmse_test = {m: {h: e0_test[h] * (1.0 - (sk[j] + 0.05 * rng.standard_normal()) / 100.0) for j, h in enumerate(H)}
             for m, sk in SKILL.items()}
for h in H:
    rmse_test["E0"][h] = e0_test[h]


def secondary(rmse: float, gain_vs_e0_pp: float) -> tuple[float, float, float]:
    mae = 0.72 * rmse * (1.0 + 0.03 * rng.standard_normal())
    r = float(np.sqrt(max(2.0 * gain_vs_e0_pp / 100.0, 0.0)) + 0.004 * rng.standard_normal())
    dacc = 0.5 + 0.4 * r + 0.003 * rng.standard_normal()
    return mae, r, dacc


# ----------------------------------------------------------------------------- §1.3 ε và số vòng
eps_rows = []
for m, e in [("LightGBM", 0.021), ("XGBoost", 0.024), ("CatBoost", 0.019), ("ExtraTrees", 0.015),
             ("AutoTS-WR", 0.031), ("AutoTS-MR", 0.034), ("LSTM", 0.058)]:
    eps_rows.append({"model": m, "std_seed (pp)": f"{e:.3f}", "ε_m = max(0.005, std) (pp)": f"{max(0.005, e):.3f}"})
eps_df = pd.DataFrame(eps_rows)

rounds = pd.DataFrame(
    rng.integers(150, 450, size=(len(FOLD_DAYS), 3)),
    index=[f"fold {i + 1} (VAL {d})" for i, d in enumerate(FOLD_DAYS)],
    columns=[f"h={h}" for h in H],
).reset_index().rename(columns={"index": "fold"})

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
    fail_pi = all(p <= 0 for p in pi)
    fail_st = all(s <= 0 for s in st)
    fail_mi = all(x <= 0 for x in mi)
    tier = "Tier 1" if (fail_pi and fail_st and fail_mi) else ("Tier 2" if (fail_pi and (fail_st or fail_mi)) else "—")
    filt_rows.append({
        "cột": name, "base": base, "lag": lag,
        "PI h1/h2/h3 (USD)": "/".join(f"{p:+.2f}" for p in pi),
        "standalone Gain vs E0 h1/h2/h3 (pp)": "/".join(f"{s:+.3f}" for s in st),
        "standalone Gain vs B0-306 (pp, median)": f"{-0.12 + 0.03 * rng.standard_normal():+.3f}",
        "MI − null h1/h2/h3 (nat)": "/".join(f"{x:+.4f}" for x in mi),
        "tier": tier, "B0* (R2)": "bỏ" if tier != "—" else "giữ",
    })
filt_df = pd.DataFrame(filt_rows)
tier_df = pd.DataFrame([
    {"bộ": "B0-306", "số cột": 306, "MedianGain vs B0-306 (pp)": "0.000", "WinRate": "—", "quyết định": "reference"},
    {"bộ": "R1 = B0 − Tier1", "số cột": 245, "MedianGain vs B0-306 (pp)": "+0.020", "WinRate": "0.60", "quyết định": "không tệ hơn"},
    {"bộ": "R2 = B0 − Tier1 − Tier2", "số cột": 197, "MedianGain vs B0-306 (pp)": "+0.031", "WinRate": "0.67", "quyết định": "**B0\\*** (cao nhất, ≥ −ε)"},
    {"bộ": "R3 = chỉ PI > 0", "số cột": 143, "MedianGain vs B0-306 (pp)": "−0.044", "WinRate": "0.33", "quyết định": "tệ hơn ε_LGBM = 0.021 → loại"},
])

# ----------------------------------------------------------------------------- §2.1 keepdrop_LightGBM
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
        "decision": "KEEP" if keep else "DROP", "|S_m| sau": size, "exp_id": f"lgbm_c{i:03d}",
    })
kd_df = pd.DataFrame(kd_rows)

# ----------------------------------------------------------------------------- §3 champion_log
champ_rows, champ = [], "LightGBM(F*)"
champ_rows.append({
    "model": "LightGBM(F*)", "F*_m (số cột ext KEEP)": "14", "champion trước": "—",
    "MedianGain vs champion (pp)": "—", "WinRate": "—", "P10Gain": "—", "WorstGain": "—", "ε_champion": f"{EPS_LGBM:.3f}",
    "decision": "champion ban đầu (§3)", "latency p50 h1 (ms)": f"{LAT['LightGBM(F*)'][0][0]:.2f}", "champion sau": "LightGBM(F*)",
})
order = ["XGBoost(F*)", "CatBoost(F*)", "TFM-POINT", "ExtraTrees(F*)", "AutoTS-WR(F*)", "AutoTS-MR(F*)", "LSTM(F*)", "Ensemble"]
for m in order:
    g = np.concatenate([gain_pp(rmse_val[m][h], rmse_val[champ][h]) for h in H])
    s = summarize(g)
    change = s["MedianGain"] > EPS_LGBM
    champ_rows.append({
        "model": m, "F*_m (số cột ext KEEP)": "—" if m in ("TFM-POINT",) else str(int(rng.integers(8, 26))),
        "champion trước": champ, "MedianGain vs champion (pp)": f"{s['MedianGain']:+.3f}", "WinRate": f"{s['WinRate']:.2f}",
        "P10Gain": f"{s['P10Gain']:+.3f}", "WorstGain": f"{s['WorstGain']:+.3f}", "ε_champion": f"{EPS_LGBM:.3f}",
        "decision": "**đổi**" if change else "giữ",
        "latency p50 h1 (ms)": "—" if m == "Ensemble" else f"{LAT[m][0][0]:.2f}",
    })
    if change:
        champ = m
    champ_rows[-1]["champion sau"] = champ
champ_df = pd.DataFrame(champ_rows)

# ----------------------------------------------------------------------------- §7.2 all_models (VAL) + TEST
val_rows = []
for m in MODELS:
    row = {"model": m}
    g_b0 = np.concatenate([gain_pp(rmse_val[m][h], rmse_val["B0-306"][h]) for h in H])
    g_ch = np.concatenate([gain_pp(rmse_val[m][h], rmse_val[CHAMPION][h]) for h in H])
    for j, h in enumerate(H):
        r_mean = float(np.mean(rmse_val[m][h]))
        ge0 = float(np.mean(gain_pp(rmse_val[m][h], e0_val[h])))
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
    if m == "Ensemble":
        p = np.sum([LAT[k][0] for k in ENSEMBLE_MEMBERS], axis=0)
        shared, dev = False, "CPU+GPU (tổng các member)"
    else:
        p, shared, dev = LAT[m]
    for j, h in enumerate(H):
        k = 1.0 if shared else (1.0 + 0.08 * j)
        p50, p95, p99 = (x * k for x in p)
        lat_rows.append({"model": m, "h": h, "p50 (ms)": f"{p50:.2f}", "p95 (ms)": f"{p95:.2f}", "p99 (ms)": f"{p99:.2f}",
                         "mean (ms)": f"{p50 * 1.12:.2f}", "max (ms)": f"{p99 * 2.1:.2f}", "shared": str(shared).lower(), "device": dev})
lat_df = pd.DataFrame(lat_rows)

# ----------------------------------------------------------------------------- Fig A: origin plot
FIG_A_MODELS = ["LightGBM(F*)", "XGBoost(F*)", "TFM-POINT", "LSTM(F*)", "Ensemble"]
COLORS = dict(zip(FIG_A_MODELS, ["tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:red"]))
origins = ["2026-01-28 08:00 UTC (VAL fold 2)", "2026-01-30 16:00 UTC (VAL fold 4)"]
fig, axes = plt.subplots(len(origins), 2, figsize=(13, 4.2 * len(origins)), gridspec_kw={"width_ratios": [2.2, 1]})
for i, label in enumerate(origins):
    r1 = rng.normal(0, SIG1, 63)
    price = PRICE * (1 + 0.002 * i) * np.exp(np.cumsum(r1))
    x = np.arange(-60, 3)  # -60..2  → index 60 = t
    price_ctx, c_t = price[:61], price[60]
    actual = price[61:64] if len(price) >= 64 else np.append(price[61:], price[-1])
    actual = price[60] * np.exp(np.cumsum(rng.normal(0, SIG1, 3)))
    preds = {m: c_t * np.exp(np.cumsum(rng.normal(0, 0.9 * SIG1 * 0.15, 3)) + 0.3 * (actual / c_t - 1) * (0.5 + rng.random())) for m in FIG_A_MODELS}
    ax = axes[i, 0]
    ax.plot(x[:61], price_ctx, color="black", lw=1.2, label="giá thật (60' trước t)")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.scatter([1, 2, 3], actual, color="black", zorder=5, s=36, label="giá thật t+1..t+3")
    ax.set_title(f"Fig A — origin t = {label}  [{FAKE}]", fontsize=9)
    ax.set_xlabel("phút so với origin t")
    ax.set_ylabel("USD")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = axes[i, 1]
    ax2.axhline(c_t, color="gray", ls="--", lw=1, label="E0 (P̂ = C_t)")
    ax2.scatter([0], [c_t], color="black", s=40, zorder=5)
    ax2.plot([0, 1, 2, 3], np.r_[c_t, actual], color="black", lw=1, marker="o", label="giá thật")
    for k, (m, p) in enumerate(preds.items()):
        off = (k - 2) * 0.06
        ax2.plot(np.array([0, 1, 2, 3]) + off, np.r_[c_t, p], color=COLORS[m], lw=0.8, marker="^", ms=6, label=m)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["t", "t+1", "t+2", "t+3"])
    ax2.set_title("zoom: 3 điểm dự báo từ origin t", fontsize=9)
    ax2.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(OUT / "fig_A_origin.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- Fig B1: RMSE bar; B2: heatmap Gain vs champion
# Chênh lệch RMSE giữa các model chỉ ~0.1% nên bar RMSE tuyệt đối không nhìn thấy gì → vẽ Gain vs E0 (pp).
fig, ax = plt.subplots(figsize=(13, 4.5))
xpos = np.arange(len(MODELS))
w = 0.27
for j, h in enumerate(H):
    vals = [float(np.median(gain_pp(rmse_val[m][h], e0_val[h]))) for m in MODELS]
    ax.bar(xpos + (j - 1) * w, vals, width=w, label=f"h={h}")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Gain vs E0 (pp), median 5 fold")
ax.set_title(f"Fig B1 — Gain vs E0 per horizon per model (RMSE tuyệt đối xem bảng)  [{FAKE}]", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_B1_gain_bar.png", dpi=130)
plt.close(fig)

hm_models = [m for m in MODELS if m not in ("E0", CHAMPION)]
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, m in zip(axes.ravel(), hm_models):
    mat = np.column_stack([gain_pp(rmse_val[m][h], rmse_val[CHAMPION][h]) for h in H])
    im = ax.imshow(mat, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"h={h}" for h in H], fontsize=8)
    ax.set_yticks(range(len(FOLD_DAYS)))
    ax.set_yticklabels([f"f{i + 1} {d}" for i, d in enumerate(FOLD_DAYS)], fontsize=7)
    ax.set_title(m, fontsize=9)
for ax in axes.ravel()[len(hm_models):]:
    ax.axis("off")
fig.suptitle(f"Fig B2 — Gain 15 ô (fold × horizon, pp) so với champion {CHAMPION}  [{FAKE}]", fontsize=9)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Gain (pp)")
fig.savefig(OUT / "fig_B2_gain_heatmap.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- Fig C: theo ngày (E0 RMSE + Gain vs E0)
days = FOLD_DAYS + TEST_DAYS
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
FIG_C_MODELS = ["B0-306", "LightGBM(F*)", "XGBoost(F*)", "TFM-POINT", "LSTM(F*)", "Ensemble"]
for j, h in enumerate(H):
    e0_days = np.r_[e0_val[h], e0_test[h] * (1 + 0.05 * rng.standard_normal(2))]
    ax = axes[0, j]
    ax.plot(days, e0_days, marker="o", color="black", lw=1.2)
    ax.axvline(4.5, color="gray", ls=":")
    ax.set_title(f"RMSE của E0 theo ngày, h={h} (USD) — mức biến động", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax = axes[1, j]
    for m in FIG_C_MODELS:
        y_val = gain_pp(rmse_val[m][h], e0_val[h])
        y_test = 100.0 * (1.0 - rmse_test[m][h] / e0_test[h]) + 0.03 * rng.standard_normal(2)
        ax.plot(days, np.r_[y_val, y_test], marker="o", ms=4, lw=1, label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(4.5, color="gray", ls=":")
    ax.set_title(f"Gain vs E0 theo ngày, h={h} (pp)", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
axes[1, 0].legend(fontsize=7)
fig.suptitle(f"Fig C — theo thời gian: VAL 5 ngày | TEST 2 ngày (bên phải đường chấm)  [{FAKE}]", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_C_by_day.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- Fig D: latency
fig, ax = plt.subplots(figsize=(12, 4.2))
lat_models = [m for m in MODELS if m not in ("E0",)]
xpos = np.arange(len(lat_models))
for k, (q, col) in enumerate(zip(["p50 (ms)", "p95 (ms)", "p99 (ms)"], ["tab:blue", "tab:orange", "tab:red"])):
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
  "15 ô = 5 fold × 3 horizon; E0 = dự báo giá không đổi (`P̂ = C_t`).\n")

A("\n## 1. §1.3 — Nhiễu seed ε_m và số vòng cố định\n")
A(md_table(eps_df))
A("\n**Giải thích.** Mỗi model chạy 3 seed trên feature baseline; `std_seed` là độ lệch chuẩn của Gain giữa các seed "
  "trên 15 ô; `ε_m` là ngưỡng \"tệ hơn\" dùng cho KEEP/DROP và champion của model đó. LSTM nhiễu seed lớn hơn tree, "
  "nên ngưỡng của nó rộng hơn — tự động, không chỉnh tay.\n")
A("\nSố vòng cố định của LightGBM (best_iteration lấy từ run baseline có ES, dùng cho mọi candidate):\n")
A(md_table(rounds))
A("\n**Giải thích.** ES trên 1.377 dòng làm best_iteration nhiễu, nên chỉ lấy một lần ở baseline rồi cố định; "
  "candidate và baseline cùng số vòng ⇒ chênh lệch Gain chỉ do feature. Run confirmation cuối bật lại ES.\n")

A("\n## 2. §1.4 — Lọc 306 feature B0 → B0\\* (`experiments/b0_filter.csv`)\n")
A("Mẫu 8 dòng (thật sẽ có 306 dòng):\n")
A(md_table(filt_df))
A("\nKiểm chứng 3 bộ lọc so với B0-306 (mỗi bộ 1 run, số vòng cố định, seed 8586):\n")
A(md_table(tier_df))
A("\n**Giải thích.** PI = RMSE tăng thêm (USD) khi xáo cột đó trong VAL (≤ 0 → model không dùng cột hữu ích); "
  "standalone = LightGBM chỉ trên một cột, Gain so với E0 và so với B0-306 (nếu một cột thắng B0-306 → cờ đỏ B0 bị nhiễu); "
  "MI − null = mutual information với z-target trên FIT trừ MI với target xáo trộn. Một cột *fail* một tiêu chí khi fail ở cả 3 horizon. "
  "Tier 1 = fail cả ba; Tier 2 = PI ≤ 0 + một tiêu chí nữa; R3 = chỉ giữ PI > 0. Chọn B0\\* = bộ không tệ hơn B0-306 có MedianGain cao nhất "
  "(ở mẫu này là R2, 197 cột). Bảng nhóm 38 base feature (gộp 8 lag) đi kèm để đọc, không dùng để quyết định.\n")

A("\n## 3. §2.1 — Vòng lặp feature của một model (`experiments/keepdrop_LightGBM.csv`)\n")
A("Mẫu 8 candidate đầu (thật: 39 dòng/model, mỗi model một file):\n")
A(md_table(kd_df))
A(f"\n**Giải thích.** Mỗi dòng = một candidate thêm vào bộ hiện tại `S_m` của model; base của Gain là chính model trên `S_m`. "
  f"Luật: `MedianGain ≥ −ε_m` → KEEP (kể cả gần như không đổi), `< −ε_m` → DROP (ε_LGBM giả = {EPS_LGBM:.3f} pp). "
  "`gain_standalone` là diagnostic (LightGBM chỉ trên cột đó vs E0): standalone > 0 nhưng vs S_m ≈ 0 ⇒ có tín hiệu nhưng trùng base. "
  "`|S_m| sau` cho thấy bộ feature lớn dần; cuối vòng lặp có safety-net (thử lại block các cột DROP) và prune permutation ≤ 0.\n")

A("\n## 4. §3 — Champion log (`experiments/champion_log.csv`)\n")
A(md_table(champ_df))
A("\n**Giải thích.** Champion ban đầu = LightGBM code gốc trên F\\*_LGBM (dòng đầu, không so sánh). Sau khi mỗi model xong vòng lặp + "
  "confirmation 3 seed, so với champion hiện tại bằng Gain trên giá 15 ô; `MedianGain > +ε_champion` → đổi champion, ngược lại giữ — "
  "cả hai trường hợp đều ghi một dòng. Ensemble xét cuối cùng, cùng luật (ở mẫu này Ensemble thắng ⇒ champion cuối = Ensemble). "
  "Cột latency chỉ là thông tin (§7.4), không phải tiêu chí.\n")

A("\n## 5. §7.2 — Bảng tổng hợp mọi model (`experiments/summary/all_models.csv`)\n")
A("### 5.1 VAL (5 fold gộp; RMSE/MAE = trung bình fold; Gain 15 ô)\n")
A(md_table(val_df))
A("\n### 5.2 TEST 2 ngày (§4, một block; refit trên FIT → 01-30, ES 01-31)\n")
A(md_table(test_df))
A("\n**Giải thích.** RMSE/MAE tính trên giá (USD) — với BTC ~80k và std return 1 phút ~0.077%, E0 ở h=1 cỡ 60 USD, h=3 cỡ 105 USD. "
  "Tín hiệu 1 phút rất nhỏ nên Gain thật chỉ cỡ 0.05–0.3 pp; Gain > ~1 pp là dấu hiệu leakage/bug. "
  "`r` và `dir-acc` tính trên thay đổi giá `P̂ − C_t` vs `C_{t+h} − C_t` (dir-acc bỏ bar giá không đổi). "
  "TFM-POINT zero-shot ở mẫu này thua E0 (Gain âm) ⇒ theo plan sẽ không chạy LoRA. TEST chỉ xem một lần, không sửa gì sau đó.\n")

A("\n## 6. §7.3 — Figure\n")
A("### Fig A — origin plot: một điểm t làm gốc, 3 điểm dự báo t+1, t+2, t+3\n")
A("![Fig A](smoke/fig_A_origin.png)\n")
A("**Giải thích.** Trái: 60 phút giá thật trước origin t và 3 điểm thật sau t. Phải: zoom quanh t — điểm đen là giá thật, "
  "tam giác màu là `P̂_{t+1..t+3}` của từng model nối từ `C_t`, đường đứt là E0. Không vẽ chuỗi dự báo liên tục; mỗi panel một origin. "
  "Origin mặc định: mỗi ngày VAL/TEST 00:00, 08:00, 16:00 UTC + 2 origin biến động lớn nhất trong ngày. "
  "Với tín hiệu thật, các tam giác sẽ nằm rất sát `C_t` (std(ŷ) ≪ std(y)) — đó là bình thường, không phải lỗi.\n")
A("### Fig B1 — Gain vs E0 per horizon per model (VAL)\n")
A("![Fig B1](smoke/fig_B1_gain_bar.png)\n")
A("**Giải thích.** Vẽ Gain (pp) thay vì RMSE tuyệt đối: chênh lệch RMSE giữa các model chỉ cỡ 0.1% nên bar RMSE trông giống hệt nhau "
  "(đã thử ở bản smoke đầu). RMSE/MAE tuyệt đối để trong bảng §5.\n")
A("### Fig B2 — heatmap Gain 15 ô so với champion\n")
A("![Fig B2](smoke/fig_B2_gain_heatmap.png)\n")
A("**Giải thích.** Mỗi ô = một (fold, horizon); xanh = tốt hơn champion, đỏ = tệ hơn. MedianGain/WinRate/P10/Worst trong các bảng trên "
  "là tóm tắt của đúng 15 ô này. Một model chỉ xanh ở 1–2 fold là dấu hiệu không ổn định.\n")
A("### Fig C — theo thời gian (VAL 5 ngày + TEST 2 ngày)\n")
A("![Fig C](smoke/fig_C_by_day.png)\n")
A("**Giải thích.** Hàng trên: RMSE của E0 theo ngày = mức biến động (std thay đổi giá) của ngày đó. Hàng dưới: Gain vs E0 theo ngày "
  "của từng model — model tốt phải nằm trên 0 ở hầu hết các ngày; một model chỉ tốt ở ngày biến động mạnh là red flag.\n")

A("\n## 7. §7.4 — Inference latency (chỉ theo dõi) (`experiments/summary/latency_summary.csv`)\n")
A(md_table(lat_df))
A("\n![Fig D](smoke/fig_D_latency.png)\n")
A("**Giải thích.** Thời gian gọi `predict` cho **một origin** (batch 1), đo ở pass riêng sau khi train (confirmation và Final), "
  "bỏ 50 lần đầu warm-up, GPU có `cuda.synchronize`. Tree đo riêng từng h (3 model); `shared = true` nghĩa là một lần gọi ra cả 3 bước "
  "(LSTM/TimesFM/AutoTS) nên h=1,2,3 cùng giá trị. Chưa gồm thời gian tính feature. Không ảnh hưởng training/loss/quyết định.\n")

A("\n## 8. Cách sinh số giả (để không nhầm với kết quả)\n")
A("- RMSE E0 per (fold, h) = 80.000 × 0.000765 × √h × (1 ± 15% nhiễu); RMSE model = E0 × (1 − skill/100) với skill giả gán sẵn "
  "(LightGBM 0.18/0.12/0.05 pp, TFM-POINT âm, Ensemble cao nhất) + nhiễu ô 0.04 pp.\n"
  "- MAE = 0.72·RMSE; r ≈ √(2·Gain_vs_E0); dir-acc ≈ 0.5 + 0.4·r; latency, PI, MI, standalone đều là hằng số + nhiễu.\n"
  "- Seed 8586; chạy lại cho cùng số. Khi có pipeline thật, script này bị thay bằng `src/plots.py` + log thật.\n")

MD_PATH.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {MD_PATH} and {len(list(OUT.glob('*.png')))} figures in {OUT}")
