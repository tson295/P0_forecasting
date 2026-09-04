"""Màu/marker cố định (§7.3): palette categorical đã validate (dataviz reference), actual luôn đen."""
from __future__ import annotations

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED = "#0b0b0b", "#898781"
WIN_STYLE = ("#2a78d6", "^")  # win = blue ▲
CHAMP_STYLE = ("#e34948", "o")  # champion = red ●
H_RAMP = ["#86b6ef", "#2a78d6", "#104281"]

# model key -> (color, marker, linestyle); nhóm A/B để mỗi panel ≤ 8 màu; AutoTS-MR dùng lại orange (khác nhóm với XGBoost)
STYLE = {
    "b0_306": ("#898781", "s", "--"),
    "b0_star": ("#52514e", "D", "--"),
    "lgbm": (PALETTE[0], "o", "-"),
    "xgb": (PALETTE[1], "^", "-"),
    "cat": (PALETTE[2], "v", "-"),
    "xgbrf": (PALETTE[3], "X", "-"),
    "autots": (PALETTE[4], "<", "-"),          # AutoTS-final (sau framework search)
    "autots_wr": (PALETTE[4], "<", ":"),       # probe (diagnostic, không vào champion/ensemble/Final)
    "lstm": (PALETTE[5], "*", "-"),
    "tfm": (PALETTE[6], "P", "-"),
    "ensemble": (PALETTE[7], "h", "-"),
    "autots_mr": (PALETTE[1], ">", ":"),
}
GROUP_A = ["lgbm", "xgb", "cat", "xgbrf", "ensemble"]
GROUP_B = ["tfm", "autots", "lstm", "b0_306", "b0_star"]  # probe autots_wr/autots_mr không vẽ ở Final
LABEL = {
    "b0_306": "B0-306", "b0_star": "B0*", "lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "xgbrf": "XGB-RF",
    "autots": "AutoTS", "autots_wr": "AutoTS-WR(probe)", "autots_mr": "AutoTS-MR(probe)", "lstm": "LSTM", "tfm": "TimesFM-LoRA", "tfm_lora_native": "TimesFM-LoRA native", "tfm_lora_xreg": "TimesFM-LoRA+XReg",
    "ensemble": "Ensemble", "e0": "E0",
}


def style(key: str) -> tuple[str, str, str]:
    return STYLE.get(key, (INK, "o", "-"))
