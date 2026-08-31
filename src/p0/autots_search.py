"""Giai đoạn (iii) của §2.2 #6 — **bake-off template GPU** cho AutoTS (audit §12.4d phương án A, §12.10).

Tách khỏi `models_autots.py` để khẳng định bằng cấu trúc: **probe WR/MR không bao giờ chạm framework `AutoTS`**.
Ở đây `AutoTS(...)` chỉ được dùng với `initial_template` do TA khai báo (mọi dòng ép GPU) và `max_generations=0`
→ không genetic search (search thật sẽ sinh regressor sklearn CPU, vi phạm invariant §0), nhưng vẫn dùng
validation nội bộ + luật chọn best của AutoTS trên **training-side của fold**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS
from .models_autots import MR_PARAMS, WR_PARAMS

NO_TRANS = {"fillna": None, "transformations": {}, "transformation_params": {}}  # 3 khoá bắt buộc = biến đổi đồng nhất
MODEL_OF = {"wr": "WindowRegression", "mr": "MultivariateRegression"}
REGRESSOR_GPU = {"LightGBM": dict(WR_PARAMS["model_params"]), "xgboost": dict(MR_PARAMS["model_params"])}


def gpu_regression_model(regressor: str, seed: int) -> dict:
    """`regression_model` ép GPU: LightGBM `device_type='gpu'`, xgboost `device='cuda'` (+ seed vì AutoTS không set cho xgboost)."""
    if regressor not in REGRESSOR_GPU:
        raise KeyError(f"regressor phải thuộc {sorted(REGRESSOR_GPU)}: {regressor}")
    mp = dict(REGRESSOR_GPU[regressor])
    if regressor.lower() in ("xgboost", "xgbregressor"):
        mp.setdefault("random_state", seed)
    return {"model": regressor, "model_params": mp}


def template_frame(specs: list[dict], seed: int, frequency: str = "min", cls_map: dict | None = None) -> "pd.DataFrame":
    """Dựng `initial_template` từ khai báo config. Params lấy từ CHÍNH `get_params()` của class AutoTS → đúng khoá tuyệt đối
    (khoá sai = TypeError lúc chạy). Mọi dòng: regressor GPU, `regression_type='User'`, `datepart_method=None`, `holiday=False`,
    transformation rỗng → feature set không bị thêm cột ngoài F_frozen."""
    import json as _json

    rows = []
    for spec in specs:
        kind = spec.get("model", "wr").lower()
        kind = {"windowregression": "wr", "multivariateregression": "mr"}.get(kind, kind)
        name = MODEL_OF[kind]
        cls = (cls_map or {}).get(name)
        if cls is None:
            from autots.models.sklearn import MultivariateRegression, WindowRegression

            cls = {"WindowRegression": WindowRegression, "MultivariateRegression": MultivariateRegression}[name]
        kw = dict(forecast_length=len(HORIZONS), frequency=frequency, regression_type="User", datepart_method=None,
                  regression_model=gpu_regression_model(spec.get("regressor", "LightGBM"), seed), n_jobs=1, random_seed=seed)
        if kind == "wr":
            kw.update(window_size=int(spec.get("window_size", 60)), output_dim="forecast_length",
                      max_windows=int(spec.get("max_windows", 200_000)), normalize_window=False, scale=False, shuffle=False)
        else:
            kw.update(holiday=False)
        params = cls(**kw).get_params()
        assert str(params.get("regression_type", "")).lower() == "user", f"{name}: regression_type phải là 'User'"
        assert params.get("datepart_method") in (None, "None"), f"{name}: datepart_method phải None (không thêm cột ngoài F_frozen)"
        assert not params.get("holiday", False), f"{name}: holiday phải False"
        rows.append({"Model": name, "ModelParameters": _json.dumps(params), "TransformationParameters": _json.dumps(NO_TRANS),
                     "Ensemble": 0})
    return pd.DataFrame(rows)


def search_best_template(df_tr: "pd.DataFrame", R_tr: "pd.DataFrame", template: "pd.DataFrame", num_validations: int,
                         seed: int, autots_cls=None) -> tuple[str, dict, "pd.DataFrame"]:
    """Bake-off trên TRAINING-SIDE của fold (df_tr kết thúc trước purge — outer VAL không bao giờ được nhìn thấy).

    `max_generations=0` → chỉ chạy đúng các dòng của `template`, không sinh dòng ngẫu nhiên; AutoTS vẫn chạy
    `num_validations` vòng validation NỘI BỘ df_tr và chọn best theo RMSE. `models_to_validate=0.99` +
    `max_per_model_class=99` để không dòng nào bị cắt khỏi validation. Trả (tên model, params, bảng mọi candidate).
    """
    import json as _json
    import random as _random

    if autots_cls is None:
        from autots import AutoTS as autots_cls  # noqa: N813
    auto = autots_cls(
        forecast_length=len(HORIZONS), frequency="min", model_list=list(MODEL_OF.values()),
        initial_template=template, max_generations=0, num_validations=int(num_validations),
        validation_method="backwards", models_to_validate=0.99, max_per_model_class=99,
        metric_weighting={"rmse_weighting": 1}, ensemble=None, no_negatives=False, constraint=None,
        drop_most_recent=0, introduce_na=False, holiday_country=None,
        transformer_list="superfast", transformer_max_depth=0,  # [] bị coi là "all" (transform.py:8980)
        random_seed=int(seed), n_jobs=1, verbose=1,  # verbose <= 0 bật filterwarnings toàn cục
    )
    auto.fit(df_tr, future_regressor=R_tr)
    _random.seed(seed)  # AutoTS set random/np.random toàn cục → trả lại trạng thái cho harness
    np.random.seed(seed)
    name = str(auto.best_model_name)
    params = auto.best_model_params if isinstance(auto.best_model_params, dict) else _json.loads(auto.best_model_params)
    trans = auto.best_model_transformation_params
    trans = trans if isinstance(trans, dict) else _json.loads(trans or "{}")
    assert str(params.get("regression_type", "")).lower() == "user", "template thắng bỏ qua F_frozen (regression_type != User)"
    assert params.get("datepart_method") in (None, "None"), "template thắng thêm cột datepart ngoài F_frozen"
    assert not params.get("holiday", False), "template thắng thêm cột holiday ngoài F_frozen"
    assert not trans.get("transformations"), "template thắng có transformer — đường ModelMonster không áp transformer"
    params["regression_model"] = gpu_regression_model(params["regression_model"]["model"], seed)  # ép lại GPU (§12.5)
    return name, params, auto.export_template(None, models="all")
