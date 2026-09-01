"""TimesFM (§2.2 #4) — TFM-POINT zero-shot + covariate loop. Theo `docs/reference/audit_timesfm.md` (timesfm 2.0.2).

- Input = chuỗi `r1` kết thúc tại origin t (context 512, chỉ τ ≤ t); `forecast(horizon=3)` → `r̂_{t+1..t+3}` → **cộng dồn** → `y_h`.
- Không train, không dùng FIT/ES (`best_iters = 0`); model trả thẳng log-return (`FitResult.is_logret = True`)
  nên KHÔNG đi qua z-space của B0 — plan §6.7: "AutoTS/TimesFM cộng dồn one-step đúng thứ tự trước khi exp".
- `infer_is_positive=False` (tắt ép dương), `force_flip_invariance=True`, `normalize_inputs=True`; head **mean** (`quantile[...,0]`)
  vì metric là RMSE (`point_forecast` là q50) — dùng chung một head cho cả đường point và đường covariate.
- Đường covariate compile riêng với `per_core_batch_size=1` (1 origin/lời gọi; để 256 sẽ pad 1 series lên 256 → ~20× chậm).
- Covariate (§2.2 #4b): `forecast_with_covariates`, **1 origin mỗi lời gọi** — API fit chung một `beta_hat` cho cả batch nên
  gộp nhiều origin sẽ để origin sau ảnh hưởng origin trước (vi phạm §6.4). Covariate **dịch 1 bar** (vị trí s mang f(s−1))
  và 3 bước tương lai giữ giá trị tại t.
"""
from __future__ import annotations

import numpy as np

from .config import HORIZONS
from .models import FitResult, SeriesBatch, _cpu_guard

CONTEXT = 512
# HAI nhánh feature selection của TimesFM (§2.2 #4b) — mỗi nhánh chạy đủ add-one → prune → confirmation,
# rồi so với nhau bằng metric project để ra TimesFM-final:
#   b0star : điểm xuất phát S = B0*  (covariate = B0* + candidate ext đang KEEP)  → model `tfm_b0`
#   ext    : điểm xuất phát S = ∅    (TimesFM native trên r1; covariate = chỉ candidate ext) → model `tfm_ext`
COVARIATE_SCOPES = {"b0star": "all", "ext": "ext"}  # scope → cột nào của colset thành covariate trong run_config
REPO_ID = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"  # pin theo audit
_CACHE: dict[tuple, object] = {}


class TimesFMModel:
    lib = "timesfm"
    supports_rounds = False  # zero-shot: không có số vòng để calibrate (§1.3)
    seed_dependent = False  # zero-shot, không có nguồn ngẫu nhiên → 3 seed cho kết quả y hệt (§1.3: ε = floor)
    input_kind = "series"

    def __init__(self, device: str = "cuda", allow_cpu: bool = False, repo_id: str = REPO_ID, revision: str = REVISION,
                 context: int = CONTEXT, max_horizon: int = 128, batch_size: int = 256, torch_compile: bool = True,
                 normalize_inputs: bool = True, force_flip_invariance: bool = True, use_mean_head: bool = True,
                 xreg_mode: str = "xreg + timesfm", xreg_force_on_cpu: bool = False, covariate_scope: str = "ext",
                 name: str = "tfm", model: object | None = None):
        _cpu_guard(device == "cuda", allow_cpu, "TimesFM")
        if covariate_scope not in COVARIATE_SCOPES:
            raise KeyError(f"covariate_scope phải thuộc {sorted(COVARIATE_SCOPES)}: {covariate_scope}")
        self.covariate_scope, self.name = covariate_scope, name
        # b0star → covariate = toàn bộ colset (B0* + ext); ext → chỉ các cột ext (baseline S = ∅ là TimesFM native)
        self.series_covariates = COVARIATE_SCOPES[covariate_scope]
        self.device, self.repo_id, self.revision = device, repo_id, revision
        self.context, self.max_horizon, self.batch_size = int(context), int(max_horizon), int(batch_size)
        self.torch_compile, self.normalize_inputs, self.flip = torch_compile, normalize_inputs, force_flip_invariance
        self.use_mean_head, self.xreg_mode, self.xreg_force_on_cpu = use_mean_head, xreg_mode, xreg_force_on_cpu
        self.train_device = self.predict_device = "GPU" if device == "cuda" else "CPU"
        self._injected = model  # chỉ dùng cho unit test (stub); production luôn load checkpoint thật

    # ------------------------------------------------------------------ checkpoint
    def forecast_config_kwargs(self, with_covariates: bool) -> dict:
        """Tham số `ForecastConfig`. Đường covariate BẮT BUỘC 1 origin/lời gọi → `per_core_batch_size = 1`:
        để nguyên 256 thì `forecast()` pad 1 series lên 256 (timesfm_2p5_base:167) ⇒ ~20× thời gian mỗi lời gọi
        (~942 ms thay vì ~45 ms → +12–20 h cho 40 run). Đường point vẫn batch đầy đủ."""
        return dict(max_context=self.context, max_horizon=self.max_horizon, normalize_inputs=self.normalize_inputs,
                    per_core_batch_size=1 if with_covariates else self.batch_size,
                    force_flip_invariance=self.flip,
                    infer_is_positive=False,  # tắt ép dương (plan §2.2 #4)
                    return_backcast=with_covariates)  # guard bắt buộc của forecast_with_covariates

    def _model(self, with_covariates: bool):
        if self._injected is not None:
            return self._injected
        key = (self.repo_id, self.revision, self.torch_compile, tuple(sorted(self.forecast_config_kwargs(with_covariates).items())))
        if key not in _CACHE:
            import timesfm

            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id, revision=self.revision, torch_compile=self.torch_compile)
            m.compile(timesfm.ForecastConfig(**self.forecast_config_kwargs(with_covariates)))
            _CACHE[key] = m
        return _CACHE[key]

    # ------------------------------------------------------------------ chuẩn bị input (chỉ τ ≤ t)
    def contexts(self, seq: SeriesBatch) -> list[np.ndarray]:
        L = self.context
        if len(seq.idx) and int(seq.idx.min()) < L - 1:
            raise ValueError(f"TimesFM: origin {int(seq.idx.min())} không đủ {L} bar context")
        out = [np.asarray(seq.r1[t - L + 1:t + 1], dtype=np.float32) for t in seq.idx]
        if out and not np.isfinite(np.stack(out)).all():
            raise ValueError("TimesFM: context chứa NaN/inf")
        return out

    def covariate_window(self, seq: SeriesBatch, t: int, j: int) -> np.ndarray:
        """Mảng dài context + 3 cho cột j tại origin t: vị trí s mang f(s−1) (dịch 1 bar), 3 bước tương lai giữ f(t)."""
        L = self.context
        past = seq.cov[t - L:t, j]  # f(t−L) … f(t−1) ↔ vị trí t−L+1 … t
        return np.concatenate([np.asarray(past, dtype=np.float32), np.full(len(HORIZONS), float(seq.cov[t, j]), np.float32)])

    # ------------------------------------------------------------------ forecast
    def head(self, res) -> np.ndarray:
        """Chọn head: `point_forecast` là **q50**, RMSE cần **mean** = `quantile_forecast[..., 0]`.
        Dùng CHUNG cho cả đường point và đường covariate — nếu khác head thì Gain "covariate vs POINT"
        sẽ lẫn chênh lệch q50-vs-mean chứ không phải tác dụng của covariate."""
        point, quant = (res[0], res[1] if len(res) > 1 else None) if isinstance(res, (tuple, list)) else (res, None)
        return np.asarray(quant)[..., 0] if (self.use_mean_head and quant is not None) else np.asarray(point)

    def _point(self, m, ctxs: list[np.ndarray]) -> np.ndarray:
        H = len(HORIZONS)
        out = []
        for s in range(0, len(ctxs), self.batch_size):
            chunk = ctxs[s:s + self.batch_size]
            res = m.forecast(horizon=H, inputs=list(chunk))  # copy: forecast() mutate list đầu vào
            out.append(np.asarray(self.head(res), dtype=np.float64)[:len(chunk), :H])
        return np.concatenate(out) if out else np.zeros((0, H))

    def _with_covariates(self, m, seq: SeriesBatch, ctxs: list[np.ndarray]) -> np.ndarray:
        H = len(HORIZONS)
        rows = []
        for k, t in enumerate(seq.idx):
            # perm (PI §2.1a): kênh j lấy cửa sổ covariate của origin khác, các kênh còn lại giữ nguyên
            dyn = {name: [self.covariate_window(seq, int(seq.perm[j][k]) if (seq.perm and j in seq.perm) else int(t), j)]
                   for j, name in enumerate(seq.cov_names)}
            res = m.forecast_with_covariates(
                inputs=[ctxs[k]],  # ĐÚNG 1 origin: xreg fit chung beta_hat cho cả batch → gộp là leakage (§6.4)
                dynamic_numerical_covariates=dyn, dynamic_categorical_covariates={},
                static_numerical_covariates={}, static_categorical_covariates={},
                xreg_mode=self.xreg_mode, normalize_xreg_target_per_input=True, ridge=0.0,
                # force_on_cpu=False → jax dùng backend mặc định (GPU khi có jax[cuda12]); xreg_lib:479 đặt
                # device=None thay vì ép jax.devices("cpu")[0], nên khối ước lượng beta_hat (pinv(XᵀX)@Xᵀ@y và
                # x_test@beta_hat) chạy trên RTX 3090. Invariant §0 "training chỉ GPU" áp dụng cho cả xreg.
                max_rows_per_col=0, force_on_cpu=self.xreg_force_on_cpu,
            )
            rows.append(np.asarray(self.head(res), dtype=np.float64).reshape(-1)[:H])
        return np.stack(rows) if rows else np.zeros((0, H))

    def predict_series(self, seq: SeriesBatch) -> np.ndarray:
        """ŷ (n, 3) = cộng dồn r̂_{t+1..t+3} (log-return), chưa qua exp."""
        use_cov = seq.cov is not None and len(seq.cov_names) > 0
        m = self._model(use_cov)
        ctxs = self.contexts(seq)
        r_hat = self._with_covariates(m, seq, ctxs) if use_cov else self._point(m, ctxs)
        return np.cumsum(np.asarray(r_hat, dtype=np.float64), axis=1).astype(np.float32)

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred: SeriesBatch, rounds, seed: int) -> FitResult:
        yhat = self.predict_series(X_pred)  # zero-shot: bỏ qua FIT/ES
        return FitResult(yhat, (0, 0, 0), [self.predict_series], is_logret=True)
