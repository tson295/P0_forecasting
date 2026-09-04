"""TimesFM (§2.2 #4). `TimesFMLoRAModel` (cuối file) là model ĐANG DÙNG (vòng expanded-data 2026-09-03/04: LoRA per fold → freeze → XReg);
`TimesFMModel` (zero-shot, `make_model("tfm_zero_shot")`) chỉ còn là lớp tham chiếu/canary — đường forecast/covariate/xreg được kế thừa
nguyên vẹn. Ghi chú dưới đây mô tả đường suy luận chung (audit `docs/reference/audit_timesfm.md`, timesfm 2.0.2).

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

import functools
import json
from pathlib import Path

import numpy as np

from .config import HORIZONS
from .models import FitResult, SeriesBatch, _cpu_guard

CONTEXT = 512
# Scope covariate: model ĐANG DÙNG (`tfm` = TimesFMLoRAModel) luôn là "ext" — S0_TFM = ∅, B0* KHÔNG bao giờ vào XReg (2026-09-03/04).
# "b0star" chỉ còn là tham chiếu lịch sử của nhánh zero-shot `tfm_b0` vòng 15 ngày (không còn model nào dùng).
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


# =============================================================================== TimesFM-LoRA (quyết định user 2026-09-03)
# pretrained → LoRA fine-tune per fold trên chuỗi r1 (FIT để học, ES để chọn epoch, VAL không bao giờ thấy) → FREEZE →
# CÙNG adapter đó cho toàn bộ XReg covariate search (thêm candidate = fit lại xreg, KHÔNG động vào trọng số).
# Theo `docs/reference/audit_timesfm_lora.md` (timesfm 2.0.2, sdist sha b03885d3…): 80 nn.Linear đích (qkv_proj/out/ff0/ff1 × 20),
# decode() bọc no_grad nên training tái hiện T:427–500 + T:119–178 có grad; compile() không torch.compile; inject sau
# load_checkpoint; base đóng băng tường minh; MỘT module dùng chung cho wrapper point (batch) và wrapper covariate (batch 1).
LORA_TARGETS = ("attn.qkv_proj", "attn.out", "ff0", "ff1")
LORA_DEFAULTS = {"r": 8, "alpha": 16.0, "dropout": 0.0, "lr": 1e-4, "weight_decay": 0.01, "batch_size": 64, "max_epochs": 20,
                 "patience": 5, "train_stride": 1, "targets": LORA_TARGETS}
_LORA_CACHE: dict[str, dict] = {}  # (repo, revision, lora cfg) → {point, cov, module, adapter, ...} — một module cho cả process
_ADAPTER_STATES: dict[str, dict] = {}  # adapter key → state dict LoRA (A/B) đã freeze
_ADAPTER_META: dict[str, dict] = {}


def _stamp(ts) -> str:
    import pandas as pd

    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y%m%dT%H%M")


class TimesFMLoRAModel(TimesFMModel):
    """TimesFM 2.5 + LoRA per fold. `rounds` = số epoch cố định (như LSTM: calibrate ES → fixed_epoch_TFM; None = ES bật).

    Adapter được cache theo (fold, seed, epoch-mode): mọi candidate của vòng add-one/prune PI ở cùng (fold, selection_seed,
    fixed epoch) nạp ĐÚNG một adapter đã freeze — không train lại; hash LoRA được kiểm sau mỗi lần predict.
    Artifact: `<adapter_dir>/<key>.pt` (state dict A/B) + `<key>.json` (meta: epoch, curve, sha256, repo/revision, cấu hình)."""

    supports_rounds = True
    seed_dependent = True

    def __init__(self, lora: dict | None = None, adapter_dir: str | None = None, **kw):
        kw["torch_compile"] = False  # audit §3: không torch.compile khi có LoRA (inject sau load_checkpoint, closure đọc self.model lúc gọi)
        super().__init__(**kw)
        self.lora = {**LORA_DEFAULTS, **(lora or {})}
        self.lora["targets"] = tuple(self.lora["targets"])
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.train_calls = 0  # số lần train thật (test: candidate không được làm tăng)

    # ------------------------------------------------------------------ module dùng chung + LoRA
    def _cache_key(self) -> str:
        return json.dumps({"repo": self.repo_id, "rev": self.revision, "lora": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.lora.items()},
                           "stub": id(self._injected) if self._injected is not None else None}, sort_keys=True)

    def _wrappers(self) -> dict:
        from .lora import freeze_except_lora, inject_lora

        key = self._cache_key()
        if key in _LORA_CACHE:
            return _LORA_CACHE[key]
        if self._injected is not None:  # unit test: stub có .model (nn.Module) + forecast/forecast_with_covariates (+ train_forward)
            point = cov = self._injected
        else:
            import timesfm

            point = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id, revision=self.revision, torch_compile=False)
            point.compile(timesfm.ForecastConfig(**self.forecast_config_kwargs(False)))
            cov = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            cov.model = point.model  # MỘT module cho cả hai đường (audit §7): point batch 256, covariate batch 1
            cov.compile(timesfm.ForecastConfig(**self.forecast_config_kwargs(True)))
        module = point.model
        replaced = inject_lora(module, self.lora["targets"], int(self.lora["r"]), float(self.lora["alpha"]), float(self.lora["dropout"]))
        n_train, n_all = freeze_except_lora(module)
        if self._injected is None and tuple(self.lora["targets"]) == LORA_TARGETS:
            # audit_timesfm_lora.md §1/§5: 4 nn.Linear × 20 layer = 80 module; mỗi layer r·(1280+3840) + 3·r·(1280+1280) = r·12800
            exp_train = int(self.lora["r"]) * 12800 * 20
            if len(replaced) != 80 or n_train != exp_train:
                raise RuntimeError(f"TimesFM-LoRA: inject lệch audit — {len(replaced)} module (mong 80), {n_train} tham số học (mong {exp_train})")
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
        _LORA_CACHE[key] = {"point": point, "cov": cov, "module": module, "replaced": replaced, "n_trainable": n_train,
                            "n_params": n_all, "adapter": None}
        return _LORA_CACHE[key]

    def _model(self, with_covariates: bool):
        w = self._wrappers()
        if w["adapter"] is None:
            raise RuntimeError("TimesFM-LoRA: chưa có adapter nào được train/nạp — fit_predict phải chạy trước predict")
        return w["cov"] if with_covariates else w["point"]

    def _load_adapter(self, key: str) -> None:
        from .lora import load_lora_state_dict

        w = self._wrappers()
        if w["adapter"] == key:
            return
        load_lora_state_dict(w["module"], _ADAPTER_STATES[key])
        for p in w["module"].parameters():
            p.requires_grad_(False)
        w["module"].eval()
        w["adapter"] = key

    def _assert_frozen(self, key: str) -> None:
        from .lora import lora_state_dict, state_sha256

        sha = state_sha256(lora_state_dict(self._wrappers()["module"]))
        if sha != _ADAPTER_META[key]["sha256"]:
            raise RuntimeError(f"TimesFM-LoRA: trọng số adapter {key} đã bị thay đổi trong lúc predict (sha lệch) — vi phạm freeze")

    # ------------------------------------------------------------------ cửa sổ train (chỉ FIT / ES)
    def windows(self, seq: SeriesBatch, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """X (n, L) = r1[t−L+1..t], Y (n, H) = y_h = cumsum(r1[t+1..t+H]) cho t ∈ seq.idx (partition đã đảm bảo t+H < T_end)."""
        L, H = self.context, len(HORIZONS)
        idx = np.asarray(seq.idx)[:: max(1, int(stride))]
        if len(idx) and int(idx.min()) < L - 1:
            raise ValueError("TimesFM-LoRA: origin không đủ context")
        if len(idx) and int(idx.max()) + H >= len(seq.r1):
            raise ValueError("TimesFM-LoRA: target vượt quá chuỗi")
        X = np.stack([seq.r1[t - L + 1:t + 1] for t in idx]).astype(np.float32) if len(idx) else np.zeros((0, L), np.float32)
        Y = np.stack([np.cumsum(seq.r1[t + 1:t + 1 + H]) for t in idx]).astype(np.float32) if len(idx) else np.zeros((0, H), np.float32)
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            raise ValueError("TimesFM-LoRA: cửa sổ train chứa NaN/inf")
        return X, Y

    # ------------------------------------------------------------------ forward có grad = ĐÚNG đường suy luận (audit §2/§9b)
    def train_forward(self, x):
        """x (B, L) float32 → r̂ (B, H) mean-head, tái hiện `_compiled_decode` (T:427–500) + `decode` (T:119–178) CÓ grad."""
        stub = self._injected
        if stub is not None and hasattr(stub, "train_forward"):
            return stub.train_forward(x)
        import torch
        from timesfm.torch import util

        module = self._wrappers()["module"]
        H = len(HORIZONS)
        if self.normalize_inputs:
            mu = torch.mean(x, dim=-1, keepdim=True)
            sigma = torch.std(x, dim=-1, keepdim=True)  # unbiased như T:438–439
            xn = util.revin(x, mu, sigma, reverse=False)
        else:
            mu = sigma = None
            xn = x

        def core(inp):
            B = inp.shape[0]
            patched = torch.reshape(inp, (B, -1, module.p))
            masks = torch.zeros_like(patched, dtype=torch.bool)
            n = torch.zeros(B, device=inp.device)
            m_ = torch.zeros(B, device=inp.device)
            s_ = torch.zeros(B, device=inp.device)
            mus, sigs = [], []
            for i in range(patched.shape[1]):
                (n, m_, s_), _ = util.update_running_stats(n, m_, s_, patched[:, i], masks[:, i])
                mus.append(m_)
                sigs.append(s_)
            cmu, csig = torch.stack(mus, dim=1), torch.stack(sigs, dim=1)
            normed = util.revin(patched, cmu, csig, reverse=False)
            normed = torch.where(masks, 0.0, normed)
            (_, _, out_ts, _), _ = module(normed, masks, None)
            ren = torch.reshape(util.revin(out_ts, cmu, csig, reverse=True), (B, -1, module.o, module.q))
            return ren[:, -1, :H, :]  # (B, H, q) — dự báo sau patch cuối, giống pf_outputs[:, -1, ...][:, :horizon]

        f = core(xn)
        if self.flip:  # force_flip_invariance: (f(x) − flip(f(−x)))/2, flip giữ kênh 0 (mean) và đảo 1..9
            g = core(-xn)
            g = torch.cat([g[..., :1], torch.flip(g[..., 1:], dims=(-1,))], dim=-1)
            f = (f - g) / 2
        ch = 0 if self.use_mean_head else module.aridx
        out = f[..., ch]
        if mu is not None:
            out = util.revin(out, mu, sigma, reverse=True)
        return out

    # ------------------------------------------------------------------ adapter per (fold, seed, epoch-mode)
    def adapter_key(self, X_fit: SeriesBatch, X_es: SeriesBatch | None, seed: int, epochs: int | None) -> str:
        fit = f"fit{_stamp(X_fit.ts[X_fit.idx[0]])}-{_stamp(X_fit.ts[X_fit.idx[-1]])}"
        es = f"_es{_stamp(X_es.ts[X_es.idx[0]])}-{_stamp(X_es.ts[X_es.idx[-1]])}" if (X_es is not None and len(X_es.idx)) else ""
        return f"tfm_lora_{fit}{es}_seed{int(seed)}_{'es' if epochs is None else f'ep{int(epochs)}'}"

    def _ensure_adapter(self, key: str, X_fit: SeriesBatch, X_es: SeriesBatch | None, seed: int, epochs: int | None) -> dict:
        import torch

        from .lora import lora_state_dict, state_sha256, train_lora

        if key in _ADAPTER_STATES:
            return _ADAPTER_META[key]
        path = (self.adapter_dir / f"{key}.pt") if self.adapter_dir else None
        if path is not None and path.exists():
            sd = torch.load(path, map_location="cpu", weights_only=True)
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if state_sha256(sd) != meta["sha256"]:
                raise RuntimeError(f"adapter {path}: sha256 không khớp meta")
            want = {"repo_id": self.repo_id, "revision": self.revision, "context": self.context,
                    "lora": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.lora.items()}}
            got = {k: meta.get(k) for k in want}
            if got != want:
                raise RuntimeError(f"adapter {path}: meta không khớp model hiện tại (repo/revision/context/lora): {got} vs {want}")
            _ADAPTER_STATES[key], _ADAPTER_META[key] = sd, meta
            return meta
        w = self._wrappers()
        module = w["module"]
        X, Y = self.windows(X_fit, int(self.lora["train_stride"]))
        if X_es is not None and epochs is None:
            Xe, Ye = self.windows(X_es, 1)
        else:
            Xe = Ye = None
        scale = float(np.std(Y[:, 0])) if len(Y) and float(np.std(Y[:, 0])) > 0 else 1.0  # hằng số ổn định optimizer, không đổi argmin

        def fwd(xb):
            return torch.cumsum(self.train_forward(xb), dim=1) / scale  # ŷ_h = cumsum(r̂) (§6.7), cùng thang với Y/scale

        self.train_calls += 1
        res = train_lora(fwd, module, X, Y / scale, Xe, (Ye / scale if Ye is not None else None), epochs=epochs,
                         max_epochs=int(self.lora["max_epochs"]), patience=int(self.lora["patience"]), lr=float(self.lora["lr"]),
                         batch_size=int(self.lora["batch_size"]), seed=int(seed), device=self.device,
                         weight_decay=float(self.lora["weight_decay"]))
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
        sd = lora_state_dict(module)
        meta = {"key": key, "best_epoch": res["best_epoch"], "mode": res["mode"], "curve": res["curve"], "sha256": res["sha256"],
                "n_trainable": res["n_trainable"], "n_params": res["n_params"], "seed": int(seed), "epochs_fixed": epochs,
                "n_windows_fit": int(len(X)), "n_windows_es": int(0 if Xe is None else len(Xe)), "target_scale": scale,
                "fit_range": [_stamp(X_fit.ts[X_fit.idx[0]]), _stamp(X_fit.ts[X_fit.idx[-1]])],
                "es_range": [_stamp(X_es.ts[X_es.idx[0]]), _stamp(X_es.ts[X_es.idx[-1]])] if X_es is not None and len(X_es.idx) else None,
                "repo_id": self.repo_id, "revision": self.revision, "lora": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.lora.items()},
                "replaced_modules": w["replaced"], "context": self.context, "torch": torch.__version__}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, path)
            path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        _ADAPTER_STATES[key], _ADAPTER_META[key] = sd, meta
        w["adapter"] = key
        return meta

    def artifact_meta(self, covariates=(), native: bool = True) -> dict:
        """Metadata tường minh cho artifact (wins/*.json): backbone, LoRA, native hay +XReg, chuỗi vào/ra (§10 hiệu chỉnh 2026-09-04)."""
        cov = list(covariates)
        return {"backbone": "timesfm-2.5-200m", "repo_id": self.repo_id, "revision": self.revision, "finetuned": True,
                "finetune_method": "LoRA", "lora": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.lora.items()},
                "native": bool(native and not cov), "covariates": cov, "input_series": "btc_1m_log_return", "context": int(self.context),
                "forecast_horizon": len(HORIZONS), "target": "cumulative_log_return_y1_y2_y3",
                "xreg": None if (native and not cov) else {"mode": self.xreg_mode, "one_origin_per_call": True, "covariate_shift_bars": 1,
                                                          "force_on_cpu": bool(self.xreg_force_on_cpu)},
                "calibration": "LoRA FIT + ES chọn epoch (= calibrate của model có epoch); adapter freeze trước candidate search"}

    def _predict_with(self, key: str, seq: SeriesBatch) -> np.ndarray:
        self._load_adapter(key)
        out = self.predict_series(seq)
        self._assert_frozen(key)
        return out

    def fit_predict(self, X_fit: SeriesBatch, z_fit, X_es: SeriesBatch, z_es, X_pred: SeriesBatch, rounds, seed: int) -> FitResult:
        epochs = int(rounds[0]) if rounds is not None else None
        key = self.adapter_key(X_fit, X_es, seed, epochs)
        meta = self._ensure_adapter(key, X_fit, X_es, seed, epochs)
        yhat = self._predict_with(key, X_pred)
        e = int(meta["best_epoch"])
        return FitResult(yhat, (e, e, e), [functools.partial(self._predict_with, key)], is_logret=True)
