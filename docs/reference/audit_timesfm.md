# AUDIT — TimesFM (cho plan §2.2 #4)

Ngày: 2026-08-29 · researcher · Trạng thái: **chưa cài package, chưa tải checkpoint, chưa chạy** (TRAINING: LOCKED).
Phương pháp: metadata PyPI + đọc **source thật** của wheel `timesfm-2.0.2-py3-none-any.whl` (tải bằng `pip download --no-deps` vào thư mục tạm, KHÔNG cài vào env), HuggingFace API, GitHub `google-research/timesfm@master`.
Mọi claim gắn với **timesfm 2.0.2**; không suy rộng sang version khác.

## 1. Version chốt

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Package | `timesfm` **2.0.2** (upload 2026-07-02), wheel `py3-none-any` (pure python) | PyPI JSON API |
| requires-python | `>=3.10` → **OK với Python 3.12.10** | wheel METADATA |
| Deps bắt buộc | `numpy>=1.26.4`, `huggingface_hub[cli]>=0.23.0`, `safetensors>=0.5.3` | wheel METADATA |
| Extra `torch` | `torch>=2.0.0` → **OK với torch 2.11+cu128** | wheel METADATA |
| Extra `xreg` (covariate) | `jax[cuda]` + `scikit-learn` | wheel METADATA |
| Lệnh cài | `pip install "timesfm[torch]==2.0.2"` (+ `jax` bản **CPU** và `scikit-learn` nếu chạy covariate) | — |

**timesfm 1.x KHÔNG dùng được**: `timesfm 1.2.0` và `1.3.0` khai báo `Requires-Python: <3.12,>=3.10` → không cài được trên Python 3.12. Kéo theo: API cũ `timesfm.TimesFm(hparams=..., checkpoint=...)`, `forecast(inputs, freq=[...])`, `forecast_on_df(...)` và checkpoint `timesfm-1.0-200m` / `timesfm-2.0-500m` **ngoài tầm** ở env hiện tại.

Wheel 2.0.2 chỉ chứa: `timesfm/{__init__,configs}.py`, `timesfm/timesfm_2p5/*`, `timesfm/torch/*`, `timesfm/flax/*`, `timesfm/utils/xreg_lib.py`. **Không có** module finetune/LoRA, không có API 1.x.

GitHub master (push 2026-08-28) có thêm `src/timesfm3/` chưa phát hành lên PyPI → **pin 2.0.2 từ PyPI**, không cài từ git.

## 2. Checkpoint

| Repo HF | gated | file | size | dùng cho |
|---|---|---|---|---|
| `google/timesfm-2.5-200m-pytorch` | **False** (không cần token) | `model.safetensors` | **925.2 MB** (fp32, ~200M param) | zero-shot + covariate (package `timesfm`) |
| `google/timesfm-2.5-200m-transformers` | **False** | `model.safetensors` | 925.2 MB | **chỉ** cho LoRA qua `transformers`+`peft` |

- Pin revision: `...-pytorch` sha `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` (lastModified 2025-10-02); `...-transformers` sha `5a9806b9b291fad9233b5249d88263f1846304d3`.
- Model card: 2025-10-02 đã fuse QKV → **phải dùng timesfm >= 2.0.x**, bản cũ load sai state_dict. 2.0.2 có `fuse_qkv=True` → khớp.
- VRAM ~1 GB weight fp32 + activation; ổn trên 3090.

## 3. API thật (đọc từ source 2.0.2)

```python
# timesfm/__init__.py
from .configs import ForecastConfig
TimesFM_2p5_200M_torch = timesfm_2p5_torch.TimesFM_2p5_200M_torch   # có nếu import được torch
TimesFM_2p5_200M_flax  = ...                                        # cần jax/flax
```

```python
# timesfm/configs.py — ForecastConfig (dataclass frozen)
ForecastConfig(max_context=0, max_horizon=0, normalize_inputs=False, window_size=0,
               per_core_batch_size=1, use_continuous_quantile_head=False,
               force_flip_invariance=True, infer_is_positive=True,
               fix_quantile_crossing=False, return_backcast=False)
```

```python
# timesfm/timesfm_2p5/timesfm_2p5_torch.py
class TimesFM_2p5_200M_torch(TimesFM_2p5, PyTorchModelHubMixin):
    DEFAULT_REPO_ID  = "google/timesfm-2.5-200m-pytorch"
    WEIGHTS_FILENAME = "model.safetensors"
    def __init__(self, torch_compile: bool = True, config: dict | None = None, **kwargs)
    def load_checkpoint(self, path: str, **kwargs)
    def compile(self, forecast_config: configs.ForecastConfig, **kwargs) -> None

# kế thừa từ timesfm_2p5_base.TimesFM_2p5:
    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]
    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates=None,
        dynamic_categorical_covariates=None, static_numerical_covariates=None,
        static_categorical_covariates=None, xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True, ridge=0.0, max_rows_per_col=0,
        force_on_cpu=False)
```

- `from_pretrained(model_id, revision=..., token=..., cache_dir=..., torch_compile=...)` do `PyTorchModelHubMixin` cung cấp; kwargs thừa truyền vào `__init__`.
- **Batch nhiều origin trong một lần gọi: CÓ.** `forecast(inputs=[ctx_1, ctx_2, ...])` — mỗi phần tử là một context window riêng, độ dài tuỳ ý; base tự cắt/pad-trái + mask về `max_context` và tự chia lô theo `global_batch_size = per_core_batch_size * device_count`. Đây là điều kiện sống còn cho 7.185 origin và nó **được hỗ trợ** (chỉ cho TFM-POINT; xem §5 cho covariate).
- Output: `point_forecast (B, horizon)`, `quantile_forecast (B, horizon, 10)` = `[mean, q10..q90]` (model card).
- **`point_forecast` = kênh 5 = q50 (median), KHÔNG phải mean** (`full_forecast[..., 5]`, `decode_index=5`). Metric của ta là RMSE nên tối ưu là **mean**: dùng `quantile_forecast[..., 0]`. Đây là lựa chọn *decode*, không phải model mới; đề xuất dùng mean và log cả hai ở run đầu.
- `max_context` phải là bội của 32 (patch len), `max_horizon` bội của 128 (output patch len); nếu không `compile()` tự làm tròn lên và in log. Context 512 (plan) hợp lệ. Với h=3: truyền `max_horizon=128`, gọi `forecast(horizon=3, ...)` → trả đúng 3 bước và **không có bước decode tự hồi quy** (`num_decode_steps = (h-1)//128 = 0`).
- `context_limit = 16384`; warmup B0 là 631 bar nên **mọi origin đều đủ 512 bar context**.

### Cờ phải đặt đúng cho return có dấu

| Cờ | Mặc định | Đặt gì | Lý do |
|---|---|---|---|
| `infer_is_positive` | **True** | **False** | đúng "tùy chọn ép dương" plan §2.2 #4: `if all(inputs>=0): forecast = max(forecast, 0)`. Với r1 gần như luôn có giá trị âm nên thực tế không kích hoạt, nhưng phải tắt tường minh |
| `force_flip_invariance` | True | giữ True | decode 2 lần (`x` và `-x`) rồi lấy `(f(x) - f(-x))/2` → dự báo phản đối xứng, hợp chuỗi return; **tốn 2x compute** |
| `normalize_inputs` | False | True (như model card) | r1 biên độ ~1e-3; model còn RevIN theo patch bên trong |
| `return_backcast` | False | False cho TFM-POINT; **bắt buộc True** cho `forecast_with_covariates` | guard trong source |
| `use_continuous_quantile_head` | False | False | chỉ cần point |

### Gotcha trong source (đọc trước khi code)

1. `forecast()` **mutate list `inputs` tại chỗ** (`inputs += [...]` để pad cho đủ batch) → truyền bản copy.
2. `_compiled_decode` gọi `self.model.decode(forecast_config.max_horizon, ...)` dùng `forecast_config` **gốc** chứ không phải `fc` đã làm tròn (closure inconsistency; vô hại khi horizon <= 128). Truyền `max_horizon=128` để hai giá trị trùng nhau.
3. `torch_compile=True` là mặc định; shape batch phải cố định để không recompile. Nếu `torch.compile` lỗi trên instance → `from_pretrained(..., torch_compile=False)`.
4. NaN đầu chuỗi bị `strip_leading_nans` + `linear_interpolation` per input → tự đảm bảo context không NaN, đừng dựa vào cơ chế này.

## 4. Pipeline tối thiểu khớp harness (TFM-POINT)

```python
import numpy as np, timesfm
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", revision=SHA, torch_compile=True)
m.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=128, normalize_inputs=True,
        per_core_batch_size=256, force_flip_invariance=True,
        infer_is_positive=False, return_backcast=False))

ctxs = [r1[t-511 : t+1] for t in idx_pred]        # r1[s] = log(C_s / C_{s-1}); chỉ tau <= t
point, quant = m.forecast(horizon=3, inputs=list(ctxs))
r_hat  = quant[..., 0]                            # (n,3) mean head; point = q50
yhat   = np.cumsum(r_hat, axis=1)                 # yhat[:, h-1] = y_h  (plan §0)
denom  = np.maximum(rv60[idx_pred], tf.volatility_floor)[:, None] * np.sqrt([1., 2., 3.])
pred_z = ((yhat / denom) - tf.mean) / tf.scale    # nghịch đảo TargetTransform
```

- Không train; không dùng `X_fit / z_fit / X_es`; `best_iters = (0, 0, 0)`.
- **Adapter phải trả `pred_z`** vì `harness.run_config` gọi `transform.decode(res.pred_z, rv60[idx_val])`; công thức trên là nghịch đảo đúng của `src/p0/transform.py` (test round-trip theo §6.7).
- Interface `fit_predict(X_fit, z_fit, X_es, z_es, X_pred, rounds, seed)` hiện **không mang chuỗi r1, timestamp, rv60, TargetTransform** → cần batch-object giống `SeqBatch` của LSTM (`harness.run_config` đã có nhánh `is_seq`). Đề xuất tối thiểu: thêm nhánh dựng `TFMBatch(r1_series, idx, rv60, transform)`; **không đổi interface của tree model**.

## 5. Covariate (plan §2.2 #4b) — CÓ API, ràng buộc nặng

**Có `forecast_with_covariates` trong 2.0.2** (chỉ TimesFM 2.5; example trong repo ghi rõ "TimesFM 1.0 does NOT support forecast_with_covariates(); that requires TimesFM 2.5 + pip install timesfm[xreg]"). Cơ chế: hồi quy tuyến tính in-context (`utils/xreg_lib.BatchedInContextXRegLinear`) ghép với TimesFM theo `xreg_mode` thuộc {"xreg + timesfm", "timesfm + xreg"}.

| Câu hỏi | Trả lời (source 2.0.2) |
|---|---|
| Phân loại covariate | `dynamic_numerical` / `dynamic_categorical` (dict tên → list theo series → mảng theo thời gian); `static_numerical` / `static_categorical` (dict tên → 1 giá trị mỗi series) |
| Dynamic có **bắt buộc giá trị tương lai**? | **CÓ.** `test_len = len(dyn_cov[i]) - len(input_i)`; mảng covariate phải dài `context + horizon`; phần `[input_len:]` là giá trị của horizon |
| Xung đột với "chỉ tau <= t"? | **Không** — plan đã quy định "giá trị cho 3 bước dự báo = giữ giá trị tại t"; giữ giá trị tại t là causal → **KHÔNG BLOCKED** |
| Ràng buộc khác | mọi series trong batch phải cùng bộ tên covariate; `return_backcast=True` bắt buộc (guard); mode "timesfm + xreg" dùng `self.model.p = 32`; cần `jax` + `scikit-learn` |

**RỦI RO LEAKAGE — điểm quan trọng nhất**: `create_covariate_matrix()` **nối phẳng (unnest) toàn bộ batch thành MỘT ma trận thiết kế và fit MỘT `beta_hat` chung cho cả batch** (`beta_hat = pinv(X'X + ridge*I) X' y`). Nếu gộp nhiều origin vào một lời gọi thì dự báo tại origin `t_i` phụ thuộc context của origin `t_j > t_i` → **vi phạm §6.4 (chỉ dùng tau <= t)**. Kết luận: khi chạy covariate, **mỗi lời gọi chỉ được chứa 1 origin** (`inputs` đúng 1 phần tử). Batch nhiều origin **chỉ hợp lệ cho TFM-POINT** (`forecast()` không có bước fit chung nào giữa các series).

**Căn thời gian của covariate** (plan §2.2: "giá trị dùng để dự báo bar s chỉ được tính từ dữ liệu <= s-1"): xreg khớp cặp *cùng chỉ số thời gian* (`x_train[s]` ↔ `target[s]`). Feature ext tại bar s dùng cả `C_s`, nên phải **dịch 1 bar**: vị trí s của mảng covariate mang giá trị `f(s-1)`; ba vị trí tương lai (t+1..t+3) mang `f(t)`. Không dịch = model học quan hệ đồng thời `f(s) → r1_s` trong đó `f(s)` chứa `C_s`, tức chứa chính `r1_s` → **leakage rõ ràng**, phải chặn trong code và test §6.4.

## 6. LoRA (plan §2.2 #4c)

- **Package `timesfm` KHÔNG có API finetune/LoRA** (wheel 2.0.2 không có module nào như vậy).
- Đường chính thức: repo có `timesfm-forecasting/examples/finetuning/finetune_lora.py` + README → dùng **HuggingFace `transformers` + `peft`**, KHÔNG dùng package `timesfm`:

```python
from transformers import TimesFm2_5ModelForPrediction  # transformers: có ở v5.4.0, KHÔNG có ở v5.2.0; latest 5.16.1
from peft import LoraConfig, get_peft_model            # peft latest 0.20.0
model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers", torch_dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                                         lora_dropout=0.05, bias="none"))
out  = model(past_values=ctx, future_values=tgt, forecast_context_len=512)
loss = out.loss                # loss nội bộ của TimesFM
yhat = out.mean_predictions    # dùng để tự tính Huber theo plan
```

- Plan yêu cầu "Huber trên `r_hat_{t+1..t+3}`" → **không dùng `out.loss`**; tự tính Huber trên `out.mean_predictions[:, :3]`. Khả thi vì example dùng vòng lặp PyTorch thuần.
- "rank 8 trên attention/FF" ↔ `target_modules="all-linear"` (README: r=4 ≈ 0.6% param, ~1.4M/232M).
- **Cảnh báo so sánh**: TFM-POINT chạy qua package `timesfm`, TFM-LoRA qua `transformers` → hai code path decode khác nhau. Nếu chạy LoRA thì **phải đo lại zero-shot bằng chính path `transformers`** để Gain chỉ phản ánh LoRA (§3). Chi phí thêm: 1 run zero-shot.
- LoRA kéo theo 3 package mới (`transformers`, `peft`, `accelerate`) + checkpoint thứ hai 925 MB. Chỉ cài khi TFM-POINT/covariate **thắng E0** (đúng điều kiện plan §2.2 #4c).

## 7. Khả thi theo plan §2.2 #4

| Mục plan | Verdict | Căn cứ |
|---|---|---|
| (a) TFM-POINT zero-shot, context 512, batch nhiều origin | **KHẢ THI** | `forecast(horizon, inputs=list_of_windows)` batch sẵn; context 512 hợp lệ |
| Tắt tùy chọn ép dương | **KHẢ THI** | `ForecastConfig(infer_is_positive=False)` |
| Cộng dồn r_hat → `y_h` → giá | **KHẢ THI** | output `(B, 3)`; `pred_z` là nghịch đảo TargetTransform |
| (b) covariate loop qua API | **KHẢ THI CÓ ĐIỀU KIỆN** | API có; nhưng **batch = 1 origin/lời gọi** (beta chung → leakage), phải **dịch covariate 1 bar**, cần `jax` + `sklearn`, `return_backcast=True` |
| covariate "chỉ đến t" (không cần giá trị tương lai) | **KHÔNG** — API bắt buộc phủ horizon; giải bằng giữ giá trị tại t (đúng như plan đã ghi) → không BLOCKED |
| (c) TFM-LoRA rank 8 | **KHẢ THI** nhưng **không qua package `timesfm`** — qua `transformers>=5.4` + `peft`, checkpoint khác |
| Checkpoint 1.0-200m / 2.0-500m | **KHÔNG KHẢ THI** trên Python 3.12 (timesfm 1.x pin `<3.12`) |
| API 1.x (`TimesFm(...)`, `forecast_on_df`) | **KHÔNG TỒN TẠI** trong 2.0.2 |

## 8. Chi phí (ƯỚC LƯỢNG, chưa đo)

| Việc | Ước lượng trên 3090 | Ghi chú |
|---|---|---|
| Tải checkpoint | 925 MB, một lần | cache HF |
| TFM-POINT 7.185 origin (5 fold VAL) | **~0.5–2 phút/run** | 29 lô x 256, mỗi lô 2 forward (flip), 16 token/series |
| TFM-POINT trên TEST 2.728 origin | < 1 phút | |
| Covariate loop (base + 39 candidate) | **~2–6 h** (15–40 ms/origin x 7.185 x 40 run) | batch 1 bắt buộc → overhead chi phối; **đo 1 fold trước khi cam kết** |
| LoRA (nếu chạy) | 1–3 h | plan ước 1–2 h |

Plan ước "TFM-POINT vài phút; loop covariate ≈ 1–1.5 h" → **loop covariate nhiều khả năng gấp 2–4 lần plan** vì ràng buộc batch = 1.

## 9. Rủi ro cài đặt

1. **`timesfm[xreg]` kéo `jax[cuda]`** — cùng máy với torch 2.11+cu128 nghĩa là hai bộ wheel `nvidia-*`, rủi ro xung đột cao. **Đề xuất: cài `jax` bản CPU + `scikit-learn` và gọi `forecast_with_covariates(..., force_on_cpu=True)`**; phép giải chỉ là `pinv` của ma trận (512 x k) nên CPU thừa sức. Đây là bước *inference* của thư viện (không lưu tham số học được) → nhất quán với §0 "predict của thư viện mặc định chạy CPU", nhưng **cần main-controller xác nhận** vì trong code có chữ `fit`.
2. `huggingface_hub[cli]` + `safetensors` sẽ được cài kèm (chưa có trong env local).
3. `torch.compile` mặc định bật → lần gọi đầu tốn vài chục giây; nếu instance thiếu triton thì `torch_compile=False`.
4. Local RTX 3050 Ti (4 GB) đủ để smoke 1–2 origin fp32 (~1 GB weight) nhưng **không** dùng chạy thật (plan §0: chạy thật trên Vast).

## 10. Điểm chưa xác minh

- Chưa chạy dòng code nào của timesfm (chưa cài, chưa tải checkpoint) → mọi kết luận là từ đọc source, **chưa có bằng chứng runtime**.
- Chưa đo tốc độ thật; các con số ở §8 là ước lượng từ FLOPs/overhead.
- Chưa xác minh `transformers` phiên bản nhỏ nhất chính xác có `timesfm2_5` (đã kiểm: có ở v5.4.0, không có ở v5.2.0; chưa kiểm v5.3.x).
- Chất lượng dự báo: chưa biết. Rủi ro đã lường: với input là r1 (gần như nhiễu trắng, mean ~0) cộng `force_flip_invariance`, dự báo có thể ~0 và TFM-POINT ≈ E0 → MedianGain vs E0 ≈ 0 → không kích hoạt LoRA. Đây là kết quả hợp lệ theo plan (§2.4), không phải bug. Nếu user muốn thử input là `log C` thay vì r1 thì đó là **đổi thiết kế §2.2 #4**, phải do user/main-controller quyết, researcher không tự đổi.

## 11. Việc kế tiếp

- **coder**: implement TFM-POINT (thay `models_pending.pending("tfm")`) theo §4: batch-object kiểu `SeqBatch`, truyền copy của `inputs`, dùng `quant[..., 0]`, trả `pred_z`. **Chưa** implement covariate cho tới khi main-controller chốt (jax CPU + batch 1 + dịch 1 bar).
- **checker**: test §6.4 cho TFM — chuỗi cắt tại t và chuỗi đầy đủ phải cho prediction giống hệt; test round-trip `pred_z` → `decode` → `y_h`; nếu làm covariate thì assert `len(inputs) == 1`.
- **infra**: pin `timesfm[torch]==2.0.2` + revision checkpoint trong `requirements.txt`; **chưa** thêm jax/transformers/peft cho tới khi cần.

---

# §12 — Chiến lược baseline covariate (audit 2026-08-31)

Ngày 2026-08-31 · researcher · **vẫn chưa cài timesfm, chưa tải checkpoint, chưa chạy dòng code timesfm nào.**
Phương pháp: đọc source thật của wheel `timesfm-2.0.2-py3-none-any.whl` (`pip download --no-deps` vào thư mục tạm, KHÔNG cài) — `timesfm/utils/xreg_lib.py` (521 dòng) và `timesfm/timesfm_2p5/timesfm_2p5_base.py` (424 dòng), `timesfm_2p5_torch.py`, `configs.py`; JAX docs cho `jnp.linalg.pinv`; GitHub `google-research/timesfm@master` cho example. Số đo runtime = **numpy tái hiện đúng đại số của `xreg_lib`** trên máy local (Intel i7-12xxx, numpy 2.4.4 + scipy-openblas) + **proxy torch** cho forward pass (không phải trọng số TimesFM). Mọi claim gắn timesfm **2.0.2**.

## 12.1 Cơ chế xreg — sự thật từ source

**Q1. Ma trận thiết kế.** `create_covariate_matrix` (xreg_lib 327–405):

| Thành phần | Số cột | Dòng |
|---|---|---|
| mỗi dynamic **numerical** covariate | **đúng 1 cột** (`_unnest(...)[:, np.newaxis]`), không nội suy, không mở rộng | 357–363 |
| mỗi static numerical | 1 cột (lặp theo `train_lens`) | 365–367 |
| mỗi categorical | one-hot (nhiều cột) | 385–396 |
| intercept | **+1 cột toàn 1.0** (`use_intercept=True` mặc định, `forecast_with_covariates` KHÔNG override) | 401–403 |

Số **hàng** `x_train` = `sum(train_lens)`, với `train_lens` do base quyết định (base 262–274):
- `xreg_mode="xreg + timesfm"` → `train_lens = [input_len]` = **512** (dòng 272)
- `xreg_mode="timesfm + xreg"` → `train_lens = [input_len − self.model.p]` = 512 − 32 = **480** (dòng 270; `input_patch_len=32`)

Số hàng `x_test` = `sum(test_lens)` = `len(mảng covariate) − input_len` = **3** (base 277–279). Vậy với k dynamic numerical, context 512, horizon 3, **1 origin**: `x_train` = **(512, k+1)**, `x_test` = **(3, k+1)**. Hàng = số điểm context, **không** phải context + horizon.

**Padding lên lũy thừa 2 (không có trong tài liệu).** `_to_padded_jax_array` (46–57) pad **cả hai chiều** lên `2**ceil(log2(n))` bằng **số 0**, áp cho `x_train`, `x_train_raw`, `flat_targets`, `x_test` (488–491). Cột: k=39 → 40 → **P=64**; k=100 → 101 → **128**; k=150 → 151 → **256**; k=306 → 307 → **512**. Cột 0 không làm sai nghiệm (pinv min-norm cho β=0 ở cột đó, `x_test` cũng 0 ở cột đó) nhưng **quyết định chi phí O(P³) và ngưỡng cắt số học của pinv** (xem Q2).

Chuẩn hoá: cột được z-hoá bằng mean/std **của context** rồi áp cùng stat cho `x_test` (374–377) → causal. Target chuẩn hoá per-input (base 379) rồi `renormalize` ở cuối (base 417–421) → causal.

**Q2. Chính quy hoá.** `ridge` mặc định **0.0** ở cả API công khai (base 209) và `fit()` (xreg_lib 415). Nghiệm (xreg_lib 492–499):

```python
beta_hat = jnp.linalg.pinv(x_train.T @ x_train + ridge * jnp.eye(x_train.shape[1]),
                           hermitian=True) @ x_train.T @ flat_targets
```

- Không có guard nào về `k` vs số hàng, không cảnh báo, không assert. `_assert_covariates` chỉ kiểm tra khớp key và khớp độ dài.
- Khi k gần/vượt số hàng → `pinv` cho nghiệm **bình phương tối thiểu chuẩn nhỏ nhất (min-norm)**, tức nội suy 512 điểm huấn luyện khi đủ hạng.
- **JAX chạy float32 mặc định** (`jax_enable_x64=False` → mảng float64 bị hạ xuống float32 khi vào `jnp`). Giải qua **phương trình chuẩn** `X'X` → bình phương số điều kiện; ở float32 đây là điểm yếu số học thật.
- `jnp.linalg.pinv` mặc định `rtol = 10 * max(rows, cols) * eps` (JAX docs). Với `X'X` là (P, P) float32: **rtol = 10·P·1.19e-7** → 7.6e-5 (P=64), 1.5e-4 (128), 3.05e-4 (256), **6.1e-4 (512)**. Trị riêng dưới `rtol·λ_max` bị cắt ⇒ **PCA-truncation ngầm**, cường độ phụ thuộc float32 **và phụ thuộc P tức phụ thuộc padding**. Hệ quả methodology: rtol **nhảy 2×** mỗi lần k+1 vượt một lũy thừa 2 → so sánh KEEP/DROP hai bên biên (k+1 = 64/65, 128/129, 256/257) trộn "feature có ích" với "mức chính quy hoá đổi". Đo được: rank hiệu dụng trên thiết kế cộng tuyến kiểu chỉ báo kỹ thuật = **27/40 (k=39), 66/101 (k=100), 79/151 (k=150), 71/307 (k=306)** — ở k=306 thư viện **âm thầm bỏ ~3/4 số hướng covariate**, và rank hiệu dụng còn **giảm** khi k tăng vì rtol tăng.
- `one_hot_encoder_drop = None if ridge > 0 else "first"` (base 349/392) — chỉ ảnh hưởng categorical.

**Q3. `max_rows_per_col`** (xreg_lib 420, 467–477): nghĩa là **cắt số HÀNG**, không phải cột. Nếu khác 0 và `nrows > ncols * max_rows_per_col` thì lấy mẫu ngẫu nhiên (jax PRNGKey **42** cố định, `replace=False`) đúng `ncols * max_rows_per_col` hàng. `0` = **tắt subsample**. Đây là knob **tăng tốc**, làm tỉ lệ n/p **tệ đi**, **không** phải cơ chế chặn under/overdetermined. Giá trị 0 hiện tại của ta là đúng. (Phụ: subsample chỉ áp cho `beta_hat`; `y_hat_context` vẫn dùng toàn bộ hàng — xreg_lib 466, 501.)

**Q4. Giới hạn số covariate.** **Không có** hard limit, assert hay cảnh báo nào theo k. Example chính thức dùng số covariate **một chữ số**: `v1/notebooks/covariates.ipynb` = 1 dyn-numerical + 1 dyn-categorical + 1 static-categorical; `timesfm-forecasting/examples/covariates-forecasting/demo_covariates.py` (418–423) = 1 dyn-numerical (`price`) + 1–3 dyn-categorical + 1–2 static-categorical. **Không có example/tài liệu nào dùng hàng chục, càng không hàng trăm covariate.**

**Q5. `xreg_mode`.** Source (docstring base 224–227 + code 322–421):

| mode | thứ tự | train_len | ghi chú |
|---|---|---|---|
| `"xreg + timesfm"` (mặc định, code ta đang dùng) | fit tuyến tính **trên chính target context** trước → TimesFM dự báo **phần dư** `target − xreg_on_context` (base 399–405) → cộng lại | 512 | dùng toàn bộ context |
| `"timesfm + xreg"` | TimesFM chạy trước (cần backcast) → fit tuyến tính trên **phần dư của backcast** (base 327–333) | 480 | mất 32 hàng (1 patch), thêm nhiễu từ backcast |

**Cảnh báo**: `explain_xreg_modes()` trong `demo_covariates.py` (431–447) mô tả **NGƯỢC** với source, và file đó còn dùng API 1.x (`TimesFmHparams`) không tồn tại trong 2.0.2 → **example đã cũ, tin source**. Với target r1 (gần nhiễu trắng, mean≈0): backcast của TimesFM gần như không giải thích được gì nên hai mode gần tương đương, nhưng `"xreg + timesfm"` giữ đủ 512 hàng, không phụ thuộc backcast, và diễn giải đúng ("hồi quy dự báo r1 theo feature, TimesFM lo phần còn lại") → **giữ `"xreg + timesfm"`**. Không mode nào sửa được vấn đề phương sai ở Q6.

**Căn hàng thực tế** (khớp `covariate_window` trong `src/p0/models_tfm.py`): base cắt train covariate = `covariate_value[input_len − train_len : input_len]` = vị trí 0..511 = `f(t−512)…f(t−1)`, target = `r1[t−511…t]` ⇒ hàng s ghép `r1_s ~ f(s−1)`. Đúng là **hồi quy dự báo một bước trên 512 bar gần nhất**, causal. Ba hàng test mang `f(t)` (giữ giá trị tại t theo plan) ⇒ **đóng góp xreg là một hằng số c cho cả 3 bước** ⇒ vào `y_h = Σ r̂` thành **h·c**.

**Q8. Batch nhiều origin mà mỗi origin có beta riêng: KHÔNG.** Ba lý do trong source: (1) `_unnest` (36–37) nối phẳng toàn batch thành MỘT ma trận và `fit()` giải MỘT `beta_hat` (492–499) — đã ghi ở §5; (2) mẹo "block-diagonal" (đặt tên covariate riêng cho từng origin, các origin khác điền 0) **không** tách được vì cột intercept toàn 1.0 trên mọi hàng (401–403) ghép các origin lại, và chuẩn hoá cột dùng mean/std trên **toàn bộ hàng** (374–377) nên cột có block-0 sau khi trừ mean sẽ khác 0 ở block khác → phá cấu trúc block; (3) chi phí: B origin × k cột → padding lũy thừa 2 (vd B=64, k=39 → 2496 → **4096** cột) → eigh 4096³ đắt hơn 64 lần giải 64³ khoảng ba bậc độ lớn. **Xác nhận lại ràng buộc 1 origin/lời gọi.**

**Q9. GPU / CPU.**
- Forward pass = torch trên CUDA (`_compiled_decode`, `timesfm_2p5_torch.py` 420–470). `force_flip_invariance=True` ⇒ **2 lần `self.model.decode`** (457–459).
- Toàn bộ xreg là **numpy + jax**: dựng ma trận, unnest, chuẩn hoá, one-hot = numpy/sklearn (thuần CPU, không có tuỳ chọn GPU); phần giải (`pad`, `X'X`, `pinv`, matmul) = jax, chạy trên `jax.default_device(cpu)` nếu `force_on_cpu=True` (479, 487), ngược lại trên backend jax mặc định (là CPU nếu chỉ cài `jax` bản CPU; là GPU nếu cài `jax[cuda]`).
- **Sự thật kỹ thuật cho invariant "training chỉ GPU"** (user quyết, researcher không tự quyết): bước xreg ước lượng `beta_hat` bằng **nghiệm đóng (closed-form least squares)** từ đúng cửa sổ context của **một origin** (τ ≤ t), **không gradient, không iteration, không hyperparameter fitting**, và **beta bị vứt sau mỗi lời gọi** — không có artifact học được nào tồn tại giữa các origin hay được tái dùng ở TEST. Về vòng đời tham số nó thuộc nhóm "in-context inference", giống `fit_data` mỗi lời gọi của AutoTS mà plan §7.4 đã chấp nhận ("AutoTS pipeline CPU quanh regression_model GPU"). Nhưng nó **đúng là ước lượng tham số từ dữ liệu** và hàm tên `fit`. Hai lựa chọn: (a) `force_on_cpu=True` + `jax` bản CPU — không đụng nvidia wheel, rủi ro cài đặt thấp; (b) `force_on_cpu=False` + `jax[cuda]` để phần này cũng chạy GPU — kéo bộ wheel `nvidia-*` thứ hai cạnh torch cu128, rủi ro xung đột cao (§9.1) và **không đổi kết quả số học** (vẫn float32).

## 12.2 Q6 — Ý nghĩa thống kê: đây là tiêu chí loại

n = **512 hàng**, p = k + 1 (kể cả intercept). Tỉ lệ n/p: **12.8** (k=39), **5.07** (k=100), **3.39** (k=150), **1.67** (k=306).

Vì đóng góp xreg là hằng số c cho cả 3 bước, sai số ước lượng vào `y_h` theo **h·(ĉ − c)**, còn benchmark E0 có RMSE ≈ σ√h. Do đó **khi feature không có tín hiệu**, tỉ lệ RMSE so với E0 là `sqrt(1 + h·L)` với `L = x₀ᵀ(XᵀX)⁺x₀` (leverage của điểm dự báo). OLS đủ hạng: `E[L] ≈ p/(n−p−1)`.

Đo bằng chính đại số của thư viện (float32, padding lũy thừa 2, cutoff pinv của JAX; 60 cửa sổ/điểm; thiết kế "tech" = cột cộng tuyến kiểu rolling mean/rms/momentum/lag sinh từ một random walk, "iid" = Gaussian độc lập):

| k | P | rank hiệu dụng (tech) | L (iid) | L (tech) | RMSE/E0 h=1 (iid / tech) | h=3 (iid / tech) |
|---|---|---|---|---|---|---|
| 1 | 2 | 2.0 | 0.0039 | 0.0041 | 1.002 / 1.002 | 1.006 / 1.006 |
| 5 | 8 | 5.0 | 0.0134 | 0.0082 | 1.007 / 1.004 | 1.020 / 1.012 |
| 10 | 16 | 8.0 | 0.0225 | 0.0158 | 1.011 / 1.008 | 1.033 / 1.023 |
| 20 | 32 | 15.0 | 0.0439 | 0.0333 | 1.022 / 1.016 | 1.064 / 1.049 |
| **39** | 64 | 27.0 | 0.0867 | 0.0591 | **1.042 / 1.029** | **1.123 / 1.085** |
| **100** | 128 | 65.9 | 0.2462 | 0.2171 | **1.116 / 1.103** | **1.319 / 1.285** |
| **150** | 256 | 79.0 | 0.4119 | 0.2438 | **1.188 / 1.115** | **1.495 / 1.316** |
| 200 | 256 | 77.8 | 0.6412 | 0.2430 | 1.281 / 1.115 | 1.710 / 1.315 |
| **306** | 512 | 71.3 | 1.5144 | 0.2086 | **1.586 / 1.099** | **2.354 / 1.275** |

(iid khớp lý thuyết `p/(n−p−1)`: k=306 → L = 1.514 đo vs 1.505 lý thuyết ⇒ simulation đúng. Monte-Carlo độc lập với target nhiễu trắng, 300 lần lặp, cho cùng kết luận: k=306 iid → 1.611 / 2.121 / 2.470 ở h=1/2/3.)

**Đọc bảng.**
1. Mốc so sánh: MEMORY (Pitfalls, từ autocorr lag-1 ≈ −0.06 trên snapshot) — tín hiệu điểm 1 phút **thật** đáng giá ~**0.1–0.2 pp** RMSE ở h=1 và ~**0.03 pp** ở h=3, tức Gain ~0.001–0.002.
2. Chi phí nhiễu ước lượng ở k=39 là **+2.9 → +4.2 pp** (h=1) và **+8.5 → +12.3 pp** (h=3) — lớn hơn tín hiệu khả dĩ **15–60 lần**. Ở k=100–306 là **+10 → +59 pp** (h=1), **+28 → +135 pp** (h=3) — lớn hơn **100–500 lần**.
3. **Điểm hoà vốn**: một cột thêm chỉ có lãi khi R² trong cửa sổ 512 bar > ≈ h·(1/n) → **> 0.2% (h=1), > 0.6% (h=3)**, tức |corr| > 4.4% / 7.6%. Hợp lý cho 1–2 feature thật tốt, **không** hợp lý cho 39/100/306 cột cùng lúc: chi phí cộng dồn tuyến tính theo p, tín hiệu thì không.
4. Với thiết kế cộng tuyến thực tế, **k ≥ 100 không thêm thông tin**: rank hiệu dụng đứng yên ở ~66–79 rồi **giảm** (79 → 77.8 → 71.3 khi k = 150 → 200 → 306) vì rtol tăng theo P. Ở k=306, **236/307 hướng covariate bị vứt trong im lặng** ⇒ vòng lặp add-one có thể thêm một cột mà fit **không đổi gì** (cột rơi dưới ngưỡng cắt) — KEEP/DROP khi đó đo **ngưỡng số học float32**, không đo feature. Đây là lỗi methodology, không phải chuyện tốc độ.

## 12.3 Q7 + Q10 — Runtime, memory

Đo trên **máy local** (Intel i7-12xxx, numpy 2.4.4 + scipy-openblas, float32), tái hiện đúng đại số của `xreg_lib`; min/median trên 25 lần:

| k | P | `create_covariate_matrix` (ms) | giải pinv (ms) | ghi chú |
|---|---|---|---|---|
| 39 | 64 | 1.47 / 1.57 | 0.58 / 0.79 | |
| 100 | 128 | 3.30 / 3.45 | 2.98 / 3.45 | |
| 150 | 256 | 5.43 / 5.76 | 10.6 / 13.2 | |
| 200 | 256 | 11.0 / 21.3 | 44.7 / 60.0 | cùng P nhưng nội dung ma trận khác → eigh chậm hơn |
| 306 | 512 | 19.2 / 30.2 | 281 / 343 | eigh 512³ float32 chi phối |

Chi phí giải ∝ **P³** với P = 2^⌈log2(k+1)⌉ (padding), **không** ∝ k³ — nhảy bậc ở k+1 = 65, 129, 257. Phần dựng ma trận là **Python thuần** (`itertools.chain` trong `_unnest`, xreg_lib 36–37), không tránh được qua API công khai.

**Forward pass batch 1** — không đo được thật (chưa cài timesfm, chưa tải checkpoint). Proxy: stack torch **cùng shape** (20 layer, d=1280, hidden 1280, 16 head, 16 token, output proj 1280→10240; 216.4M tham số) trên RTX 3050 Ti Laptop, 2 decode (flip invariance): **batch 1 = 45.2 ms**, batch 8 = 43.0 ms, batch 64 = 237 ms, **batch 256 = 942 ms**. Quy đổi sang 3090 theo băng thông (936 vs ~192 GB/s; vùng batch 1 là memory/launch-bound): **≈ 5–15 ms/origin**, lấy **10 ms** làm giá trị trung tâm; cộng ~3 ms overhead python của `forecast()` + dispatch/transfer jax.

**Chi phí một run VAL (5 fold × 1437 = 7185 origin) và một vòng lặp 40 run (base + 39 candidate):**

| Chiến lược | k điển hình | ms/origin | phút/run | **giờ cho 40 run** |
|---|---|---|---|---|
| **3** — chỉ ext §2.3 | 1 → 39 (P ≤ 64) | ~15 | 1.8 | **1.2 h** (0.9–1.6 h theo dải forward) |
| **2** — subset B0* 100 | 100 → 139 (P=128) | ~20 | 2.4 | **1.6 h** (1.4–2.1 h) |
| **2** — subset B0* 150 | 150 → 189 (P=256) | ~32–95 | 3.8–11 | **2.6–7.6 h** |
| **1** — toàn bộ B0* (=306) | 306 → 345 (P=512) | ~313–386 | 37–46 | **25–31 h** |

Ba lưu ý bắt buộc:
1. Nếu `|B0*|` sau §1.4 rơi vào 200–256 thì strategy 1 ≈ 5.6–7.6 h; nếu vẫn 306 thì ≈ 25–31 h. **Chi phí của strategy 1 chưa biết được cho tới khi §1.4 chạy xong** → không thể freeze trước run như user yêu cầu.
2. Toàn bộ phần đắt (eigh) là **CPU đơn luồng trong khi GPU đứng chờ** → xung đột trực tiếp với §0 "Vast tính giờ … không idle".
3. **Bug chi phí trong code hiện tại** (xem 12.5): `per_core_batch_size=256` dùng chung cho path covariate ⇒ mỗi lời gọi 1 origin bị `forecast()` pad lên đủ 256 series (base 167–168) ⇒ trả tiền forward batch 256 cho mỗi origin. Theo proxy, thêm ~150–250 ms/origin trên 3090 ⇒ **+12–20 h cho 40 run, với mọi strategy**.

**Q10 (strategy 3 cụ thể).** Vòng lặp add-one làm k đi từ 0 (TFM-POINT) → 1 → … ≤ 39; P ≤ 64 nên phần xreg luôn < 2.5 ms/origin. Tính hợp lệ thống kê ở k=39 (trường hợp xấu nhất, giữ hết): +2.9…4.2 pp RMSE ở h=1, +8.5…12.3 pp ở h=3 khi không có tín hiệu. Nhưng **luật §2.1 tự giới hạn**: TimesFM zero-shot deterministic ⇒ ε_TFM ≈ 0 ⇒ KEEP đòi MedianGain ≥ 0 ⇒ candidate vô dụng bị DROP ngay ở bước thêm nó (chi phí +0.2…0.6 pp đã lớn hơn 0). Kỳ vọng thực tế: k cuối cùng nhỏ (0–5 cột), chi phí ≤ +1 pp. Đây là khác biệt lớn nhất so với strategy 1/2: ở S1/S2 **toàn bộ 100–306 cột được nạp vô điều kiện, không cột nào phải qua cửa KEEP/DROP**.

## 12.4 Bảng so sánh 3 strategy + khuyến nghị

| Tiêu chí | **S1: toàn bộ B0\*** (k≈306) | **S2: subset B0\*** (k≈100–150) | **S3: chỉ 39 ext §2.3, add-one** |
|---|---|---|---|
| **Correctness / thống kê** | **HỎNG.** n/p = 1.67; nhiễu ước lượng +10…59 pp (h=1), +28…135 pp (h=3) khi không có tín hiệu; rank hiệu dụng 71/307 → 236 hướng bị float32 vứt im lặng; KEEP/DROP đo ngưỡng số học, không đo feature | **XẤU.** n/p = 3.4–5.1; +10…19 pp (h=1), +28…50 pp (h=3); rank hiệu dụng bão hoà ~66–79; rtol nhảy 2× khi k+1 vượt 128/256 → biên so sánh không đồng nhất | **CHẤP NHẬN ĐƯỢC.** n/p ≥ 12.8; +0.2 pp (k=1) → +4.2 pp (k=39, xấu nhất); mọi cột phải qua cửa MedianGain ≥ −ε; điểm hoà vốn R² > 0.2%/0.6% nêu rõ được trong báo cáo |
| **Leakage safety** | Như nhau cho cả ba: 1 origin/lời gọi + dịch 1 bar là bắt buộc và đủ (§5, §12.1 Q8). S1/S2 có **thêm** rủi ro vận hành: 306 cột × 515 điểm/origin phải dịch và kiểm NaN đúng — bề mặt lỗi lớn hơn ~8× | như S1 | Bề mặt nhỏ nhất; test §6.4 đã có cho đúng các cột ext |
| **GPU-only** | Tệ nhất: ~343 ms eigh CPU vs ~10 ms forward GPU ⇒ **~97% thời gian GPU idle** | ~3–60 ms CPU vs 10 ms GPU ⇒ 25–86% idle | ~2 ms CPU vs 10 ms GPU ⇒ GPU chi phối, khớp tinh thần §0 |
| **Runtime thực tế (40 run)** | **25–31 h** (chưa chốt được, phụ thuộc \|B0\*\|) | 1.6–7.6 h | **1.2 h** |
| **Khớp plan** | **Đổi thiết kế** — trái §2.2 #4b và §2.1 dòng 186 | **Đổi thiết kế** — như S1, cộng thêm quy tắc chọn subset mới (không có trong plan) | **Đúng plan hiện hành**, không cần sửa gì |

**KHUYẾN NGHỊ: chọn S3.** Ba lý do theo thứ tự quan trọng:
1. **Thống kê**: hồi quy in-context chỉ có **512 hàng**. Ở k ≥ 100, phương sai ước lượng làm RMSE xấu đi 10–135 pp trong khi tín hiệu 1 phút khả dĩ chỉ 0.1–0.2 pp — S1/S2 **được đảm bảo thua E0** trước khi bàn TimesFM tốt hay dở, tức chúng không trả lời được câu hỏi nào cả (§0: mỗi run phải trả lời một câu hỏi).
2. **Đo cái gì**: ở k lớn, cutoff float32 của `pinv` vứt phần lớn hướng covariate và ngưỡng cutoff còn đổi theo padding lũy thừa 2 ⇒ KEEP/DROP không còn là phép đo về feature.
3. **Plan**: §2.2 #4b ("§2.1 với candidate §2.3 làm covariate theo phút, **xuất phát không covariate**") và §2.1 dòng 186 ("TimesFM không có feature dạng cột: xuất phát không covariate, thử thêm lần lượt candidate §2.3") **đã** là S3. S1/S2 là mở rộng scope. Dòng 168 ("Input = B0\* + …, với LSTM/TimesFM-covariate là chuỗi theo phút của cùng cột") là câu tổng quát viết cho model dạng bảng và **mâu thuẫn với hai câu đặc thù TimesFM ở trên** → đề nghị main-controller làm rõ dòng 168 thành "(TimesFM: chỉ các cột ext đang KEEP + f, không có B0\*)". Đây là làm rõ mâu thuẫn nội tại, không phải đổi thiết kế.

S2 chỉ nên xét lại nếu (và chỉ nếu) canary cho thấy S3 **thắng E0 rõ** và user muốn thử thêm — khi đó bắt buộc kèm `ridge > 0` (hiện `ridge=0.0`), mà ridge là **tham số mới cần calibrate** ⇒ thêm scope, cần user quyết riêng.

## 12.5 Hai lỗi trong `src/p0/models_tfm.py` phải sửa trước khi chạy (độc lập với strategy)

1. **`per_core_batch_size` cho path covariate.** `_model(with_covariates=True)` vẫn compile với `per_core_batch_size=self.batch_size` (=256). `forecast()` pad `inputs` lên bội của `global_batch_size` (base 167–168) ⇒ mỗi origin chạy forward batch 256. Sửa: instance dùng cho covariate phải compile với `per_core_batch_size=1`. Ảnh hưởng đo được (proxy): 45 ms → 942 ms mỗi lời gọi trên 3050 Ti.
2. **Head sai ở path covariate.** `_with_covariates` lấy `res[0]` = `new_point_outputs` = **q50** (`decode_index=5`), trong khi `_point` dùng `quant[..., 0]` = **mean**. Metric là RMSE ⇒ phải dùng mean ở cả hai, nếu không Gain "TFM-cov vs TFM-POINT" lẫn cả "q50 vs mean" vào kết quả. Sửa: dùng `res[1]` (`new_quantile_outputs`, kênh 0 = mean) khi `use_mean_head=True`.

Ghi chú nhỏ (không phải lỗi): `forecast()` mutate list `inputs` tại chỗ — code ta luôn truyền list literal mới nên an toàn; hai instance model (POINT và covariate) ⇒ ~1.85 GB VRAM, vẫn thoải mái trên 3090.

## 12.6 Canary phải chạy trên GPU Vast TRƯỚC khi cam kết (đo, không đoán)

Điều kiện: user cho phép cài `timesfm[torch]==2.0.2` + `jax` (CPU) + `scikit-learn` và tải checkpoint. Canary = **1 fold VAL, ~200 origin liên tiếp**, chỉ VAL, không chạm TEST.

1. **Shape & rtol**: gọi `BatchedInContextXRegLinear(...).fit(debug_info=True)` cho 1 origin với k=1 và k=39; in `x_train.shape`, `x_test.shape`, dtype thật của mảng jax, số trị riêng vượt ngưỡng. Phải thấy **(512, 2^⌈log2(k+1)⌉)**, **(4, …)**, **float32**. Nếu dtype là float64 (có ai bật `jax_enable_x64`) thì §12.2 phải tính lại.
2. **Causality (§6.4)**: cắt chuỗi và covariate tại t rồi chạy lại → prediction phải **giống hệt**. Và assert `len(inputs) == 1` trong mọi lời gọi covariate.
3. **Dịch 1 bar**: dựng chuỗi giả với `f(s) = r1_s` (leak cố ý) và `f(s) = r1_{s−1}`; bản leak phải cho Gain lớn bất thường, bản đúng thì không. Test này bắt lỗi lệch bar tốt hơn mọi review bằng mắt.
4. **`assert np.isfinite(covariate_window(...)).all()`** cho origin VAL đầu tiên (cột ext lookback tới 1440 phút, cửa sổ covariate lùi thêm 512 bar ⇒ cần ≥ 1952 bar lịch sử; VAL sớm nhất cách đầu data ~12.2k bar nên an toàn, nhưng phải assert chứ không giả định).
5. **Thời gian thật/origin**: đo riêng (a) `forecast_with_covariates` tổng, (b) forward batch 1 qua `forecast()` không covariate, (c) hiệu = phần xreg. Xác nhận (b) ∈ 5–15 ms và (c) ≈ 2–3 ms ở k=39. Nếu (a) > 100 ms ở k=39 thì lỗi `per_core_batch_size` chưa được sửa.
6. **Head**: log cả `point` (q50) và `quantile[..., 0]` (mean) cho 200 origin; xác nhận adapter dùng mean ở **cả hai** path.
7. **Không tín hiệu = không hại**: chạy k=1 với một cột **nhiễu trắng thuần** làm covariate; RMSE so với TFM-POINT phải xấu đi ~0.2–0.6 pp đúng như §12.2 dự đoán. Lệch nhiều ⇒ mô hình phương sai ở §12.2 sai, phải đánh giá lại trước khi chạy vòng lặp.
8. **ε_TFM**: chạy TFM-POINT 3 lần (3 evaluation seed §1.3) và xác nhận ε_TFM ≈ 0 (zero-shot deterministic). Nếu > 0 đáng kể thì có nguồn phi-determinism (torch.compile / kernel) phải ghi nhận trước khi áp luật KEEP ≥ −ε.

Nếu canary #5 cho thấy (b) > 30 ms/origin trên 3090 thì tổng của S3 lên ~3 h — vẫn chấp nhận được; nhưng S1 sẽ lên > 35 h ⇒ kết luận không đổi.
