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
