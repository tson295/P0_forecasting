# AUDIT — TimesFM LoRA per fold + XReg frozen (quyết định user 2026-09-03)

Ngày: 2026-09-03 · researcher · **Không cài package, không tải checkpoint, không training.**
Phương pháp: `pip download timesfm==2.0.2 --no-deps --no-binary :all:` → sdist `timesfm-2.0.2.tar.gz`
(sha256 `b03885d3467b09314a4b461e1f6729375844061dd31fd41e3c35aa503c333da3`), đọc source; **smoke CPU/GPU local**
(torch 2.11.0+cu128, RTX 3050 Ti 4 GB) bằng cách import **đúng file** `timesfm_2p5/timesfm_2p5_torch.py` từ sdist
với stub `safetensors`/`huggingface_hub` (chỉ để import), module **random-init** (không checkpoint) — để đếm tham số,
liệt kê `named_modules`, kiểm tra grad, đo VRAM. PEFT: source tag `v0.20.0` + docs; transformers: `main` ngày 2026-09-03.
Số dòng dưới đây là của sdist 2.0.2: `T` = `timesfm/timesfm_2p5/timesfm_2p5_torch.py`, `B` = `.../timesfm_2p5_base.py`,
`X` = `timesfm/torch/transformer.py`, `D` = `timesfm/torch/dense.py`, `U` = `timesfm/torch/util.py`, `R` = `timesfm/utils/xreg_lib.py`.
Mọi claim gắn version; "unknown" = chưa xác minh.

## 1. Object model (Q1)

- `TimesFM_2p5_200M_torch.from_pretrained(...)` = `PyTorchModelHubMixin.from_pretrained` → `_from_pretrained` (T:304–364)
  → `instance = cls(config=config, **model_kwargs)` (T:357) → `__init__` tạo `self.model = TimesFM_2p5_200M_torch_module()` (T:282)
  → `instance.load_checkpoint(path, torch_compile=...)` (T:361–363). **Trả về wrapper `TimesFM_2p5_200M_torch` (không phải nn.Module);
  `nn.Module` nằm ở thuộc tính `.model`, class `TimesFM_2p5_200M_torch_module`** (T:36–256).
- Kiến trúc (T:45–69, hằng số T:45–54): `p=32, o=128, os=1024, m=4, x=20 layer, h=16 head, md=1280, hd=80, q=10, aridx=5`.
  `tokenizer = ResidualBlock(64→1280→1280, bias)` · `stacked_xf = ModuleList[20 × Transformer]` · `output_projection_point = ResidualBlock(1280→1280→1280, no bias)`
  · `output_projection_quantiles = ResidualBlock(1280→1280→10240, no bias)`. FF hidden = 1280 (không phải 4×).
- **Danh sách `nn.Linear` (đo bằng `named_modules()` trên class thật, 89 module):**
  - `tokenizer.hidden_layer (64→1280)`, `tokenizer.output_layer (1280→1280)`, `tokenizer.residual_layer (64→1280)` (D:29–43, có bias)
  - với mỗi `i ∈ 0..19`: `stacked_xf.{i}.attn.qkv_proj (1280→3840, fuse_qkv=True, X:199–200)`, `stacked_xf.{i}.attn.out (1280→1280, X:205)`,
    `stacked_xf.{i}.ff0 (1280→1280, X:335)`, `stacked_xf.{i}.ff1 (1280→1280, X:340)` — tất cả `bias=False` (B:112)
  - `output_projection_point.{hidden_layer,output_layer,residual_layer}` (1280→1280 ×3), `output_projection_quantiles.{hidden_layer (1280→1280), output_layer (1280→10240), residual_layer (1280→10240)}`
- **Không có projection nào là raw parameter / `F.linear`**: attention dùng `F.scaled_dot_product_attention(scale=1.0)` không tham số (X:132–151);
  tham số ngoài Linear chỉ là 140 tensor nhỏ: `RMSNorm.scale` (pre/post attn/ff ln 1280; `attn.query_ln/key_ln` 80) và `attn.per_dim_scale.per_dim_scale` (80) (X:154–166, 207–209, 314–331).
- **Tham số: 231.289.280** (99,95 % nằm trong `nn.Linear`) — "200M" là tên gọi; khớp 925,2 MB fp32 của checkpoint (audit_timesfm §2) và "~232M" trong README example.

## 2. Forward dùng cho training + tiền xử lý (Q2)

- `forward(inputs, masks, decode_caches=None)` (T:86–113): `inputs (B, N, 32)` float, `masks (B, N, 32)` bool (True = pad); ghép `cat([inputs, masks], -1)` → (B, N, 64) → tokenizer (T:92–93);
  attention nhận `masks[..., -1]` làm patch-mask (T:102); trả `((input_emb (B,N,1280), output_emb (B,N,1280), output_ts (B,N,1280), quantile_spread (B,N,10240)), caches)`.
  `output_ts` reshape `(B, N, 128, 10)` (T:171–174). Attention **causal theo patch** (`make_attn_mask` X:32–53: `q_index ≥ kv_index`) → output tại patch i = dự báo 128 điểm ngay sau patch i (decoder-only).
- **`decode()` bọc toàn bộ trong `torch.no_grad()` (T:118)** → không dùng được cho training; training phải **tái hiện** T:119–178 (prefill; với horizon 3, `num_decode_steps = (3−1)//128 = 0` (T:120) → không AR).
- Chuỗi tiền xử lý phải tái hiện **y hệt** để train = serve:
  1. `forecast()` B:175–183: `linear_interpolation(strip_leading_nans(x))`; dài ≥ `max_context` → lấy 512 cuối, mask toàn False (ta luôn đưa đúng 512, không NaN — `models_tfm.contexts`).
  2. `_compiled_decode` T:427–441: float32 lên device; `infer_is_positive` (T:433–436) tắt; `normalize_inputs=True`: `mu = mean(dim=-1)`, **`sigma = torch.std(dim=-1)` = unbiased (correction=1)**, `revin` (U:77–94; `sigma < 1e-6 → chia 1`).
  3. `decode` T:125–142: running stats **tích luỹ theo patch** (`update_running_stats` U:33–74, phương sai population, chỉ đếm điểm không mask) → `context_mu/sigma (B, N)`; T:166–167 `revin` từng patch + mask→0; T:168–170 `self(...)`; T:171–174 `revin(reverse)` với stats của **chính patch đó** (patch i chỉ dùng stats ≤ i → causal trong cửa sổ).
  4. T:448 `pf_outputs[:, -1, ...]` = (B, 128, 10) dự báo sau patch cuối; **flip** T:456–468: `decode(−inputs)`, `flip_quantile_fn` giữ kênh 0, đảo 1..9 (T:453–454), lấy `(f(x) − f(−x))/2`; T:477 cắt `[:, :horizon]`; `return_backcast` T:479–483 nối backcast `pf_outputs[:, :-1, :32, :]` phía trước (chỉ cần cho guard xreg, mode "xreg + timesfm" không dùng: B:407 lấy `[-max_horizon:]`);
     T:499–500 `revin(reverse)` với (mu, sigma) của bước 2; T:509–510 `.detach().cpu().numpy()` → `(full[..., 5], full)`.
- **Kênh 0 = mean (xác nhận trong 2.0.2):** `flip_quantile_fn` tách kênh 0 khỏi 1..9 (mean phản đối xứng dưới flip, quantile đảo thứ tự) (T:453–454); `fix_quantile_crossing` chỉ ép đơn điệu trên 1..9 quanh 5 (T:485–497); `use_continuous_quantile_head` chỉ đụng [1,2,3,4,6,7,8,9] neo kênh 5 (T:470–476); `decode_index=5` (B:95) = q50 cho AR và `point_forecast`. Ngoài package: model card `[mean, q10..q90]` (audit_timesfm §3); v1 finetune chính thức `predictions_mean = predictions[..., 0]` (`v1/src/finetuning/finetuning_torch.py`, GitHub master). **`quantile_forecast[..., 0]` hiện dùng là đúng**; loss LoRA phải đặt trên đúng kênh này.
- **Package 2.0.2 KHÔNG có finetune/LoRA/loss**: sdist chỉ có `configs.py`, `timesfm_2p5/*`, `torch/*`, `flax/*`, `utils/xreg_lib.py`, `tests/*` (grep `finetun|lora` trong `src/`: 0 kết quả). README (sdist README.md:32–33, 65) chỉ trỏ tới `timesfm-forecasting/examples/finetuning/` trên GitHub = **transformers + peft**, `loss = outputs.loss` nội bộ, checkpoint `...-transformers`, bf16, `LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0.05, bias="none")`, lr 1e-4, 10 epoch, batch 32, context 64, horizon 13 (đọc từ `finetune_lora.py` + README trên master 2026-09-03).
- **Loss "chính thức" đọc được (không thuộc package 2.0.2):**
  - transformers `main` `modeling_timesfm2_5.py`: `normalized_targets = revin(future_values, mu_global, sigma_global)`; `mse_loss = F.mse_loss(normalized_preds[:, :, decode_index], normalized_targets)`;
    `_quantile_loss` = pinball `mean(max((q−1)e, q e))` trên các kênh `≠ decode_index` **ghép theo thứ tự với `quantiles=[0.1..0.9]`**; `loss = mse + quantile`. `TimesFm2_5Config.decode_index = 5`, `attention_dropout = 0.0`.
    ⇒ Với layout `[mean, q10..q90]` của 2.0.2, loss này ép **kênh 5 (q50) về mean bằng MSE** và **kênh 0 (mean) về q=0.1 bằng pinball** — mâu thuẫn layout (đọc source, chưa chạy; có thể converter checkpoint `-transformers` đã đổi thứ tự kênh — **unknown**). Hệ quả chắc chắn: `out.mean_predictions` của path transformers = kênh `decode_index=5`, KHÔNG phải kênh mean mà harness dùng → **không nhập loss này**. (Sửa ghi chú audit_timesfm §6: `out.mean_predictions` là q50.)
  - v1 (archive, API 1.x, không cài được trên py3.12): MSE `mean((x − y)²)` trên `predictions[..., 0]` của patch cuối, quantile loss tuỳ chọn; defaults lr 1e-4, wd 0.01, 20 epoch, batch 32. Gần nhất với đề xuất §9(b).

## 3. `compile(ForecastConfig)` và `torch.compile` (Q3)

- `compile()` (T:377–512) **không gọi `torch.compile`**, không đụng trọng số/`requires_grad`/`eval`: chỉ tính `global_batch_size` (T:386–388), làm tròn context/horizon (T:393–408), gán `self.forecast_config` (T:419) và closure `_compiled_decode` → `self.compiled_decode` (T:421, 512). Closure đọc `self.model` **lúc gọi** (T:445, 458) ⇒ mọi thay đổi module sau `compile()` (inject LoRA, cập nhật A/B) tự động có hiệu lực.
- `torch.compile` chỉ ở `load_checkpoint` (T:298–302) khi `torch_compile=True`: `self.model.forward = torch.compile(self.model.forward)` (gán thuộc tính instance bọc bound method). `from_pretrained(..., torch_compile=False)` bỏ hẳn (tests/test_model_loading.py:77–88).
- `module.load_checkpoint` (T:79–84): `load_state_dict(strict=True)`, `.to(device)`, `.eval()`. **Không có `requires_grad_(False)` ở bất kỳ đâu** (grep) ⇒ sau load, **cả 231M tham số đều `requires_grad=True`** → code training phải tự đóng băng base (PEFT làm tự động; wrapper tự viết phải làm tay). `.eval()` không ảnh hưởng số học (không dropout/batchnorm). `strict=True` ⇒ **inject LoRA SAU `load_checkpoint`**, không bao giờ gọi lại `load_checkpoint` trên module đã inject.
- Inject SAU `torch.compile` có bị "bỏ qua im lặng" không? Smoke local (torch 2.11.0+cu128, `backend="eager"`, layer `timesfm.torch.transformer.Transformer` thật thu nhỏ): thay `attn.qkv_proj` bằng LoRA wrapper sau compile → output đổi (max|Δ| = 0,168); cập nhật in-place `lora_B` → output đổi; dynamo `unique_graphs: 2` (recompile đúng). **Không bị bỏ qua**, nhưng chỉ kiểm với backend eager; inductor trên Vast chưa kiểm (unknown). Khuyến nghị: instance LoRA dùng `torch_compile=False` cho cả train lẫn vòng lặp covariate; nếu cần tốc độ thì `torch.compile` **một lần sau khi inject + freeze**, kèm canary "output có adapter ≠ output pristine".
- `torch.no_grad` ở `decode` (T:118) chỉ chạm inference — đúng ý; training không đi qua `decode`.

## 4. Cầu nối tới XReg — chuỗi gọi (Q4)

`models_tfm.py:118 forecast_with_covariates(...)` → B:198–423: guard `return_backcast` (B:240–243) → `from ..utils import xreg_lib` (B:245) → mode `"xreg + timesfm"` (B:371–421):
`targets = r1 window` (B:373–376) → `xreg_lib.normalize` (R:61–64: `(x − mean)/std`, std population, `< 1e-6 → 1`) (B:378–379) → `BatchedInContextXRegLinear(...).fit(...)` (B:380–398, chỉ numpy/jax) →
**`self.forecast(horizon=max_horizon, inputs=[target − xreg_on_context])`** (B:399–405) → `forecast` (B:155–196) → `self.compiled_decode(...)` (B:188) → closure `_compiled_decode` (T:421–510) → `self.model.decode` (T:445; T:458 cho flip) → `self(normed_inputs, patched_masks, caches)` (T:168–170) → `nn.Module.__call__` → `forward` (T:86–113) → `self.stacked_xf[i](...)` (T:100–104) → `Transformer.forward` (X:354–370) → `MultiHeadAttention.forward` dùng `self.qkv_proj` (X:236), `self.out` (X:303); FF dùng `self.ff0`/`self.ff1` (X:367) ⇒ **các module đã bọc LoRA nằm đúng trên đường đi**. Kết quả cộng `xreg` (B:406–416) và `renormalize` (B:417–421).
- `self.model` chỉ xuất hiện ở B:270 (`self.model.p`); `xreg_lib` không nhận model, chỉ nhận mảng (B:380–389). `load_checkpoint`/`load_state_dict` chỉ được gọi ở T:79–84, T:287–302, T:361 (từ `_from_pretrained`) — **không có gì trong đường xreg nạp lại trọng số gốc**.
- Sắc thái phải ghi rõ: ở k ≥ 1, input của model là **phần dư `target_norm − xreg_on_context`**, không phải r1 thô; `normalize_inputs=True` chuẩn hoá lại lần nữa (T:438–441). Adapter train trên r1 thô rồi dùng trên phần dư = **lệch phân phối train/serve** vốn có của thiết kế "train một lần, freeze, tái dùng" — không phải bug, là rủi ro phải nêu (§9e). Ở k = 0 (`tfm_ext` baseline) không có phần dư (`forecast_with_covariates` raise nếu không covariate B:248–258 → harness đi `forecast()`).

## 5. PEFT (Q5)

- Target đều là `torch.nn.Linear` (X:200, 205, 335, 340; D:29–43) → được `LoraModel._create_new_module` hỗ trợ (peft 0.20.0 `tuners/lora/model.py`: "Only the following modules are supported: torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d/2d/3d, transformers Conv1D, torch.nn.MultiheadAttention").
- `peft.inject_adapter_in_model(peft_config, model, adapter_name="default", low_cpu_mem_usage=False, state_dict=None) -> nn.Module` (peft 0.20.0 `src/peft/mapping.py`, re-export `peft.functional`): **in-place**, không trả `PeftModel`. `BaseTuner.inject_adapter` kết thúc bằng `self._mark_only_adapters_as_trainable(model)` (vô điều kiện) = `requires_grad=False` cho mọi param không chứa `"lora_"`; `BaseTuner.__init__` gán `model.peft_config = {...}` ⇒ `get_peft_model_state_dict(model)` / `set_peft_model_state_dict(model, sd)` (`utils/save_and_load.py`, đọc `model.peft_config[adapter_name]`) **dùng được trên nn.Module thường**; key bỏ tên adapter (`...lora_A.weight`); `set_...` trả `load_result(missing_keys, unexpected_keys)` → phải assert rỗng.
- **Không dùng `target_modules="all-linear"`**: `_maybe_include_all_linear_layers` chọn mọi `nn.Linear`/Conv1D và chỉ loại output layer khi model là `transformers.PreTrainedModel` (`get_output_embeddings`) — với module thường sẽ bọc cả tokenizer (64→1280) và hai head (kể cả 1280→10240). Dùng `target_modules=["qkv_proj", "out", "ff0", "ff1"]` (khớp hậu tố tên) → đúng 80 module; assert `len(targeted_module_names) == 80`.
- `LoraConfig` 0.20.0 mặc định: `r=8, lora_alpha=8, lora_dropout=0.0, bias="none", init_lora_weights=True` (A kaiming-uniform, **B = 0** → adapter khởi đầu = identity), `use_rslora=False` (scaling `alpha/r`).
- Version: peft **0.20.0** (2026-07-28), `python>=3.10`, `torch>=1.13.0` **không chặn trên**, **`transformers` là dependency BẮT BUỘC** (unpinned) + `accelerate>=0.21.0` + `huggingface_hub>=0.25.0` + `safetensors` ⇒ `pip install peft` kéo transformers 5.x. Release notes 0.20.0 không nêu ma trận torch đã test; **tương thích runtime với torch 2.11 = unknown** (chưa cài, chưa import).
- **Wrapper tự chứa (đã chạy thật trên class module 2.0.2, torch 2.11, CPU/GPU local, random-init):**
  ```python
  class LoRALinear(torch.nn.Module):            # W x + (alpha/r)·B(A x); B = 0 lúc khởi tạo
      def __init__(self, base: torch.nn.Linear, r=8, alpha=16):
          super().__init__(); self.base, self.scaling = base, alpha / r
          self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features)); torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
          self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r))
      def forward(self, x): return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
  for p in module.parameters(): p.requires_grad_(False)
  for L in module.stacked_xf: L.attn.qkv_proj, L.attn.out, L.ff0, L.ff1 = (LoRALinear(L.attn.qkv_proj), LoRALinear(L.attn.out), LoRALinear(L.ff0), LoRALinear(L.ff1))
  ```
  Kết quả đo: 80 module bọc, **2.048.000 tham số trainable (0,885 % của 231,29 M)** ở r=8 (mỗi layer `8·(1280+3840) + 3·8·(1280+1280) = 102.400`); grad chỉ vào `lora_A/lora_B`, `base.weight.grad = None`; state_dict có 160 tensor `stacked_xf.{i}.attn.qkv_proj.lora_A/…`. Lưu: `safetensors.torch.save_file({k: v for k, v in sd.items() if ".lora_" in k})`; nạp: `load_state_dict(sd, strict=False)` + assert `unexpected_keys == []` và mọi key lora đều có trong sd.
- **Khuyến nghị: wrapper tự chứa** (không thêm transformers/accelerate; ~40 dòng gồm save/load; toán học trùng PEFT với `lora_dropout=0`). PEFT chấp nhận được nếu user đồng ý thêm dependency; hai đường cho cùng kết quả.

## 6. Tất định / seed (Q6)

- **Inference: không có nguồn ngẫu nhiên** trong `timesfm/torch/*` và `T` (grep `dropout|rand|manual_seed`: chỉ `jax.random` ở R:470–471 cho `max_rows_per_col` — ta đặt 0 = tắt — và init flax). Danh sách module type của model: không có `Dropout`. Còn lại chỉ là phi tất định kernel GPU theo shape (MEMORY đã coi zero-shot là tất định, ε = sàn) → giữ cùng batch shape giữa các run.
- **Training**: (1) init `lora_A` (torch RNG) → `torch.manual_seed(seed)` ngay trước inject; (2) thứ tự minibatch → `torch.Generator().manual_seed(seed)`; (3) `lora_dropout` — đề xuất 0 (mặc định PEFT; example chính thức 0,05); (4) kernel backward không tất định (SDPA/cuBLAS) → nếu cần bit-reproducible: `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (có thể ép SDPA về math kernel, chậm hơn); nếu không, nhiễu run-to-run là một phần của ε đo bằng 3 eval seed; (5) không `torch.compile` trong training.
- Áp đúng §1.3 plan: `calib_seed=8586` → ES chọn epoch; `eval_seeds=8587/8588/8589` → 3 adapter/fold → ε_TFM-LoRA; **`selection_seed=8587` → adapter FROZEN duy nhất** cho baseline + 39 candidate + prune PI. `models_tfm.py:33–34` (`supports_rounds=False`, `seed_dependent=False`) phải đổi thành True/True cho model LoRA (zero-shot giữ nguyên).

## 7. Kích thước / chi phí (Q7)

- Base 231,29 M fp32 = 925 MB; LoRA r=8 trên 80 module = 2,05 M (0,885 %); r=4 = 1,02 M. Example chính thức (transformers): r=4, alpha=8, dropout 0,05, "all-linear" ≈ 1,4 M/232 M (0,6 %, gồm tokenizer + head), lr 1e-4, 10 epoch, batch 32, bf16; v1: lr 1e-4, wd 0,01, 20 epoch, batch 32. Tác giả không công bố default cho 2.5 ngoài example này.
- **Đề xuất**: r=8, alpha=16, target `qkv_proj/out/ff0/ff1` × 20 layer, dropout 0, AdamW lr 1e-4 wd 0,01, **fp32** (giữ số học đồng nhất với đường zero-shot; không bf16 như example). Không search hyperparameter (ngoài scope); chỉ epoch do ES chọn.
- **VRAM đo local** (RTX 3050 Ti, torch 2.11, module random-init + wrapper r=8, context 512, forward có flip + backward + AdamW step; script gọi f(−x) hai lần = **3 forward**, là cận trên): weights+LoRA 890 MB allocated; đỉnh − weights = **94 / 95 / 87 MB per sample** ở B = 1/2/4. Suy ra bản 2-forward ≈ 60–65 MB/sample (tỷ lệ, **chưa đo riêng**). Mỗi worker cố định ≈ 0,9 GB weights + 0,3–0,5 GB CUDA context/allocator.
  - 5 worker đồng thời trên 24 GB: 5 × 1,4 = 7 GB cố định → còn ~17 GB → **B ≤ ~35/worker** (cận trên 95 MB) hoặc ~50 (65 MB). Cộng thêm jax của xreg nếu chạy cùng lúc (allocator jax lớn dần dù `PREALLOCATE=false`).
  - Tuần tự 1 worker: B=128 → ~13,5 GB (cận trên) an toàn; B=256 → 18–26 GB → rủi ro. **Khuyến nghị: train fold tuần tự trong một tiến trình, B=64–128**; hoặc 5 worker với B ≤ 32.
  - Thời gian/step trên 3090: **unknown** (không đo được từ 3050 Ti); FIT có 9.887–15.647 origin/fold → 155–245 step/epoch ở B=64; tổng = 5 fold × epoch × 4 lượt (1 calib + 3 eval seed) — cần canary 1 fold × 1 epoch trước khi cam kết.
- `models_tfm._CACHE` (:66–76) hiện tạo **hai** instance `from_pretrained` (point batch 256 / covariate batch 1) = 2 module. Với LoRA phải **dùng chung một module**: `m_cov = TimesFM_2p5_200M_torch(torch_compile=False); m_cov.model = m_point.model; m_cov.compile(fc_cov)` — hợp lệ vì `compile()` chỉ dựng closure trên wrapper (T:386–419), `forecast_config/global_batch_size` là thuộc tính wrapper, không phải module. Tiết kiệm 925 MB và bảo đảm hai đường dùng đúng một bộ trọng số.

## 8. Phần xreg/jax phải giữ nguyên (Q8)

Không đổi bất kỳ mục nào của audit_timesfm §9.1(b)/§12 và MEMORY 2026-09-01: `xreg_mode="xreg + timesfm"`, `ridge=0.0`, `max_rows_per_col=0`, `normalize_xreg_target_per_input=True`, `force_on_cpu=False` + `jax[cuda12]==0.11.1` + **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** (càng bắt buộc khi torch đang train), `per_core_batch_size=1` cho instance covariate, `return_backcast=True`, **1 origin/lời gọi**, covariate **dịch 1 bar**, 3 bước tương lai giữ f(t), head mean `res[1][..., 0]` cho cả hai đường. LoRA không chạm bất kỳ dòng nào của `xreg_lib`: xreg chạy trên numpy/jax trước và sau lời gọi model (B:373–421), model chỉ nhận `target − xreg_on_context`. `create_covariate_matrix` vẫn CPU thuần (giới hạn đã ghi).

## 9. Verdict + thiết kế cầu nối tối thiểu

**Verdict: KHẢ THI với package `timesfm==2.0.2` (torch), checkpoint `-pytorch` rev `1d952420…`, KHÔNG cần transformers, KHÔNG cần checkpoint thứ hai.** Điều kiện kỹ thuật cứng: (i) không dùng `decode()` để train (no_grad T:118) — tái hiện T:427–500 + T:119–178 có grad; (ii) inject sau `load_checkpoint`, `torch_compile=False`; (iii) đóng băng base tường minh; (iv) một module dùng chung cho hai wrapper; (v) loss trên kênh 0.

(a) **Inject**: 80 `nn.Linear` `stacked_xf.{i}.attn.qkv_proj / attn.out / ff0 / ff1` (i = 0..19). Không bọc tokenizer và hai head: chỉ kênh 0 của point head nhận loss, bọc head làm lệch ngữ nghĩa các kênh còn lại mà `flip_quantile_fn`/`fix_quantile_crossing` giả định. Thứ tự: `from_pretrained(rev, torch_compile=False)` → `compile(fc)` → `torch.manual_seed(seed)` → inject (wrapper §5 hoặc `inject_adapter_in_model`) → `requires_grad_(False)` cho mọi param không `lora_` → assert trainable = 2.048.000.

(b) **Train** (per fold, `fit_predict(X_fit, z_fit, X_es, z_es, X_pred, rounds, seed)`, `X_fit/X_es` là `SeriesBatch` với `idx_fit/idx_es` — harness.py:247; `z_*` bỏ qua vì `is_logret=True`):
- Cửa sổ: `ctx = r1[t−511..t]`, target `r1[t+1..t+3]`, t ∈ `idx_fit` (đã đảm bảo `t+3 < T_end` của FIT → target không rời FIT); ES: t ∈ `idx_es` (đã trừ purge 60'). VAL/TEST không bao giờ vào training.
- Forward train = **đúng hàm suy luận**: normalize_inputs (mean, std unbiased, `revin`) → patch (B,16,32), mask False → running stats T:129–142 → `revin` → `module(...)` → `revin(reverse)` per patch → `[:, -1, :3, :]` → flip `(f(x) − flip(f(−x)))/2` → kênh 0 → `revin(reverse)` với (mu, sigma) → `r̂ (B,3)` → `ŷ_h = cumsum`.
- **Loss đề xuất: MSE trên `ŷ_h` vs `y_h = cumsum(r1[t+1..t+3])`, đơn vị log-return thô, trung bình 3 horizon**, nhân hằng số (vd `1/var_FIT(r1)`) chỉ để optimizer ổn định (không đổi argmin). Lý do: metric project là RMSE giá per horizon ≈ `C_t·|ŷ_h − y_h|` → MSE thô cân các origin như metric; MSE trong không gian chuẩn hoá (loss transformers) cân theo `1/σ_w²` = ưu tiên cửa sổ yên. Không dùng `out.loss` (pinball 9 kênh không dùng + mapping kênh nghi sai §2 + code path/checkpoint khác). Huber sẽ thêm δ phải calibrate → thêm knob, không đề xuất. Chỉ dùng **patch cuối** (đúng context 512 như serve); dùng cả 16 vị trí = 16× supervision nhưng 15 vị trí có context < 512 không bao giờ xuất hiện lúc serve — ghi nhận, không đề xuất.
- Epoch: theo cơ chế LSTM §2.2 #7 (patience 5, ≤ 50 epoch): run `calib_seed` → mỗi epoch đo RMSE giá trên `idx_es` bằng đúng đường suy luận → `fixed_epoch_TFM` per fold, `rounds = (e, e, e)`; `eval_seeds` train số epoch cố định → ε; `selection_seed` adapter → FROZEN cho toàn bộ add-one/prune; confirmation 3 seed bật ES như LSTM. Không có luật mới — là áp §1.3 cho model có epoch.

(c) **Freeze + save + reload**: sau train `requires_grad_(False)` lora, `module.eval()`; `sd = {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items() if ".lora_" in k}` → `safetensors.torch.save_file(sd, experiments/runs/<exp>/lora_<fold>_<seed>_e<epochs>.safetensors)` + sidecar JSON (r, alpha, target list, seed, epochs, fold, repo/revision, sdist sha, torch version, hash sd). Reload: `from_pretrained(rev, torch_compile=False)` → inject cùng config → `load_state_dict(sd, strict=False)` → assert `unexpected_keys == []` và `set(missing) ∩ lora_keys == ∅` → canary: forward cửa sổ cố định (lưu lúc save) trùng tới 1e-6. `_CACHE` key = (repo, revision, fold, seed, epochs, lora cfg) → cùng module cho point path và covariate path (§7).

(d) **Tái dùng trong `forecast_with_covariates`**: không đổi `models_tfm.py:118–127`; chuỗi §4 đi qua module đã inject; mỗi candidate chỉ fit lại `beta_hat` (R:492–499, vứt sau lời gọi), LoRA không có grad/optimizer/`requires_grad`, `decode` chạy `no_grad` (T:118). Thêm assert: hash các tensor `lora_*` trước và sau toàn bộ vòng lặp add-one + prune PI phải bằng nhau.

(e) **Rủi ro**: (1) lệch train/serve: adapter học trên r1 thô, dùng trên phần dư OLS khi k ≥ 1 (§4); (2) 231M model, 10–16k cửa sổ/fold, r1 ≈ nhiễu trắng → khả năng collapse về 0 = E0 hoặc overfit — ES trên ES-partition là chốt chặn; kết quả "LoRA ≈ E0" là hợp lệ; (3) ε_TFM-LoRA > sàn → KEEP/DROP rộng hơn nhánh zero-shot (nhất quán §1.3, cần ghi nhận khi đọc); (4) VRAM khi jax + torch train cùng GPU (§7); (5) `torch_compile=False` làm inference chậm bao nhiêu = unknown, đo canary; (6) không bao giờ gọi `load_checkpoint` sau inject (strict=True); (7) giữ fp32; (8) inductor + inject = unknown (chỉ kiểm eager); (9) nếu chọn PEFT: import peft/transformers 5.x với torch 2.11 chưa kiểm.

## 10. Unknown (chưa xác minh, không đoán)

- Hành vi `torch.compile` backend inductor (Vast) sau inject; tốc độ inference khi `torch_compile=False`.
- peft 0.20.0 import/chạy trên torch 2.11 + transformers 5.x.
- Thời gian step/epoch trên 3090; VRAM bản 2-forward (suy tỷ lệ từ đo 3-forward).
- Mapping kênh trong loss của transformers `main` có thật sự lệch layout hay converter đã đổi thứ tự.
- Tên key trong `model.safetensors` (chưa tải) — `strict=True` ở T:82 bảo đảm khớp module một khi load được.

## 11. Việc kế tiếp

- **Session chính**: viết `TimesFMLoRAModel` (subclass `TimesFMModel`, `supports_rounds=True`, `seed_dependent=True`) theo §9; hàm `train_forward` tái hiện T:427–500 + T:119–178; wrapper §5; cache adapter theo (fold, seed, epochs); share module cho hai wrapper; không đổi đường xreg. Cần user cho phép cài `timesfm[torch]==2.0.2` (+ `jax[cuda12]==0.11.1`, `scikit-learn`) trên Vast — đã trong quyết định cũ; **không cần peft** nếu dùng wrapper.
- **checker**: (1) adapter B=0 → prediction bit-identical với đường zero-shot hiện tại (cả point lẫn covariate); (2) `train_forward` (no_grad) == `compiled_decode` cùng cửa sổ ≤ 1e-6; (3) hash lora trước/sau vòng lặp 39 candidate + PI; (4) target training ⊂ FIT, ES ⊂ ES, không chạm VAL (§6.4); (5) reload canary; (6) `len(targeted) == 80`, trainable == 2.048.000; (7) ε_TFM-LoRA từ 3 adapter — báo cáo, không dùng seed nào làm mốc.
