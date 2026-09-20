Bạn đang làm việc trên một máy Vast.ai Linux có **1 × NVIDIA RTX 4090 24GB**.

Bạn có:

* `GITHUB_TOKEN`: có quyền clone/pull/push GitHub repo.
* `HF_TOKEN`: có quyền tạo/push Hugging Face repo.
* Internet access.
* Repo GitHub:

```text
https://github.com/tson295/Pretrain_Model
```

Đây là một task **hoàn chỉnh và self-contained**. Không cần tìm prompt cũ, không cần suy diễn “giữ config trước đó”, không tự chọn architecture/hyperparameter khác.

Mục tiêu:

1. Clone repo.
2. Sửa chính xác những điểm được mô tả bên dưới.
3. Chạy toàn bộ unit tests + CUDA smoke tests.
4. Base-train từ random initialization 5 learned models:

   * OF/OFI-LSTM
   * HFformer
   * PatchTST
   * ModernTCN
   * LiT-L10
5. Evaluate E0 baseline.
6. Tận dụng RTX 4090 hiệu quả bằng cách chạy nhiều model song song nếu profiling chứng minh có lợi.
7. Lưu đầy đủ `best` và `last` checkpoints.
8. Lưu prediction files để sau này có thể tái tạo toàn bộ figures mà không cần inference lại.
9. Push source code về GitHub.
10. Push checkpoints, metrics, predictions và run metadata lên Hugging Face.
11. Sau cùng báo cáo đầy đủ kết quả và đường dẫn artifact.

Không được hyperparameter tuning.
Không được thay architecture ngoài những thay đổi được chỉ định bên dưới.
Không dùng test set để chọn checkpoint, epoch hoặc architecture.
Không download pretrained model weights.
Tất cả 5 learned models đều được **khởi tạo random và train từ đầu trên BTC L10 train split**.

---

# A. CLONE REPO VÀ AUTHENTICATION

Clone:

```bash
git clone https://github.com/tson295/Pretrain_Model.git
cd Pretrain_Model
```

Dùng `GITHUB_TOKEN` để authenticate cho push.

Ưu tiên GitHub CLI:

```bash
printf '%s' "$GITHUB_TOKEN" | gh auth login --with-token
gh auth setup-git
```

Không:

* echo token ra log;
* ghi token vào source;
* commit token;
* embed token trực tiếp vào Git remote URL nếu tránh được.

Kiểm tra:

```bash
git status
git rev-parse HEAD
nvidia-smi
python --version
```

Ghi lại commit SHA ban đầu.

---

# B. ENVIRONMENT

Tạo virtual environment.

RTX 4090 cần CUDA-enabled PyTorch.

Không được vô tình cài CPU-only PyTorch.

Sau khi setup, verify:

```python
import torch

assert torch.cuda.is_available()
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory)
```

GPU phải là RTX 4090 với khoảng 24GB VRAM.

Cài các dependency trong `requirements.txt` và thêm `huggingface_hub` nếu repo chưa có.

Không download pretrained weights từ Hugging Face hoặc nguồn bên ngoài.

---

# C. DATASET — FREEZE CHÍNH XÁC FILE

Dataset bắt buộc phải là:

```text
BTCUSDT_L10_oct2023.csv
```

Expected SHA256:

```text
e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab
```

Nếu `$LOB_CSV` tồn tại, dùng file đó.

Nếu `$LOB_CSV` không tồn tại nhưng `$HF_DATASET_REPO` tồn tại, dùng `$HF_TOKEN` tải đúng file `BTCUSDT_L10_oct2023.csv` từ repo đó.

Nếu không tìm được đúng file thì **dừng**, không tự lấy dataset khác.

Trước khi làm bất cứ training nào:

```bash
sha256sum "$LOB_CSV"
```

Phải có:

```text
actual_sha256 ==
e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab
```

Expected audit:

```text
rows                    = 678362
median dt               ≈ 1.236 s
p99 dt                  ≈ 1.683 s
max dt                  ≈ 6.221 s
number of gaps > 2 s    = 60
segment transitions     = 1
```

Nếu SHA không đúng: dừng.

Nếu SHA đúng nhưng statistics không đúng: dừng và báo lỗi preprocessing.

---

# D. TASK — FREEZE

Mid-price:

```text
mid_t = (best_bid_t + best_ask_t) / 2
```

Direct targets:

```text
y_1m = log(mid_(t+60s)  / mid_t)
y_2m = log(mid_(t+120s) / mid_t)
y_3m = log(mid_(t+180s) / mid_t)
```

Output learned model:

```text
[B, 3]
```

theo thứ tự:

```text
[pred_return_1m, pred_return_2m, pred_return_3m]
```

Không recursive forecasting.

Không forecast phút 1 rồi feed prediction đó để forecast phút 2.

---

# E. TARGET LOOKUP — FREEZE

Timestamp irregular.

Không dùng:

```text
origin + fixed number of rows
```

Phải tìm:

```python
target_index = np.searchsorted(
    timestamps,
    origin_timestamp + horizon,
    side="left",
)
```

cho:

```text
60 s
120 s
180 s
```

Observation được chọn phải là observation đầu tiên:

```text
timestamp >= origin_timestamp + horizon
```

Target tolerance:

```text
2.0 seconds
```

Nếu overshoot > 2.0s thì sample invalid.

---

# F. CONTINUITY — FREEZE

Một edge giữa snapshot `i` và `i+1` là bad nếu:

```text
dt > 2.0 seconds
OR
segment_id changes
```

Lưu ý:

```text
dt == 2.0 seconds
```

vẫn hợp lệ.

Một sample chỉ hợp lệ nếu **không có bad edge nào từ history start cho tới target 3 phút**.

Tức phải sạch trên toàn đoạn:

```text
history_start
→ origin
→ target_1m
→ target_2m
→ target_3m
```

Dùng prefix sum của bad edges để kiểm tra O(1).

Không scan toàn window trong mỗi `__getitem__`.

Không forward-fill.
Không interpolation.
Không reorder timestamps.
Không deduplicate âm thầm.

---

# G. TRAIN / VALIDATION / TEST SPLIT — FREEZE

Không dùng random split.

Không tự tính lại split theo một policy khác.

Dùng chính xác chronological boundaries sau:

```text
train_end =
2023-10-07T22:01:22.396Z

validation_end =
2023-10-09T09:35:56.2555Z
```

Expected row ranges:

```text
train:
[0, 477086)

validation:
[477086, 578798)

test:
[578798, 678362)
```

Mọi history và mọi target của sample phải nằm hoàn toàn trong cùng một split.

Không được để train label đi vào validation.

Không được dùng validation/test để fit corpus-level statistics.

---

# H. HISTORY + STRIDE — FREEZE

Base experiment chỉ chạy:

```text
history_seconds = 60
history_rows = None
stride_seconds = None
```

Không chạy 120s hoặc 180s history trong lần này.

Code hiện tại phải resolve history bằng:

```text
history_rows =
ceil(
    history_seconds /
    median_continuous_train_dt
)
```

Với dataset này expected:

```text
history_rows = 49
```

Khi:

```text
stride_seconds = None
```

giữ logic:

```text
stride_seconds =
history_seconds / 6
```

tức:

```text
10.0 seconds
```

Sau đó convert sang rows bằng current cadence logic.

Expected:

```text
stride_rows = 8
```

Không thay thành stride 1.
Không đổi stride để tăng số training samples.

Expected valid sample counts gần chính xác:

```text
train       = 58774
validation  = 12427
test        = 12145
```

Nếu sample counts lệch đáng kể thì audit dataset logic trước khi train.

---

# I. FEATURE NORMALIZATION

## I.1 Models KHÔNG phải HFformer

OF/OFI-LSTM, LiT và raw-input preprocessing nơi thích hợp tiếp tục dùng **train-only preprocessing** hiện có, trừ normalization nội bộ vốn là một phần architecture của PatchTST/ModernTCN.

Không dùng validation/test statistics.

---

# J. HFFORMER NORMALIZATION — FREEZE CHÍNH XÁC

Đây là thay đổi bắt buộc.

HFformer **KHÔNG ĐƯỢC dùng train-global Standardizer**.

HFformer feature array phải đi vào Dataset/model ở raw numerical feature scale trước window-local normalization.

Mỗi input sample có:

```text
x shape = [B, T, 38]
```

Ngay trước HFformer backbone, normalize **từng feature trên time dimension của chính sample đó**:

```python
x_fp32 = x.float()

mean = x_fp32.mean(
    dim=1,
    keepdim=True,
)

std = x_fp32.std(
    dim=1,
    keepdim=True,
    unbiased=False,
)

x_norm = (
    x_fp32 - mean
) / (
    std + 1e-5
)
```

Freeze:

```text
unbiased = False
eps = 1e-5
```

Không:

```text
global normalize
→ rồi local normalize lần nữa
```

Không dùng saved train-global mean/std cho HFformer features.

HFformer normalization phải phụ thuộc **duy nhất vào history hiện tại của sample**.

Update metadata để checkpoint ghi rõ:

```text
normalization =
window-local per-feature z-score
over history dimension
unbiased=False
eps=1e-5
no train-global standardizer
```

Unit test phải chứng minh HFformer không phụ thuộc validation/test corpus statistics.

---

# K. FREEZE ARCHITECTURE — TẤT CẢ 5 MODEL

Không được hiểu “giữ architecture cũ”.
Dùng chính xác các config dưới đây.

---

# K1. OF/OFI-LSTM

Tên code:

```text
ofi_lstm
```

Base experiment lần này dùng:

```text
of_representation = "of"
```

Input không collapse thành OFI10.

Input channels:

```text
bid_OF_1
...
bid_OF_10

ask_OF_1
...
ask_OF_10
```

Tổng:

```text
20 channels
```

OF calculation giữ logic hiện tại:

Bid:

```text
price improves:
    current qty

price unchanged:
    current qty - previous qty

price worsens:
    -previous qty
```

Ask dùng logic symmetric theo hướng giảm giá là improve.

Architecture:

```text
input channels = 20
LSTM layers = 2
LSTM hidden size = 64
batch_first = True
dropout = 0.1
prediction head = Linear(64, 3)
```

Expected parameter count:

```text
55,491
```

Sau instantiate:

```python
sum(
    p.numel()
    for p in model.parameters()
)
```

phải bằng:

```text
55491
```

Nếu không bằng, dừng và kiểm tra architecture.

Không tăng hidden size.

---

# K2. HFFORMER

Input features chính xác:

```text
bid_price_1..9         = 9
bid_qty_1..9           = 9
ask_price_1..9         = 9
ask_qty_1..9           = 9
lagged_log_return      = 1
weighted_mid_price_L1  = 1
```

Tổng:

```text
38 features
```

Weighted mid:

```text
(
    ask_price_1 * bid_qty_1
    +
    bid_price_1 * ask_qty_1
)
/
(
    bid_qty_1 + ask_qty_1
)
```

Zero total quantity fallback:

```text
ordinary mid
```

Architecture:

```text
input_projection:
    Linear(38, 36)

d_model:
    36

attention heads:
    6

Transformer encoder layers:
    2

FFN dimension:
    64

dropout:
    0.3

attention:
    causal = True

position encoding:
    NONE

activation:
    Stable Spiking PReLU

spike simulation dt:
    0.001

encoder style:
    post-norm

final encoder LayerNorm:
    yes

decoder:
    Linear(36, 1)
    PReLU
    transpose time
    Linear(history_rows=49, 3)
```

Separate Linear modules phải tiếp tục tồn tại:

```text
q_proj
k_proj
v_proj
out_proj
fc1
fc2
```

để future LoRA có thể inject.

Expected parameter count:

```text
22,026
```

Normalization không tính vào parameter count.

Sau instantiate phải đúng:

```text
22026
```

Không scale HFformer lên chỉ vì model nhỏ.

---

# K3. PATCHTST

Input:

```text
40 raw L10 channels
```

Feature order:

```text
bid_price_1..10
bid_qty_1..10
ask_price_1..10
ask_qty_1..10
```

Architecture freeze:

```text
num_input_channels = 40
context_length = 49

patch_length = 16
patch_stride = 8

d_model = 128
num_attention_heads = 16
num_hidden_layers = 3

ffn_dim = 256

attention_dropout = 0.2
ff_dropout = 0.2
positional_dropout = 0.2
path_dropout = 0.2

head_dropout = 0.0

share_embedding = True
channel_attention = False

norm_type = "batchnorm"
pre_norm = True

activation_function = "gelu"

positional_encoding_type = "sincos"

scaling = "std"

do_mask_input = False
```

Use:

```text
transformers.PatchTSTModel
```

khởi tạo từ config local.

Không:

```text
from_pretrained(remote_model)
```

Không tải external weights.

Backbone output:

```text
[B, channels, patches, d_model]
```

Regression head:

```text
Flatten from dim 1
Dropout(0.0)
Linear(
    channels * patches * d_model,
    3,
)
```

Không denormalize output sang price.

Target vẫn là 3 log returns.

Expected parameter count tại history_rows=49:

```text
477,059
```

Sau instantiate phải bằng:

```text
477059
```

Nếu không đúng, không train.

---

# K4. MODERNTCN

Input:

```text
40 raw L10 channels
```

Feature order giống PatchTST:

```text
bid_price_1..10
bid_qty_1..10
ask_price_1..10
ask_qty_1..10
```

Architecture freeze:

```text
history_rows = 49

patch_length = 16
patch_stride = 8

dims =
[256, 256, 256, 256]

number of stages =
4

blocks per stage =
[1, 1, 1, 1]

large kernels =
[31, 29, 27, 13]

small kernels =
[5, 5, 5, 5]

ffn_ratio =
2

downsample_ratio =
2

dropout =
0.05

head_dropout =
0.0

instance_norm =
True
```

Retain:

```text
shared patch Conv1d stem
BatchNorm
4 stages
depthwise large/small kernel branches
ConvFFN1
ConvFFN2
GELU
residual blocks
stage downsampling
```

Regression head:

```text
Flatten
Dropout(0.0)
Linear(..., 3)
```

Expected parameter count tại history_rows=49:

```text
50,568,195
```

Sau instantiate phải bằng:

```text
50568195
```

Nếu không đúng, dừng.

Không giảm architecture để tiết kiệm VRAM.

---

# K5. LiT-L10 — CAPACITY MỚI

Không dùng config LiT cũ 121,107 parameters.

Input:

```text
[B, T, side=2, depth=10, field=2]
```

Trong đó:

```text
side:
    bid
    ask

depth:
    L1..L10

field:
    price
    quantity
```

Base history:

```text
T = 49
```

Structured patch:

```text
temporal_patch = 4
```

Mỗi token phải span:

```text
4 consecutive snapshots
×
all 10 levels
×
price + quantity
```

của **một side**.

Tức raw content/token:

```text
4 * 10 * 2
=
80 numbers
```

Nếu `49` không chia hết cho `4`, giữ current behavior:

```text
left-pad bằng oldest historical snapshot
```

để đủ temporal patch.

Expected number temporal patches:

```text
ceil(49 / 4)
=
13
```

Bid và ask tạo token riêng:

```text
13 * 2
=
26 tokens
```

Architecture freeze:

```text
d_model = 128
position_dim = 32

content projection dimension =
96

patch_projection =
Linear(80, 96)

learned position embedding =
[1, 26, 32]

content projection
+
position embedding
are CONCATENATED

final token dim =
96 + 32
=
128
```

Transformer:

```text
layers = 4
heads = 8
d_model = 128
ffn_dim = 256
dropout = 0.1
causal attention = False
post-norm EncoderLayer
```

Separate projections:

```text
q_proj
k_proj
v_proj
out_proj
fc1
fc2
```

Sau Transformer:

```text
bid token
+
ask token
của cùng temporal patch
```

được concatenate:

```text
128 + 128
=
256
```

LSTM:

```text
input_size = 256
hidden_size = 128
num_layers = 1
batch_first = True
```

Prediction head:

```text
Linear(128, 3)
```

Expected **exact parameter count** với history_rows=49:

```text
736,547
```

Sau instantiate phải bằng:

```text
736547
```

Nếu parameter count không đúng, không bắt đầu training.

---

# L. E0 BASELINE

E0 không train.

Prediction:

```text
pred_return_1m = 0
pred_return_2m = 0
pred_return_3m = 0
```

Tương đương:

```text
predicted future mid = current mid
```

E0 phải sử dụng **đúng cùng origins và targets** với learned models.

---

# M. FREEZE TRAINING HYPERPARAMETERS

Đây là base run.

Không tune.

Dùng giống nhau cho 5 learned models trừ architecture:

```text
epochs = 30

batch_size = 128

learning_rate = 1e-4

weight_decay = 1e-4

optimizer = AdamW

loss = MSE

scheduler = CosineAnnealingLR

gradient_clip = 1.0

gradient_accumulation = 1

seed = 42
```

### Batch size contract

Base batch size của **tất cả 5 learned models**:

```text
128
```

Cụ thể:

```text
OF/OFI-LSTM = 128
HFformer    = 128
PatchTST    = 128
ModernTCN   = 128
LiT         = 128
```

Không được:

```text
128 → 256 → 512 ...
```

chỉ để lấp VRAM.

Không dùng `--auto-batch-size` cho full run.

Chỉ được giảm batch dưới 128 nếu:

```text
single model
running ALONE
at batch=128
```

bị CUDA OOM.

Nếu trường hợp đó xảy ra:

1. ghi rõ model;
2. ghi peak memory/OOM;
3. giảm về 64;
4. nếu vẫn OOM → 32;
5. báo trong final report.

Không tăng batch trên 128 trong experiment này.

---

# N. PRECISION

Trên RTX 4090:

```text
BF16 preferred
TF32 enabled
```

Nếu BF16 không available thì fallback FP16.

Không thay architecture.

Mọi loss accumulation/metric quan trọng phải dùng precision đủ ổn định.

HFformer window normalization phải tính FP32 như đã chỉ định.

---

# O. DATALOADER / SYSTEM TUNING

Đây là **systems tuning**, không phải model hyperparameter tuning.

Được phép benchmark:

```text
num_workers
prefetch_factor
torch.compile
concurrent process grouping
```

chỉ dựa trên:

```text
samples/sec
step time
GPU utilization
CPU utilization
VRAM usage
```

Không dùng validation loss để quyết định systems settings.

DataLoader phải dùng khi CUDA:

```text
pin_memory=True
persistent_workers=True
non_blocking=True
```

Không pandas trong `__getitem__`.

Không materialize overlapping windows.

---

# P. GPU UTILIZATION — KHÔNG THAY BATCH

Vì nhiều model nhỏ, một job có thể không sử dụng hết RTX 4090.

Không giải quyết bằng cách tăng batch > 128.

Thay vào đó được phép chạy **nhiều independent training processes song song trên cùng GPU**.

Mỗi process vẫn giữ:

```text
batch_size = 128
seed = 42
architecture frozen
training hyperparameters frozen
```

---

# Q. SYSTEM PROFILING TRƯỚC FULL TRAINING

Sau CUDA smoke tests, benchmark ngắn bằng temporary model instances.

Không dùng checkpoint thật.
Không làm thay đổi weights của actual training runs.

Mỗi benchmark khoảng:

```text
50–100 forward/backward/optimizer steps
```

với:

```text
batch_size = 128
```

Đo từng model đơn:

```text
OF/OFI-LSTM
HFformer
PatchTST
ModernTCN
LiT
```

Ghi:

```text
peak allocated VRAM
peak reserved VRAM
average GPU utilization
samples/sec
mean step time
```

---

# R. CONCURRENCY PROFILING

Sau single-job profiling, thử concurrent groups.

Batch của từng model vẫn = 128.

Được thử tối đa 5 models/processes nếu memory cho phép, nhưng phải benchmark tăng dần:

```text
2 jobs
3 jobs
4 jobs
5 jobs
```

Không mặc định rằng nhiều hơn luôn tốt hơn.

Candidate lightweight groups nên bao gồm thử nghiệm như:

```text
OF/OFI-LSTM + HFformer

OF/OFI-LSTM + HFformer + LiT

OF/OFI-LSTM + HFformer + LiT + PatchTST
```

ModernTCN phải benchmark:

```text
alone
```

trước.

Sau đó có thể thử ModernTCN cùng một hoặc nhiều model nhẹ nếu tổng VRAM đủ.

Mỗi process:

```text
CUDA_VISIBLE_DEVICES=0
```

nhưng cùng dùng GPU 0.

Concurrency được chọn bằng:

1. aggregate samples/sec lớn hơn sequential;
2. GPU utilization cao hơn;
3. không OOM;
4. không làm aggregate throughput giảm;
5. tổng VRAM an toàn.

Safety margin khi chạy nhiều process:

```text
ít nhất 3 GiB VRAM free
```

Không cần lấp 24/24 GB.

Nếu:

```text
GPU utilization > 85–90%
```

và aggregate throughput không cải thiện khi thêm process thì không thêm nữa.

Không dùng CUDA MPS trừ khi một bounded benchmark chứng minh throughput tăng.

Lưu quyết định cuối vào:

```text
reports/vast/concurrency_benchmark.json
```

---

# S. TORCH.COMPILE

`torch.compile` là systems optimization, không phải architecture change.

Benchmark từng model ngắn.

Chỉ bật cho full training nếu:

```text
compile works reliably
AND
outputs/loss remain finite
AND
throughput materially improves
```

HFformer dùng `pytorch-spiking`, nên nếu compile có vấn đề:

```text
HFformer chạy eager
```

Không sửa semantics model để ép compile.

Ghi compile status từng model vào final report.

---

# T. METRICS — FREEZE

Chỉ báo cáo 4 metric families:

```text
RMSE
MAE
standard R²
RMSE Gain vs E0
```

theo từng horizon:

```text
1m
2m
3m
```

## Standard R²

Giữ:

```text
R² =
1
-
sum((y - pred)^2)
/
sum((y - mean(y))^2)
```

Đây là standard R².

Không đổi nó thành R²_OS.

## E0 RMSE

Vì E0 predicts return = 0:

```text
RMSE_E0 =
sqrt(
    mean(y²)
)
```

## Gain vs E0

```text
RMSE_Gain_vs_E0 =
1
-
RMSE_model / RMSE_E0
```

Interpretation:

```text
> 0 : model better than E0
= 0 : equal E0
< 0 : worse than E0
```

E0:

```text
gain = 0
```

Metric JSON phải có:

```json
{
  "rmse": [r1, r2, r3],
  "mae": [m1, m2, m3],
  "r2": [r1, r2, r3],
  "rmse_e0": [e1, e2, e3],
  "rmse_gain_vs_e0": [g1, g2, g3]
}
```

Không thêm R²_OS.

Unit tests:

```text
perfect prediction:
RMSE = 0
Gain = 1

E0 prediction:
Gain = 0
```

---

# U. CHECKPOINT SELECTION

Train mỗi learned model tối đa 30 epochs.

Sau mỗi epoch:

```text
train metrics
validation metrics
```

Checkpoint selection criterion:

```text
validation mean MSE across the 3 horizons
```

Không dùng test set.

Lưu:

```text
best
last
```

---

# V. CHECKPOINT FORMAT — BẮT BUỘC

Mỗi run:

```text
checkpoints/
  <model>/
    <run_name>/
      best/
      last/
```

Cả `best` và `last` phải chứa:

```text
model.safetensors
config.json
experiment.json
preprocessing.json
feature_schema.json
target_config.json
split_manifest.json
data_stats.json
environment.json

optimizer.pt
scheduler.pt
trainer_state.pt
```

`last` phải resume exact training được.

`best` phải dùng được cho:

```text
inference
full fine-tuning
head-only fine-tuning
future LoRA fine-tuning
```

Checkpoint phải lưu architecture config đầy đủ.

Sau save phải verify:

```text
load checkpoint
→ same input
→ prediction exact / numerically identical within expected deterministic tolerance
```

---

# W. FUTURE LoRA COMPATIBILITY

Không train LoRA trong task này.

Nhưng checkpoint/model code phải tiếp tục expose Transformer modules bằng names:

```text
q_proj
k_proj
v_proj
out_proj
fc1
fc2
```

Không merge projections theo cách làm future LoRA injection khó hơn.

Base checkpoint phải chứa **unmodified full base weights**.

Không merge LoRA adapter vì chưa có LoRA trong phase này.

---

# X. RUN NAMES

Freeze:

```text
ofi_lstm_60s_base
hfformer_60s_base
patchtst_60s_base
moderntcn_60s_base
lit_60s_base
```

E0:

```text
e0_60s
```

---

# Y. PREDICTION ARTIFACTS — BẮT BUỘC

Sau khi mỗi model train xong:

1. load `best`;
2. inference deterministic;
3. save predictions.

Save cho:

```text
train
validation
test
```

Việc chạy test chỉ được thực hiện **sau khi model training và best checkpoint selection đã hoàn tất**.

Tạo:

```text
artifacts/
  <model>/
    <run_name>/
      train_predictions.csv.gz
      validation_predictions.csv.gz
      test_predictions.csv.gz

      train_metrics.json
      validation_metrics.json
      test_metrics.json

      training_history.jsonl
      run_summary.json
```

E0 phải có prediction files cùng schema.

Prediction table phải chứa tối thiểu:

```text
origin_index
origin_timestamp_ns
origin_mid

target_index_1m
target_timestamp_ns_1m
target_mid_1m
true_return_1m
pred_return_1m
pred_mid_1m

target_index_2m
target_timestamp_ns_2m
target_mid_2m
true_return_2m
pred_return_2m
pred_mid_2m

target_index_3m
target_timestamp_ns_3m
target_mid_3m
true_return_3m
pred_return_3m
pred_mid_3m
```

Trong đó:

```text
pred_mid_h =
origin_mid * exp(pred_return_h)
```

Prediction rows phải deterministic và ordered theo origin timestamp.

Không save chỉ summary metric.

Mục tiêu là sau này có thể tạo lại:

```text
actual vs predicted plots
return plots
error plots
scatter plots
horizon comparisons
time-series visualizations
```

mà không cần chạy model lại.

---

# Z. GPU MONITORING

Trong full training chạy GPU monitor background.

Sample mỗi khoảng:

```text
2–5 seconds
```

Save:

```text
reports/vast/gpu_usage.csv
```

Columns tối thiểu:

```text
timestamp
gpu_util_percent
memory_used_mb
memory_total_mb
temperature_c
power_draw_w
```

Nếu chạy concurrent jobs, log thêm process/PID mapping riêng.

Tạo:

```text
reports/vast/
    hardware.json
    cuda_smoke.json
    single_job_benchmark.json
    concurrency_benchmark.json
    training_schedule.json
    final_training_summary.json
```

---

# AA. .GITIGNORE — SỬA BẮT BUỘC

Đảm bảo `.gitignore` có ít nhất:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.pyc
.DS_Store

checkpoints/
artifacts/
runs/
data/

*.csv
*.csv.gz

.env
```

Prediction files và checkpoints:

```text
MUST remain local + Hugging Face only.
MUST NOT be committed to GitHub.
```

Không dùng `git add -f` cho artifacts.

Trước mỗi GitHub push:

```bash
git status
```

verify không có:

```text
CSV
CSV.GZ
checkpoint
model.safetensors
optimizer.pt
HF token
GitHub token
```

staged.

---

# AB. TESTS TRƯỚC FULL TRAINING

Bắt buộc chạy:

```bash
python -m compileall -q src train.py scripts tests
pytest -q
```

Sau đó real-data prepare validation.

Expected:

```text
SHA exact
rows exact
history_rows = 49
stride_rows = 8

train samples = 58774
validation samples = 12427
test samples = 12145
```

Sau đó instantiate 5 models và assert parameter counts:

```text
OF/OFI-LSTM:
55,491

HFformer:
22,026

PatchTST:
477,059

ModernTCN:
50,568,195

LiT:
736,547
```

Sau đó CUDA smoke:

```text
one small forward
one backward
finite predictions
finite MSE
finite gradients
```

cho cả 5 learned models.

Không full train nếu một test fail.

---

# AC. HF TOKEN / HUGGING FACE DESTINATION

Dùng:

```text
HF_TOKEN
```

Không in token.

Nếu environment variable:

```text
HF_REPO_ID
```

đã tồn tại thì push tới repo đó.

Nếu không tồn tại:

1. dùng HF API để lấy authenticated username;
2. tạo private repo:

```text
<authenticated_username>/Pretrain_Model
```

nếu chưa tồn tại.

Không upload raw BTC CSV lên model repo.

---

# AD. HUGGING FACE ARTIFACT STRUCTURE

Upload theo cấu trúc:

```text
e0/
  e0_60s/
    artifacts/

ofi_lstm/
  ofi_lstm_60s_base/
    best/
    last/
    artifacts/

hfformer/
  hfformer_60s_base/
    best/
    last/
    artifacts/

patchtst/
  patchtst_60s_base/
    best/
    last/
    artifacts/

moderntcn/
  moderntcn_60s_base/
    best/
    last/
    artifacts/

lit/
  lit_60s_base/
    best/
    last/
    artifacts/

reports/
```

Tạo HF README/model card ghi:

```text
Git commit SHA
dataset SHA256
dataset date interval
L10 depth
history = 60s / resolved 49 rows
stride = resolved 8 rows
horizons = 60/120/180s
split timestamps
gap rule
architecture
parameter count
batch size = 128
optimizer/loss/lr
best epoch
validation metrics
test metrics
hardware
precision
compile yes/no
concurrency schedule
```

---

# AE. GITHUB WORKFLOW

Sau khi sửa code và tests pass:

Commit code trước training.

Ví dụ commit bao gồm:

```text
HFformer exact window normalization
LiT 736,547-param configuration
fixed architecture contracts
RMSE gain vs E0
prediction export
Vast profiling/orchestration
HF artifact upload
.gitignore artifact policy
```

Push GitHub.

Record:

```bash
git rev-parse HEAD
```

Đó là **training source commit SHA**.

Full training phải chạy từ đúng commit đó.

Nếu sau training chỉ thêm final report/documentation:

* commit report riêng;
* không thay training source code âm thầm.

---

# AF. TRAINING SCHEDULE

Sau profiling, tạo một deterministic schedule.

Ví dụ:

```text
Group 1:
some safe concurrent lightweight models

Group 2:
remaining models

ModernTCN:
alone or concurrent only if benchmark proves aggregate throughput improves
```

Không hard-code group nếu systems benchmark cho thấy group khác tốt hơn.

Nhưng **training parameters của từng model không được thay đổi**.

Concurrency chỉ quyết định:

```text
which frozen run executes at the same time
```

không quyết định:

```text
batch
architecture
lr
epochs
stride
split
```

Save final schedule trước launch vào:

```text
reports/vast/training_schedule.json
```

---

# AG. FAILURE POLICY

Nếu một concurrent group OOM:

1. stop group;
2. không giảm architecture;
3. không giảm batch ngay;
4. chạy các jobs với ít concurrency hơn.

Chỉ giảm batch <128 nếu **chính model đó chạy một mình** vẫn OOM.

Nếu một process crash:

* inspect log;
* nếu checkpoint `last` valid và experiment contract không đổi → resume;
* không restart từ epoch 0 vô lý.

Nếu data/hash/split mismatch:

```text
STOP.
```

Không “fix” bằng cách tự đổi dataset.

---

# AH. SAU TRAINING

Sau khi cả 5 learned models hoàn tất:

1. verify `best`;
2. verify `last`;
3. evaluate train;
4. evaluate validation;
5. evaluate held-out test;
6. export predictions;
7. calculate:

   * RMSE
   * MAE
   * standard R²
   * RMSE Gain vs E0
8. upload HF;
9. verify uploaded artifacts tồn tại;
10. chỉ sau verify HF thành công mới được cân nhắc cleanup local.

Không xóa local checkpoints hoặc prediction files trước khi verify HF upload.

---

# AI. FINAL REPORT — BẮT BUỘC

Sau toàn bộ task, báo cáo lại:

```text
1. initial Git commit
2. training-source Git commit
3. final Git commit nếu khác
4. Hugging Face repo URL/id

5. dataset path
6. dataset SHA256
7. dataset rows
8. split timestamps
9. split row ranges
10. history_rows
11. stride_rows
12. train/val/test sample counts

13. GPU model
14. CUDA version
15. PyTorch version
16. precision
17. torch.compile status từng model

18. exact parameter count:
    OF/OFI-LSTM
    HFformer
    PatchTST
    ModernTCN
    LiT

19. batch size từng model
20. num_workers từng model
21. concurrency groups đã dùng

22. single-job VRAM từng model
23. concurrent-group peak VRAM
24. average GPU utilization
25. aggregate samples/sec

26. training duration từng model
27. best epoch từng model

28. validation:
    RMSE 1m/2m/3m
    MAE 1m/2m/3m
    R² 1m/2m/3m
    RMSE Gain vs E0 1m/2m/3m

29. test:
    RMSE 1m/2m/3m
    MAE 1m/2m/3m
    R² 1m/2m/3m
    RMSE Gain vs E0 1m/2m/3m

30. local checkpoint paths
31. local prediction paths
32. HF checkpoint paths
33. HF prediction artifact paths

34. jobs nào crash
35. jobs nào resume
36. bất kỳ deviation nào khỏi contract
```

Nếu không có deviation, ghi rõ:

```text
No experiment-contract deviations.
```

Không chỉ nói “training successful”.
Phải báo exact numbers và artifact locations.
