# Bàn giao triển khai prompt.md

> **Trạng thái đã thay đổi.** Tài liệu này mô tả phase chuẩn bị (không train).
> Base experiment đã được chạy thật sau đó: kết quả, benchmark GPU, checkpoint,
> prediction artifact và Hugging Face repo nằm ở [FINAL_REPORT.md](FINAL_REPORT.md),
> [contract_check.json](contract_check.json) và [vast/](vast/). Hai thay đổi
> architecture bắt buộc của lần chạy đó — HFformer window-local normalization và
> LiT 736,547 parameters — đã ghi đè phần mô tả tương ứng bên dưới.

Đã implement sáu model và pipeline chuẩn bị train. **Không chạy epoch training,
không chạy optimizer step, không benchmark GPU, không push Hugging Face.**
Các backward check chỉ dùng một batch tối đa hai samples/model; unit test dùng
synthetic data. Không chọn hyperparameter dựa trên validation.

## 1. File tạo mới

Workspace ban đầu chỉ có `prompt.md` và ba ảnh; không có code hay Git repository.
Không sửa các file ban đầu hoặc CSV ở project bên cạnh.

```text
.gitignore
requirements.txt
README.md
THIRD_PARTY.md
train.py
configs/default.json
src/config.py
src/data/{dataset.py,preprocessing.py,ofi.py}
src/models/{common.py,ofi_lstm.py,hfformer.py,patchtst.py,moderntcn.py,lit.py}
src/models/__init__.py                  # registry + architecture defaults
src/training/{trainer.py,checkpoint.py}
src/utils/metrics.py
src/__init__.py và các package __init__.py
scripts/validate.py
tests/{conftest.py,test_data.py,test_models_checkpoint.py,test_cli.py,__init__.py}
reports/validation.json
reports/IMPLEMENTATION.md
```

Đã tạo `.venv` local để kiểm tra dependency và chạy test. Code chia riêng data,
model, trainer; config tập trung. Chi tiết command ở [README](../README.md).

## 2–5. Dataset, thống kê, continuity và target

Đọc thực tế file:
`/Users/son/Projects/P0_LOB/data/processed/BTCUSDT_L10_oct2023.csv`.

SHA256: `e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab`.

| Thống kê | Kết quả |
|---|---:|
| Rows | 678,362 |
| Median dt toàn bộ | 1.236s |
| p99 dt | 1.683s |
| Max dt | 6.221s |
| Gaps dt > 2s | 60 |
| Segment transitions | 1 |
| Median continuous dt của train | 1.235s |

Mapping đã xác nhận: `timestamp_ms` là milliseconds; `segment_id` dùng làm
boundary; các cột `bid_price_i`, `bid_qty_i`, `ask_price_i`, `ask_qty_i`, i=1..10
được map theo tên. `datetime_utc` và `nonce` không dùng làm feature. Giá/qty được
tính bằng float64 trước khi normalize và chuyển array contiguous float32.

`load_csv()` trong `src/data/preprocessing.py` tạo:
`bad_edges = (dt > max_gap_seconds) | segment_changed` và prefix sum của bad edges.
`LOBDataset.__init__()` trong `src/data/dataset.py` kiểm tra
`bad_prefix[target_3m] - bad_prefix[history_start] == 0`. Không scan từng history.
Edge đúng 2s được chấp nhận; segment change bị reject dù dt nhỏ.

Dataset lưu array + origin/target indices; `__getitem__` slice history trực tiếp,
tính `log(mid[target]/mid[origin])`; không pandas, không lưu overlapping windows.
Prefix check bao phủ cả history, origin và mọi target. Toàn bộ history/targets
nằm trong cùng split. Flow/lag feature reset tại gap/segment/split boundary.

Target là observation đầu tiên có timestamp `>= t + {60,120,180}s`, dùng
`np.searchsorted(..., side='left')` trên timestamp int64 nanosecond. Overshoot
không quá 2s (configurable); thiếu một target thì loại cả sample. Một sample train
thực tế có target delays `[60.246, 120.120, 180.805]s`, không phải offset rows cố định.
Max overshoot trên toàn bộ valid samples không quá 1.842s trong audit 60s history.

Split mặc định 70/15/15 theo **elapsed timestamp duration**:

- Train rows `[0,477086)`; cutoff `2023-10-07T22:01:22.396Z`.
- Validation rows `[477086,578798)`; cutoff `2023-10-09T09:35:56.2555Z`.
- Test rows `[578798,678362)`.

Cutoff explicit có thể cấu hình. Manifest lưu cutoff, row ranges, source SHA256,
sample counts và hash origin indices. Mean/std chỉ fit raw feature rows thuộc train.

| History seconds | Rows (ceil) | Stride rows | Train samples | Validation | Test |
|---|---:|---:|---:|---:|---:|
| 60 | 49 | 8 | 58,774 | 12,427 | 12,145 |
| 120 | 98 | 16 | 29,278 | 6,175 | 6,039 |
| 180 | 146 | 24 | 19,447 | 4,095 | 4,004 |

120s cho 98 rows vì dùng ceil với median train 1.235s; không hard-code con số xấp
xỉ 97 trong prompt. Các history là fixed-row approximations, không resampling.
`--stride-seconds` đổi stride; `--history-rows 64` hỗ trợ LiT paper-style.

## 6–8. Features, architecture, giữ nguyên và custom

| Model | Features | Default architecture |
|---|---|---|
| E0 | Cùng valid origins/labels, bỏ qua input | Return vector zero |
| OFI-LSTM | Bid OF1..10 + ask OF1..10; option bid−ask OFI10 | 2-layer LSTM hidden64, dropout .1, Linear64→3 |
| HFformer | Bid price9 + bid qty9 + ask price9 + ask qty9 + lag log-return + weighted mid = 38 | d36, 6 heads, 2 encoders, FFN64, dropout .3, spiking PReLU, final LayerNorm, linear temporal decoder→3 |
| PatchTST | 40 raw fields theo thứ tự price-bid/qty-bid/price-ask/qty-ask | HF backbone, patch16/stride8, d128, 16 heads, 3 layers, FFN256, dropout .2, joint head→3 |
| ModernTCN | Như PatchTST | Patch16/stride8, 4 stages d256, 1 block/stage, large kernels31/29/27/13 + small5, ConvFFN ratio2, head→3 |
| LiT | Tensor `[T,2 sides,10 levels,2 price/qty]` | Patch 4 snapshots × toàn bộ 10 levels một side; projection48 concat position16; 2 encoders d64/4heads/FFN128; LSTM64; Linear→3 |

OF: bid cải thiện giá → current qty, giá không đổi → qty delta, giá xấu hơn →
negative previous qty; ask dùng dấu cải thiện ngược lại. HF lag-return là
`log(mid_t/mid_(t-1))`; weighted mid là
`(ask_price*bid_qty + bid_price*ask_qty)/(bid_qty+ask_qty)` ở L1, zero-volume fallback=mid.

- HFformer giữ cấu trúc public notebook, actual spiking activation và không dùng
  position encoding. Default lấy final notebook loop, vì paper table không khớp.
  Correct batch/time axis; deterministic spike initial phase; fix PReLU surrogate
  parameter gradient; split Q/K/V để LoRA. Head dự đoán trực tiếp ba return.
- PatchTST dùng `transformers.PatchTSTModel` thật, không custom Transformer thay
  thế. Giữ temporal patches và independent channels/shared weights. Custom joint
  return head; không denormalize return sang price/quantity.
- ModernTCN adapt official stem, depthwise branches, ConvFFN1/2, residuals,
  BatchNorm và stages. Custom regression head/padding; bỏ calendar/decomposition/
  unused multiscale/deployment fusion. Singleton final minibatch dùng running BN.
- LiT theo paper; giữ side/depth structure đến patching, learned position concat,
  encoder→LSTM. L20→L10, head không softmax. Encoder/LSTM widths là adaptation
  defaults công khai, không claim exact paper replication.
- OFI-LSTM chưa có paper cụ thể được chỉ định; representation OF20 là default,
  hidden64/layers2 là untuned choices.

Nguồn chính thức, commit IDs, deviations và ModernTCN MIT notice đầy đủ trong
[THIRD_PARTY.md](../THIRD_PARTY.md). Tất cả model train from scratch; không tải weights.

## 9. Kết quả kiểm tra

| Model | Real input shape | Output | Params | Forward/backward | Safetensors roundtrip |
|---|---|---|---:|---|---|
| E0 | `[2,49,40]` | `[2,3]` | 0 | Forward; không có gradient | Exact |
| OFI-LSTM | `[2,49,20]` | `[2,3]` | 55,491 | Pass | Exact |
| HFformer | `[2,49,38]` | `[2,3]` | 22,026 | Pass | Exact |
| PatchTST | `[2,49,40]` | `[2,3]` | 477,059 | Pass | Exact |
| ModernTCN | `[2,49,40]` | `[2,3]` | 50,568,195 | Pass | Exact |
| LiT | `[2,49,2,10,2]` | `[2,3]` | 121,107 | Pass | Exact |

Syntax/import checks pass; **37 unit tests passed** (`pytest -q`, 3.49s).
CLI `--prepare-only` cũng đã chạy thành công trên CSV thật, xuất metadata vào
`runs/inspection/` mà không instantiate trainer hay chạy training.
Unit tests cover irregular targets, first-after lookup,
inclusive tolerance, exactly-2s edges, segment/split rejection, independently
computed continuity oracle, column mapping, train-only scaling, saved scaling reuse,
OF signs/reset, architecture shapes at 49/64/98/146 rows, HF PReLU gradients and
batch independence, singleton ModernTCN batch, LoRA module names, frozen backbone,
checkpoint/RNG roundtrip, sampler resume, and smoke-mode training/probe guards.

Full default architectures were smoke-tested on the real data; alternate-history
unit tests use smaller PatchTST/ModernTCN widths to keep testing tiny. No accuracy,
validation-loss tuning, or training-quality evaluation was performed.

## 10–11. Checkpoint, fine-tuning, LoRA, Hugging Face

Best/last đều lưu safetensors weights, resolved model config, complete experiment,
preprocessing/normalizer, feature schema, target config, split manifest, data stats,
dependency versions, optimizer, scheduler và trainer state. Trainer state chứa
epoch/global step/best score, scaler, freeze mode và RNG states. Resume ở epoch
boundary và validate source/splits/preprocessing/config trước khi tiếp tục.

Full fine-tune dùng `--init-from`; head-only thêm `--head-only`, freeze cả parameter,
dropout và BN của backbone. `q_proj/k_proj/v_proj/out_proj` và FFN modules là Linear
có tên để gắn LoRA sau này; chưa implement hay train LoRA adapters trong phase này.

Local `save_pretrained`/`from_pretrained` hoạt động và đã verify exact predictions
cho cả sáu model. Artifact folder có thể upload lên HF Hub cùng code/dependencies;
chưa đăng ký `AutoModel`, chưa implement remote Hub download, chưa push bất kỳ file.

## 12. Các điểm cần biết trước khi train thật

- Đây là máy CPU/macOS, chưa verify RTX4090 AMP/compile/VRAM/throughput. BF16→FP16,
  TF32, pin/persistent/prefetch workers, accumulation và bounded batch probe đã
  implement. Probe opt-in, mặc định giữ 2GiB headroom, không chạy ở phase này.
- ModernTCN default có khoảng 50.6M params theo official run config; kiểm tra batch
  vừa VRAM trên máy train. Không giảm architecture dựa trên validation.
- Các split hiện tại là provisional; chọn cutoff cuối cùng trước base training.
  Resume không đổi split/optimizer contract; đổi experiment dùng full fine-tune.
- LiT exact unpublished hyperparameters, HFformer paper/notebook mismatch và
  unspecified OFI-LSTM paper được ghi rõ, không che dưới claim “exact replication”.
- Không có real training checkpoint. Smoke weights chỉ là initialized weights và
  đã được xóa cùng temporary directories sau khi roundtrip test.
- CSV ngoài workspace: command phải cung cấp `--csv`, hoặc sửa path config cho máy
  train. Source CSV vẫn giữ nguyên.

## 13. Commands training sau này — chỉ ghi, chưa chạy

```bash
source .venv/bin/activate
export LOB_CSV=/Users/son/Projects/P0_LOB/data/processed/BTCUSDT_L10_oct2023.csv
python train.py --model ofi_lstm --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name of_60s
python train.py --model hfformer --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name hf_60s
python train.py --model patchtst --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name patch_60s
python train.py --model moderntcn --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name tcn_60s
python train.py --model lit --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name lit_60s
```

Commands prepare/smoke/E0/evaluation/resume/fine-tune/head-only chi tiết ở README.
Đổi `--history-seconds 120|180`, stride, workers, accumulation, compile hoặc probe
chỉ khi chủ động chạy experiment mới. Không có command training nào ở trên được chạy.
