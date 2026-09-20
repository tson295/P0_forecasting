Bạn đang làm việc trên local repo hiện tại của project BTC LOB forecasting.

## Mục tiêu

Triển khai code để chuẩn bị train các model sau trên dataset L10 hiện tại:

1. E0 baseline
2. OFI + LSTM
3. HFformer
4. PatchTST
5. ModernTCN
6. LiT adapted từ L20 xuống L10

**KHÔNG chạy training thực tế ở bước này.**
Chỉ implement code, kiểm tra syntax/import/shape bằng smoke test rất nhỏ nếu cần.

Dataset hiện tại là:

`BTCUSDT_L10_oct2023.csv`

Data có khoảng 678k snapshots, khoảng cách giữa các snapshot chủ yếu ~1.24s.

---

## 1. Bài toán chung

Dùng LOB L10 để dự đoán trực tiếp return tại:

* 1 phút
* 2 phút
* 3 phút

Target:

$$
y_h(t)=\log\frac{MP_{t+h}}{MP_t}
$$

với:

* `MP_t = (best_bid + best_ask) / 2`
* `h ∈ {60s, 120s, 180s}`

Không recursive forecasting.

Output của mỗi learned model phải là vector:

```python
[pred_1m, pred_2m, pred_3m]
```

E0 baseline:

```python
pred_1m = pred_2m = pred_3m = 0
```

tương đương dự đoán future mid-price bằng mid-price hiện tại.

---

## 2. Dataset/windowing

Không materialize toàn bộ overlapping windows ra disk.

Dataset class chỉ giữ raw arrays + valid origin indices và slice window khi `__getitem__`.

Timestamp trong data là irregular, khoảng 1.24s/snapshot.

### Continuity rule

Tính:

```python
dt[i] = timestamp[i+1] - timestamp[i]
```

Nếu bất kỳ:

```python
dt > 2.0 seconds
```

thì coi đó là discontinuity.

Một sample chỉ hợp lệ nếu:

1. toàn bộ history window không chứa gap > 2s;
2. đoạn từ origin tới target 3 phút cũng không đi qua gap > 2s;
3. không đi qua boundary/segment khác nếu data có segment identifier.

Nên precompute prefix sum của `bad_gap` để kiểm tra window O(1), không scan từng window mỗi lần.

---

## 3. History và stride

Thiết kế config để hỗ trợ:

```text
history_seconds ∈ {60, 120, 180}
```

Quy đổi history sang số rows dựa trên median sampling interval của TRAIN split, không hard-code 1 second.

Gần đúng hiện tại:

```text
60s  ≈ 49 rows
120s ≈ 97 rows
180s ≈ 146 rows
```

Stride mặc định tương ứng:

```text
60s  -> khoảng 10s
120s -> khoảng 20s
180s -> khoảng 30s
```

tức với sampling hiện tại xấp xỉ:

```text
8 / 16 / 24 rows
```

Nhưng code nên derive từ `median_dt`, không hard-code.

Phải có config để đổi stride sau này.

---

## 4. Time-based targets

Không lấy target đơn giản bằng `i + 48`, `i + 97`, ... nếu timestamp irregular.

Với mỗi origin timestamp `t`, tìm observation đầu tiên tại hoặc ngay sau:

```text
t + 60s
t + 120s
t + 180s
```

Có tolerance configurable.

Ví dụ:

```python
target_tolerance_seconds = 2.0
```

Nếu không tìm được target đủ gần hoặc target interval đi qua discontinuity thì drop sample đó.

Mọi target lookup phải causal và deterministic.

---

## 5. Split

Thiết kế split theo timestamp, không random split.

Hiện tại chưa cần tối ưu split cuối cùng, nhưng pipeline phải hỗ trợ:

```text
train
validation
test
```

theo chronological order.

Không normalization/statistics nào được dùng thông tin validation/test.

Save split manifest để reproduce được.

---

# MODEL IMPLEMENTATIONS

## 6. OFI + LSTM

Implement multi-level order flow từ L10.

Với mỗi level tính bid-side OF và ask-side OF dựa vào thay đổi price/quantity giữa hai snapshot liên tiếp.

Có thể support cả:

```text
OF representation: 20 channels
= bid_OF_1 ... bid_OF_10
+ ask_OF_1 ... ask_OF_10
```

và optional:

```text
OFI representation: 10 channels
= bid_OF_i - ask_OF_i
```

Default trước mắt dùng representation sát paper nhất có thể.

Pipeline:

```text
raw L10
-> OF/OFI
-> normalization train-only
-> LSTM
-> FC
-> 3 returns
```

Model code phải tách riêng khỏi dataset code.

---

## 7. HFformer

Ưu tiên reuse/adapt architecture từ public HFformer implementation thay vì tự sáng tạo model mới.

Feature set sát paper:

* bid prices level 1–9
* bid quantities level 1–9
* ask prices level 1–9
* ask quantities level 1–9
* lagged log-return
* weighted mid-price

Tổng 38 features.

Data mình có L10 nên HFformer mặc định chỉ dùng L1–L9 để sát paper.

Giữ architecture paper càng nhiều càng tốt:

```text
input
-> Transformer encoder
-> prediction head
```

Nếu implementation paper có activation/normalization đặc biệt thì giữ lại.

Chỉ thay đổi phần bắt buộc:

```text
classification / original prediction head
-> regression head size 3
```

Không fine-tune hyperparameter ở bước này.

---

## 8. PatchTST

Ưu tiên dùng official/open implementation hoặc Hugging Face implementation nếu dễ maintain.

Đây là supervised model from scratch cho data hiện tại, không load pretrained external checkpoint ở phase này.

Input raw L10 multivariate series:

```text
bid_price_1..10
bid_qty_1..10
ask_price_1..10
ask_qty_1..10
```

= 40 channels.

Output:

```text
3 direct return predictions
```

Giữ kiến trúc PatchTST nguyên bản nhiều nhất có thể:

* patching theo temporal dimension
* channel-independent processing
* Transformer encoder

Chỉ custom input/output/config cần thiết.

---

## 9. ModernTCN

Ưu tiên adapt official implementation.

Input mặc định giống PatchTST:

```text
40 raw L10 channels
```

Giữ architecture ModernTCN:

```text
patch embedding
-> ModernTCN blocks
-> forecast/regression head
```

Output:

```text
3 returns
```

Không tune architecture trong bước này.

---

## 10. LiT

Không có official implementation đáng tin cậy, nên implement dựa trên architecture trong paper:

```text
LOB price/volume tensor
-> structured spatial-temporal patches
-> linear patch projection
-> learnable positional embedding
-> Transformer encoder
-> LSTM
-> regression head
```

Paper dùng L20 nhưng data hiện tại là L10.

Adapt:

```text
L20 -> L10
```

nhưng giữ nguyên ý tưởng spatial structure.

Không flatten raw LOB ngay từ đầu.

Tổ chức input thành price/volume channels sao cho bid/ask/depth structure vẫn được bảo toàn.

Default history có thể hỗ trợ paper-style 64 snapshots, nhưng toàn pipeline vẫn phải tương thích với history configurable bên trên.

Output head:

```python
Linear(..., 3)
```

không softmax.

---

# TRAINING INFRASTRUCTURE

## 11. Chưa train nhưng phải chuẩn bị trainer

Implement training infrastructure đầy đủ để sau này chỉ cần chạy command.

Ví dụ:

```bash
python train.py --model ofi_lstm ...
python train.py --model hfformer ...
python train.py --model patchtst ...
python train.py --model moderntcn ...
python train.py --model lit ...
```

Nhưng **không chạy những command training này bây giờ**.

Có thể thêm:

```bash
python train.py --model lit --smoke-test
```

để chỉ lấy vài batch và chạy forward/backward 1–2 step nếu thật sự cần kiểm tra code.

Không được vô tình chạy toàn bộ epoch.

---

## 12. RTX 4090 24GB optimization

Code chuẩn bị cho 1 × RTX 4090 24GB.

Không được để GPU bị starvation do dataloader kém.

Chuẩn bị:

* BF16 ưu tiên nếu supported
* fallback FP16
* TF32
* pin_memory
* persistent_workers
* configurable num_workers
* prefetch_factor
* non_blocking GPU transfer
* contiguous numeric arrays
* không dùng pandas trong `__getitem__`
* optional `torch.compile`
* gradient accumulation configurable

Implement optional auto batch-size probing:

```text
128 -> 256 -> 512 -> ...
```

để tìm batch size lớn hợp lý trước training thật.

Nhưng chỉ implement, **không benchmark GPU lâu ở bước này**.

Không ép VRAM đầy 100%; nên giữ khoảng 1.5–2GB headroom.

---

# CHECKPOINT / FUTURE LORA

## 13. Checkpoint phải dùng lại được cho fine-tuning/LoRA

Đây là yêu cầu quan trọng.

Model implementation phải được viết sao cho checkpoint base-training sau này có thể:

1. resume full training;
2. full fine-tune;
3. freeze backbone + train prediction head;
4. gắn LoRA vào Transformer attention/MLP modules.

Đối với Transformer models, đặt tên module rõ ràng nếu implementation cho phép:

```text
q_proj
k_proj
v_proj
out_proj
```

và FFN modules rõ ràng.

Không viết architecture theo kiểu khiến sau này rất khó inject LoRA.

Checkpoint format dự kiến:

```text
checkpoints/<model>/<run_name>/
    best/
        model.safetensors
        config.json
        preprocessing.json
        feature_schema.json
        target_config.json

    last/
        model.safetensors
        trainer_state.pt
        optimizer.pt
        scheduler.pt
        config.json
```

Có thể điều chỉnh format nếu có lý do kỹ thuật tốt hơn, nhưng phải giữ đủ thông tin để reproduce và fine-tune.

---

## 14. Hugging Face compatibility

Chuẩn bị code để checkpoint sau này push lên Hugging Face dễ dàng.

Không cần push bây giờ.

Ít nhất phải save:

```text
model weights
model config
feature schema
normalization/preprocessing information
history configuration
target definition
training/split manifest
```

Ưu tiên `safetensors`.

Nếu hợp lý, viết helper:

```python
save_pretrained(...)
from_pretrained(...)
```

hoặc abstraction tương đương.

---

# CODE QUALITY

## 15. Không over-engineer

Code phải:

* rõ ràng;
* dễ đọc;
* dễ chạy;
* ít dependency không cần thiết;
* model riêng, dataset riêng, training riêng;
* config tập trung;
* không tạo framework khổng lồ.

Có thể dùng cấu trúc kiểu:

```text
src/
    data/
        dataset.py
        preprocessing.py
        ofi.py
    models/
        ofi_lstm.py
        hfformer.py
        patchtst.py
        moderntcn.py
        lit.py
    training/
        trainer.py
        checkpoint.py
    utils/
        metrics.py

train.py
configs/
```

Nhưng nếu repo hiện tại đã có structure tốt thì tích hợp theo structure hiện có, không phá repo chỉ để giống layout trên.

---

# TEST BẮT BUỘC

Sau khi code xong:

1. chạy syntax/import checks;
2. load CSV;
3. xác nhận column mapping;
4. tính lại basic stats:

   * number of rows
   * median dt
   * p99 dt
   * max dt
   * number of gaps > 2s
5. tạo Dataset cho ít nhất một history configuration;
6. lấy vài samples;
7. kiểm tra target timestamps 1/2/3 phút;
8. kiểm tra không sample nào cross gap > 2s;
9. instantiate tất cả model;
10. chạy dummy/small-batch forward cho từng model;
11. xác nhận output shape:

```python
[B, 3]
```

12. nếu cần, chạy đúng 1 backward step trên synthetic/small data để chắc chắn graph hoạt động.

**Không chạy epoch training thật.**

---

# SAU KHI HOÀN THÀNH

Không tự bắt đầu training.

Hãy báo cáo lại chính xác:

1. Những file đã tạo/sửa.
2. Dataset/windowing hiện hoạt động như thế nào.
3. Số rows và thống kê `dt` thực tế đọc được.
4. Rule `gap > 2s` được implement ở đâu.
5. Cách tìm target 1/2/3 phút trên timestamp irregular.
6. Feature chính xác của từng model:

   * OFI-LSTM
   * HFformer
   * PatchTST
   * ModernTCN
   * LiT
7. Architecture/config hiện tại của từng model.
8. Những phần nào giữ nguyên từ paper/code gốc và những phần nào bắt buộc phải custom.
9. Kết quả smoke test của từng model, gồm input/output shape.
10. Checkpoint system đã chuẩn bị thế nào cho future LoRA/full fine-tuning.
11. Hugging Face save/load đã chuẩn bị tới đâu.
12. Các vấn đề hoặc ambiguity còn tồn tại trước khi train thật.
13. Các command chính xác mà sau này sẽ dùng để bắt đầu training, **nhưng không chạy chúng**.

Không được tự ý tune hyperparameter dựa trên validation.
Không được tự ý thêm model mới.
Không được train thật.
