# Report 1 — Cách làm AutoTS và TimesFM hiện tại

Ngày: 2026-09-10  
Trạng thái: báo cáo code đang có, chưa đề xuất sửa và chưa triển khai  
Phạm vi: AutoTS và TimesFM cũ của pipeline OHLCV  
Ngoài phạm vi: order book, các model ML/DL còn lại  
Kiểm thử trong phiên: không chạy smoke test, canary, training, inference hoặc test suite

## 1. Mục tiêu chung của pipeline cũ

Repo hiện dự báo BTC theo phút với target:

```text
y_h(t) = log(C[t+h] / C[t]), h ∈ {1, 2, 3} phút
P_hat[t+h] = C[t] × exp(y_hat_h)
```

Metric được tính trên giá bằng RMSE/MAE. E0 là dự báo giá không đổi. Validation gồm năm fold time-series; mỗi fold dùng FIT, ES, purge rồi outer VAL. Dữ liệu hiện tại chỉ có OHLCV/amount một phút và OHLCV/amount năm phút, không có order book.

AutoTS và TimesFM không dùng `TargetTransform` của các tree model. Hai model dự báo ba one-step log-return `r_hat[t+1:t+3]`, sau đó cộng dồn:

```text
y_hat_1 = r_hat_1
y_hat_2 = r_hat_1 + r_hat_2
y_hat_3 = r_hat_1 + r_hat_2 + r_hat_3
```

## 2. AutoTS hiện tại

### 2.1 Hai model probe cố định

Code chính nằm ở [`src/p0/models_autots.py`](../src/p0/models_autots.py). Project không dùng genetic search mặc định cho bước probe mà gọi trực tiếp hai class của AutoTS 1.0.4:

| Nhánh | AutoTS model | Regression backend | Cấu hình chính |
|---|---|---|---|
| `autots_wr` | `WindowRegression` | LightGBM GPU | window 60, output cùng lúc 3 horizon, 400 trees |
| `autots_mr` | `MultivariateRegression` | XGBoost GPU | feature time-series nội bộ, 400 trees |

Input target là một series `r1`. Regressor là các feature OHLCV/amount đã được chuẩn hóa bằng thống kê FIT và điền NaN sau chuẩn hóa.

Trong expanded-data:

- WR bắt đầu từ 72 cột B0 và 21 ext cũ đã khóa.
- MR bắt đầu từ 72 cột B0 và 8 ext cũ đã khóa.
- Mỗi nhánh còn thử tuần tự 163 candidate ngắn.

Mỗi candidate chạy toàn bộ năm fold. Trong từng fold:

1. Dựng dataframe `r1` và dataframe regressor.
2. Fit AutoTS model đúng một lần trên FIT.
3. Với mỗi origin outer VAL, gọi `fit_data(df <= t)` để thay phần data được nhìn thấy nhưng không refit regression model.
4. Gọi `predict(forecast_length=3)`.
5. Cộng dồn ba return và tính metric giá.

### 2.2 Alignment causal của regressor

Hai class AutoTS ghép regressor khác nhau nên code có hai phép dịch riêng:

- MultivariateRegression: hàng target tại `s` dùng feature `f(s-1)`.
- WindowRegression: AutoTS ghép regressor tại vị trí đầu của window, nên adapter lưu `R[s] = f(s + window_size - 1)`; lúc forecast chỉ truyền feature đã biết tại origin `f(t)`.

Code còn tránh bug AutoTS 1.0.4 ở `future_regressor.reindex(df)` bằng cách gọi `fit_data(df_slice)` không kèm regressor rồi gán lại đúng phần đuôi `regressor_train` cần cho predict.

### 2.3 Feature loop của hai probe

Mỗi nhánh WR/MR đi qua quy trình chung:

```text
S0 của nhánh
→ baseline và đo seed noise
→ add-one tuần tự 163 candidate
→ KEEP/DROP theo MedianGain và epsilon
→ permutation importance các cột mới
→ raw vs pruned confirmation bằng 3 seed
→ F_WR_best hoặc F_MR_best
```

Hai kết quả này chỉ là probe để tìm feature set; chúng chưa phải AutoTS-final và không được đưa trực tiếp vào champion.

### 2.4 AutoTS framework bake-off

Sau hai probe, [`src/p0/autots_search.py`](../src/p0/autots_search.py) và `cmd_autots_search` chạy thêm một tầng lựa chọn:

1. Freeze `F_WR_best` và `F_MR_best`.
2. Với từng frozen set, dựng các template WR/MR có backend GPU do project khai báo.
3. Gọi framework `AutoTS` với `max_generations=0`, `transformer_max_depth=0`, `num_validations=10`.
4. Chia template theo nhóm có cùng cách shift regressor.
5. Chọn template trong training-side của từng fold.
6. Refit template thắng và rolling predict outer VAL.
7. Chọn WR/MR cuối bằng metric của project, sau đó confirmation ba evaluation seed.
8. Ghi `wins/autots.json` làm AutoTS-final.

`max_generations=0` nghĩa là đây là bake-off template cố định, không phải genetic evolution. Lý do là generation mới của AutoTS có thể sinh regression backend CPU và ghi đè cấu hình GPU của project.

### 2.5 Kết quả và chi phí đã biết

Artifact vòng 15 ngày chọn `F_WR_best|wr:60` làm AutoTS-final, nhưng `MedianGain_vs_E0 = -2.0578 pp`. Nghĩa là AutoTS cũ không thắng E0 ở vòng này.

Nguồn thời gian chạy dài trong expanded-data:

```text
163 candidate × 5 fold × 4.320 origin
= 3.520.800 origin-prediction cho mỗi nhánh WR hoặc MR
```

Số đo lịch sử ghi khoảng 11,74 ms/origin cho WR và 28,19 ms/origin cho MR. Riêng rolling prediction tương ứng khoảng 11,5 giờ WR và 27,6 giờ MR nếu chạy tuần tự. Các số này chưa gồm fit theo candidate, baseline, prune, confirmation và AutoTS bake-off. Plan hiện tại ước lượng riêng bake-off khoảng 5–9 giờ.

Vì FIT expanded-data dài 120 ngày và số cột lớn, ước lượng từ vòng 15 ngày có xu hướng thấp hơn runtime thực tế.

### 2.6 Đường lệnh hiện tại

Nếu chạy đúng pipeline cũ, thứ tự là:

```bash
python run.py loop --config configs/p0_full.json --model autots_wr
python run.py loop --config configs/p0_full.json --model autots_mr
python run.py autots-search --config configs/p0_full.json
```

Các lệnh chỉ được ghi lại để mô tả code hiện tại; chúng không được chạy trong phiên báo cáo này.

## 3. TimesFM hiện tại

### 3.1 Backbone và output

Code nằm ở [`src/p0/models_tfm.py`](../src/p0/models_tfm.py). Model `tfm` hiện trỏ tới `TimesFMLoRAModel`; lớp zero-shot `TimesFMModel` chỉ còn là reference.

Backbone:

- Package `timesfm==2.0.2`.
- Checkpoint `google/timesfm-2.5-200m-pytorch` với revision pin.
- Context 512 return một phút.
- Forecast horizon API compile là 128 nhưng chỉ lấy ba bước đầu.
- `normalize_inputs=True`, `infer_is_positive=False`, flip invariance bật.
- Dùng mean head `quantile_forecast[..., 0]`; `point_forecast` là q50.

Native path nhận từng context `r1[t-511:t]`, dự báo ba return và cộng dồn thành ba target log-return.

### 3.2 LoRA per fold

TimesFM đang được fine-tune bằng LoRA tự cài trong repo, không dùng PEFT:

- Rank 8, alpha 16, dropout 0.
- Inject vào `qkv_proj`, attention output và hai linear FFN của 20 block.
- Khoảng 2,048 triệu tham số học; backbone đóng băng.
- AdamW, learning rate `1e-4`, weight decay `0.01`, batch 64.
- Tối đa 20 epoch, patience 5.
- Loss là MSE trên vector MIMO `y_hat_1..3` sau khi cộng dồn return.

Mỗi fold dùng FIT để train, ES để chọn epoch và không nhìn outer VAL. Adapter được lưu theo fold/seed/epoch, hash trước và sau inference để bảo đảm đã freeze.

Plan hiện tại có thể tạo bảy adapter mỗi fold: một calibration, ba evaluation và ba confirmation; tổng cộng 35 lượt adapter trên năm fold.

### 3.3 XReg sau LoRA

Sau khi adapter được freeze, feature ext đi qua `forecast_with_covariates` của TimesFM:

1. Covariate tại hàng target `s` là feature `f(s-1)`.
2. Ba bước tương lai giữ feature tại origin `f(t)`.
3. `xreg_mode="xreg + timesfm"`, target được normalize theo input, `ridge=0`.
4. XReg fit một hồi quy in-context, trừ phần giải thích được khỏi return, TimesFM forecast residual rồi cộng phần XReg lại.

API fit một `beta_hat` chung cho mọi series trong một lời gọi. Nếu batch nhiều rolling origin, context của origin sau sẽ ảnh hưởng origin trước và gây leakage. Vì vậy code buộc:

```text
1 origin / forecast_with_covariates call
per_core_batch_size = 1
```

Đây là quyết định đúng về causal nhưng là nút thắt runtime lớn nhất.

### 3.4 Feature loop và TimesFM-final

TimesFM expanded-data bắt đầu với `S0 = empty`; B0 cũ không đi vào XReg. Luồng hiện tại:

```text
pretrained TimesFM
→ LoRA FIT/ES per fold và seed
→ freeze adapter
→ baseline LoRA native
→ add-one 163 ext candidate qua XReg
→ permutation importance
→ raw-vs-pruned confirmation
→ hệ thống B = cùng LoRA + XReg(F_win)
→ so hệ thống B với hệ thống A = LoRA native
→ TimesFM-final
```

XReg không được coi là model riêng. A và B phải dùng cùng adapter đã freeze.

### 3.5 Kết quả và chi phí đã biết

Artifact vòng 15 ngày chọn TimesFM không covariate; `MedianGain_vs_E0 = -1.9958 pp`. Nhánh OHLCV/XReg cũ chưa tạo được skill so với E0.

Artifact lịch sử trên RTX 3090 ghi khoảng 682 ms/origin cho XReg hai covariate; một lần đo cũ khác khoảng 1.011 ms, tức hơn một giây cho mỗi origin. Chỉ riêng add-one expanded-data ở mức đo 682 ms:

```text
163 × 5 × 4.320 = 3.520.800 XReg calls
3.520.800 × 0,682 s ≈ 667 giờ GPU tuần tự
```

Hai GPU chỉ giảm cận này khoảng một nửa nếu luôn được sử dụng hoàn hảo. Baseline, các bộ feature tích lũy, prune, confirmation và LoRA training chưa nằm trong 667 giờ. Vì vậy việc vượt xa ETA cũ là hệ quả trực tiếp của cấu trúc tính toán, không phải chỉ do GPU chậm.

### 3.6 Đường lệnh hiện tại

Pipeline cũ chạy:

```bash
python run.py loop --config configs/p0_full.json --model tfm
python run.py tfm-final --config configs/p0_full.json
```

Các lệnh này không được chạy trong phiên báo cáo.

## 4. Kết luận về hai đường cũ

AutoTS và TimesFM hiện tại là hai pipeline hoàn chỉnh cho OHLCV, nhưng cả hai đều nhân runtime theo candidate × fold × origin. TimesFM còn nhân thêm adapter/seed và không thể batch XReg rolling-origin bằng API hiện tại. Hai đường này không có representation order flow, elapsed time hay sequence order-book event trong [`docs/IDEA.md`](IDEA.md).

Vì vậy nên coi chúng là nhánh cũ độc lập để lưu tài liệu và quyết định chạy riêng. Không gắn order book vào hai pipeline này trong proposal mới. Thiết kế order-book cho ML/DL hiện tại nằm ở [`REPORT_2_ORDERBOOK_ML_DL.md`](REPORT_2_ORDERBOOK_ML_DL.md).
