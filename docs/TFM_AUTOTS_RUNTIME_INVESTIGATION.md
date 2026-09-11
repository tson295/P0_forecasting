# Điều tra runtime TimesFM/XReg và AutoTS cũ

Branch điều tra ban đầu: `investigate/tfm-autots-runtime`, tạo từ `OB` tại `109cbc2`.
Implementation tiếp theo đã viết trên `tfm_autots`; xem `TFM_AUTOTS_PHASE.md`.
Các mô tả dưới đây ghi nhận code trước sửa. Chưa có test hoặc số đo runtime mới.
Phạm vi: đọc pipeline OHLCV `src/p0`; chưa sửa implementation, không test/training/inference/benchmark.
Đây là kết luận từ code và tài liệu audit trong repo, không phải số đo mới hoặc xác nhận runtime trên Vast.

## TimesFM: cần phân biệt LoRA training với rolling prediction

`src/p0/models_tfm.py`:

- `_ensure_adapter()` gọi `train_lora()` theo batch (default 64), rồi freeze và cache adapter.
  Key gồm khoảng FIT/ES, seed và epoch mode; không gồm candidate feature set.
  Vì vậy mỗi lần harness gọi `fit_predict()` không đồng nghĩa train lại LoRA cho từng origin/candidate.
- `_point()` đã chia origins theo `batch_size` (default 256).
- `_with_covariates()` lặp từng origin và gọi `forecast_with_covariates(inputs=[ctxs[k]])`.
  Wrapper covariate cố ý đặt `per_core_batch_size=1`, tránh padding một origin lên batch lớn.
- Cấu hình mặc định là `xreg + timesfm`. Theo audit source TimesFM 2.0.2 trong repo:
  fit XReg trên context → lấy residual context → TimesFM forecast residual → cộng XReg forecast trở lại.
  “TimesFM chạy trước, XReg bù residual sau” là mode khác: `timesfm + xreg`.

Code giải thích API XReg hiện gộp các series của một lời gọi để fit một beta chung. Với các rolling origins
khác thời điểm, gộp như vậy có thể để context origin sau ảnh hưởng prediction origin trước.
Không được chỉ đổi `inputs=[...]` thành nhiều origins hoặc tăng batch size rồi coi là tối ưu tương đương.

Hướng batching giữ phương pháp hiện tại:

```text
B origin contexts/covariates
  → B hồi quy XReg độc lập: mỗi origin có normalization, intercept và beta riêng
  → B residual contexts
  → TimesFM forecast theo batch
  → cộng XReg forecast tương ứng, khôi phục scale và cumsum đúng thứ tự
```

Có thể batch TimesFM sau khi đã tính XReg riêng cho từng origin; tiếp đó mới vectorize các phép giải XReg
độc lập trên GPU. Không dùng pooled beta. Giữ ridge, padding, ngưỡng pinv, normalization và head theo contract
cũ khi đánh giá tính tương đương; đổi solver/precision có thể đổi kết quả. Không đổi mode để tối ưu runtime.
Residual phụ thuộc feature set nên không cache cùng một TimesFM output xuyên mọi candidate có XReg.

## AutoTS: đã thấy những nguồn chi phí, chưa định lượng nút thắt lớn nhất

`src/p0/models_autots.py`:

- `fit_predict()` dựng model và gọi `m.fit()` một lần cho mỗi lượt fold/feature-set/seed.
- `_make_predictor()` lặp từng origin: tạo DataFrame tối đa 400 bar → `fit_data()` → dựng future regressor
  → `predict(3)`. MR còn dựng/gán `regressor_train` mỗi origin.
- `fit_data()` cập nhật context dùng khi forecast; không phải refit estimator. Không quy chậm cho việc
  train lại model ở mỗi origin chỉ vì tên phương thức có chữ fit.
- Tuy vậy rolling prediction vẫn có nhiều lời gọi Python/pandas/model với input nhỏ. GPU fit không tự
  loại bỏ chi phí chuẩn bị dữ liệu hoặc làm vòng prediction tuần tự thành batched inference.

`src/p0/loop.py` và `src/p0/cli.py`:

- Add-one search thử candidate tuần tự vì feature set thay đổi sau KEEP. Mỗi candidate chạy các fold,
  fit estimator rồi rolling predict. Fold có thể được scheduler phân phối; không phải toàn bộ pipeline
  luôn tuần tự trên một GPU.
- Permutation importance gọi predictor thêm theo feature/repeat; confirmation chạy các seed.
- `autots_search.py` còn gọi AutoTS template search có validation nội bộ; default config hiện là 10 validations.
  Sau đó template thắng được refit và chấm outer VAL. Đây là chi phí khác với rolling prediction.

Hướng cần xử lý: tách chi phí dựng dữ liệu, estimator fit, rolling prediction và template search từ log thật
đã có; nếu thiếu thì ghi rõ chưa biết. WR có thể dựng feature rows của nhiều origins rồi predict batch nếu
tái hiện đúng feature/alignment của AutoTS. MR cần giữ phụ thuộc recursive trong mỗi origin, nhưng có thể
batch nhiều origins độc lập theo từng forecast step. Không gộp nhiều origins thành một chuỗi forecast dài.

Chưa có bằng chứng mới để kết luận fit hay prediction chiếm bao nhiêu phần trăm, hoặc hứa mức tăng tốc.
Việc tiếp theo là chốt batching giữ nguyên phương pháp trên nhánh này; không thay `src_OB` hoặc luồng Vast OB.
