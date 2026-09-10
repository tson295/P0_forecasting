# Order-book Direct forecasting

Pipeline mới trong `src_OB`; `src/p0` cũ được giữ nguyên. Không có smoke command hoặc feature search.
Code được viết theo yêu cầu, chưa chạy, chưa test, chưa training local.

## Dữ liệu và nhãn

- BTCUSDT Binance spot, L2 **10 level mỗi phía**.
- Historical dataset cố định **[2024-09-10 00:00:00, 2026-09-10 00:00:00) UTC**, đúng 730 ngày. Không stream realtime, không tự rút xuống 1–1,5 năm. `download` chỉ tải archive, `prepare` chỉ đọc file lịch sử và `train` chỉ đọc prepared memmap.
- Nguồn đã hỗ trợ: Tardis `book_snapshot_25`, lấy 10 level đầu. Đây là snapshot do nhà cung cấp replay từ feed, không phải REST snapshot hiện tại.
- Mỗi snapshot có **một** mid-price: `MP_t = (bid_price_0 + ask_price_0) / 2`. Không tính một mid riêng cho từng level.
- `local_timestamp` (microseconds UTC) là thời điểm thông tin sẵn có. `timestamp` sàn được đọc nhưng không dùng để nhìn trước arrival time.
- Tính OF theo price/volume ở từng level trước khi lọc mid-price. Bid flow và ask flow có dấu liquidity supply; `OFI = bid_OF - ask_OF`.
- Các dòng liên tiếp cùng mid bị bỏ khỏi tập origin/sequence model; flow trong đoạn bị bỏ được cộng tới lần mid đổi kế tiếp. Giữ toàn bộ timestamp/raw mid riêng để lấy nhãn đúng thời gian.
- Baseline input `[bid_OF, ask_OF, OFI] × 10`, cộng elapsed time giữa hai mid-change snapshot, elapsed time giữa hai raw snapshot cuối và update count (log1p): **33 feature**. Flow giữ đơn vị volume gốc trước bước scaler trên FIT.
- `include_distances=false` mặc định. Có thể bật thành `true` rồi tạo một `prepared_dir` khác để thêm `bid_distance_bps` và `ask_distance_bps` cho input đa biến: **53 feature**. Distance không thuộc baseline OF/OFI; TimesFM zero-shot luôn bỏ qua toàn bộ covariate.
- Nhãn `log(mid(t+h)/mid(t))`, h = **60, 120, 180 giây**, không phải h dòng sau khi lọc. `mid(t+h)` là quote cuối đã tới tại thời điểm đó; không lấy quote đầu tiên sau horizon.
- Khi tính **RMSE, MAE, R²**, đổi về `predicted_price = mid(t) * exp(predicted_log_return)`. Metric trên giá USDT; giá ở đây là mid L2, không phải OHLCV close.
- Các khoảng feed im lặng >10 giây được tách segment; window/label không vượt segment. Age tối đa khi lấy quote là 10 giây. Đây là ngưỡng cấu hình, không chứng minh đã phát hiện mọi mất gói: CSV snapshot không có sequence ID.

OF ở mỗi level `i`, với `p` là price và `q` là volume:

| Biến động price của level | bid_OF | ask_OF |
|---|---|---|
| Tăng | `q_bid(t)` | `-q_ask(t-1)` |
| Giữ nguyên | `q_bid(t)-q_bid(t-1)` | `q_ask(t)-q_ask(t-1)` |
| Giảm | `-q_bid(t-1)` | `q_ask(t)` |

Tính từng raw transition trước, cộng bid_OF/ask_OF tới snapshot được giữ tiếp theo, rồi lấy
`OFI_i = accumulated_bid_OF_i - accumulated_ask_OF_i`. So sánh price của từng phía riêng biệt.

## Models

Mỗi `(fold, family, horizon)` có model/adapter riêng và scalar output, không dùng prediction h=1 để suy ra h=2/3.

| Family | Input và cách dùng |
|---|---|
| LightGBM, XGBoost, CatBoost, XGB-RF | Cùng feature cố định: snapshot mới nhất, mean 10/100 event, tổng OFI theo level; Direct riêng từng h |
| LSTM | 100 mid-change event với OF/elapsed time; 1 lớp hidden 64, scalar linear head |
| AutoTS | Native AutoTS tự search backend LightGBM/XGBoost, tham số và window trong family `WindowRegression`; mọi candidate fit bằng GPU, nhận OF/OFI/elapsed cố định |
| TimesFM zero-shot | Chỉ chuỗi một mid-price, đổi sang log và center theo giá tại origin; pretrained univariate, không train, không OF/XReg |
| TimesFM LoRA | Adapter riêng mỗi h; cùng pretrained mean decoder + đầu residual OF/elapsed nhỏ, fine-tune đồng thời; không XReg search |

Không thêm DeepLOB/DeepLOB-inspired vì không có model này trong `src/p0`.

### AutoTS và chính sách GPU

AutoTS không có bảo đảm GPU cho toàn bộ catalogue. Các implementation sklearn như RandomForest,
ExtraTrees, KNN, SVM, ElasticNet chạy CPU; những model thống kê cũng không tự chuyển sang GPU.
Vì vậy search hiện giới hạn `WindowRegression` với **hai backend GPU** LightGBM/XGBoost, không mở toàn catalogue.
Không fix sẵn backend thắng, số cây, learning rate hay window. Default: 12 candidate ban đầu, 3 generation,
2 vòng validation bổ sung. AutoTS sinh/chấm/chọn tham số; không search subset feature hoặc learned transform.

Adapter trong `autots_native.py` chỉ giữ GPU allowlist và căn feature quan sát được theo window của candidate.
Regressor mang nhãn thời gian `t+h` nhưng giá trị luôn là feature đã biết tại `t`, không lấy OF tương lai.
Search dùng các block 64 origin nằm trong outer FIT, mỗi block cách cutoff fit ít nhất 6 ngày;
mỗi origin vẫn được dự báo direct một bước, context được cập nhật bằng quan sát lịch sử, không refit ở gap/VAL.
Model thắng được refit trên outer FIT; outer VAL không tham gia chọn model. Lưu toàn bộ bảng kết quả search.

GPU-only ở đây áp dụng cho **fit model**. Đọc file, tạo feature, scaler, điều phối AutoTS và tính metric dùng CPU.
XGBoost nhận CUDA arrays, chặn cảnh báo chuyển CPU và yêu cầu CUDA backend; LightGBM/CatBoost đặt GPU backend
tường minh. Thiếu GPU/build hỗ trợ thì dừng job, không retry bằng CPU. LightGBM/CatBoost có thể predict trên CPU;
đây không phải fallback training. Những guard này chỉ chạy trong job thật, không có trial fit hay smoke test.

CPU không đồng nghĩa mọi fit đều lâu: model tuyến tính nhỏ có thể nhanh, model nặng/candidate nhiều có thể lâu.
AutoTS tổng thời gian còn nhân theo candidate, validation, fold và horizon; hiện chưa có số đo để đưa ETA.

### TimesFM LoRA

Không còn vòng feature search/XReg. Một `(fold, horizon)` train **một adapter và OF head cùng một optimizer**.
Default 5 fold × 3 horizon = **15 lần fine-tune**, không phải một model chung toàn bộ thí nghiệm.
Base checkpoint được freeze; default 5 epoch, batch 32, context 512. Bỏ search giảm số lần fit, nhưng chưa thể
khẳng định chạy rất nhanh trên 2 năm LOB: còn phụ thuộc số origin, GPU/VRAM và tốc độ đọc dữ liệu.

AutoTS/TimesFM cần chuỗi đều để horizon có nghĩa thời gian: ở mỗi h, context price lấy cách nhau h giây;
forecast **1 step = t+h**. Origin chấm điểm vẫn là các lần mid đổi. Không giả lập irregular event index thành phút.
AutoTS fit trên price grid h giây của FIT; regressor căn theo window của API. Zero-shot TimesFM giữ checkpoint gốc
và không được mô tả là model đã học OF. LoRA dùng bộ decode có gradient và LoRA helper đã có trong `src/p0`,
chỉ import các hàm đó, không gọi training/search harness cũ.

## Walk-forward

Default: rolling FIT 120 ngày, **gap 6 ngày**, VAL 7 ngày; 5 fold, bước 30 ngày, đi theo thời gian.
Nhãn cuối cùng của training phải nằm trước train_end. Window training không bắt đầu trước train_start.
Không early-stop/tune trên outer VAL; số cây/epoch của các model ngoài AutoTS khóa trong config.
AutoTS chọn tham số bằng validation nội bộ FIT. Scaler và target scale chỉ lấy từ FIT.
Gap 6 ngày tuân theo yêu cầu; nhãn dài tối đa 3 phút tự nó không đòi gap 5 ngày.

Tất cả model chấm trên cùng origin mask được quyết định bởi danh sách `models` trong config, kể cả khi tách job bằng `--models`.
Nguồn thiếu ngày không được tự lấp vào hoặc coi các ngày đầu tháng rời rạc là 2 năm liên tục.

## Dùng trên Vast

Từ root repo, dùng image Vast CUDA 12, cài CUDA-enabled PyTorch và GPU-enabled LightGBM trước, rồi:

```bash
pip install -r src_OB/requirements-vast.txt
```

Default LightGBM `device_type=cuda`; nếu build trên Vast dùng OpenCL, đổi `tree.lightgbm_device` thành `gpu`.
Không CPU fallback. Không có lệnh training nào được chạy trên local trong phiên này.

Tải lịch sử **[2024-09-10, 2026-09-10)** (2 năm). Đặt `TARDIS_API_KEY` trong môi trường máy tải dữ liệu trước:

```bash
python -m src_OB download --config configs/orderbook.json
```

Downloader tải từng ngày, ghi file tạm rồi rename, có retry và tiếp tục các ngày đã hoàn tất.
401/403 báo thiếu quyền, 404 báo thiếu ngày; không tự thay nguồn/market hoặc mua dữ liệu.
**Hiện chưa có order-book dataset trên đĩa workspace; chưa tải lịch sử trong phiên này.**
Downloader cần API key có quyền đủ 730 ngày; không thay bằng các sample miễn phí đầu tháng.

Các đường dẫn đã cấu hình, không phải dữ liệu đã tồn tại:

- Raw: `data/orderbook/binance/BTCUSDT/YYYY-MM-DD.csv.gz` và `download_manifest.json`.
- Prepared: `data/orderbook/prepared/{raw_ts,raw_mid,raw_segment,ts,mid,segment,features}.bin` và `manifest.json`.
- Kết quả training: `experiments/orderbook/`.

Thiếu bất kỳ ngày nào trong khoảng freeze thì `prepare` dừng. Manifest schema v2 lưu range và layout OF/OFI;
prepared dataset cũ có distance layout khác cần được tạo lại sang một thư mục mới.

Sau khi có raw data, chuẩn bị memmap một lần trên Vast/máy lưu dữ liệu:

```bash
python -m src_OB prepare --config configs/orderbook.json
```

`prepare` là xử lý dataset thật, không phải test. Đọc CSV theo chunk, ghi feature và raw-price memmap trên đĩa;
không tạo sẵn tensor `[toàn bộ sample, context, feature]` trong VRAM. Thư mục prepared mới phải chưa tồn tại;
đổi `prepared_dir` khi cần tạo phiên bản khác. Các lỗi input được raise trong đường chạy thật.

Chạy trên GPU Vast, từng fold/horizon tuần tự:

```bash
P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train --config configs/orderbook.json
```

Hoặc chia fold cho hai GPU thành hai job:

```bash
P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train --folds fold1,fold2,fold3
P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=1 python -m src_OB train --folds fold4,fold5
```

`--models lgbm,xgb,lstm` chọn subset để chạy, không chọn feature. Checkpoint và predictions ghi ở
`experiments/orderbook/foldN/model/h60s` (tương tự h120s/h180s), gồm model, prediction giá, metric và config.
Thư mục kết quả đã tồn tại sẽ không bị ghi đè. Dùng `output_dir` mới nếu chạy lại; không có tự động resume training dở.
Tree matrix nằm trong RAM host; cửa sổ neural chỉ chuyển batch lên GPU. Nhu cầu đĩa/RAM và thời gian chạy thực tế chưa đo.

## Nguồn

- [Tardis schema](https://docs.tardis.dev/downloadable-csv-files/data-types): snapshot L2 đã dựng, timestamps microsecond, lấy 10/25 level.
- [Tardis API](https://docs.tardis.dev/downloadable-csv-files/api): lịch sử liên tục cần API key; miễn phí chỉ ngày đầu mỗi tháng.
- [AutoTS regressors](https://github.com/winedarksea/AutoTS/blob/1.0.4/autots/models/sklearn.py): native regressor generation và WindowRegression.
- [AutoTS search](https://github.com/winedarksea/AutoTS/blob/1.0.4/autots/evaluator/auto_ts.py): model search và custom validation indexes.
- [IDEA](../docs/IDEA.md): nguồn ý tưởng OF và elapsed time; yêu cầu Direct mới thay thế đề xuất MIMO cũ.
