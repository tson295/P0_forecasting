# Order-book Direct forecasting

Pipeline mới trong `src_OB`; `src/p0` cũ được giữ nguyên. Không có smoke command hoặc feature search.
Vast 2026-09-10: download + prepare thật đã chạy; archive pinned chỉ dựng được 300,6 s book hợp lệ nên
data **BLOCKED**, chưa training (bằng chứng: `experiments/orderbook_hf/DATA_REPORT.md`, `RUN_REPORT.md`).

## Dữ liệu và nhãn

- BTCUSDT Binance spot, L2 **10 level mỗi phía**.
- Historical dataset: **MaximumLeverage/crypto-lob-stream** trên Hugging Face, pin revision trong config. Không giả định 2 năm; theo dataset card, public reconstructable data chỉ từ khoảng **2026-06-03**. Coverage chính xác lấy từ timestamp/state reconstruct được, không lấy ngày đầu/cuối theo tên file.
- Không stream realtime. `download` tải snapshot + depth archive; `prepare` replay offline; `train` chỉ đọc prepared memmap. Tardis và yêu cầu freeze 730 ngày cũ đã được thay thế.
- Mỗi snapshot có **một** mid-price: `MP_t = (bid_price_0 + ask_price_0) / 2`. Không tính một mid riêng cho từng level.
- Nguồn có `timestamp_ms`; nội bộ đổi sang microseconds UTC. Đây là event/receipt time tùy phiên bản collector cũ, không tự gọi nó là local arrival time được bảo đảm.
- Tính OF theo price/volume ở từng level trước khi lọc mid-price. Bid flow và ask flow có dấu liquidity supply; `OFI = bid_OF - ask_OF`.
- Các dòng liên tiếp cùng mid bị bỏ khỏi tập origin/sequence model; flow trong đoạn bị bỏ được cộng tới lần mid đổi kế tiếp. Giữ toàn bộ timestamp/raw mid riêng để lấy nhãn đúng thời gian.
- Baseline input `[bid_OF, ask_OF, OFI] × 10`, cộng elapsed time giữa hai mid-change snapshot, elapsed time giữa hai raw snapshot cuối và update count (log1p): **33 feature**. Flow giữ đơn vị volume gốc trước bước scaler trên FIT.
- Baseline không nhận `bid_distance_bps`/`ask_distance_bps`. Các model multivariate nhận OF/OFI và timing; TimesFM zero-shot luôn bỏ qua toàn bộ covariate.
- Nhãn `log(mid(t+h)/mid(t))`, h = **60, 120, 180 giây**, không phải h dòng sau khi lọc. `mid(t+h)` là quote cuối đã tới tại thời điểm đó; không lấy quote đầu tiên sau horizon.
- Khi tính **RMSE, MAE, R²**, đổi về `predicted_price = mid(t) * exp(predicted_log_return)`. Metric trên giá USDT; giá ở đây là mid L2, không phải OHLCV close.
- Thêm `rmse_gain_vs_e0 = 1 - RMSE_model / RMSE_E0` và `r2_os_vs_e0 = 1 - (RMSE_model / RMSE_E0)^2`, cũng trên giá gốc. E0 hiện có là zero-return → giá dự đoán bằng `mid(t)`, được tính một lần/fold/horizon và dùng chung đúng các origin cho mọi model. Tái sử dụng hàm gain cũ, xuất dưới dạng tỷ lệ (0.1 = cải thiện 10%), không nhân 100. Nếu E0 có lỗi bằng 0 thì hai tỷ số để null/ô trống và ghi trạng thái undefined, không ép thành 0 hoặc vô cực. `R2` thông thường vẫn giữ riêng.
- Sequence gap, timestamp đảo thứ tự/gap >10 giây, book crossed/thiếu top 10 hoặc top 10 đi vào phần depth đã mất thông tin đều kết thúc segment và chờ snapshot mới. Age tối đa khi lấy quote là 10 giây **trong cùng segment**, không kéo dài quote ra sau điểm cuối segment.

### Reconstruct đúng từ depth diff

Mỗi row depth là một price-level update, không phải một full-book snapshot. Replay theo `last_update_id`;
gom toàn bộ row chung `(timestamp_ms, first_update_id, last_update_id)` thành một message, kể cả khi nhóm
bị chia giữa Arrow batch/file. Snapshot được đọc theo thời gian, chỉ dùng anchor đã xuất hiện trước event.
Khi chưa có book sống, snapshot thay toàn bộ cache và bắt đầu segment mới, không carry OF từ trước anchor.

Với snapshot/update ID hiện tại `S`, bỏ message có `last_update_id <= S`; message tiếp theo phải bao phủ
`S+1` trong `[first_update_id, last_update_id]`. Khoảng ID bỏ trống làm hủy book và chờ snapshot hợp lệ kế tiếp.

Replay v2 (sửa checker W1/I1, `REPLAY_VERSION = 2`, ghi trong manifest/reconstruction): message đến trong lúc
chờ snapshot được giữ trong một cửa sổ `max_feed_gap_seconds`; khi snapshot `S` được quan sát, các message
`u > S` đã nhận trước nó (quy trình Binance: buffer stream rồi lấy snapshot) được áp sau snapshot nếu nối đúng
`S+1`, và book gộp chỉ được phát tại timestamp snapshot, không phát state quá khứ. Book đang sống liên tục theo ID
không neo lại vào snapshot cũ hơn (tránh replay trùng). Snapshot đi trước chỉ được bỏ qua khi message depth kế tiếp
được book sống chấp nhận (`u > last_id`, `U ≤ last_id+1`, cách message trước ≤ `max_feed_gap_seconds`); nếu không,
segment đóng với `snapshot_ahead_unconfirmed` (message kế tiếp là bản cũ/trùng), `sequence_gap` hoặc
`invalid_timestamp_gap` rồi neo tại snapshot. Gap ID được kiểm trước quy tắc timestamp để reset ghi đúng nguyên nhân;
message gây đứt được đệm làm ứng viên bridge, buffer giữ qua một lần neo lỗi và chỉ xóa sau khi neo thành công;
message đệm nằm trước một hard gap khai báo không được dùng làm bridge cho snapshot sau gap. Trong cùng một
timestamp, snapshot có `lastUpdateId` lớn nhất được thử trước và mỗi timestamp chỉ neo một lần; snapshot còn lại cùng
timestamp bị bỏ và đếm vào `snapshot_at_or_before_reset`. Manifest prepared ghi `replay_version`, `code_commit`,
`code_uncommitted_paths` (đường dẫn chưa commit trong `src_OB/`, `configs/`, lấy lúc bắt đầu prepare; `null` nếu git
lỗi) và `config_sha256` (hash của config đã resolve đường dẫn tuyệt đối, khác sha256 của file config); `run.json` của
mỗi cell chép lại các field này cùng commit code lúc train. Commit code trước khi prepare/train để provenance sạch.
Book đang sống không làm mới độ sâu từ snapshot. Prepared HF v3 hiện có được giữ nguyên (replay v1).
Quantity mới ghi đè quantity hiện tại; quantity bằng 0 xóa đúng price đó. Áp dụng đủ bid và ask của message
rồi prune cache về tối đa **1.000 level mỗi phía**, sau đó mới lấy top 10 và tính mid/OF.
Không cắt cache còn 10 level: các level phía dưới vẫn cần khi best levels biến mất.

Cache bị prune không được hồi sinh ghost level cũ. Nếu top 10 đi sâu qua ranh giới đã bị quên, pipeline
yêu cầu snapshot mới; book crossed cũng reset, không tự xóa level để che một chuỗi diff bị thiếu.
Hard gap chỉ áp khi nguồn công bố, qua `known_hard_gaps_utc` trong config: HF khai báo **2026-07-05 20:56 → 21:39 UTC**
(reset bất kể update IDs, bỏ event nằm trong khoảng này); nguồn độc lập mặc định không có hard gap nào.

Historical June–August 2026 có thể thiếu depth updates do collector cũ. Không forward-fill qua các khoảng
thiếu. `segments.json` ghi khoảng hợp lệ thực tế và nguyên nhân kết thúc; `reconstruction.json` ghi số message,
reset, prune và known gap. IDs/timestamps không chứng minh các message còn lại hoàn toàn không có thiếu row
âm thầm; pipeline không tuyên bố sửa được dữ liệu chưa được thu thập.

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

Adapter trong `autots_native.py` giữ GPU allowlist, căn feature quan sát được theo window của candidate,
và dùng native window maker riêng trong từng segment liên tục rồi fit chung một GPU estimator.
Không nối cuối segment trước với đầu segment sau để tạo window; gap trên regular grid để thiếu, không fill.
Training matrix luôn lấy lại từ raw timeline, không sử dụng giá bị AutoTS nội suy ở bước làm sạch bảng.
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
Tối đa 5 fold × 3 horizon = **15 lần fine-tune**; số fold thực tế phụ thuộc coverage.
Base checkpoint được freeze; default 5 epoch, batch 32, context 512. Bỏ search giảm số lần fit, nhưng chưa thể
khẳng định chạy rất nhanh: còn phụ thuộc số origin, GPU/VRAM và tốc độ đọc dữ liệu.

AutoTS/TimesFM cần chuỗi đều để horizon có nghĩa thời gian: ở mỗi h, context price lấy cách nhau h giây;
forecast **1 step = t+h**. Origin chấm điểm vẫn là các lần mid đổi. Không giả lập irregular event index thành phút.
AutoTS fit trên price grid h giây của FIT; regressor căn theo window của API. Zero-shot TimesFM giữ checkpoint gốc
và không được mô tả là model đã học OF. LoRA dùng bộ decode có gradient và LoRA helper đã có trong `src/p0`,
chỉ import các hàm đó, không gọi training/search harness cũ.

## Walk-forward

Default cho archive ngắn: rolling FIT **21 ngày**, **gap 6 ngày**, VAL **3 ngày**; tối đa 5 fold,
bước 7 ngày. Điểm kết thúc lấy từ state cuối reconstruct được. Pipeline chỉ tạo số fold vừa coverage
thực tế và ghi coverage/revision vào run metadata; không đủ một fold thì dừng, không pad hoặc tạo lịch sử.
Nhãn cuối cùng của training phải nằm trước train_end. Window training không bắt đầu trước train_start.
Không early-stop/tune trên outer VAL; số cây/epoch của các model ngoài AutoTS khóa trong config.
AutoTS chọn tham số bằng validation nội bộ FIT. Scaler và target scale chỉ lấy từ FIT.
Gap 6 ngày tuân theo yêu cầu; nhãn dài tối đa 3 phút tự nó không đòi gap 5 ngày.

Tất cả model chấm trên cùng origin mask được quyết định bởi danh sách `models` trong config, kể cả khi tách job bằng `--models`.
Calendar coverage không đồng nghĩa liên tục. Window của LSTM, TimesFM và AutoTS cùng nhãn phải nằm trong
segment hợp lệ. Nếu các segment quá ngắn cho context cấu hình (TimesFM hiện 512 điểm), tập origin có thể
rỗng; pipeline báo thiếu dữ liệu, không giảm context ngầm hoặc nối qua gap để đủ sample.

## Dùng trên Vast

Từ root repo, dùng image Vast CUDA 12, cài CUDA-enabled PyTorch và GPU-enabled LightGBM trước, rồi:

```bash
pip install -r src_OB/requirements-vast.txt
```

Default LightGBM `device_type=cuda`; nếu build trên Vast dùng OpenCL, đổi `tree.lightgbm_device` thành `gpu`.
Không CPU fallback. Không có lệnh training nào được chạy trên local trong phiên này.

Tải public archive, không cần Tardis key:

```bash
python -m src_OB download --config configs/orderbook.json
```

Downloader chọn đúng `depth/binance/BTCUSDT/*.parquet` và `snapshots/binance/BTCUSDT/*.parquet`,
không tải trades/ETH/SOL. Pin revision, ghi file tạm rồi rename, có retry và tiếp tục file đã hoàn tất;
manifest ghi danh sách file và metadata kích thước từ HF.

**Đã tải xong 4 file, 704.186.850 byte**, revision `873f31e729ae23b1c309cd5dcb33feed27c407de`:

- Raw hiện có: `data/orderbook/hf_crypto_lob_stream/depth/binance/BTCUSDT/{2026-06,2026-07}.parquet`
  và `snapshots/binance/BTCUSDT/{2026-06,2026-07}.parquet` dưới cùng raw root.
- Download manifest hiện có: `data/orderbook/hf_crypto_lob_stream/download_manifest.json`.
- Prepared tại `data/orderbook/prepared_hf/`: raw/kept memmap, `manifest.json` schema v3,
  `segments.json`, `reconstruction.json` (prepare thật trên Vast 2026-09-10).
- Kết quả thật: depth chỉ gồm 1.392 run ID-liên tục ≈300 s (5 phút/giờ, lỗi flush-overwrite của collector v1
  theo card nguồn); 1/38 snapshot nối được; 3.214 raw state, 61 origin, segment hợp lệ dài nhất 300,6 s;
  4 fold theo lịch nhưng 0 origin FIT/VAL → không cell nào train được với phương pháp hiện tại.
- Kết quả training sẽ nằm tại `experiments/orderbook_hf/`.

Schema v2 cũ không dùng cho HF diff archive; luôn tạo prepared directory mới.

Sau khi có raw data, chuẩn bị memmap một lần trên Vast/máy lưu dữ liệu:

```bash
pip install -r src_OB/requirements-data.txt
python -m src_OB prepare --config configs/orderbook.json
```

`prepare` là xử lý dataset thật, không phải test. DuckDB external sort với memory limit 1 GB sắp depth theo ID,
Arrow đọc theo chunk, snapshot/message được replay thành state rồi ghi feature và raw-price memmap trên đĩa;
không tạo sẵn tensor `[toàn bộ sample, context, feature]` trong VRAM. Thư mục prepared mới phải chưa tồn tại;
đổi `prepared_dir` khi cần tạo phiên bản khác. Các lỗi input được raise trong đường chạy thật.

Sau prepare, ghi `DATA_REPORT.md`/`data_report.json` vào `output_dir` (đọc raw Parquet và prepared data;
đếm origin bằng cùng hàm `select_origins` mà `train` dùng; không fit/infer):

```bash
python -m src_OB data-report --config configs/orderbook.json
```

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
`experiments/orderbook_hf/foldN/model/h60s` (tương tự h120s/h180s), gồm model, prediction giá, metric và config.
Thư mục kết quả đã tồn tại sẽ không bị ghi đè. Dùng `output_dir` mới nếu chạy lại; không có tự động resume training dở.
Tree matrix nằm trong RAM host; cửa sổ neural chỉ chuyển batch lên GPU. Nhu cầu đĩa/RAM và thời gian chạy thực tế chưa đo.

## Kết quả và inference latency

Mỗi `foldN/model/hHs/` ghi `metrics.json`, bảng một dòng `metrics.csv`, `predictions.parquet`,
`latency.json`, trace `inference_latency.csv`, checkpoint và trạng thái run. Các metric E0 gain và latency
cũng đi vào `summary/per_fold_per_horizon.csv` và `summary/by_model_horizon.csv`, cập nhật sau mỗi cell hoàn tất.
Summary chỉ dùng cell đã có `completed.json`, có khóa ghi cho nhiều job fold dùng chung output directory.

Summary theo model/horizon báo mean/min/max qua fold và số fold có giá trị. Các cột `pooled_*` tính từ
MSE được gộp theo số origin; chúng khác mean gain qua fold và được ghi riêng. Không tạo model baseline mới.
Để dựng lại bảng từ artifact đã có, không train/infer lại:

```bash
python -m src_OB summarize --config configs/orderbook.json
```

Latency được đo **ngay trong lượt inference thật trên Vast**, không benchmark pass, không warmup, không
predict lặp để đối chiếu. Default chọn đều tối đa 1024 origin trên cùng tập VAL chạy batch 1; prediction
của chúng vẫn dùng trong RMSE. Những origin còn lại chạy batch 32, AutoTS gọi API từng origin. Mỗi origin
được dự đoán đúng một lần. Có đồng bộ CUDA trước và sau đồng hồ `perf_counter` để không chỉ đo enqueue.

- `inference_mean_ms`, `inference_p50_ms`, `inference_p95_ms`, `inference_p99_ms`: latency request batch 1 trên các origin được chọn.
- `inference_upper_bound_ms`: **max quan sát được của các request batch 1 được chọn**, không phải hard bound hay confidence bound. `inference_upper_bound_kind` ghi rõ điều này.
- `inference_batch_p95_ms`, `inference_batch_p99_ms`, `inference_batch_max_ms`: latency của tất cả call, kèm số prediction/call trong trace; không gọi batch-time/batch-size là latency từng request.
- `inference_total_seconds`, `inference_amortized_ms_per_prediction`: tổng thời gian các call và chi phí trung bình phân bổ/prediction.
- `inference_first_call_ms`: call đầu, giữ cả ảnh hưởng khởi tạo/lazy allocation; không loại warmup. Percentile của summary được tính từ trace gộp, không lấy trung bình p95/p99 các fold.

Phạm vi đồng hồ: đọc prepared context/feature, gom input, scaler, chuyển CPU/GPU, predict và lấy log-return
về host. Không gồm download, prepare toàn dataset, load checkpoint, training/search, ghi artifact, đổi log-return
thành giá hoặc tính metric. Với TimesFM, native backend có thể pad batch 1 theo cấu hình; số đo phản ánh call
thật của pipeline. `duration_seconds` vẫn là thời gian toàn cell, không được dùng thay inference latency.

Git không ignore artifact dưới `experiments/`, `reports/`, `results/`, `outputs/`; đã bỏ ignore `reports/figures/`.
Prediction Parquet và model joblib trong experiments dùng Git LFS như các checkpoint khác; metric/CSV/JSON
được track thông thường. Code không tự commit/push artifact khi training.

## Nguồn

- [HF dataset](https://huggingface.co/datasets/MaximumLeverage/crypto-lob-stream): schema, coverage, prune và known gap.
- [Collector/replay source](https://github.com/Goodie-Goody/crypto_lob_stream_pypi): snapshot depth 1000, absolute quantity, collector limitations.
- [Binance Spot depth](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams): U/u sequence và depth update protocol.
- [AutoTS regressors](https://github.com/winedarksea/AutoTS/blob/1.0.4/autots/models/sklearn.py): native regressor generation và WindowRegression.
- [AutoTS search](https://github.com/winedarksea/AutoTS/blob/1.0.4/autots/evaluator/auto_ts.py): model search và custom validation indexes.
- [IDEA](../docs/IDEA.md): nguồn ý tưởng OF và elapsed time; yêu cầu Direct mới thay thế đề xuất MIMO cũ.
