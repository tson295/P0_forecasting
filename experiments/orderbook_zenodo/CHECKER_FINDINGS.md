# Checker findings — run Zenodo 20046390 (`configs/orderbook_zenodo.json`)

Checker chỉ đọc code, data/metadata và log thật; không train/prepare/infer/test. Script đọc-only của checker nằm trong
scratchpad session (không thuộc repo).

## Lượt Z-pre — sau prepare + data-report thật, trước training (2026-09-11)

Kết luận: **không ERROR**; READY 5 fold khớp kiểm tra độc lập.

| ID | Mức | Nội dung (evidence) | Xử lý |
|---|---|---|---|
| Z-P1 | PASS | md5 tar tính lại `58507a0f…` (308.618.431 B) = config = download manifest = files API; license datacite CC-BY-4.0/CC-BY-NC-4.0 + điều khoản phi thương mại; Spot (`symbol` "BTC/USDT", lưới giá 0,01, spread 0,01 ở 99,5% mẫu, qty bước 1e-5) | — |
| Z-P2 | PASS | adapter `snapshots.py`: top-10 hợp lệ, timestamp/nonce đi lùi thì bỏ + đóng segment, bước > 10 s, dòng hỏng; 1.439.159 dòng = 1.439.157 nhận + 1 JSON hỏng + 1 snapshot đi lùi (đều 10-01); 2 segment | — |
| Z-P3 | PASS | tính lại OF/OFI/mid từ 3.000 dòng raw 10-02: 815 origin, sai số max 6e-8 trên 33 cột; kept = đầu segment + mid đổi; raw timeline riêng | — |
| Z-P4 | PASS | nhãn as-of ≤ q, age ≤ 10 s, cùng segment; đếm lại 17 counter × 5 fold: 0 lệch; nhãn FIT cuối < train_end; VAL 10-11→10-21 không chồng; gap 86.400 s > 180 s | — |
| Z-P5 | PASS (code/data) | AutoTS 3 split nội bộ cuối FIT, −2 d, −4 d, purge 1 d; lịch sử ≥ 1.850 điểm lưới ở h180 (min 91); API AutoTS 1.0.4 nhận tuple custom validation | — |
| Z-P6 | PASS | nhịp origin mean 7,63 s (median 2,5 s, max 1.769 s); mọi VAL origin đủ context 512×h; mid phẳng dài nhất 10-15 06:22–06:51 là thị trường tĩnh thật (nonce tăng, top-10 đổi) | — |
| Z-P7 | PASS | guard GPU/Vast train.py giữ nguyên; launcher đúng env; manifest code_commit c68fc45, uncommitted rỗng, config_sha256 khớp; checkpoint TimesFM `1d952420…` đã cache | — |
| Z-W1 | WARN | quyết định 2026-09-11 (gap 1 d) chưa ghi vào CLAUDE.md, checker.md, VAST_SESSION_PROMPT.md:95, MEMORY, README:131 | **Đã sửa** (commit tài liệu sau khi train start; không đổi code) |
| Z-I1 | INFO | datacite ghi "5-second"; thực tế ~1,24 s (~68.900 snapshot/ngày); timestamp là đồng hồ collector; UTC khớp tới phút (spike ETF giả 10-16 13:25→13:30) | ghi vào RUN_REPORT |
| Z-I2 | INFO | đầu 10-01 bất thường (1 dòng hỏng, 1 đi lùi, book 101/99 ask); ngoài mọi FIT/VAL/context | ghi vào RUN_REPORT |
| Z-I3 | INFO | FIT hiệu dụng ~7,9 d: context TimesFM h180 (25,55 h) phải nằm trong FIT nên origin FIT đầu tiên ~01:33 ngày thứ 2 | ghi vào RUN_REPORT |
| Z-I4 | INFO | fold2 VAL chỉ 7.621 origin (cuối tuần) | report nêu mean-fold và pooled |
| Z-I5 | INFO | `Data` không guard `snapshot_adapter_version` | nếu sửa `snapshots.py` phải tăng version + thêm guard |
| Z-I6 | INFO | manifest `config_sha256` = sha JSON đã resolve (`78aa19…`); environment.json = sha bytes file (`2995bd…`) | ghi rõ trong RUN_REPORT |
| Z-I7 | INFO | HF_HOME global trỏ `/workspace/.hf_home` (root) | mọi lệnh recovery phải export HF_HOME như launcher |
| Z-I8 | INFO | raw tar untracked, không gitignore | stage theo scope, không `git add -A` |

Chưa có bằng chứng lúc này: fit GPU thật, API runtime AutoTS/TimesFM/LoRA, metric/E0, latency, 120 cell.

## Lượt ZF — cuối run (sau TRAIN_EXIT=0 16:18:14Z và SUMMARIZE_EXIT=0, 2026-09-11)

Kết luận: **không ERROR**. Checker chỉ đọc; load checkpoint/memmap với `CUDA_VISIBLE_DEVICES=""`, không fit/predict.

| ID | Mức | Nội dung (evidence) | Xử lý |
|---|---|---|---|
| ZF-P1 | PASS | 5 fold (train.log) → 120 cell, đủ 120, 0 `failed.json`; mỗi cell có completed, run, metrics json/csv (khớp nhau), predictions (n = val origins DATA_REPORT, 0 NaN), latency json/csv; checkpoint: model.joblib (60 cây + 15 AutoTS, kèm search_results/selected_model), model.pt (15 LSTM), adapter.pt (15 LoRA); zero-shot pin revision | — |
| ZF-P2 | PASS | train_code d8797c5 cho 15 cell lần 1, 2a5a1c3 cho 105 cell, 0 file chưa commit; code 2a5a1c3 = HEAD; mỗi thư mục cell chỉ nằm trong đúng một commit (không ghi đè); created_at khớp từng lần chạy; prior_attempts đúng; hai attempt còn đủ | — |
| ZF-P3 | PASS | tính lại cả 120 cell từ predictions: RMSE/MAE/R² sai lệch ≤ 4e-16, gain/r2_os ≤ 5e-13; E0 = origin_price; origin/actual/E0 giống hệt giữa 8 family trong mỗi fold/h; actual = mid as-of t+h (tuổi ≤ 10 s), origin nằm trong VAL | — |
| ZF-P4 | PASS | per_fold_per_horizon (120 hàng) và by_model_horizon (24 hàng) khớp per-cell; mean theo fold và pooled là cột riêng; không lẫn attempt | — |
| ZF-P5 | PASS | CUDA sync quanh đồng hồ (latency.py:66-72); p50/p95/p99 trên request batch 1 lấy mẫu; max = `observed_max_not_hard_bound`; batch stats riêng | — |
| ZF-P6 | PASS | config hiệu lực: lgbm 15/15 và AutoTS-LightGBM `device_type cuda`, `linear_tree 0`, `max_bin 255`; XGBoost `cuda:0` hist; CatBoost GPU; search AutoTS lần 3: 0 candidate linear_tree/goss/max_bin > 255; log lần 3 không có CPU/fallback | — |
| ZF-P7 | PASS | AutoTS: 3 vòng validation nội bộ, target ≤ 23:59 ngày train_end, purge 1 d; refit trên FIT; regressor căn đúng; chọn 10 XGBoost gbtree + 5 LightGBM gbdt, không dart | — |
| ZF-P8 | PASS | zero-shot chỉ log-mid; LoRA 15/15 adapter (2.048.000 tham số) + OF head + scaler + repo/revision; một AdamW chung (timesfm.py:71-72); không XReg | — |
| ZF-W1 | WARN | chẩn đoán "ngoại suy" chỉ đúng một phần: đúng cho fold1/3/5; fold2 (bias ≈ −3) và fold4 (bias −11…−20) vẫn 1,8–3,2× E0 vì cây hồi quy mức giá thô bị kẹp quanh mức giá dày đặc của FIT | **Đã sửa** RUN_REPORT §3b/§6 theo bảng 15 cell |
| ZF-W2 | WARN | câu "DART trên CUDA cho giá trị sai" không có evidence: có candidate dart smape 0,03–0,1 bên cạnh 61–200 | **Đã sửa** RUN_REPORT §3b: nguyên nhân chưa xác định, bị loại khi chấm, không cell nào chọn dart |
| ZF-I1 | INFO | run.json giữ `status: "started"`; 15 cell lần 1 không có key `prior_attempts` | ghi RUN_REPORT §4, không sửa cell completed |
| ZF-I2 | INFO | mean theo fold và pooled ngược dấu ở tfm_zero_shot h120/h180; R² của E0 cao (fold1 h60 ≈ 0,976) | ghi RUN_REPORT §5 |
| ZF-I3 | INFO | vị trí 0 luôn trong mẫu batch 1 nên max thường là lần gọi đầu (xgb fold1 h60 222,7 ms); batch thực tế 7–30 prediction | ghi RUN_REPORT §5 |
| ZF-I4 | INFO | LoRA train loss 0,019–0,078 nhưng thua E0 và zero-shot 15/15; LSTM tương tự; target các origin chồng lấp | ghi RUN_REPORT §5; không tune bằng outer VAL |
| ZF-I5 | INFO | `verbosity=-1` che cảnh báo C++ của LightGBM; bằng chứng GPU là config hiệu lực (ZF-P6); CPU fit của attempt1 suy từ replay seed | ghi RUN_REPORT §3 |
| ZF-I6 | INFO | zero-shot không tự ghi repo/revision trong artifact cell; truy qua commit 2a5a1c3 và log tải checkpoint | ghi RUN_REPORT §4 |
