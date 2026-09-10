# RUN_REPORT — goal Order Book trên Vast (2026-09-10)

**Trạng thái: BLOCKED — goal CHƯA hoàn tất.** Archive HF pinned đã được tải và prepare thật, nhưng chỉ dựng
được **300,6 giây** order book hợp lệ trên toàn bộ tháng 6–7/2026. Với phương pháp cố định, không fold/model/horizon
nào có origin FIT/VAL hợp lệ: **0/96 cell**. Lượt tiếp nối (mục 11) không tìm được nguồn thay thế phù hợp. Chưa có
training, metric, E0 gain, latency hay summary.

## 1. Việc đã chạy thật

| Bước | Lệnh | UTC | Kết quả |
|---|---|---|---|
| Môi trường | `run_meta/install_env.sh` (venv `/home/ubuntu/venv-ob`) | 16:18 → 16:24 lỗi; retry LightGBM → 16:37:10 | Lần 1: build LightGBM CUDA lỗi ở nvlink (mục 6); retry với NCCL shared: EXIT 0 |
| Download | `python -m src_OB download --config configs/orderbook.json` | 16:16:56 → 16:17:12 | EXIT 0; 4 file, 704.186.850 byte; sha256 khớp `source_lfs_sha256` |
| Prepare | `python -m src_OB prepare --config configs/orderbook.json` | 16:21:53 → 16:23:50 | EXIT 0; `data/orderbook/prepared_hf/` schema v3 |
| Data report | `python -m src_OB data-report --config configs/orderbook.json` | → 16:29:32 | `DATA_REPORT.md`, `data_report.json`: **BLOCKED** |
| Checker | agent `checker` đọc data/metadata/code (bước 4) | ~16:31 → ~16:45 | Không ERROR; B1 blocker xác nhận — `CHECKER_FINDINGS.md` |

Không chạy `train`/`summarize`: bước 5 của goal chỉ chạy khi data khả dụng; `select_origins` (cùng hàm `train` dùng)
trả 0 origin FIT/VAL cho cả 4 fold, nên `train` sẽ dừng ở fold1 (`ValueError: fold1: không còn sample…`) trước khi
tạo bất kỳ cell nào (checker P7: raise trước `mkdir`). Không có smoke/canary/test/probe fit nào được chạy.

## 2. Coverage, segment và origin (số thật)

- Depth thô: 4.225.572 message / 119.488.794 row, 2026-06-04 00:54 → 2026-07-31 23:56 UTC.
  Theo policy replay (U ≤ u trước + 1, bước ≤ 10 s) chỉ có **1.392 run**, mỗi run **≈300 s** (q0.01–q1:
  299,9–300,4 s), bắt đầu ở phút :49–:55 mỗi giờ, cách nhau 3.034–3.828 s và mất hàng chục nghìn–hàng chục triệu
  update ID. Mọi điểm cắt run đều là gap ID (checker P2). Tổng thời gian có depth **115,8 h / 1.391 h (8,3%)**.
- Snapshot: 38 (≈2 ngày/lần); chỉ **1** snapshot (2026-07-05 20:50:12) có depth message nối được `last_update_id+1`
  (tra thuần theo ID, không điều kiện thời gian). 37 snapshot còn lại rơi vào khoảng trống: message kế tiếp đến sau
  303–3.348 s và thiếu 62.181–11.671.754 ID; 18 snapshot không có depth nào trong ±600 s.
- Replay v1 (`reconstruction.json`): 4.222.272/4.225.572 message ở trạng thái chờ snapshot; 38 segment, 37 segment chỉ
  có state snapshot. Nhãn kết thúc: 35 `invalid_timestamp_gap` (check timestamp chạy trước check contiguity; các điểm
  này đồng thời là gap ID lớn), 2 `snapshot_reanchor`, 1 `known_hard_gap`; `counts.snapshot_reanchor = 38` là số
  snapshot đã áp.
- Segment duy nhất > 1 state: 2026-07-05 20:50:12,355 → 20:55:12,975 (300,62 s, 3.177 state, **24 origin**
  mid-change, spread 0,01 ở mọi state). Run depth thực tế hết lúc 20:55:12,975; segment được ghi `known_hard_gap`
  vì message kế tiếp theo ID (21:54:59) nằm sau hard gap 20:56–21:39. Tổng: 3.214 raw state, 61 origin
  (gồm 38 state đầu segment).
- Walk-forward (coverage theo lịch 2026-06-05 06:29 → 2026-07-30 17:14): **4 fold theo lịch, 0 fold hợp lệ**. Mọi
  mask đều 0: nhãn + context 100 origin trong segment, context trong FIT, context giá TimesFM/AutoTS 512 × h. VAL của
  cả 4 fold chỉ chứa segment một state, nên rỗng **kể cả nếu bỏ TimesFM và context = 1** (checker B1). Cột
  `valid_segment_seconds_*` ≈ 1e-5 s là quy ước `end_exclusive = last + 1 µs`, nghĩa thực là 0 s book dùng được.
- Context cần: 100 origin mid-change liên tiếp (segment tốt nhất chỉ có 24); nhãn t+h trong segment; TimesFM/AutoTS
  (mask chung 512 điểm, span 511×h) cần segment liên tục 8,53 h (h60), 17,07 h (h120), 25,6 h (h180) tính cả nhãn.
  Segment dài nhất 300,6 s.
- Giới hạn khác: timestamp là receipt time của collector, có burst (89 message obsolete trong 135 ms sau snapshot,
  7 lần đổi mid đầu trong 0,1 s); ID/timestamp không chứng minh không thiếu row bên trong message.

Chi tiết: `DATA_REPORT.md`, `data_report.json`, `run_meta/raw_archive_continuity.json` (sinh bởi
`run_meta/raw_continuity.py`).

## 3. Cell

Expected theo pipeline = 4 fold theo lịch (0 fold có origin FIT/VAL hợp lệ) × 8 family × 3 horizon = **96**;
completed **0**; thiếu **96** (fold1–fold4 × lgbm, xgb, cat, xgbrf, lstm, autots, tfm_zero_shot, tfm_lora × h60s,
h120s, h180s). Không có `per_fold_per_horizon.csv`/`by_model_horizon.csv`, metric giá, `rmse_gain_vs_e0`,
`r2_os_vs_e0` hay latency vì chưa có cell hoàn tất.

## 4. Nguyên nhân gốc

Card của repo nguồn `Goooddy/crypto-lob-stream` (cập nhật 2026-09-01, sau thời điểm repo pinned được nhân bản)
ghi: collector v1 dùng cho Binance BTCUSDT/ETHUSDT/SOLUSDT tháng 6–8/2026 có lỗi đặt tên flush khiến các lần flush
sau trong cùng giờ **ghi đè** lần trước; snapshot không bị ảnh hưởng. File tải về khớp đúng dấu vết đó: mỗi giờ chỉ
còn một đoạn ~5 phút. Card nguồn ước tính giữ lại 93%/88% volume cho tháng 6/7, nhưng phép đo trực tiếp trên file
(checker tái lập độc lập) cho thấy mọi giờ đều chỉ còn ~300 s. Repo pinned `MaximumLeverage/crypto-lob-stream` chỉ có
một commit (= revision pin); repo nguồn có thêm `2026-08.parquet` với kích thước tương đương (card nguồn: tháng 8 còn
~52%), nên không cứu được. Bằng chứng: `run_meta/upstream_evidence.md`.

Replay không thể khôi phục phần update đã không được lưu: không có snapshot nào nằm trong 1.391/1.392 run để neo,
dựng ngược từ snapshot tương lai bị cấm (diff quantity tuyệt đối cũng không đảo ngược được), và kể cả khi neo được,
run dài tối đa 300 s.

## 5. Quyết định cần từ user (danh sách duy nhất)

Theo phương pháp hiện tại, cần depth BTCUSDT **Spot** liên tục (snapshot + diff, hoặc snapshot ≤ 10 s) đủ dài: mỗi
fold cần 30 ngày theo lịch (FIT 21 + gap 6 + VAL 3), 5 fold cần 58 ngày, và segment liên tục ≥ 25,6 h quanh cả FIT và
VAL (TimesFM h180). Không nguồn miễn phí/đã có quyền nào đạt (`SOURCE_REPORT.md`). Chọn một:

1. **Cấp nguồn có khóa/trả phí hoặc quyền truy cập** cho ≥ 30–58 ngày Spot liên tục: Tardis API key, Binance
   historical data access, Crypto Lake/Kaiko, hoặc duyệt dataset gated (vd. `Lazy108/binance-polymarket-orderflow`,
   Spot snapshot 1 s/20 level, hiện 22 ngày).
2. **Đổi phương pháp tường minh** cho một ứng viên gần nhất: FIT ngắn hơn để dùng Zenodo 20046390 (21 ngày Spot 5 s,
   chỉ phi thương mại), hoặc context TimesFM/h180 khác cho `predict-quant/binance-spot-orderbook` (run dài nhất 22,64 h,
   VAL ≤ 7,5 h, không license).
3. **Thu thập mới** ≥ 30 ngày (1 fold) / ≥ 58 ngày (5 fold) bằng collector liên tục (vd. `crypto-lob-stream` ≥ 0.9.2),
   rồi kiểm continuity thật trước khi prepare.

Session này không đổi revision/nguồn sang dữ liệu không đạt, không nối gap, không giảm context/gap/FIT và không tạo
dữ liệu để lách blocker.

## 6. Môi trường, lỗi đã sửa và provenance

- Máy: Vast container `C.50503596`, 96 CPU, 125 GB RAM, 1 × RTX 3090 24 GB
  (`GPU-7745473c-82d7-079d-7a96-db04ea3fcee6`, compute 8.6), driver 595.84, image CUDA 12.8.1 (nvcc 12.8).
  `/workspace` không phải volume: recycle/destroy mất toàn bộ container.
- Python 3.12.3, venv `/home/ubuntu/venv-ob` dựng bằng `run_meta/install_env.sh` (log `logs/env_install.log`):
  torch 2.11.0+cu128 (cuDNN 9.19, CUDA available), LightGBM 4.7.0 build `USE_CUDA=ON`, xgboost 3.4.1
  (`USE_CUDA=True`), catboost 1.2.10 (1 GPU), cupy-cuda12x 14.2.0, timesfm 2.0.2, autots 1.0.4, statsmodels 0.15.0,
  numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1, duckdb 1.5.5. Không cài JAX/XReg.
  Đầy đủ: `run_meta/pip_freeze.txt`, `run_meta/environment.json` (ghi bằng `run_meta/record_env.py`, chỉ đọc
  metadata/thiết bị, không fit).
- Config `configs/orderbook.json` lúc download/prepare/data-report: sha256 `72d68928bdcb9eb54087164086b715e79cba09581c28d776fd8ee6acc3c00187`.
  Từ `1e2f3ce` config thêm `known_hard_gaps_utc` (sha256 `b73919e40ca8c88f…`), cùng semantics cho HF.
  Code lúc chạy: HEAD `a5872b9` + thay đổi ở mục 7 (commit `e169df1`).
- Lỗi đã sửa: build LightGBM CUDA lần 1 (16:24) dừng ở `cmake_device_link` —
  `nvlink fatal: Input file '/usr/lib/x86_64-linux-gnu/libnccl_static.a:common.cu.o' ABI version '8' is incompatible
  with target ABI version '7'` (NCCL 2.31.2 static của image mới hơn nvcc 12.8). Sửa bằng
  `-C cmake.define.BUILD_WITH_SHARED_NCCL=ON` (FindNCCL của LightGBM chọn `libnccl.so`); build xong 16:37:10,
  `lib_lightgbm.so` link `libnccl.so.2`. Backend `device_type=cuda` giữ nguyên theo config; chưa có fit nào chạy.
- Pipeline download/prepare/data-report không gặp lỗi runtime. Runtime thật: download 16 s, prepare 117 s.

## 7. Thay đổi code trong lượt đầu

- `src_OB/data.py`: `select_origins` — tách nguyên văn logic chọn origin FIT/VAL từ `train.py` (checker P7: ngữ
  nghĩa y hệt HEAD) để `train` và báo cáo dùng cùng một mask.
- `src_OB/report.py` + lệnh `python -m src_OB data-report`: đọc raw Parquet (DuckDB), prepared memmap/metadata và
  `select_origins`; ghi `DATA_REPORT.md`/`data_report.json`. Không fit/infer.
- `src_OB/README.md`: trạng thái thật và lệnh `data-report`. (Thay đổi của lượt tiếp nối: mục 11.)

## 8. Checker lượt đầu

`CHECKER_FINDINGS.md` (nguyên văn). Tóm tắt: PASS P1–P7 (danh tính archive, replay contract, không bug ordering
snapshot, segment 20 tính lại khớp, OF/OFI khớp `features.bin` tới 4,2e-7, nhãn as-of, `select_origins`);
**B1** blocker dữ liệu xác nhận; **W1** WARN: replay bỏ message bridge nhận trước timestamp snapshot và kiểm
timestamp trước contiguity — không ảnh hưởng archive này (đã sửa ở lượt tiếp nối, replay v2); I1–I6 INFO (đã ghi vào
mục 2). GPU-only lúc fit, scaler/AutoTS FIT-only và latency: chưa có bằng chứng vì chưa có fit.

## 9. Artifact và git

- Commit trên branch `OB`: code ở mục 7 và 11, `.claude/MEMORY.md`, `experiments/orderbook_hf/` (`DATA_REPORT.md`,
  `data_report.json`, `RUN_REPORT.md`, `SOURCE_REPORT.md`, `CHECKER_FINDINGS.md`, `BACKUP_STATUS.md`, `logs/`,
  `run_meta/`), prepared metadata `data/orderbook/prepared_hf/{manifest,segments,reconstruction}.json` và 7 memmap `.bin`
  (Git LFS, 7 object, 86.652 B; cả thư mục prepared ~130 KB).
- Không commit: 4 raw Parquet (704 MB; tải lại đúng revision bằng `python -m src_OB download`, sha256 trong
  `download_manifest.json`), venv, checkpoint TimesFM pretrained (chưa tải vì chưa train).
- Push: lúc đầu instance không có credential HTTPS (`could not read Username for 'https://github.com'`). Ở lượt
  tiếp nối, SSH key `~/.ssh/id_ed25519` xác thực được `tson295`; `remote.origin.pushurl` trỏ SSH; mọi commit đã lên
  `origin/OB` (LFS 7/7). Trạng thái backup chi tiết: `BACKUP_STATUS.md`.
- tmux session `ob` (window download/env/prepare/lgbm) còn giữ pane log trên instance.

## 10. Phần chưa đạt

- 96/96 cell, `summarize`, bảng by-model/horizon, E0 gains, latency p95/p99/max, runtime training và checker sau
  training: chưa có — bị chặn bởi B1 và không có nguồn thay thế phù hợp (mục 11).
- Chính sách GPU-only lúc fit và các API AutoTS 1.0.4 / TimesFM 2.0.2 / LoRA decoder trong pipeline OB vẫn chưa
  được thực thi trong lượt chạy thật nào.
- Replay v2 đã có trong code nhưng chưa có lượt prepare thật nào dùng nó. Trước prepare v2 thật đầu tiên cần đóng
  R3-I1, R3-I2, R3-I4, `reset_after` của R3-I5 và F-02 (ghi SHA code/hash config vào manifest hoặc tăng
  `REPLAY_VERSION`) — xem `CHECKER_FINDINGS.md`.

## 11. Tiếp nối 2026-09-10 (goal khôi phục data, bắt đầu 17:39 UTC)

**Trạng thái vẫn BLOCKED.** Không có nguồn thay thế phù hợp; không download/prepare/train mới; 0 cell. Commit đã push:
`a04d17b` 17:43, `603738e` 18:13, `1e2f3ce` 18:38, `7d65cd6`, `500ad0d`, `3f41869` 19:02, `84acea0` và commit chứa
bản báo cáo này (xác nhận bằng `git ls-remote`, `BACKUP_STATUS.md`).

1. **Lưu việc:** remote `OB` có thêm `109cbc2` (prompt mới, chỉ docs); merge không xung đột thành `a04d17b` và push
   qua SSH kèm LFS 7/7.
2. **W1/I1 → replay v2** (`src_OB/reconstruct.py`, `REPLAY_VERSION = 2`): message chờ snapshot được đệm trong cửa sổ
   `max_feed_gap_seconds`; message `u > S` nối đúng `S+1` được áp sau snapshot và book gộp chỉ phát tại timestamp
   snapshot; book đang sống không neo lại snapshot cũ (không replay trùng); snapshot đi trước chỉ bị bỏ qua khi message
   kế tiếp được book sống chấp nhận, nếu không thì đóng segment đúng lý do rồi neo; buffer sống qua anchor lỗi; kiểm ID
   trước timestamp; hard gap chuyển sang `known_hard_gaps_utc` trong config (HF khai báo 2026-07-05; nguồn độc lập mặc
   định rỗng) và không bridge qua hard gap. `prepare` ghi `replay_version`. Không prepare lại HF (goal cấm lặp bằng chứng
   blocker). Theo suy luận của checker lượt 2–3 và cuối run từ evidence (chưa chạy thật), trên archive HF v2 chỉ đổi nhãn
   reset (37 segment → `sequence_gap`, 1 `known_hard_gap`), không đổi state/segment. Replay v2 **chưa được lượt chạy
   thật nào thực thi**.
3. **Tìm nguồn** (3 vòng: 17:39–18:11, ~18:33–18:36, ~18:54–19:02 UTC; ~46 phút): 27 nhóm ứng viên trong
   `SOURCE_REPORT.md` (HF theo tên/tag/full-text/tác giả, Zenodo, figshare, Dataverse, GitHub, AWS Open Data, Binance
   Vision, Tardis, Crypto Lake, Crypto Chassis, CryptoDataDownload, Kaggle); evidence từng vòng trong
   `run_meta/source_search/`. Gần nhất: `predict-quant/binance-spot-orderbook` (Spot diff 100 ms; theo luật pipeline
   2 fold, context VAL tốt nhất 7,41–7,51 h < 8,52 h, FIT h180 22,59 h < 25,55 h — `pq_spot_folds.out`; snapshot không
   timestamp, không license), Zenodo 20046390 (Spot 5 s, 100 level, liên tục nhưng 21 ngày < 30 ngày cho một fold, phi
   thương mại) và `Lazy108/binance-polymarket-orderflow` (Spot snapshot 1 s/20 level nhưng gated manual, 22 ngày). Tardis
   không key chỉ phục vụ ngày đầu mỗi tháng (HTTP 401 cho ngày khác).
4. **Checker:** lượt 2 không ERROR — T-W1/T-I3 đã sửa, T-W2 (code v2 chưa push) đã push, T-W3 (ứng viên sót) đã bổ sung.
   Lượt 3 (đọc lại `1e2f3ce`) không ERROR/WARN; README cập nhật (R3-I5); R3-I3 đã sửa sau đó. Cuối run (HEAD `84acea0`)
   không ERROR, xác nhận điều kiện BLOCKED đủ; F-07 (evidence vòng 3, Lazy108, fold/context predict-quant) đã bổ sung;
   F-08/F-09 (câu chữ MEMORY/RUN_REPORT) đã sửa; F-02/F-10 để trước prepare v2 thật đầu tiên (mục 10). Chi tiết:
   `CHECKER_FINDINGS.md`.
5. **Cell thiếu:** toàn bộ (96 theo 4 fold lịch của HF; chưa có nguồn mới nên chưa có fold mới).
6. **Quyết định tối thiểu khi user quay lại:** mục 5. Next step khi có data: đóng các INFO ở mục 10 → config riêng
   (`raw_dir`/`prepared_dir`/`output_dir` mới) + downloader/adapter → download → prepare (replay v2) → `data-report` →
   checker → `train` → `summarize` → checker → báo cáo.
7. **Máy:** instance Vast vẫn chạy, GPU rảnh; không stop/destroy.
