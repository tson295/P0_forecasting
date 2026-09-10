# Checker findings — bước 4 (sau prepare thật, trước training), 2026-09-10

Agent `checker` (chỉ đọc code/raw Parquet/memmap/metadata/log; không prepare/train/test lại; segment 20 được
tính lại độc lập từ 3.176 message thật, không replay toàn dataset). Session chính lưu nguyên văn dưới đây.

**Kết luận:** không có ERROR correctness trong prepare/data/report; BLOCKED là thật, không có đường hợp lệ trong
phương pháp hiện tại để có origin.

## PASS (có bằng chứng)

- **P1 Archive identity**: repo/revision `873f31e7…` khớp `configs/orderbook.json:4-5`, `download_manifest.json`,
  `prepared_hf/manifest.json`; sha256sum lại 4 file trên Vast khớp `source_lfs_sha256`; depth 119.488.794 row,
  snapshots 76.000 row (38 × 1000 bid + 1000 ask), 100% `exchange=binance`, `asset=BTCUSDT`. Spot: tick 0,01 (level
  snapshot cách đúng 1 cent), min qty 1e-05, update ID ~9,5e10; không có cột market, nhãn "Spot" dựa trên card + đặc
  điểm này.
- **P2 Replay contract**: code gom message (ts,U,u) xuyên batch (`reconstruct.py:26-42`), quantity tuyệt đối/0 xóa/
  prune sau cả message (`99-120`), top 10 thứ tự đúng (`122-129`), một mid (`prepare.py:59`), snapshot thay cache/
  reanchor (`131-155`), gap ID/ts/hard gap (`74-82`, `168-176`, `218-221`). Raw theo thứ tự ID: 4.224.180 chuyển tiếp
  `U = u_prev+1`, 0 overlap, 0 (U,u) trùng khác ts, 0 ts đảo; đúng 1.391 gap ID ⇒ 1.392 run; mọi điểm cắt run đều là
  gap ID (quy tắc 10 s không cắt thêm). Counts khớp: 4.222.272 chờ + 89 obsolete + 3.176 áp + 35 bị bỏ khi reset =
  4.225.572.
- **P3 "Chỉ 1 snapshot nối được" không phải bug ordering**: query thuần ID (không điều kiện thời gian) — chỉ
  S=96980990616 (2026-07-05 20:50:12.355) có message `U ≤ S+1 ≤ u`. 37 snapshot còn lại: message kế tiếp sau
  303,8–3.347,6 s, thiếu 62.181–11.671.754 ID; 18 snapshot không có depth nào trong ±600 s. Quy tắc >10 s, thứ tự
  snapshot/depth hay buffer trước snapshot đều không làm mất dữ liệu dựng được ở archive này.
- **P4 Segment 20 đúng**: 89 message u ≤ S đến sau snapshot 0,214–0,349 s (burst catch-up của collector) bị bỏ
  đúng là obsolete; bridge ở +0,349 s; 3.176 message liền mạch tới 20:55:12.975, bước thời gian max 1,995 s; message
  kế tiếp theo ID là 21:54:59; không có event nào trong 20:56–21:39. Tính lại độc lập → `raw_ts`/`raw_mid` y hệt
  (3.177 state); spread 0,01 ở 100% state; 23 lần đổi best bid, 23 lần đổi best ask ⇒ ít đổi mid là thật, không phải
  level cũ kẹt. 300,62 s = state snapshot + run thô 300,406 s (run dài nhất toàn archive).
- **P5 OF/OFI + timeline**: `prepare.py:66-83` flow từng transition, cộng dồn trước same-mid drop, reset đầu segment,
  OFI = bid − ask. OF tính lại khớp `features.bin` (sai lệch max 4,2e-7); tổng update 3.120 = 3.176 − 56 state đuôi
  sau lần đổi mid cuối (bị bỏ, không mang sang segment khác); tổng elapsed khớp 295,394 s. Raw memmap (3.214) tách
  khỏi kept (61), `raw_ts` tăng dần.
- **P6 Nhãn as-of**: `data.py:63-72`, `74-98` — quote cuối ≤ t+h, tuổi ≤ 10 s, cùng segment, không vượt
  `segment_end`; nhãn train trước `train_end`; gap 6 ngày, `config.py` chặn ≤ 5.
- **P7 `select_origins`**: ngữ nghĩa y hệt `HEAD:src_OB/train.py` (chỉ đưa `length` bất biến ra ngoài vòng, thêm
  counts); thứ tự mask giữ nguyên, population vẫn `cfg["models"]`. Số 0 trong DATA_REPORT đúng: tổng kept = 61 <
  context 100 và mỗi segment ≤ 24 kept ⇒ `indices()` đã trả 0 cho cả train và VAL trước mask TimesFM.
  `train.py:37-39` raise trước `mkdir` (dòng 55) nên không sinh cell dở.

## Findings

- **B1 — BLOCKER dữ liệu (xác nhận, không phải lỗi code).** Evidence: `data_report.json` folds, `segments.json` id 20,
  `reconstruction.json`. VAL rỗng ở cả 4 fold **kể cả khi context = 1 và bỏ TimesFM**: mọi cửa sổ VAL chỉ chứa
  segment một state (≤ 3 µs); segment nhiều state duy nhất (07-05) nằm trong FIT fold 2–4 / gap fold1. LSTM/tree cần
  100 origin trong segment (max 24). TimesFM 512×h cần 8,53/17,07/25,6 h; AutoTS 90×h cần 1,48/2,97/4,45 h; run thô
  dài nhất 300,4 s. Không có đường hợp lệ: mọi điểm cắt là gap ID (đổi ngưỡng/ordering/grouping không kéo dài được
  run), 37/38 snapshot không có bridge ở bất kỳ thời điểm nào, dựng ngược từ snapshot tương lai bị cấm và diff absolute
  không đảo ngược được, repo pin chỉ một commit. Fix: quyết định nguồn dữ liệu/phương pháp của user (RUN_REPORT §5).
- **W1 — WARN (không ảnh hưởng archive này).** `reconstruct.py:223-228`, `168-176`: snapshot chỉ được áp khi có depth
  event với ts ≥ ts snapshot; nếu message bridge nhận được trước snapshot (quy trình Binance: buffer stream → GET
  snapshot) nó bị bỏ vào "waiting", và nếu bridge trễ > 10 s thì bị cắt do check timestamp đứng trước check
  contiguity. Fix cho nguồn dữ liệu mới: giữ buffer theo ID các message có u > S, áp sau snapshot với timestamp state =
  max(ts); tạo `prepared_dir` mới.
- **I1 — INFO, nhãn reset dễ đọc sai**: 35 segment kết thúc với `invalid_timestamp_gap` nhưng thực chất cũng là gap
  ID lớn (`sequence_gap` không bao giờ xuất hiện); `counts.snapshot_reanchor = 38` đếm mọi snapshot được áp
  (`reconstruct.py:140`), chỉ 2 segment thật sự kết thúc do reanchor. Fix: ghi cả hai cờ / đổi tên count.
- **I2 — INFO**: segment 20 ghi `known_hard_gap` nhưng run depth thực tế đã hết lúc 20:55:12.975; xử lý đúng và thận
  trọng, chỉ cần ghi chính xác trong RUN_REPORT.
- **I3 — INFO**: timestamp là receipt time, có burst (89 message trong 135 ms; 7 lần đổi mid đầu trong 0,1 s); feature
  elapsed và nhãn as-of chịu ảnh hưởng — ghi là giới hạn.
- **I4 — INFO**: `valid_segment_seconds_fit/val` = 1,3e-05/2e-06 s là do quy ước `end_exclusive = last + 1 µs`;
  nghĩa thực là 0 s book dùng được; 4 fold chỉ là fold theo lịch.
- **I5 — INFO, giới hạn collector**: ID/ts không chứng minh không thiếu row bên trong message; segment 20 không có dấu
  hiệu bất thường nhưng đó không phải bằng chứng. Checker không kiểm lại card upstream qua mạng; phép đo trên file
  (1.392 run ≈300 s, 115,8 h / 1.391 h) được tái lập độc lập.
- **I6 — INFO, env/provenance**: `environment.json` — torch cu128 có CUDA, xgboost `USE_CUDA=True`, catboost 1 GPU,
  LightGBM 4.7.0 CUDA (log EXIT=0 16:37), không JAX; config sha256 `72d68928…` khớp. GPU-only lúc fit, scaler/AutoTS
  FIT-only, latency: **chưa có bằng chứng** vì chưa có fit (đúng với trạng thái blocked). Số trong RUN_REPORT §1–3
  khớp (96 cell kỳ vọng, 0 hoàn tất).

## Xử lý của session chính

- Không có ERROR cần sửa. B1 được báo cho user như blocker, không lách bằng cách nối gap/giảm context/đổi revision.
- W1, I1: chưa sửa code vì không đổi kết quả trên archive này và prepared hiện tại phải giữ nguyên contract; sửa khi
  prepare nguồn dữ liệu mới vào `prepared_dir` mới. I2–I4 được ghi rõ trong RUN_REPORT §2.
