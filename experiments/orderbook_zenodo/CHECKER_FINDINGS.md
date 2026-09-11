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
