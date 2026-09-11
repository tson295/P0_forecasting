# BACKUP_STATUS — run Zenodo 20046390

Đích backup: `origin` = GitHub `tson295/P0_forecasting`, branch `OB`. Push qua SSH (`remote.origin.pushurl`). Object
lớn (`data/orderbook/**/*.bin`, `experiments/**/*.parquet|.pt|.joblib`) đi qua Git LFS. Không force push, không
`git add -A`, không commit secret.

| Commit | Nội dung | Xác minh |
|---|---|---|
| c68fc45 | config/adapter/downloader Zenodo, fold 9/1/2/2 | — |
| ade1e37 | environment + launcher | — |
| d8797c5 | prepared memmaps (LFS) + DATA_REPORT READY | `ls-remote` = d8797c5… |
| 688fefd | sửa GOSS/linear_tree + `--resume` + docs Z-W1 | `ls-remote` = 688fefd… |
| 2a5a1c3 | `max_bin ≤ 255` + `PYTHONFAULTHANDLER` | `ls-remote` = 2a5a1c3… |
| 60100fc → 1b5ab07 | backup từng phần: cell fold1, attempts, bản nháp report | `ls-remote` sau mỗi push |
| 40b4c45 | toàn bộ 120 cell, summary CSV, log train/summarize; 184 object LFS (314 MB) | `ls-remote` = 40b4c4556b7ee51ed3bd81a2611ac83638e54961 |

| e171d03 | RUN_REPORT COMPLETE, CHECKER_FINDINGS lượt ZF, BACKUP_STATUS, MEMORY | `ls-remote` = e171d03180a1bc1e76c085767d5bb2e9de1bc227 |

| ecfc67d | evidence quét Zenodo theo ngày + BACKUP_STATUS | `ls-remote` = ecfc67dbf06c174ec28c7134bd2cd5ce78bd05e1 |
| e8ca931 | AutoTS adapter v2 (log-return quanh origin) + README | push cùng 37aae35 |
| 37aae35 | `git mv` 15 cell AutoTS v1 → `superseded/autots_raw_price/` + README | `ls-remote` = 37aae35970bc9c50ebdd396af628baa03e1d5c97 |
| 6108a1d | shim pandas weighted sample cho AutoTS 1.0.4 | `ls-remote` = 6108a1de91de8a3f5bc6ff77c8da93d2dee4e021 |
| 749712b | 15 cell AutoTS v2, attempt fold3/autots/h120s, summary sinh lại, log; 30 object LFS | `ls-remote` = 749712bb4324b6a7819621699b2742e2ab476722 |

| d31b31d | RUN_REPORT AutoTS v2, CHECKER_FINDINGS lượt ZA, MEMORY | `ls-remote` = d31b31d1bb439745137590c4322412ca5b8315c3 |
| 62b56a4 | `src_OB/visualize.py` + figures (30 ảnh path, 2 heatmap, index/csv); 32 object LFS | `ls-remote` = 62b56a41840bddfc13749b260e7d3bed63e7c976 |

Commit chứa bản cập nhật file này nằm ngay sau 62b56a4 và được kiểm tra bằng `git ls-remote origin refs/heads/OB`.

Không nằm trong git (có chủ đích):
- Raw `data/orderbook/zenodo_20046390/btcusdt_lob_oct2023.tar.gz` (308.618.431 B): tải lại được từ Zenodo record 20046390,
  md5 `58507a0fe055f0f2d3a498c15fc7d8d2` đã pin trong config và `download_manifest.json`.
- Checkpoint pretrained TimesFM `google/timesfm-2.5-200m-pytorch` @ `1d952420…` (theo policy, không commit checkpoint
  pretrained gốc; tải lại theo revision đã pin). Adapter LoRA đã train thì có trong git.
- Venv `/home/ubuntu/venv-ob`: dựng lại bằng `experiments/orderbook_hf/run_meta/install_env.sh`; phiên bản đã ghi trong
  `run_meta/environment.json` và `pip_freeze.txt`.
- `summary/results.lock` (file khóa rỗng).
