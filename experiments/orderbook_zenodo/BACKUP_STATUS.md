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

Commit cuối cùng (report, checker findings, MEMORY) được ghi trong mục "Commit cuối" của `RUN_REPORT.md` và kiểm tra
bằng `git ls-remote origin refs/heads/OB`.

Không nằm trong git (có chủ đích):
- Raw `data/orderbook/zenodo_20046390/btcusdt_lob_oct2023.tar.gz` (308.618.431 B): tải lại được từ Zenodo record 20046390,
  md5 `58507a0fe055f0f2d3a498c15fc7d8d2` đã pin trong config và `download_manifest.json`.
- Checkpoint pretrained TimesFM `google/timesfm-2.5-200m-pytorch` @ `1d952420…` (theo policy, không commit checkpoint
  pretrained gốc; tải lại theo revision đã pin). Adapter LoRA đã train thì có trong git.
- Venv `/home/ubuntu/venv-ob`: dựng lại bằng `experiments/orderbook_hf/run_meta/install_env.sh`; phiên bản đã ghi trong
  `run_meta/environment.json` và `pip_freeze.txt`.
- `summary/results.lock` (file khóa rỗng).
