# BACKUP_STATUS — 2026-09-10

- **Đích ngoài instance:** GitHub `git@github.com:tson295/P0_forecasting.git`, branch `OB`. Instance có SSH key
  `~/.ssh/id_ed25519` (xác thực được tài khoản `tson295`); `remote.origin.pushurl` trỏ SSH, fetch vẫn HTTPS.
- **Đã xác nhận:** `a04d17b` (merge `e169df1` + remote `109cbc2`) đã push; LFS 7/7 object (87 KB) đã upload;
  `git ls-remote origin refs/heads/OB` = `a04d17b80d6bbd3445bde890d1e7614373860701` (~17:45 UTC).
- **Commit chứa file này:** được push ngay sau khi tạo và xác nhận bằng `git ls-remote`; kết quả ghi ở báo cáo cuối
  của session. Nếu push thất bại, trạng thái là BACKUP_PENDING cho commit đó.
- **Không backup ra ngoài (có chủ đích):** 4 raw Parquet HF (704.186.850 byte; tải lại đúng revision bằng
  `python -m src_OB download`, sha256 pin trong `download_manifest.json`), venv `/home/ubuntu/venv-ob`
  (tái lập bằng `run_meta/install_env.sh`), tmux pane log, scratch. Không có dữ liệu nguồn mới nào được tải.
- Không có đích backup khác được cấu hình (rclone chưa cấu hình, không có HF token); không upload repo/raw lên dịch vụ
  công cộng khác.
- **Máy:** instance Vast `C.50503596` vẫn chạy (không stop/destroy); `/workspace` không phải volume.
