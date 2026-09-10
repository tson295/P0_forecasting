# BACKUP_STATUS — 2026-09-10

- **Đích ngoài instance:** GitHub `git@github.com:tson295/P0_forecasting.git`, branch `OB`. Instance có SSH key
  `~/.ssh/id_ed25519` (xác thực được tài khoản `tson295`); `remote.origin.pushurl` trỏ SSH, fetch vẫn HTTPS.
- **Đã xác nhận:** `a04d17b` (merge `e169df1` + remote `109cbc2`) đã push; LFS 7/7 object (87 KB) đã upload;
  `git ls-remote origin refs/heads/OB` = `a04d17b80d6bbd3445bde890d1e7614373860701` (~17:45 UTC).
  Commit báo cáo tìm nguồn `603738e1316e6d55c0fd0732f4c6ba9d82a463e9` đã push và khớp `ls-remote` (18:13 UTC; checker T-P4
  xác nhận thêm 7/7 LFS object có trên remote). Code replay v2 + phần sửa theo checker lượt 2:
  `1e2f3ce88b97916e90fad04c5a8e2d009ee215be` (commit 18:38:47 UTC) đã push và khớp `ls-remote`. Commit ghi checker lượt 3 và file
  này là commit cuối của lượt tiếp nối; SHA của nó được xác nhận bằng `ls-remote` trong báo cáo cuối của session.
- **Commit chứa file này:** được push ngay sau khi tạo và xác nhận bằng `git ls-remote`; kết quả ghi ở báo cáo cuối
  của session. Nếu push thất bại, trạng thái là BACKUP_PENDING cho commit đó.
- **Không backup ra ngoài (có chủ đích):** 4 raw Parquet HF (704.186.850 byte; tải lại đúng revision bằng
  `python -m src_OB download`, sha256 pin trong `download_manifest.json`), venv `/home/ubuntu/venv-ob`
  (tái lập bằng `run_meta/install_env.sh`), tmux pane log, scratch. Không có dữ liệu nguồn mới nào được tải.
- Không có đích backup khác được cấu hình (rclone chưa cấu hình, không có HF token); không upload repo/raw lên dịch vụ
  công cộng khác.
- **Máy:** instance Vast `C.50503596` vẫn chạy (không stop/destroy); `/workspace` không phải volume.
