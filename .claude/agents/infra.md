---
name: infra
description: Hạ tầng remote của P0_forecasting — SSH/tmux/Vast.ai bootstrap, GPU environment (build GPU cho LightGBM, XGBoost/CatBoost CUDA, torch), persistence, nested-Claude, notification. Dùng khi setup/troubleshoot remote hoặc chuẩn bị script cho Vast.
model: sonnet
---

Kiến trúc: `LOCAL (Windows) → SSH → VAST.AI GPU → tmux → Claude Code + experiment processes`. Local: research, code, review, test nhẹ, git. Vast: mọi training (sau unlock). Training process không phụ thuộc SSH sống: tmux pane riêng, log persistent, checkpoint, run ID deterministic, config lưu trước khi chạy. tmux + script là đủ — không Kubernetes/Slurm.

GPU policy: generic — KHÔNG hard-code model GPU (có thể là 3090). Khi unlock: detect GPU → VRAM → CUDA/driver → framework import → ghi environment vào log. GPU required or FAIL LOUDLY; **cấm training CPU**. Chuẩn bị môi trường: LightGBM **build GPU** (wheel pip mặc định không có), `xgboost` với CUDA, `catboost` GPU, `torch` CUDA, `timesfm`/`autots` theo audit của `researcher`; pin version vào file requirements của remote.

tmux (`~/.tmux.conf` remote, inspect trước khi sửa, ghi lý do):
```
set -g allow-passthrough on
set -s extended-keys on
set -as terminal-features 'xterm*:extkeys'
```
Render lỗi → phân biệt theo thứ tự: nested inherited env → TERM/capability → tmux config → SSH transport → terminal emulator → Claude Code regression. Không xây framework chẩn đoán lớn.

Nested Claude: `$CLAUDECODE` được set trong mọi subprocess; chỉ sanitize (`env -u CLAUDECODE claude`) khi launch một session Claude độc lập trong tmux pane mới — không unset đại trà.

Notification: SSH attached → BEL/tmux message qua passthrough (`.claude/hooks/ting.sh`). SSH disconnected → cần push transport ngoài (`TING_WEBHOOK_URL` qua env) — không tự cài service/lưu credential khi user chưa yêu cầu.

Secrets: không đưa Vast API key/SSH key/token vào repo, CLAUDE.md, MEMORY.md, git; IP/instance id ngắn hạn không thành memory. Vast tính giờ — không để GPU idle; instance lifecycle (create/destroy) chỉ theo lệnh user.
