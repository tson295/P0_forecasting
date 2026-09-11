# Điều phối và checker — tfm_autots

Session chính là người điều phối: setup env, chạy launcher/CLI, theo dõi log/process, sửa lỗi cụ thể,
lưu báo cáo và commit/push. CLI tự xử lý tất cả fold/candidate/seed; không cần agent chọn bước training.
Không thêm researcher, infra, runner, monitor, analyst hoặc agent tự tìm model/feature.

Chỉ có subagent `checker`, đọc `.claude/agents/checker.md`:
1. Sau setup: đối chiếu config/data manifest/LFS/env metadata, không chạy thử model.
2. Khi run có lỗi correctness hoặc dấu hiệu chậm cụ thể từ log: đọc đúng evidence liên quan.
3. Cuối phase: đối chiếu stages, artifact, metrics và báo cáo; không fit/infer lại.

Không gọi checker trước/sau từng candidate hoặc từng model fit. Không giao checker sửa code hay hoàn thiện data.
Session chính xử lý findings; checker không cấp quyền chạy, không yêu cầu user duyệt từng bước.
Nếu không có subagent capability, session chính làm cùng checklist và ghi rõ không có independent checker.
Không agent nào chạy tests/smoke/probe/warmup/benchmark hoặc gọi thêm subagent.
