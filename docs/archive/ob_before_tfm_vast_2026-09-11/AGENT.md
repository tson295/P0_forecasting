# Agent cho luồng Order Book

Chỉ `checker` còn nằm trong `.claude/agents/` và được gọi trong luồng OB.
Các vai trò cũ đã lưu tại `docs/archive/claude_legacy_2026-09-10/`, không tự kích hoạt.

Session chính thiết lập môi trường, chạy CLI, sửa lỗi và giữ goal tới khi hoàn tất.
`src_OB.train` tự chạy các fold/model/horizon; AutoTS tự tìm trong search space đã cho phép.
Không giao agent quyết định thứ tự training, feature, hyperparameter hoặc winner.

Gọi checker khi:

1. Prepare thật đã có manifest/segments: đọc data contract, coverage và tính khả dụng của cấu hình trước train.
2. Có lỗi correctness cụ thể trong run: đọc evidence và báo nguyên nhân, không chạy thử model.
3. Training hoàn tất: đối chiếu các cell/artifact/summary bắt buộc, không fit/infer lại.

Checker chỉ báo finding có evidence, phân biệt ERROR/WARN/INFO/PASS/chưa có bằng chứng.
Session chính lưu finding vào report của run và xử lý ERROR; WARN không tạo một vòng xin duyệt mới.
Không cần gọi checker trước/sau từng model. Nếu môi trường không hỗ trợ subagent, session chính đọc cùng
checklist và ghi rõ không có independent checker; không tạo một hệ thống agent thay thế.

Không agent nào chạy smoke/canary/test/probe fit, benchmark hoặc subagent khác.
Checker không cấp quyền training, không điều phối pipeline, không thay user quyết định methodology.
