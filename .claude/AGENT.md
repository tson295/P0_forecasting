# AGENT REGISTRY

Agent native nằm ở `.claude/agents/*.md`. Mọi agent làm theo plan chính thức `docs/RESEARCH_PLAN.md` và hiến pháp `.claude/CLAUDE.md`; khi mâu thuẫn, plan + CLAUDE.md thắng. Không agent nào được tự mở rộng protocol/governance/stage/rule, thêm model hay metric.

| Agent | Vai trò | Model | Tools |
|---|---|---|---|
| `main-controller` | **Điều khiển**: xác định bước, ra work order, giữ TRAINING lock, cập nhật MEMORY, phân xử mâu thuẫn | inherit | Read/Grep/Glob/Bash/Edit/Write |
| `coder` | **Code**: implement theo §8, unit test tí hon, bàn giao checker | inherit | all |
| `researcher` | **Research**: audit API/version, giả thuyết feature, verdict methodology theo luật plan | inherit | Read/Grep/Glob/Bash/WebFetch/WebSearch |
| `checker` | **Checking**: checklist §6 + review code + test + reproducibility; phủ quyết; không sửa code | inherit | Read/Grep/Glob/Bash |
| `runner` | Chạy các bước trên Vast — chỉ khi `TRAINING: UNLOCKED`; checker pre-run PASS mới chạy | inherit | all |
| `analyst` | Đọc log thật (b0_filter, keepdrop_*, champion_log, all_models, latency) → khuyến nghị theo luật plan | inherit | all |
| `infra` | SSH/tmux/Vast bootstrap, GPU env (build GPU LightGBM, CUDA XGBoost/CatBoost, torch), persistence, notification | sonnet | all |

## Cách phối hợp

```
user ⇄ main session ─ dùng main-controller để xác định bước + ra work order
      ├─ researcher → audit / giả thuyết / verdict
      ├─ coder      → code + test tí hon
      ├─ checker    → PASS/FAIL (phủ quyết) — độc lập với coder
      ├─ runner     → chạy trên Vast (chỉ khi UNLOCKED, sau checker PASS)
      ├─ analyst    → đọc log thật → khuyến nghị
      └─ infra      → Vast / tmux / GPU env
```

- Việc code: work order → coder → checker → sửa → main-controller cập nhật MEMORY.
- Một run: main-controller xác định bước (§8, base, số vòng/ε) → checker pre-run → runner → analyst → champion log / MEMORY.
- Subagent không tự gọi subagent khác: work order do main session thực thi. Effort đặt theo từng lần gọi khi cần.
- `runner` từ chối chạy khi `MEMORY.md` còn `TRAINING: LOCKED`; unlock chỉ bằng lệnh user rõ ràng.

Tài liệu: chính thức = `README.md`, `docs/RESEARCH_PLAN.md`, `.claude/CLAUDE.md`, `.claude/MEMORY.md`, file này; lưu trữ (hết hiệu lực) = `docs/archive/`; tham khảo = `docs/reference/`; layout mẫu số giả = `reports/smoke_visualize.md`. Hooks/statusline: `.claude/settings.json` + `.claude/hooks/`.
