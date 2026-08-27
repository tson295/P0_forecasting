# AGENT REGISTRY

Agent native nằm ở `.claude/agents/*.md`. Mọi agent làm theo plan đơn giản hóa `docs/RESEARCH_PLAN.md` (rev hiện hành) và hiến pháp rút gọn `.claude/CLAUDE.md`; khi mâu thuẫn, plan + CLAUDE.md thắng. Không agent nào được tự mở rộng protocol/governance/stage/rule.

| Agent | Trách nhiệm | Model | Trạng thái |
|---|---|---|---|
| `research-methodology` | Kiểm tra một run/quyết định đúng luật plan (bước nào, base nào, KEEP/DROP §2.1, champion §3, lọc B0 §1.4), xử lý mâu thuẫn | inherit | active |
| `leakage-auditor` | Checklist §6 của plan (input, target, time alignment, leakage, biên, metric trên giá, decode, hợp lý); quyền phủ quyết một run | inherit | active |
| `tree-engineer` | Code theo §8: adapter, split, 39 feature §2.3, metric trên giá, filter_b0 §1.4, vòng lặp §2.1, runner LightGBM/XGBoost/CatBoost/ExtraTrees | inherit | active |
| `timesfm-engineer` | §2.2 #4: audit version/covariate API, TFM-POINT, covariate loop nếu có, LoRA khi thắng E0 | inherit | active |
| `validation-metrics` | §0 metric trên giá + Gain 15 ô, §1.2 fold, §1.3 ε/số vòng, §7 log/all_models, §7.4 latency | inherit | active |
| `test-debug` | Unit/smoke test không-training theo §8; validate config/JSON/frontmatter; determinism | sonnet | active |
| `remote-infra` | SSH/tmux/Vast bootstrap, GPU detect, persistence, notification | sonnet | active |
| `experiment-analyst` | Đọc log thật (keepdrop_*, champion_log, all_models, latency) và ra khuyến nghị theo luật plan | inherit | chờ có run thật |
| `gpu-training-runner` | Chạy các bước trên Vast — chỉ sau khi user unlock training | inherit | **LOCKED** |

Ghi chú: `gpu-training-runner` từ chối chạy khi `MEMORY.md` còn `TRAINING: LOCKED`; unlock chỉ bằng lệnh user rõ ràng. Hooks/statusline nằm trong `.claude/settings.json` + `.claude/hooks/`. Layout mẫu (số giả) của mọi bảng/figure: `reports/smoke_visualize.md`.
