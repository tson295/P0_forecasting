# AGENT REGISTRY

Agent native nằm ở `.claude/agents/*.md`. Mọi agent làm theo plan chính thức `docs/RESEARCH_PLAN.md` và hiến pháp `.claude/CLAUDE.md`; khi mâu thuẫn, plan + CLAUDE.md thắng. Không agent nào được tự mở rộng protocol/governance/stage/rule, thêm model hay metric.

## Nguyên tắc: chỉ giữ agent làm việc mà pipeline KHÔNG tự làm

Pipeline đã deterministic và tự ép luật trong code (`src/p0/cli.py`): S0_m khoá + Candidate_m (`lock-s0`), add-one KEEP/DROP theo `MedianGain ≥ −ε_m`, prune PI (chỉ cột mới), confirmation 3 seed, TimesFM-LoRA → freeze → XReg → `tfm-final`, AutoTS probe → `autots-search`, champion `> +ε_champion`, ensemble, Final, `visualize` hậu kỳ — chạy tuần tự bằng lệnh (2026-09-03). CLI cũng tự chặn: `TRAINING: LOCKED` (đọc MEMORY), GPU preflight, sha256 §6.1, LF phủ HF, `--smoke/--allow-cpu` chỉ cho dataset tổng hợp, `loop` đầu tiên bắt buộc là `lgbm`, `loop` cần `s0/` từ `lock-s0`, không vẽ trong training. Điều phối những bước đó bằng agent không thêm giá trị — nên vai trò `main-controller`, `coder`, `runner` **đã bỏ** (2026-08-31): bước hiện tại đọc ở plan §8 + `.claude/MEMORY.md` "Exact Next Step"; lệnh chạy ở plan §8 + `docs/VAST_SESSION_PROMPT.md`; viết code cần full context nên do session chính làm.

Giữ lại đúng bốn vai trò làm việc mà code không làm thay được:

| Agent | Vai trò | Khi nào gọi | Model | Tools |
|---|---|---|---|---|
| `checker` | **Verify độc lập KHÔNG TƯƠNG TÁC**: checklist §6 (leakage, biên, target, alignment, metric trên giá, decode, seed/ε, S0/candidate, LoRA), review code, chạy unit/smoke CPU, schema log, reproducibility. Không sửa code. Mọi finding ghi `experiments/<run>/checker_log.jsonl` qua `scripts/checker_record.py` (PASS/INFO/WARN/ERROR); ERROR = chặn run tới khi sửa, WARN/INFO = ghi rồi tiếp tục; **không bao giờ hỏi user "tiếp hay dừng"**. | trước khi nhận code mới, trước mỗi run thật, khi kết quả bất thường | inherit | Read/Grep/Glob/Bash |
| `researcher` | **Audit API/version trước khi code** (ghi `docs/reference/audit_<lib>.md`) + trọng tài methodology theo luật plan. | trước khi code một thư viện mới hoặc đổi version; khi cần verdict đúng/sai theo plan | inherit | Read/Grep/Glob/Bash/WebFetch/WebSearch |
| `analyst` | **Sau một full run**: đọc kết quả thật (vòng 15 ngày ở `experiments/15d/`, vòng expanded-data ở `experiments/full/`), phát hiện anomaly / failure / phụ thuộc regime, đánh giá theo luật plan, đề xuất experiment/feature kế tiếp có căn cứ. | sau `loop`/`ensemble`/`final` có log thật | inherit | all |
| `infra` | **GPU/env troubleshooting trên Vast**: build LightGBM GPU, CUDA/driver, jax[cuda12] chung GPU với torch (PREALLOCATE=false), Git LFS, tmux/persistence khi bootstrap fail. | khi `scripts/vast_bootstrap.sh` hoặc preflight fail | sonnet | all |

## Cách phối hợp

```
user ⇄ session chính (đọc plan §8 + MEMORY để biết bước kế tiếp; tự viết code; tự chạy lệnh)
      ├─ researcher → audit API/version / verdict methodology   (trước khi code)
      ├─ checker    → PASS/FAIL + phủ quyết                      (trước khi nhận code, trước run thật)
      ├─ analyst    → đọc kết quả thật → finding + đề xuất       (sau full run)
      └─ infra      → GPU/env trên Vast                          (khi bootstrap/preflight fail)
```

- Việc code: session chính viết → `checker` review → sửa → cập nhật MEMORY.
- Một run: kiểm tra bước theo plan §8 → `checker` pre-run (ghi checker_log; `scripts/checker_record.py --blocking` phải sạch ERROR) → chạy `python run.py <step>` trong tmux → `analyst` đọc kết quả. Bất biến cứng (checksum, biên, target, S0 malformed, GPU/CPU fallback, TEST lần hai, LOCKED) do code tự ép và dừng — không có prompt hỏi user.
- Subagent không tự gọi subagent khác. Không agent nào được tự unlock training: `TRAINING: UNLOCKED` chỉ do user ra lệnh rõ ràng, và `cli.gate()` là chốt chặn thật.

Tài liệu: chính thức = `README.md`, `docs/RESEARCH_PLAN.md`, `.claude/CLAUDE.md`, `.claude/MEMORY.md`, file này; lưu trữ (hết hiệu lực) = `docs/archive/`; tham khảo = `docs/reference/`; layout mẫu số giả = `reports/smoke_visualize.md`. Hooks/statusline: `.claude/settings.json` + `.claude/hooks/`.
