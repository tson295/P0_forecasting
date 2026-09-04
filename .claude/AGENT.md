# AGENT REGISTRY

Agent native nằm ở `.claude/agents/*.md`. Mọi agent làm theo plan chính thức `docs/RESEARCH_PLAN.md` và hiến pháp `.claude/CLAUDE.md`; khi mâu thuẫn, plan + CLAUDE.md thắng. Không agent nào được tự mở rộng protocol/governance/stage/rule, thêm model hay metric.

## Nguyên tắc: chỉ giữ agent làm việc mà pipeline KHÔNG tự làm

Pipeline đã deterministic và tự ép luật trong code (`src/p0/cli.py`): S0_m khoá + Candidate_m (`lock-s0`), add-one KEEP/DROP theo `MedianGain ≥ −ε_m`, prune PI (chỉ cột mới), confirmation 3 seed, TimesFM-LoRA → freeze → XReg → F_win → `tfm-final` (hệ thống A baseline vs B +XReg), AutoTS probe → `autots-search`, champion `> +ε_champion` (hoãn → `champion-replay` thứ tự cố định), ensemble, Final, `visualize` hậu kỳ — chạy bằng lệnh, hoặc `orchestrate` điều phối DAG nhánh trên scheduler 2 GPU đối xứng (2026-09-04c). Lịch chạy/GPU chỉ đổi wall-clock, không đổi quyết định khoa học. CLI cũng tự chặn: `TRAINING: LOCKED` (đọc MEMORY), GPU preflight, sha256 §6.1, LF phủ HF, `--smoke/--allow-cpu` chỉ cho dataset tổng hợp, `loop` đầu tiên bắt buộc là `lgbm`, `loop` cần `s0/` từ `lock-s0`, không vẽ trong training. Điều phối những bước đó bằng agent không thêm giá trị — nên vai trò `main-controller`, `coder`, `runner` **đã bỏ** (2026-08-31): bước hiện tại đọc ở plan §8 + `.claude/MEMORY.md` "Exact Next Step"; lệnh chạy ở plan §8 + `docs/VAST_SESSION_PROMPT.md`; viết code cần full context nên do session chính làm.

## Pha VẬN HÀNH (quyết định user 2026-09-04d)

Project đã rời pha nghiên cứu/thiết kế. Đường chạy bình thường **KHÔNG gọi `researcher`** và **KHÔNG gọi `analyst`
trong lúc training**. Agent vận hành chính khi run đang chạy là **`run-monitor`** (chỉ quan sát).

| Agent | Trạng thái | Gọi khi | Model | Tools |
|---|---|---|---|---|
| `checker` | ACTIVE | ĐÚNG hai điểm: trước `orchestrate`, và trước `final` (TEST) | inherit | Read/Grep/Glob/Bash |
| `run-monitor` | ACTIVE (chính trong lúc run) | trong khi `orchestrate` chạy, để biết đang ở đâu / GPU nào bận / có lỗi gì | inherit | Read/Grep/Glob/Bash |
| `infra` | ON-DEMAND | bootstrap/CUDA/driver/backend GPU/Git LFS/OOM hỏng, hoặc run dừng với `ref=USER_DECISION_REQUIRED` | sonnet | all |
| `analyst` | POST-RUN ONLY | sau khi `orchestrate` có artifact thật; và sau `final` | inherit | all |
| `researcher` | DORMANT | chỉ khi USER yêu cầu đổi methodology/model/feature/thư viện hoặc khảo cứu kỹ thuật mới | inherit | Read/Grep/Glob/Bash/WebFetch/WebSearch |

`run-monitor` chỉ đọc `scheduler_log.jsonl` / `orchestrate_log.jsonl` / `checker_log.jsonl` / `log.csv` + `nvidia-smi`;
nó KHÔNG được sửa code, methodology, feature, seed, ε, hyperparameter, KHÔNG quyết KEEP/DROP hay champion.

**Ngoại lệ tương tác DUY NHẤT**: sự cố **tài nguyên GPU** (§10) — `checker_log.gpu_stop` dừng an toàn, giữ artifact,
không CPU fallback, không đổi tham số, rồi HỎI USER (exit 3, ERROR có `ref=USER_DECISION_REQUIRED`). Mọi vi phạm bất
biến khoa học khác vẫn dừng tự động, không hỏi, không có tuỳ chọn "chạy tiếp".

Chi tiết bốn vai trò cũ (vẫn đúng về nội dung công việc):

| Agent | Vai trò | Khi nào gọi | Model | Tools |
|---|---|---|---|---|
| `checker` | **Verify độc lập KHÔNG TƯƠNG TÁC**: checklist §6 (leakage, biên, target, alignment, metric trên giá, decode, seed/ε, S0/candidate, LoRA), review code, chạy unit/smoke CPU, schema log, reproducibility. Không sửa code. Mọi finding ghi `experiments/<run>/checker_log.jsonl` qua `scripts/checker_record.py` (PASS/INFO/WARN/ERROR); ERROR = chặn run tới khi sửa, WARN/INFO = ghi rồi tiếp tục; **không bao giờ hỏi user "tiếp hay dừng"**. | trước khi nhận code mới, trước mỗi run thật, khi kết quả bất thường | inherit | Read/Grep/Glob/Bash |
| `researcher` | **Audit API/version trước khi code** (ghi `docs/reference/audit_<lib>.md`) + trọng tài methodology theo luật plan. | trước khi code một thư viện mới hoặc đổi version; khi cần verdict đúng/sai theo plan | inherit | Read/Grep/Glob/Bash/WebFetch/WebSearch |
| `analyst` | **Sau một full run**: đọc kết quả thật (vòng 15 ngày ở `experiments/15d/`, vòng expanded-data ở `experiments/full/`), phát hiện anomaly / failure / phụ thuộc regime, đánh giá theo luật plan, đề xuất experiment/feature kế tiếp có căn cứ. | sau `loop`/`ensemble`/`final` có log thật | inherit | all |
| `infra` | **GPU/env troubleshooting trên Vast**: build LightGBM GPU, CUDA/driver, jax[cuda12] chung GPU với torch (PREALLOCATE=false), Git LFS, tmux/persistence khi bootstrap fail. | khi `scripts/vast_bootstrap.sh` hoặc preflight fail | sonnet | all |

## Cách phối hợp

```
đường chạy VẬN HÀNH (2026-09-04d):
  git clone + git lfs pull → bootstrap → gpu-probe → check-data → lock-s0
        → checker (preflight)  → USER UNLOCK → orchestrate ──┬─ run-monitor quan sát (chỉ đọc)
                                                             └─ infra (chỉ khi GPU/env hỏng)
        → mọi đại diện đã lưu → champion-replay (thứ tự cố định) → ensemble
        → analyst (đọc kết quả VAL) → checker (trước Final) → final (TEST 1 lần) → visualize → analyst (tổng kết)
  researcher: KHÔNG nằm trong đường này — chỉ khi user yêu cầu đổi methodology/khảo cứu.
```

- Việc code: session chính viết → `checker` review → sửa → cập nhật MEMORY.
- Một run: `checker` preflight (ghi checker_log; `scripts/checker_record.py --blocking` phải sạch ERROR) → `python run.py orchestrate` trong tmux → `run-monitor` theo dõi định kỳ → sau khi xong `analyst` đọc kết quả VAL → `checker` trước `final` → `final` → `visualize`. Bất biến cứng (checksum, biên, target, S0 malformed, GPU/CPU fallback, TEST lần hai, LOCKED) do code tự ép và dừng — không có prompt hỏi user.
- Subagent không tự gọi subagent khác. Không agent nào được tự unlock training: `TRAINING: UNLOCKED` chỉ do user ra lệnh rõ ràng, và `cli.gate()` là chốt chặn thật.

Tài liệu: chính thức = `README.md`, `docs/RESEARCH_PLAN.md`, `.claude/CLAUDE.md`, `.claude/MEMORY.md`, file này; lưu trữ (hết hiệu lực) = `docs/archive/`; tham khảo = `docs/reference/`; layout mẫu số giả = `reports/smoke_visualize.md`. Hooks/statusline: `.claude/settings.json` + `.claude/hooks/`.
