"""Checker log KHÔNG tương tác (quyết định user 2026-09-04): mọi finding ghi vào `experiments/<run>/checker_log.jsonl`.

- Bất biến cứng do code ép (checksum, biên leakage, target ngoài partition, artifact S0/Candidate malformed,
  TEST chạy lần hai, TRAINING LOCKED, LF không phủ HF): `hard_fail` ghi ERROR rồi thoát ngay — KHÔNG hỏi user. Đây là thí nghiệm
  KHÔNG HỢP LỆ, không phải lựa chọn tài nguyên → không bao giờ có tuỳ chọn "chạy tiếp".
- **NGOẠI LỆ DUY NHẤT (quyết định user 2026-09-04d, §10): SỰ CỐ TÀI NGUYÊN GPU** — không có GPU, GPU được giao biến mất,
  CUDA không dùng được, backend không train được trên GPU, phát hiện CPU fallback, định tuyến GPU sai, worker CUDA không
  khởi động/chết, OOM khiến đường GPU không đi tiếp được: `gpu_stop` DỪNG AN TOÀN (giữ nguyên artifact đã xong, ghi log),
  KHÔNG CPU fallback, KHÔNG tự đổi batch/hyperparameter/methodology, rồi **HỎI USER muốn xử lý thế nào** (exit code 3).
- Finding tư vấn (tương quan cao, nghi dư thừa, gain bất thường, quan sát runtime, ghi chú methodology không vi phạm bất biến):
  `record` với WARN/INFO rồi tiếp tục — KHÔNG hỏi user.
- Agent `checker` cũng ghi finding qua `scripts/checker_record.py` (cùng schema); ERROR = chặn run cho tới khi sửa, WARN = ghi và đi tiếp.
Mỗi bản ghi: timestamp, stage, model, severity, check_id, message, file, ref.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import NoReturn

LOG_NAME = "checker_log.jsonl"
_WRITE_LOCK = threading.Lock()  # orchestrate: nhiều nhánh ghi finding song song
SEVERITIES = ("PASS", "INFO", "WARN", "ERROR")


def log_path(exp_dir: Path | str) -> Path:
    return Path(exp_dir) / LOG_NAME


def record(exp_dir: Path | str | None, stage: str, severity: str, check_id: str, message: str, model: str = "",
           file: str = "", ref: str = "") -> dict:
    """Ghi một finding (append JSONL). exp_dir None → chỉ trả bản ghi (không ghi file)."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity phải thuộc {SEVERITIES}: {severity}")
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "model": model, "severity": severity,
           "check_id": check_id, "message": message, "file": file, "ref": ref}
    if exp_dir is not None:
        p = log_path(exp_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def hard_fail(exp_dir: Path | str | None, stage: str, check_id: str, message: str, model: str = "", file: str = "",
              ref: str = "") -> NoReturn:
    """Bất biến cứng bị vi phạm: ghi ERROR rồi dừng ngay (SystemExit), không hỏi user."""
    record(exp_dir, stage, "ERROR", check_id, message, model, file, ref)
    sys.exit(f"[{check_id}] {message}")


GPU_STOP_EXIT = 3  # exit code riêng: "dừng vì tài nguyên GPU, đang chờ user quyết" (khác 1 = lỗi/bất biến thường)


def gpu_stop(exp_dir: Path | str | None, stage: str, check_id: str, message: str, model: str = "",
             detail: str = "", options=()) -> NoReturn:
    """Sự cố TÀI NGUYÊN GPU (§10) — ngoại lệ DUY NHẤT được dừng và HỎI USER.

    Ghi ERROR (ref=USER_DECISION_REQUIRED) → in khối thông báo rõ ràng → thoát với `GPU_STOP_EXIT`.
    KHÔNG chuyển sang CPU, KHÔNG đổi batch/hyperparameter/methodology, KHÔNG xoá artifact đã hoàn tất.
    """
    record(exp_dir, stage, "ERROR", check_id, f"[CẦN USER QUYẾT] {message}" + (f" | {detail}" if detail else ""),
           model=model, ref="USER_DECISION_REQUIRED")
    opts = list(options) or [
        "sửa/đổi GPU (driver, instance khác, GPU rảnh) rồi chạy lại ĐÚNG bước vừa dừng",
        "chạy tiếp trên MỘT GPU: export P0_GPU_DEVICES=0 (chậm hơn, khoa học không đổi)",
        "dừng hẳn run này và báo lại",
    ]
    lines = ["", "=" * 78,
             "DỪNG AN TOÀN: SỰ CỐ TÀI NGUYÊN GPU — CẦN USER QUYẾT (không tự xử lý)",
             "=" * 78,
             f"Bước: {stage}" + (f" | model: {model}" if model else ""),
             f"Sự cố: {message}"]
    if detail:
        lines.append(f"Chi tiết: {detail}")
    lines += ["",
              "- KHÔNG có CPU fallback (training chỉ GPU).",
              "- KHÔNG tự đổi batch / hyperparameter / seed / methodology để 'chạy cho xong'.",
              f"- Artifact đã hoàn tất được giữ nguyên: {exp_dir}",
              "", "Bạn muốn xử lý thế nào?"]
    lines += [f"  {i + 1}. {o}" for i, o in enumerate(opts)]
    lines += ["=" * 78, ""]
    print("\n".join(lines), flush=True)
    sys.exit(GPU_STOP_EXIT)


def read(exp_dir: Path | str) -> list[dict]:
    p = log_path(exp_dir)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def blocking_errors(exp_dir: Path | str, stage: str | None = None) -> list[dict]:
    """ERROR chưa được đóng bởi một PASS cùng check_id ghi sau đó (checker/code dùng để biết run có bị chặn không)."""
    rows = read(exp_dir)
    open_err: dict[str, dict] = {}
    for r in rows:
        if stage is not None and r.get("stage") != stage:
            continue
        key = f"{r.get('stage')}|{r.get('model')}|{r.get('check_id')}"
        if r["severity"] == "ERROR":
            open_err[key] = r
        elif r["severity"] == "PASS" and key in open_err:
            del open_err[key]
    return list(open_err.values())
