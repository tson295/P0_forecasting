"""Chính sách repo: data canonical trong git-LFS + không artifact thí nghiệm nào bị ignore + vai trò agent pha vận hành.

Yêu cầu user 2026-09-04d, mục 15 (1–5) và (31–35):
(1)(2) hai CSV 2 năm KHÔNG bị gitignore · (3) cả hai đi Git LFS · (4) sha256 khớp anchor ·
(5) artifact đại diện dưới experiments/** không bị ignore (kiểm bằng `git check-ignore`, không đọc mắt) ·
(31) đường chạy bình thường không gọi `researcher` · (32) `analyst` chỉ hậu-run · (33) `run-monitor` không sửa được gì ·
(34) `checker` chạy trước `orchestrate` và trước `final` · (35) sự cố tài nguyên GPU là ngoại lệ hỏi-user DUY NHẤT.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HF = "data/BTC_1m_2y.csv"
LF = "data/BTC_5m_2y.csv"
ANCHOR = ROOT / "data" / "data_checksums_2y.json"


def _git(*args, check=False):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=check)


def _is_lfs_pointer(p: Path) -> bool:
    return p.exists() and p.stat().st_size < 1024 and p.read_bytes(  # pointer ≈ 130 byte
    ).startswith(b"version https://git-lfs")


# ----------------------------------------------------------------------------- (1)(2) không bị ignore
@pytest.mark.parametrize("rel", [HF, LF])
def test_canonical_csv_is_not_gitignored(rel):
    r = _git("check-ignore", "-q", rel)
    assert r.returncode == 1, f"{rel} đang bị gitignore — clone + `git lfs pull` sẽ KHÔNG có data (check-ignore: {r.stdout})"


def test_other_data_csv_still_ignored():
    """Chỉ hai file canonical được đưa vào git; CSV data khác vẫn nằm ngoài."""
    assert _git("check-ignore", "-q", "data/BTC_hf_1min.csv").returncode == 0
    assert _git("check-ignore", "-q", "data/some_random_dump.csv").returncode == 0


# ----------------------------------------------------------------------------- (3) Git LFS
@pytest.mark.parametrize("rel", [HF, LF])
def test_canonical_csv_goes_through_git_lfs(rel):
    attr = _git("check-attr", "filter", "diff", "merge", "--", rel).stdout
    assert "filter: lfs" in attr, f"{rel} chưa có filter=lfs trong .gitattributes: {attr}"
    assert "diff: lfs" in attr and "merge: lfs" in attr
    listed = _git("lfs", "ls-files").stdout
    assert rel in listed, f"{rel} không nằm trong `git lfs ls-files`: {listed[:400]}"


def test_lfs_rules_are_narrow_not_all_csv():
    """Không LFS-hoá mọi data/*.csv — chỉ đúng hai file canonical."""
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    lfs_lines = [ln for ln in attrs.splitlines() if "filter=lfs" in ln and ln.strip().startswith("data/")]
    assert sorted(ln.split()[0] for ln in lfs_lines) == [HF, LF]
    assert "data/*.csv filter=lfs" not in attrs


# ----------------------------------------------------------------------------- (4) sha256 khớp anchor
@pytest.mark.parametrize("key,rel,rows", [("hf", HF, 1_051_201), ("lf", LF, 210_239)])
def test_canonical_csv_hash_matches_anchor(key, rel, rows):
    entry = json.loads(ANCHOR.read_text(encoding="utf-8"))["files"][key]
    assert entry["path"] == rel and int(entry["report"]["rows"]) == rows
    f = ROOT / rel
    assert f.exists(), f"thiếu {rel} — chạy `git lfs pull`"
    if _is_lfs_pointer(f):
        pytest.skip(f"{rel} mới là LFS pointer (chưa `git lfs pull`) — bỏ qua kiểm sha256 nội dung")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha == entry["sha256"], f"{rel}: sha256 {sha} ≠ anchor {entry['sha256']}"
    assert int(entry["bytes"]) == f.stat().st_size


def test_lf_derivation_sidecar_matches_committed_hf():
    side = json.loads((ROOT / "data" / "BTC_5m_2y.derivation.json").read_text(encoding="utf-8"))
    files = json.loads(ANCHOR.read_text(encoding="utf-8"))["files"]
    assert side["source_sha256"] == files["hf"]["sha256"]  # LF trong repo dẫn xuất từ ĐÚNG HF trong repo
    assert side["lf_sha256"] == files["lf"]["sha256"] and int(side["rows_lf"]) == int(files["lf"]["report"]["rows"])


# ----------------------------------------------------------------------------- (5) experiments/** không bị ignore
EXPERIMENT_PATHS = [
    "experiments/full/log.csv", "experiments/full/calib/lgbm_base.json", "experiments/full/runs/exp1/run.json",
    "experiments/full/runs/exp1/pred_val.npz", "experiments/full/s0/lgbm.json", "experiments/full/s0/candidates_lgbm.json",
    "experiments/full/wins/lgbm.json", "experiments/full/wins/lgbm_seed0.npz", "experiments/full/wins/tfm.json",
    "experiments/full/wins/tfm_lora_baseline.json", "experiments/full/wins/tfm_lora_xreg.json",
    "experiments/full/lora/tfm_lora_fold1_seed8587_es.pt", "experiments/full/lora/tfm_lora_fold1_seed8587_es.json",
    "experiments/full/final/index.json", "experiments/full/final/TEST_SENTINEL.json", "experiments/full/final/lgbm.npz",
    "experiments/full/summary/all_models_test.csv", "experiments/full/summary/latency_summary.csv",
    "experiments/full/summary/fig_final_heatmaps.png", "experiments/full/scheduler_log.jsonl",
    "experiments/full/orchestrate_log.jsonl", "experiments/full/checker_log.jsonl", "experiments/full/champion_log.csv",
    "experiments/full/champion.json", "experiments/full/champion_replay.csv", "experiments/full/champion_replay.json",
    "experiments/full/gpu_probe.json", "experiments/full/keepdrop_lgbm.csv", "experiments/full/prune_pi_lgbm.csv",
    "experiments/full/autots_templates/best_FINAL.json", "experiments/full/cache/x.npz", "experiments/15d/wins/lgbm.json",
]


def test_no_canonical_experiment_artifact_is_gitignored():
    r = _git("check-ignore", "-v", "--no-index", *EXPERIMENT_PATHS)
    assert r.returncode == 1 and not r.stdout.strip(), f"artifact thí nghiệm bị ignore:\n{r.stdout}"


def test_experiment_binaries_are_lfs_tracked():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pat in ("experiments/**/*.npz", "experiments/**/*.pt", "experiments/**/*.png"):
        assert f"{pat} filter=lfs" in attrs


# ----------------------------------------------------------------------------- (31)–(35) agent pha vận hành
AGENTS = ROOT / ".claude" / "agents"


def _front(name: str) -> str:
    return (AGENTS / f"{name}.md").read_text(encoding="utf-8")


def test_run_monitor_agent_exists_and_is_read_only():
    """(33) run-monitor không có tool sửa file và tự cấm đổi methodology/kết quả."""
    txt = _front("run-monitor")
    tools = re.search(r"^tools: \[(.+)\]$", txt, re.M).group(1)
    assert "Edit" not in tools and "Write" not in tools and "NotebookEdit" not in tools
    assert set(t.strip() for t in tools.split(",")) <= {"Read", "Grep", "Glob", "Bash"}
    low = txt.lower()
    for phrase in ("không sửa code", "scheduler_log.jsonl", "orchestrate_log.jsonl", "nvidia-smi"):
        assert phrase in low or phrase in txt
    assert "keep/drop" in low and "champion" in low  # nói rõ không tự quyết


def test_analyst_is_post_run_only():
    txt = _front("analyst")
    assert "POST-RUN ONLY" in txt and "run-monitor" in txt


def test_researcher_is_dormant_outside_normal_flow():
    txt = _front("researcher")
    assert "DORMANT" in txt and "KHÔNG nằm trong đường chạy bình thường" in txt


def test_checker_runs_before_orchestrate_and_before_final():
    txt = _front("checker")
    assert "trước `orchestrate`" in txt and "trước `final`" in txt


def test_normal_flow_docs_do_not_call_researcher():
    """(31)(34) đường chạy vận hành trong AGENT.md: có checker/run-monitor, researcher đứng ngoài."""
    reg = (ROOT / ".claude" / "AGENT.md").read_text(encoding="utf-8")
    flow = reg[reg.index("đường chạy VẬN HÀNH"):reg.index("```", reg.index("đường chạy VẬN HÀNH"))]
    assert "run-monitor" in flow and "checker" in flow and "orchestrate" in flow and "champion-replay" in flow
    assert "researcher: KHÔNG nằm trong đường này" in flow
    assert "POST-RUN ONLY" in reg or "POST-RUN" in reg


def test_gpu_resource_failure_is_the_only_user_prompt_exception():
    """(35) tài liệu chính thức phải nói rõ: chỉ sự cố tài nguyên GPU mới được hỏi user."""
    for path in (ROOT / ".claude" / "CLAUDE.md", ROOT / ".claude" / "AGENT.md", AGENTS / "checker.md"):
        txt = path.read_text(encoding="utf-8")
        assert "DUY NHẤT" in txt and "GPU" in txt, path
        assert "USER_DECISION_REQUIRED" in txt or "HỎI USER" in txt.upper(), path
    claude = (ROOT / ".claude" / "CLAUDE.md").read_text(encoding="utf-8")
    assert "KHÔNG CPU fallback" in claude and "không có tuỳ chọn" in claude.lower()
