"""RunConfig: mọi tham số của một giai đoạn (dataset, fold, seed, candidate, model) đọc từ JSON."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HORIZONS = (1, 2, 3)
HMAX_SEC = 3 * 60  # target xa nhất: t + 3 phút phải nằm trong partition (§0 quy tắc biên)
STEP_SEC = 60


def config_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


@dataclass
class RunConfig:
    dataset_label: str
    hf_csv: str
    lf_csv: str | None
    val_days: list[str] | None = None  # split 15 ngày (§1.2): VAL = các ngày UTC; None khi dùng `split` rolling
    test_start: str | None = None
    test_end: str | None = None
    split: dict[str, Any] | None = None  # data đầy đủ (§5): {"mode": "rolling_from_end", n_folds, val_days, fit_days, es_days, test_days}
    checksums: str = "data/data_checksums.json"  # anchor §6.1 của snapshot mà config này dùng
    prev_run_dir: str | None = None  # vòng trước (wins/, b0_star.json, keepdrop_*) → S0_m khoá (2026-09-03); None = từ B0*
    fold_workers: int = 1  # §9: số process chạy song song 5 fold (env P0_FOLD_WORKERS ưu tiên); 1 = tuần tự
    gpu_devices: list[int] | None = None  # §9 (2026-09-04c): GPU vật lý làm worker, vd [0, 1] — ĐỐI XỨNG, không gán vai trò ML/DL
    gpu_slots_per_device: int = 1  # số task nặng đồng thời trên MỖI GPU (mặc định 1 — không oversubscribe VRAM)
    max_branches: int | None = None  # số nhánh model chạy đồng thời trong `orchestrate` (None = số worker GPU)
    defer_champion: bool = False  # §14: nhánh chỉ sinh artifact đại diện; champion so lại ở `champion-replay` (thứ tự cố định)
    short_candidates: list[str] | None = None  # giới hạn pool C_short (None = toàn bộ `features_short.SHORT_COLUMNS`)
    es_hours: int = 23
    purge_minutes: int = 60
    calib_seed: int = 8586  # seed0 — CHỈ dùng cho run ES tìm số vòng cố định (§1.3); không dùng để đo ε, không dùng để selection
    eval_seeds: tuple[int, ...] = (8587, 8588, 8589)  # seed1/2/3 — đo ε (§1.3) và confirmation 3 seed (§2.1b)
    selection_seed: int | None = None  # MỘT seed cố định cho MỌI bước selection (R1–R4, baseline + 39 candidate, prune PI); None → eval_seeds[0]
    eps_floor_pp: float = 0.005
    experiments_dir: str = "experiments"
    candidates: list[str] = field(default_factory=list)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_order: list[str] = field(default_factory=lambda: ["lgbm", "xgb", "cat", "tfm", "xgbrf", "autots_wr", "autots_mr", "lstm"])
    require_gpu: bool = True
    root: str = "."

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        d["eval_seeds"] = tuple(int(s) for s in d.get("eval_seeds", (8587, 8588, 8589)))
        d.setdefault("root", str(path.resolve().parent.parent))
        cfg = cls(**d)
        if cfg.split is None and (not cfg.val_days or not cfg.test_start):
            raise ValueError(f"{path}: cần `val_days` + `test_start` (split 15 ngày) hoặc `split` (rolling_from_end)")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval_seeds"] = list(self.eval_seeds)
        return d

    @property
    def sel_seed(self) -> int:
        """Seed dùng cho mọi bước selection (một giá trị duy nhất → chênh lệch RMSE chỉ do feature set)."""
        return int(self.selection_seed if self.selection_seed is not None else self.eval_seeds[0])

    def hash(self) -> str:
        """Hash cấu hình KHÔNG gồm đường dẫn máy (root, experiments_dir) → cùng config cho cùng hash ở local và Vast."""
        d = self.to_dict()
        # Chỉ-thực-thi (không đổi một con số nào của kết quả): số worker/GPU, số nhánh song song, hoãn champion (replay
        # cho ra đúng chuỗi so sánh của luật §3). Loại khỏi hash để cùng một thí nghiệm có cùng config_hash trên mọi máy.
        for k in ("root", "experiments_dir", "fold_workers", "gpu_devices", "gpu_slots_per_device", "max_branches", "defer_champion"):
            d.pop(k, None)
        return config_hash(d)

    def path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        return q if q.is_absolute() else Path(self.root) / q

    @property
    def exp_dir(self) -> Path:
        return self.path(self.experiments_dir)  # type: ignore[return-value]

    def model_params(self, name: str) -> dict[str, Any]:
        return dict(self.models.get(name, {}))
