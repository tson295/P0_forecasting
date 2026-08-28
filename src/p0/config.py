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
    val_days: list[str]
    test_start: str
    test_end: str | None = None
    es_hours: int = 23
    purge_minutes: int = 60
    seeds: tuple[int, ...] = (8586, 8587, 8588)
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
        d["seeds"] = tuple(int(s) for s in d.get("seeds", (8586, 8587, 8588)))
        d.setdefault("root", str(path.resolve().parent.parent))
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d

    def hash(self) -> str:
        """Hash cấu hình KHÔNG gồm đường dẫn máy (root, experiments_dir) → cùng config cho cùng hash ở local và Vast."""
        d = self.to_dict()
        for k in ("root", "experiments_dir"):
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
