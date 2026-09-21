"""Central, JSON-serializable experiment configuration. No tuned parameters."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

MODELS = ("e0", "ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")


@dataclass
class DataConfig:
    csv_path: str = "BTCUSDT_L10_oct2023.csv"
    timestamp_column: str = "timestamp_ms"
    timestamp_unit: str = "ms"
    segment_column: str | None = "segment_id"
    # Canonical name -> actual CSV name; unspecified fields use canonical names.
    column_mapping: dict = field(default_factory=dict)
    history_seconds: int = 60
    history_rows: int | None = None  # Explicit paper-style override, e.g. LiT 64.
    # Opt-in, recorded repairs for a file whose rows are not already in time order.
    # Defaults keep the original guarantee: never reorder or deduplicate silently.
    sort_by_timestamp: bool = False
    duplicate_timestamp_policy: str = "error"  # error | keep_first | keep_last
    stride_seconds: float | None = None
    max_gap_seconds: float = 2.0
    target_tolerance_seconds: float = 2.0
    horizons_seconds: tuple = (60, 120, 180)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    # Walk-forward: the last test_fraction of elapsed time is held out, the rest is cut
    # into folds+1 equal blocks, and fold k trains on blocks [0,k) and validates on block k.
    split_scheme: str = "chronological"  # chronological | walk_forward
    folds: int = 3
    fold: int = 1                        # 1-based
    test_fraction: float = 0.15
    # No sample may touch a record within embargo_seconds of the next split's first record.
    embargo_seconds: float = 0.0
    train_end: str | None = None  # ISO-8601, exclusive UTC boundary.
    validation_end: str | None = None
    of_representation: str = "of"

    def validate(self):
        # The window is resolved from the measured cadence, so the second count is
        # only required to be positive and to cover at least two snapshots.
        if self.history_seconds < 1:
            raise ValueError("history_seconds must be a positive number of seconds")
        if self.duplicate_timestamp_policy not in ("error", "keep_first", "keep_last"):
            raise ValueError("duplicate_timestamp_policy must be error, keep_first or keep_last")
        if self.split_scheme not in ("chronological", "walk_forward"):
            raise ValueError("split_scheme must be chronological or walk_forward")
        if self.split_scheme == "walk_forward":
            if self.folds < 2 or not 1 <= self.fold <= self.folds:
                raise ValueError("walk_forward needs folds >= 2 and 1 <= fold <= folds")
            if not 0 < self.test_fraction < 1:
                raise ValueError("test_fraction must be in (0, 1)")
        if self.embargo_seconds < 0:
            raise ValueError("embargo_seconds must be nonnegative")
        if self.history_rows is not None and self.history_rows < 2:
            raise ValueError("history_rows must be >= 2")
        if self.stride_seconds is not None and self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")
        if self.max_gap_seconds <= 0 or self.target_tolerance_seconds < 0:
            raise ValueError("Invalid gap/tolerance")
        if tuple(self.horizons_seconds) != (60, 120, 180):
            raise ValueError("This task has exactly the 60/120/180-second horizons")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1-self.train_fraction:
            raise ValueError("Invalid chronological split fractions")
        if bool(self.train_end) != bool(self.validation_end):
            raise ValueError("Supply both timestamp split boundaries")
        if self.of_representation not in ("of", "ofi"):
            raise ValueError("of_representation must be of or ofi")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    prefetch_factor: int = 2
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    precision: str = "auto"
    device: str = "auto"
    compile: bool = False
    auto_batch_size: bool = False
    probe_max_batch_size: int = 4096
    vram_headroom_gb: float = 2.0
    seed: int = 42
    loss: str = "mse"
    quantile: float = 0.5
    run_name: str = "base"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        if min(self.epochs, self.batch_size, self.gradient_accumulation, self.prefetch_factor) < 1:
            raise ValueError("Epoch, batch, accumulation, prefetch must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid training parameters")
        if self.loss not in ("mse", "quantile") or not 0 < self.quantile < 1:
            raise ValueError("Invalid loss/quantile")
        if self.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError("Unknown precision")
        if self.vram_headroom_gb < 1.5 or self.probe_max_batch_size < 128:
            raise ValueError("Probe requires >=1.5 GiB headroom and max batch >=128")
        if Path(self.run_name).name != self.run_name or self.run_name in ("", ".", ".."):
            raise ValueError("run_name must be a single directory name")


@dataclass
class Config:
    model: str = "ofi_lstm"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    model_kwargs: dict = field(default_factory=dict)

    def validate(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        self.data.validate()
        self.training.validate()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        return cls(model=obj.get("model", "ofi_lstm"),
                   data=DataConfig(**obj.get("data", {})),
                   training=TrainConfig(**obj.get("training", {})),
                   model_kwargs=obj.get("model_kwargs", {}))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
