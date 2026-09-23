"""WF3 samples, frozen class labels, and GPU-resident window batching.

Everything that decides WHICH samples exist and WHAT the model sees is the WF3 code:
src/data/dataset.prepare_data loads and repairs the file, cuts the walk-forward fold,
applies purge + embargo, selects valid origins (history validity, gap rule, target
lookup, tolerance), builds the model's features and fits the train-only standardizer.

This module adds only:
  * delta_h = mid[target_h] - mid[origin], in USD, from the same origin/target indices
    (snapped to 1e-6 USD, src/cls/labels.displacement);
  * the frozen class of each delta (src/cls/labels.py), never refitted here;
  * batching: the standardized feature matrix lives on the GPU and a batch of windows
    is gathered there, x[origin-history+1 : origin+1], which is exactly what
    LOBDataset.__getitem__ returns. The training order is torch.randperm(n) from a
    Generator seeded like the WF3 trainer's RandomSampler, so the permutation is the
    one the WF3 DataLoader would have drawn.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from src.config import Config, DataConfig, TrainConfig
from src.data.dataset import prepare_data
from . import HORIZONS
from .labels import displacement

# Feature set per architecture: OFI-LSTM keeps its 20 order-flow channels, ModernTCN and
# the Transformer read the 40 raw L10 price/quantity columns; all use the WF3 train-only
# z-score (src/data/preprocessing.GLOBAL_STANDARDIZER_MODELS).
FEATURE_MODEL = dict(ofi_lstm="ofi_lstm", moderntcn="moderntcn", transformer="transformer")


def wf3_config(arch, data_fields, csv_path):
    data = DataConfig(**data_fields)
    data.csv_path = csv_path
    return Config(model=FEATURE_MODEL[arch], data=data, training=TrainConfig(), model_kwargs={})


@dataclass
class Split:
    name: str
    origins: np.ndarray         # [n] int64 raw row of each origin (WF3 LOBDataset.origins)
    targets: np.ndarray         # [n, 3] int64 raw row of each 60/120/180 s target
    delta: np.ndarray           # [n, H] float64 USD displacement for the run's horizons
    labels: np.ndarray          # [n, H] int64 frozen class
    origin_sha256: str

    def __len__(self):
        return len(self.origins)


def fingerprint(origins, targets):
    return hashlib.sha256(np.ascontiguousarray(origins).tobytes()
                          + np.ascontiguousarray(targets).tobytes()).hexdigest()


class ClassificationData:
    def __init__(self, arch, data_fields, csv_path, label_set, horizons, device):
        self.config = wf3_config(arch, data_fields, csv_path)
        prepared = prepare_data(self.config)
        self.prepared = prepared
        self.metadata = prepared.metadata
        self.history_rows = prepared.history_rows
        self.horizons = [int(h) for h in horizons]
        self.columns = [HORIZONS.index(h) for h in self.horizons]
        self.label_set = label_set
        raw = prepared.raw
        self.timestamps, self.mid = raw.timestamps, raw.mid
        self.splits = {}
        for name, ds in prepared.datasets.items():
            targets = ds.target_indices
            delta = displacement(raw.mid[targets[:, self.columns]], raw.mid[ds.origins][:, None])
            labels = np.stack([label_set[h].assign(delta[:, j]) for j, h in enumerate(self.horizons)], 1)
            self.splits[name] = Split(name, ds.origins, targets, delta, labels,
                                      hashlib.sha256(ds.origins.tobytes()).hexdigest())
        x = prepared.datasets["train"].x  # one standardized matrix shared by every split
        self.channels = int(np.prod(x.shape[1:]))
        self.feature_shape = list(x.shape[1:])
        self.device = device
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self.offsets = torch.arange(-(self.history_rows-1), 1, device=device)
        self._device_splits = {}
        # The book tensor is no longer needed once features exist; keep mids and stamps.
        raw.book = None

    def n_classes(self):
        return [self.label_set[h].n_classes for h in self.horizons]

    def device_split(self, name):
        if name not in self._device_splits:
            s = self.splits[name]
            self._device_splits[name] = dict(
                origins=torch.from_numpy(s.origins).to(self.device),
                labels=torch.from_numpy(s.labels).to(self.device),
                delta=torch.from_numpy(s.delta).to(self.device))
        return self._device_splits[name]

    def windows(self, origins):
        """[B] origin rows -> [B, history, features...], identical to LOBDataset.__getitem__."""
        return self.x[origins[:, None]+self.offsets]

    def batches(self, name, batch_size, order=None):
        """Yield (x, labels, delta, index) in `order` (a CPU LongTensor) or sequentially."""
        split = self.device_split(name)
        n = len(self.splits[name])
        order = (torch.arange(n) if order is None else order).to(self.device)
        for start in range(0, n, batch_size):
            index = order[start:start+batch_size]
            origins = split["origins"][index]
            yield self.windows(origins), split["labels"][index], split["delta"][index], index

    def summary(self):
        return {name: dict(samples=len(s), origin_index_sha256=s.origin_sha256,
                           sample_fingerprint=fingerprint(s.origins, s.targets))
                for name, s in self.splits.items()}
