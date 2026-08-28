"""LSTM-DMH (§2.2 #7): context 512 phút, input mỗi phút = fine feature B0 còn trong B0* (+ rv60) + ext đang KEEP;
1 lớp LSTM hidden 64; head linear 3 output (ŷ_1..3 trong z-space); Huber(delta 0.9) trên z; Adam lr 1e-3; batch 256;
≤ 50 epoch; ES patience 5 trên ES set (run calibrate/confirmation) hoặc số epoch cố định (`fixed_epoch_LSTM`, §1.3).
Training chỉ GPU; `allow_cpu=True` chỉ cho unit test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FitResult


@dataclass
class SeqBatch:
    """Ma trận feature theo phút (n_grid, d) + chỉ số origin; cửa sổ [i−L+1, i] lấy khi tạo batch (không copy trước)."""

    feats: np.ndarray  # (n_grid, d) float32, đã chuẩn hoá train-only, NaN → 0
    idx: np.ndarray  # chỉ số origin trên lưới
    perm: dict[int, np.ndarray] | None = None  # PI (§1.4/§2.1a): kênh j lấy cửa sổ của origin perm[j][k] thay cho idx[k]

    def slice(self, a: int, b: int) -> "SeqBatch":
        return SeqBatch(self.feats, self.idx[a:b], None if not self.perm else {j: np.asarray(v)[a:b] for j, v in self.perm.items()})

    def with_perm(self, perm: dict[int, np.ndarray]) -> "SeqBatch":
        """Bản sao chỉ khác `perm` (không copy feats/idx) — dùng cho permutation importance."""
        return SeqBatch(self.feats, self.idx, perm)


class LSTMModel:
    name = "lstm"
    lib = "torch"
    supports_rounds = True  # "rounds" = số epoch cố định (fixed_epoch_LSTM)

    def __init__(self, device: str = "cuda", allow_cpu: bool = False, context: int = 512, hidden: int = 64, lr: float = 1e-3,
                 batch_size: int = 256, max_epochs: int = 50, patience: int = 5, huber_delta: float = 0.9):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            if not allow_cpu:
                raise RuntimeError("LSTM: không có CUDA — training CPU bị cấm (plan §0).")
            device = "cpu"
        if device != "cuda" and not allow_cpu:
            raise RuntimeError("LSTM: training trên CPU bị cấm (plan §0). allow_cpu=True chỉ cho unit test.")
        self.device, self.context, self.hidden, self.lr = device, context, hidden, lr
        self.batch_size, self.max_epochs, self.patience, self.delta = batch_size, max_epochs, patience, huber_delta
        self.train_device = self.predict_device = "GPU" if device == "cuda" else "CPU"

    def _windows(self, feats_t, idx: np.ndarray, perm: dict[int, np.ndarray] | None = None):
        import torch

        L = self.context
        i = torch.as_tensor(np.asarray(idx), device=feats_t.device)
        offs = torch.arange(-L + 1, 1, device=feats_t.device)
        gather = (i[:, None] + offs[None, :]).clamp_min(0)
        x = feats_t[gather]  # (b, L, d)
        if perm:  # PI: thay TOÀN BỘ cửa sổ của kênh j bằng cửa sổ của origin khác (đúng logic xáo cột giữa các origin)
            for j, alt in perm.items():
                a = torch.as_tensor(np.asarray(alt), device=feats_t.device)
                g = (a[:, None] + offs[None, :]).clamp_min(0)
                x[:, :, j] = feats_t[g][:, :, j]
        return x

    def fit_predict(self, seq_fit: SeqBatch, z_fit: np.ndarray, seq_es: SeqBatch, z_es: np.ndarray, seq_pred: SeqBatch,
                    rounds, seed: int) -> FitResult:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device(self.device)
        feats_t = torch.as_tensor(seq_fit.feats, dtype=torch.float32, device=dev)
        d = feats_t.shape[1]

        class Net(nn.Module):
            def __init__(self, d_in, hid):
                super().__init__()
                self.lstm = nn.LSTM(d_in, hid, batch_first=True)
                self.head = nn.Linear(hid, 3)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = Net(d, self.hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.HuberLoss(delta=self.delta)
        zf = torch.as_tensor(z_fit, dtype=torch.float32, device=dev)
        ze = torch.as_tensor(z_es, dtype=torch.float32, device=dev)
        n_epochs = int(rounds[0]) if rounds is not None else self.max_epochs
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def eval_loss(seq: SeqBatch, z_t) -> float:
            net.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(seq.idx), self.batch_size):
                    b = seq.idx[s:s + self.batch_size]
                    out = net(self._windows(feats_t, b))
                    tot += float(loss_fn(out, z_t[s:s + self.batch_size]).item()) * len(b)
                    cnt += len(b)
            return tot / max(cnt, 1)

        best_state, best_loss, best_epoch, bad = None, float("inf"), 0, 0
        for epoch in range(1, n_epochs + 1):
            net.train()
            perm = torch.randperm(len(seq_fit.idx), generator=gen).numpy()
            for s in range(0, len(perm), self.batch_size):
                sel = perm[s:s + self.batch_size]
                out = net(self._windows(feats_t, seq_fit.idx[sel]))
                loss = loss_fn(out, zf[sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if rounds is None:  # early stopping theo epoch trên ES set
                cur = eval_loss(seq_es, ze)
                if cur < best_loss - 1e-9:
                    best_loss, best_epoch, bad = cur, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                best_epoch = epoch
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        def predict(seq: SeqBatch) -> np.ndarray:
            outs = []
            with torch.no_grad():
                for s in range(0, len(seq.idx), self.batch_size):
                    sub = seq.slice(s, s + self.batch_size)
                    outs.append(net(self._windows(feats_t, sub.idx, sub.perm)).cpu().numpy())
            return np.concatenate(outs).astype(np.float32) if outs else np.zeros((0, 3), np.float32)

        preds = predict(seq_pred)
        self._net, self._feats_t = net, feats_t

        def predictor(seq_or_idx):
            seq = seq_or_idx if isinstance(seq_or_idx, SeqBatch) else SeqBatch(seq_fit.feats, np.asarray(seq_or_idx))
            return predict(seq)

        return FitResult(preds, (best_epoch, best_epoch, best_epoch), [predictor])
