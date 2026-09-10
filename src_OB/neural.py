"""Existing plain LSTM, trained separately for each horizon on OF/OFI."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .latency import infer


class LSTM(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.recurrent = nn.LSTM(width, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.recurrent(x)[0][:, -1]).squeeze(-1)


def fit_scaler(data, ids):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Only feature rows preceding training origins, not validation or future rows.
    first = max(0, int(ids[0]) - data.cfg["context"] + 1)
    last = int(ids[-1]) + 1
    for s in range(first, last, 100000):
        scaler.partial_fit(data.features[s:min(last, s + 100000)])
    return scaler.mean_.astype(np.float32), np.maximum(scaler.scale_, 1e-6).astype(np.float32)


def run(name, cfg, data, train_ids, val_ids, horizon, out):
    torch.manual_seed(cfg["seed"])
    p = cfg["neural"]
    mean, std = fit_scaler(data, train_ids)
    target, _ = data.target(train_ids, horizon)
    target_scale = max(float(target.std()), 1e-8)
    net = LSTM(data.width, p["hidden"]).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    rng = np.random.default_rng(cfg["seed"])

    def batch(ids):
        x = (data.windows(ids) - mean) / std
        return torch.as_tensor(x, dtype=torch.float32, device="cuda")

    for epoch in range(p["epochs"]):
        net.train()
        order = rng.permutation(len(train_ids))
        total = 0.
        for s in range(0, len(order), p["batch_size"]):
            pick = order[s:s + p["batch_size"]]
            y = torch.as_tensor(target[pick] / target_scale, dtype=torch.float32, device="cuda")
            loss = nn.functional.huber_loss(net(batch(train_ids[pick])), y, delta=0.9)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            total += loss.item() * len(pick)
        print(f"{name} h={horizon}s epoch={epoch + 1} train_loss={total / len(train_ids):.6g}", flush=True)
    torch.save({"state_dict": net.state_dict(), "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std), "target_scale": target_scale,
                "model": name, "horizon_seconds": horizon, "config": cfg}, out / "model.pt")
    net.eval()
    def predict(ids):
        return net(batch(ids)).cpu().numpy() * target_scale
    with torch.inference_mode():
        return infer(cfg, val_ids, out, predict)
