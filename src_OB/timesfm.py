"""Native zero-shot and independent horizon-specific LoRA; no XReg feature search."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .neural import fit_scaler
from .latency import infer

REPO = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def backbone(cfg):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO, revision=REVISION, torch_compile=False)
    model.model.to("cuda")
    model.compile(timesfm.ForecastConfig(max_context=cfg["tfm"]["context"], max_horizon=128,
                                        per_core_batch_size=cfg["tfm"]["batch_size"],
                                        normalize_inputs=True, infer_is_positive=False,
                                        force_flip_invariance=True, return_backcast=False))
    return model


def contexts(data, ids, horizon, length):
    # Physical time spacing h: native one-step forecast is precisely t+h, even after dedup.
    log_prices, valid = data.price_context(ids, horizon, length)
    if not valid.all():
        raise ValueError("TimesFM context thiếu quote; không điền future/backfill.")
    # Subtract in float64 before conversion: retain small BTC returns accurately.
    return (log_prices - log_prices[:, -1:]).astype(np.float32)


class DifferentiableForecast:
    """Reuse the already implemented differentiable mean decode, without legacy training/search."""
    def __init__(self, wrapper):
        self._injected = None
        self.normalize_inputs = True
        self.flip = True
        self.use_mean_head = True
        self.wrapper = wrapper

    def _wrappers(self):
        return {"module": self.wrapper.model}

    def __call__(self, x):
        from src.p0.models_tfm import TimesFMLoRAModel
        # Existing decoder returns 3 values; use only first for this horizon's model.
        return TimesFMLoRAModel.train_forward(self, x)[:, 0]


def run(name, cfg, data, train_ids, val_ids, horizon, out):
    from src.p0.lora import inject_lora, freeze_except_lora, lora_state_dict

    torch.manual_seed(cfg["seed"])
    p = cfg["tfm"]
    wrapper = backbone(cfg)  # pristine instance for each (fold, horizon, variant)
    mean = std = encoder = None
    if name == "tfm_lora":
        inject_lora(wrapper.model, ("attn.qkv_proj", "attn.out", "ff0", "ff1"),
                    r=p["rank"], alpha=p["alpha"], dropout=0.)
        freeze_except_lora(wrapper.model)
        mean, std = fit_scaler(data, train_ids)
        # Small causal OF + elapsed-time residual head, fit jointly with LoRA.
        # Zero-shot remains the genuine pretrained univariate baseline.
        encoder = nn.Sequential(nn.Linear(data.width, 32), nn.GELU(), nn.Linear(32, 1)).cuda()
        nn.init.zeros_(encoder[-1].weight)
        nn.init.zeros_(encoder[-1].bias)
        decode = DifferentiableForecast(wrapper)
        params = [v for v in wrapper.model.parameters() if v.requires_grad] + list(encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=p["lr"])
        target, _ = data.target(train_ids, horizon)
        target_scale = max(float(target.std()), 1e-8)
        rng = np.random.default_rng(cfg["seed"])
        for epoch in range(p["epochs"]):
            wrapper.model.train()
            encoder.train()
            order = rng.permutation(len(train_ids))
            total = 0.
            for s in range(0, len(order), p["batch_size"]):
                pick = order[s:s + p["batch_size"]]
                ids = train_ids[pick]
                x = torch.as_tensor(contexts(data, ids, horizon, p["context"]), device="cuda")
                f = torch.as_tensor((data.features[ids] - mean) / std, device="cuda")
                predicted_return = decode(x) - x[:, -1] + encoder(f).squeeze(-1) * target_scale
                y = torch.as_tensor(target[pick], dtype=torch.float32, device="cuda")
                loss = ((predicted_return - y) / target_scale).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                total += loss.item() * len(pick)
            print(f"tfm_lora h={horizon}s epoch={epoch + 1} train_loss={total / len(train_ids):.6g}", flush=True)
        torch.save({"lora": lora_state_dict(wrapper.model), "of_head": encoder.state_dict(),
                    "feature_mean": torch.from_numpy(mean), "feature_std": torch.from_numpy(std),
                    "target_scale": target_scale, "repo": REPO, "revision": REVISION,
                    "horizon_seconds": horizon, "config": cfg}, out / "adapter.pt")
        encoder.eval()
    wrapper.model.eval()
    def predict(ids):
        x = contexts(data, ids, horizon, p["context"])
        _, quantile = wrapper.forecast(horizon=1, inputs=list(x.copy()))
        delta = np.asarray(quantile)[:len(ids), -1, 0].astype(np.float64)
        if encoder is not None:
            f = torch.as_tensor((data.features[ids] - mean) / std, device="cuda")
            delta += encoder(f).squeeze(-1).cpu().numpy() * target_scale
        return delta
    with torch.inference_mode():
        return infer(cfg, val_ids, out, predict)
