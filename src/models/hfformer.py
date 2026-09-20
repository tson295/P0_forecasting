"""Behavioral adaptation of Nakols/HFformerV2's 7_HFformerv2.ipynb.

See THIRD_PARTY.md for exact source and intentional corrections. The spiking
library expects [batch,time,feature], unlike the original notebook's T,B,F.
"""
import torch
from torch import nn
from pytorch_spiking import SpikingActivation
from .common import ForecastModel, EncoderLayer


class WindowLocalZScore(nn.Module):
    """Per-sample, per-feature z-score across the history dimension.

    Frozen contract: never a corpus statistic, unbiased=False, eps=1e-5, and the
    result depends only on this sample's own history. The module therefore holds
    no parameters and no buffers.

    The reduction accumulates in FP64 before the FP32 result. That is the same
    formula, evaluated without reduction error: at raw L10 price scale (~2.7e4,
    FP32 ulp ~2e-3) a batched FP32 mean is off by a few ulp, and dividing that
    residue by eps=1e-5 turns a constant window -- whose true z-score is 0 --
    into ~600. Worse, the FP32 reduction kernel varies with tensor shape, so the
    value would depend on the batch rather than on the sample alone.
    """
    eps = 1e-5

    def forward(self, x):
        x_fp32 = x.float()
        window = x_fp32.double()
        mean = window.mean(dim=1, keepdim=True)
        std = window.std(dim=1, keepdim=True, unbiased=False)
        return ((window-mean)/(std+self.eps)).float()


class StableSpikingPReLU(nn.Module):
    def __init__(self, dt):
        super().__init__()
        # Fixed initial voltage prevents batch-dependent random predictions.
        self.spiking = SpikingActivation(nn.PReLU(), dt=dt,
                                         spiking_aware_training=True)

    def forward(self, x):
        # The upstream spike accumulator benefits from FP32 even under AMP.
        phase = (torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)+0.5)/x.shape[-1]
        self.spiking.initial_state = phase.expand(x.shape[0], -1).clone()
        with torch.autocast(device_type=x.device.type, enabled=False):
            # The upstream custom autograd omits activation-parameter gradients.
            # Preserve its spike forward and analog surrogate, including PReLU weight.
            rates = self.spiking.activation(x.float())
            with torch.no_grad():
                spikes = self.spiking(x.float())
            result = rates+(spikes-rates).detach()
        return result.to(x.dtype)


class HFformer(ForecastModel):
    def __init__(self, config):
        super().__init__(config)
        c = config
        # Raw feature scale in, window-local z-score here: never a saved standardizer.
        self.normalization = WindowLocalZScore()
        self.input_projection = nn.Linear(c["channels"], c["d_model"])
        self.encoder = nn.Sequential(*[
            EncoderLayer(c["d_model"], c["heads"], c["ffn_dim"], c["dropout"],
                         StableSpikingPReLU(c["spike_dt"]), causal=c["causal"])
            for _ in range(c["layers"])])
        self.encoder_norm = nn.LayerNorm(c["d_model"])
        self.head = HFHead(c["d_model"], c["history_rows"])

    def forward(self, x):
        x = self.normalization(x)
        return self.head(self.encoder_norm(self.encoder(self.input_projection(x))))


class HFHead(nn.Module):
    def __init__(self, dim, history):
        super().__init__()
        self.feature_projection = nn.Linear(dim, 1)
        self.activation = nn.PReLU()
        self.time_projection = nn.Linear(history, 3)

    def forward(self, x):
        x = self.feature_projection(x).transpose(1, 2)
        return self.time_projection(self.activation(x)).squeeze(1)
