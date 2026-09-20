"""Behavioral adaptation of Nakols/HFformerV2's 7_HFformerv2.ipynb.

See THIRD_PARTY.md for exact source and intentional corrections. The spiking
library expects [batch,time,feature], unlike the original notebook's T,B,F.
"""
import torch
from torch import nn
from pytorch_spiking import SpikingActivation
from .common import ForecastModel, EncoderLayer


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
        self.input_projection = nn.Linear(c["channels"], c["d_model"])
        self.encoder = nn.Sequential(*[
            EncoderLayer(c["d_model"], c["heads"], c["ffn_dim"], c["dropout"],
                         StableSpikingPReLU(c["spike_dt"]), causal=c["causal"])
            for _ in range(c["layers"])])
        self.encoder_norm = nn.LayerNorm(c["d_model"])
        self.head = HFHead(c["d_model"], c["history_rows"])

    def forward(self, x):
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
