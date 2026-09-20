"""LiT L10 adaptation: explicit side/depth/price-volume input, structured patches."""
import math
import torch
from torch import nn
from .common import ForecastModel, EncoderLayer


class LiT(ForecastModel):
    def __init__(self, config):
        super().__init__(config)
        c = config
        self.temporal_patch = c["temporal_patch"]
        self.patch_count = math.ceil(c["history_rows"]/self.temporal_patch)
        # Each patch spans all ten levels of ONE side and a short time interval.
        self.patch_projection = nn.Linear(self.temporal_patch*10*2, c["d_model"]-c["position_dim"])
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_count*2, c["position_dim"]))
        nn.init.normal_(self.position_embedding, std=.02)
        self.encoder = nn.Sequential(*[
            EncoderLayer(c["d_model"], c["heads"], c["ffn_dim"], c["dropout"])
            for _ in range(c["layers"])])
        # Concatenate bid/ask tokens at the SAME time before the temporal LSTM.
        self.lstm = nn.LSTM(2*c["d_model"], c["lstm_hidden"], c["lstm_layers"],
                            batch_first=True, dropout=c["dropout"] if c["lstm_layers"] > 1 else 0.)
        self.head = nn.Linear(c["lstm_hidden"], 3)

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[2:]) != (2, 10, 2):
            raise ValueError("LiT requires [batch,time,side=2,depth=10,field=2]")
        b, t = x.shape[:2]
        pad = (-t) % self.temporal_patch
        if pad:
            x = torch.cat([x[:, :1].expand(-1, pad, -1, -1, -1), x], dim=1)
        # Time-major tokens: bid(t0), ask(t0), bid(t1), ask(t1), ...
        x = x.reshape(b, self.patch_count, self.temporal_patch, 2, 10, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, self.patch_count*2, -1)
        x = self.patch_projection(x)
        x = torch.cat([x, self.position_embedding.expand(b, -1, -1)], dim=-1)
        x = self.encoder(x).reshape(b, self.patch_count, -1)
        x, _ = self.lstm(x)
        return self.head(x[:, -1])
