from torch import nn
from .common import ForecastModel


class OFILSTM(ForecastModel):
    def __init__(self, config):
        super().__init__(config)
        c = config
        self.backbone = nn.LSTM(c["channels"], c["hidden_size"], c["layers"],
                                batch_first=True, dropout=c["dropout"] if c["layers"] > 1 else 0.)
        self.head = nn.Linear(c["hidden_size"], 3)

    def forward(self, x):
        x, _ = self.backbone(x)
        return self.head(x[:, -1])
