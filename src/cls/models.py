"""Classification versions of the three architectures.

OFI-LSTM and ModernTCN are the WF3 classes themselves (src/models), built with the WF3
backbone configuration; only `head` is replaced, by one Linear head per horizon that
emits class logits. The Transformer is the new conventional encoder baseline.

Every model returns a list of logits tensors, one per horizon it predicts, in the order
of `model_config["horizons"]`.
"""
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from src.models import DEFAULTS
from src.models.common import ForecastModel
from src.models.ofi_lstm import OFILSTM
from .fast_moderntcn import FastModernTCN

ARCHITECTURES = ("ofi_lstm", "moderntcn", "transformer")

# The WF3 backbone configuration (configs/<model>_wf3_f*.json model_kwargs over the
# src/models DEFAULTS), so each backbone is exactly the one the regression run trained.
WF3_BACKBONE = {
    "ofi_lstm": DEFAULTS["ofi_lstm"] | dict(hidden_size=384, layers=3),
    "moderntcn": copy.deepcopy(DEFAULTS["moderntcn"]),
}
TRANSFORMER_DEFAULTS = dict(d_model=256, heads=8, layers=4, ffn_dim=1024, dropout=0.1,
                            instance_norm=True, pooling="mean",
                            positional_encoding="sinusoidal")
BACKBONE_DEFAULTS = WF3_BACKBONE | {"transformer": TRANSFORMER_DEFAULTS}


class ClassificationHeads(nn.Module):
    """One independent Linear head per horizon on a shared representation."""
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_features, int(n)) for n in n_classes])

    def forward(self, z):
        return [head(z) for head in self.heads]


class OFILSTMClassifier(OFILSTM):
    """WF3 OFI-LSTM backbone (last hidden state of a stacked LSTM) + classification heads."""
    def __init__(self, config):
        super().__init__(config)
        self.head = ClassificationHeads(config["hidden_size"], config["n_classes"])


class ModernTCNClassifier(FastModernTCN):
    """WF3 ModernTCN backbone + flatten/dropout (as in WF3) + classification heads."""
    def __init__(self, config):
        super().__init__(config)
        in_features = self.head[-1].in_features
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(config["head_dropout"]),
                                  ClassificationHeads(in_features, config["n_classes"]))


def sinusoidal_positions(length, dim):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)*(-math.log(10000.0)/dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position*frequency)
    table[:, 1::2] = torch.cos(position*frequency)
    return table


class TransformerClassifier(ForecastModel):
    """input [B, T, F] -> per-window instance norm -> Linear -> + sinusoidal positions ->
    pre-norm TransformerEncoder -> mean over time -> classification heads."""
    def __init__(self, config):
        super().__init__(config)
        c = config
        if c["pooling"] != "mean" or c["positional_encoding"] != "sinusoidal":
            raise ValueError("The Transformer baseline is mean-pooled with sinusoidal positions only")
        self.input_proj = nn.Linear(c["channels"], c["d_model"])
        self.register_buffer("positions", sinusoidal_positions(c["history_rows"], c["d_model"]),
                             persistent=False)
        self.input_dropout = nn.Dropout(c["dropout"])
        layer = nn.TransformerEncoderLayer(c["d_model"], c["heads"], c["ffn_dim"], c["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c["layers"], norm=nn.LayerNorm(c["d_model"]),
                                             enable_nested_tensor=False)
        self.head = ClassificationHeads(c["d_model"], c["n_classes"])

    def forward(self, x):
        if self.model_config["instance_norm"]:
            # The same per-window normalization ModernTCN applies to its input.
            x = (x-x.mean(1, keepdim=True))/x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        z = self.input_dropout(self.input_proj(x)+self.positions[:x.shape[1]])
        z = self.encoder(z)
        return self.head(z.mean(1))


CLASSES = dict(ofi_lstm=OFILSTMClassifier, moderntcn=ModernTCNClassifier,
               transformer=TransformerClassifier)


def build_classifier(arch, history_rows, channels, n_classes, horizons, **overrides):
    if arch not in CLASSES:
        raise ValueError(f"Unknown architecture {arch}")
    unknown = overrides.keys()-BACKBONE_DEFAULTS[arch].keys()
    if unknown:
        raise ValueError(f"Unknown {arch} options: {sorted(unknown)}")
    if len(n_classes) != len(horizons):
        raise ValueError("One class count per predicted horizon")
    config = dict(name=arch, history_rows=int(history_rows), channels=int(channels),
                  **(copy.deepcopy(BACKBONE_DEFAULTS[arch]) | overrides),
                  n_classes=[int(n) for n in n_classes], horizons=[int(h) for h in horizons])
    return CLASSES[arch](config)


def build_from_config(config):
    config = copy.deepcopy(config)
    return build_classifier(config.pop("name"), config.pop("history_rows"), config.pop("channels"),
                            config.pop("n_classes"), config.pop("horizons"), **config)


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def save_weights(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              str(directory/"model.safetensors"))
    (directory/"config.json").write_text(json.dumps(model.model_config, indent=2)+"\n")


def load_classifier(directory, device="cpu"):
    directory = Path(directory)
    model = build_from_config(json.loads((directory/"config.json").read_text()))
    model.load_state_dict(load_file(str(directory/"model.safetensors")), strict=True)
    return model.to(device).eval()
