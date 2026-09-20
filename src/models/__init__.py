import torch
from .common import ForecastModel

DEFAULTS = {
    "e0": {},
    "ofi_lstm": dict(hidden_size=64, layers=2, dropout=.1),
    # Final multi-horizon loop in the public notebook (not the contradictory paper table).
    "hfformer": dict(d_model=36, heads=6, layers=2, ffn_dim=64, dropout=.3,
                     causal=True, spike_dt=.001),
    "patchtst": dict(d_model=128, heads=16, layers=3, ffn_dim=256, dropout=.2,
                     head_dropout=0., patch_length=16, patch_stride=8, scaling="std"),
    "moderntcn": dict(dims=[256, 256, 256, 256], blocks=[1, 1, 1, 1],
                      large_kernels=[31, 29, 27, 13], small_kernels=[5, 5, 5, 5],
                      ffn_ratio=2, patch_length=16, patch_stride=8, downsample_ratio=2,
                      dropout=.05, head_dropout=0., instance_norm=True),
    "lit": dict(d_model=64, position_dim=16, heads=4, layers=2, ffn_dim=128,
                dropout=.1, temporal_patch=4, lstm_hidden=64, lstm_layers=1),
}


class E0(ForecastModel):
    def forward(self, x):
        return x.new_zeros((x.shape[0], 3))


def build_model(name, history_rows, channels, **kwargs):
    import copy
    from .ofi_lstm import OFILSTM
    from .hfformer import HFformer
    from .patchtst import PatchTST
    from .moderntcn import ModernTCN
    from .lit import LiT
    classes = dict(e0=E0, ofi_lstm=OFILSTM, hfformer=HFformer,
                   patchtst=PatchTST, moderntcn=ModernTCN, lit=LiT)
    if name not in classes:
        raise ValueError(f"Unknown model {name}")
    unknown = kwargs.keys()-DEFAULTS[name].keys()
    if unknown:
        raise ValueError(f"Unknown {name} model options: {sorted(unknown)}")
    c = dict(name=name, history_rows=history_rows, channels=channels,
             **(copy.deepcopy(DEFAULTS[name]) | kwargs))
    if history_rows < 2 or channels < 1:
        raise ValueError("Invalid input shape")
    if name == "lit" and not 0 < c["position_dim"] < c["d_model"]:
        raise ValueError("LiT position_dim must be between zero and d_model")
    return classes[name](c)
