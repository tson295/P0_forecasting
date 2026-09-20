"""Hugging Face PatchTST encoder, randomly initialized; direct-return head."""
from torch import nn
from transformers import PatchTSTConfig, PatchTSTModel
from .common import ForecastModel


class PatchTST(ForecastModel):
    def __init__(self, config):
        super().__init__(config)
        c = config
        if c["history_rows"] < c["patch_length"]:
            raise ValueError("PatchTST history must be >= patch_length")
        hf = PatchTSTConfig(
            num_input_channels=c["channels"], context_length=c["history_rows"],
            patch_length=c["patch_length"], patch_stride=c["patch_stride"],
            d_model=c["d_model"], num_attention_heads=c["heads"],
            num_hidden_layers=c["layers"], ffn_dim=c["ffn_dim"],
            attention_dropout=c["dropout"], ff_dropout=c["dropout"],
            positional_dropout=c["dropout"], path_dropout=c["dropout"],
            share_embedding=True, channel_attention=False, norm_type="batchnorm",
            pre_norm=True, activation_function="gelu", positional_encoding_type="sincos",
            scaling=c["scaling"], do_mask_input=False)
        self.backbone = PatchTSTModel(hf)
        patches = (c["history_rows"]-c["patch_length"])//c["patch_stride"]+1
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(c["head_dropout"]),
                                  nn.Linear(c["channels"]*patches*c["d_model"], 3))

    def forward(self, x):
        # HF output: [batch, channels, temporal patches, embedding].
        # Returns are not raw prices, so never RevIN-denormalize the output.
        return self.head(self.backbone(past_values=x).last_hidden_state)
