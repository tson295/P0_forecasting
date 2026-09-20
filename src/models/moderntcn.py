"""Adapted official ModernTCN convolution structure. MIT attribution in THIRD_PARTY.md."""
import math
from torch import nn
from torch.nn import functional as F
from .common import ForecastModel


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics when a final minibatch has one scalar per channel."""
    def forward(self, x):
        if self.training and x.numel()//x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                self.bias, training=False, momentum=0., eps=self.eps)
        return super().forward(x)


class LargeKernel(nn.Module):
    def __init__(self, channels, large, small):
        super().__init__()
        if large % 2 != 1 or small % 2 != 1 or small > large:
            raise ValueError("ModernTCN kernels must be odd, small <= large")
        def branch(k):
            return nn.Sequential(nn.Conv1d(channels, channels, k, padding=k//2,
                                           groups=channels, bias=False), SafeBatchNorm1d(channels))
        self.large, self.small = branch(large), branch(small)

    def forward(self, x):
        return self.large(x)+self.small(x)


class ModernBlock(nn.Module):
    def __init__(self, channels, dim, ratio, large, small, dropout):
        super().__init__()
        width = channels*dim
        self.dw = LargeKernel(width, large, small)
        self.norm = nn.BatchNorm1d(dim)
        def ffn(groups):
            return nn.Sequential(nn.Conv1d(width, width*ratio, 1, groups=groups),
                                 nn.Dropout(dropout), nn.GELU(),
                                 nn.Conv1d(width*ratio, width, 1, groups=groups), nn.Dropout(dropout))
        self.conv_ffn1, self.conv_ffn2 = ffn(channels), ffn(dim)

    def forward(self, x):
        b, m, d, n = x.shape
        residual = x
        x = self.dw(x.reshape(b, m*d, n)).reshape(b*m, d, n)
        x = self.norm(x).reshape(b, m*d, n)
        x = self.conv_ffn1(x).reshape(b, m, d, n).permute(0, 2, 1, 3)
        x = self.conv_ffn2(x.reshape(b, d*m, n)).reshape(b, d, m, n)
        return residual+x.permute(0, 2, 1, 3)


class ModernTCN(ForecastModel):
    def __init__(self, config):
        super().__init__(config)
        c = config
        dims = c["dims"]
        if not len(dims) == len(c["blocks"]) == len(c["large_kernels"]) == len(c["small_kernels"]):
            raise ValueError("ModernTCN stage settings must have matching lengths")
        self.stem = nn.Sequential(nn.Conv1d(1, dims[0], c["patch_length"], stride=c["patch_stride"]),
                                  nn.BatchNorm1d(dims[0]))
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(dims[i-1]),
                          nn.Conv1d(dims[i-1], dims[i], c["downsample_ratio"], stride=c["downsample_ratio"]))
            for i in range(1, len(dims))])
        self.stages = nn.ModuleList([
            nn.Sequential(*[ModernBlock(c["channels"], dim, c["ffn_ratio"],
                                        c["large_kernels"][i], c["small_kernels"][i], c["dropout"])
                            for _ in range(c["blocks"][i])]) for i, dim in enumerate(dims)])
        patches = math.ceil(c["history_rows"]/c["patch_stride"])
        for _ in self.downsample:
            patches = math.ceil(patches/c["downsample_ratio"])
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(c["head_dropout"]),
                                  nn.Linear(c["channels"]*dims[-1]*patches, 3))

    def forward(self, x):
        c = self.model_config
        b, t, m = x.shape
        if c["instance_norm"]:
            x = (x-x.mean(1, keepdim=True))/x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        x = x.transpose(1, 2).reshape(b*m, 1, t)
        # Replication padding supports arbitrary derived histories, retaining newest row.
        patches = math.ceil(t/c["patch_stride"])
        pad = max(0, (patches-1)*c["patch_stride"]+c["patch_length"]-t)
        x = self.stem(F.pad(x, (0, pad), mode="replicate"))
        x = x.reshape(b, m, x.shape[1], x.shape[2])
        for i, stage in enumerate(self.stages):
            if i:
                _, _, d, n = x.shape
                x = x.reshape(b*m, d, n)
                pad = (-n) % c["downsample_ratio"]
                x = self.downsample[i-1](F.pad(x, (0, pad), mode="replicate"))
                x = x.reshape(b, m, x.shape[1], x.shape[2])
            x = stage(x)
        return self.head(x)
