"""Kernel-level reimplementation of the WF3 ModernTCN forward pass.

The architecture, the parameters and the state_dict are the WF3 ModernTCN's, untouched:
every module below is the WF3 module and only the arithmetic that consumes its weights
changes. cuDNN runs the grouped 1x1 convolutions (groups=channels=40 and groups=dim=256)
as one tiny GEMM per group, about 2,000 kernel launches per training step, and runs the
31-tap depthwise convolutions over sequences of at most 7 patches. Here a grouped 1x1
convolution is one batched matmul over its groups, and a depthwise convolution on a
length-n sequence is a per-channel n x n Toeplitz matmul built from the same kernel taps
(the taps that only ever multiply zero padding get zero gradient, exactly as in the
convolution). tests/test_cls_fast_moderntcn.py checks outputs and gradients against the
WF3 module on the same weights.
"""
import math

import torch
from torch.nn import functional as F

from src.models.moderntcn import ModernTCN


def _pointwise(conv, z):
    """conv: Conv1d(cin, cout, 1, groups=g) on z laid out [g, cin/g, L]; returns [g, cout/g, L]."""
    g, cout = conv.groups, conv.out_channels
    weight = conv.weight.view(g, cout//g, -1)
    y = torch.bmm(weight if torch.is_autocast_enabled() else weight.to(z.dtype), z)
    if conv.bias is not None:
        y = y+conv.bias.view(g, cout//g, 1).to(y.dtype)
    return y


def _ffn(ffn, z):
    """WF3 conv FFN = Conv1d(groups) -> Dropout -> GELU -> Conv1d(groups) -> Dropout."""
    return ffn[4](_pointwise(ffn[3], ffn[2](ffn[1](_pointwise(ffn[0], z)))))


def _toeplitz(conv, n, device):
    """Depthwise Conv1d(c, c, k, padding=k//2, groups=c) as per-channel [c, s, t] matrices."""
    k, p = conv.kernel_size[0], conv.padding[0]
    t = torch.arange(n, device=device)
    tap = t.view(n, 1)-t.view(1, n)+p           # tap[s, t] = s - t + p (input s, output t)
    valid = ((tap >= 0) & (tap < k)).to(conv.weight.dtype)
    return conv.weight[:, 0, :][:, tap.clamp(0, k-1)]*valid  # [c, s, t]


def _depthwise(conv, xc):
    """xc laid out [c, b, n] -> [c, b, n]; out[c, b, t] = sum_s xc[c, b, s] * w[c, s - t + p]."""
    matrix = _toeplitz(conv, xc.shape[-1], xc.device)
    return torch.bmm(xc, matrix if torch.is_autocast_enabled() else matrix.to(xc.dtype))


def replicate_right(x, pad):
    """F.pad(x, (0, pad), mode='replicate') for a right-only pad, as an index gather."""
    if pad <= 0:
        return x
    n = x.shape[-1]
    index = torch.arange(n+pad, device=x.device).clamp(max=n-1)
    return x.index_select(-1, index)


def block_forward(block, x):
    """WF3 ModernBlock.forward on x [b, m, d, n], same modules, batched-matmul arithmetic."""
    b, m, d, n = x.shape
    xc = x.reshape(b, m*d, n).transpose(0, 1)    # [c, b, n] view, c = m*d
    large, small = block.dw.large, block.dw.small
    # Each branch's BatchNorm sees the WF3 layout [b, c, n], so its statistics are unchanged.
    y = (large[1](_depthwise(large[0], xc).transpose(0, 1))
         + small[1](_depthwise(small[0], xc).transpose(0, 1)))          # [b, c, n]
    y = block.norm(y.reshape(b*m, d, n))                               # [b*m, d, n]
    z = y.reshape(b, m, d, n).permute(1, 2, 0, 3).reshape(m, d, b*n)    # groups = m
    z = _ffn(block.conv_ffn1, z)                                       # [m, d, b*n]
    z = _ffn(block.conv_ffn2, z.transpose(0, 1).contiguous())          # groups = d: [d, m, b*n]
    return x+z.reshape(d, m, b, n).permute(2, 1, 0, 3)


def backbone_features(model, x):
    """Everything in ModernTCN.forward before the head, returning [b, m, d, n]."""
    c = model.model_config
    b, t, m = x.shape
    if c["instance_norm"]:
        x = (x-x.mean(1, keepdim=True))/x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    x = x.transpose(1, 2).reshape(b*m, 1, t)
    patches = math.ceil(t/c["patch_stride"])
    pad = max(0, (patches-1)*c["patch_stride"]+c["patch_length"]-t)
    x = model.stem(replicate_right(x, pad))
    x = x.reshape(b, m, x.shape[1], x.shape[2])
    for i, stage in enumerate(model.stages):
        if i:
            _, _, d, n = x.shape
            x = x.reshape(b*m, d, n)
            pad = (-n) % c["downsample_ratio"]
            x = model.downsample[i-1](replicate_right(x, pad))
            x = x.reshape(b, m, x.shape[1], x.shape[2])
        for block in stage:
            x = block_forward(block, x)
    return x


class FastModernTCN(ModernTCN):
    """WF3 ModernTCN, same modules and state_dict, batched-matmul forward."""
    def forward(self, x):
        return self.head(backbone_features(self, x))
