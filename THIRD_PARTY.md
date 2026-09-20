# Architecture provenance and deviations

No external pretrained weights are downloaded or loaded. Model defaults are fixed
implementation choices, not selected using validation data.

## HFformer

Primary reference: [Nakols/HFformerV2, 7_HFformerv2.ipynb](https://github.com/Nakols/HFformerV2/blob/14ea36a4bc6fff7be6bf82270e5c71ceabdc413e/7_HFformerv2.ipynb),
commit `14ea36a4bc6fff7be6bf82270e5c71ceabdc413e`.
Paper: [Exploring the Advantages of Transformers for High-Frequency Trading](https://arxiv.org/abs/2302.13850).

`src/models/hfformer.py` is a behavioral reimplementation of the notebook's
encoder/linear-decoder structure, not vendored notebook code. Defaults follow its
final forecasting loop: embedding 36, six heads, two encoders, FFN 64, dropout .3,
causal attention. The paper table and notebook configurations differ; this code
explicitly chooses the runnable notebook configuration. No positional encoding.
Post-norm layers and final LayerNorm are preserved. Decoder remains a per-token
linear map to one feature, PReLU, and a temporal linear map, now to three returns.

Uses the actual [`pytorch-spiking` 0.1.0](https://github.com/nengo/pytorch-spiking)
SpikingActivation/PReLU forward (simulation dt .001, **not the market sampling
interval**) and its analog surrogate derivative. Deliberate correctness changes:

- Use batch-first tensors so the spiking time axis is time, not the batch axis.
- Initialize voltage with fixed uniformly spaced phases per neuron, independently
  for every sample/call, rather than random mutable state. Predictions are repeatable
  and independent of other examples in a batch.
- Express the same surrogate with a straight-through expression to retain gradients
  for the PReLU slope; the library's custom backward omits that parameter's gradient.
- Separate attention projections for LoRA; FP32 spike accumulation under AMP.

Feature order is explicit in each saved schema, not the notebook's interleaved
order. Features are 36 L1–L9 price/quantity fields, one snapshot-lag log return,
and L1 quantity-weighted mid. Windowing, normalization, targets, loss reduction,
and trainer are task-specific. Optional quantile loss exists, but default is MSE
for three point predictions, not multiple quantiles.

## PatchTST

Uses the maintained [Hugging Face PatchTSTModel](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/patchtst)
from pinned `transformers==4.57.6`, instantiated directly from config.
Original project: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST).

Retains shared univariate patch embedding, channel-independent Transformer,
batch normalization, pre-norm, positional encoding, GELU, and input instance
scaling. Each channel is encoded separately with shared weights. A custom joint
flatten/linear head maps the 40 encoded variables to three returns. Output is
not rescaled into price/quantity units. No external checkpoint or pretraining.

## ModernTCN

Adapted from [official long-term forecasting implementation](https://github.com/luodhhh/ModernTCN/blob/56a9a2c018385cd5acef015378cae7f084d1b11c/ModernTCN-Long-term-forecasting/models/ModernTCN.py),
commit `56a9a2c018385cd5acef015378cae7f084d1b11c` (MIT license below).

Retains shared patch-convolution/BatchNorm stem, four stages and downsampling,
large/small depthwise convolution branches with BatchNorm, per-variable grouped
ConvFFN1, per-embedding cross-variable ConvFFN2, GELU, and block residuals.
Defaults use official `run.py` dimensions/blocks/kernel sizes (256 in four stages,
one block each, large kernels 31/29/27/13, small kernels 5, expansion 2).

Adapts the output to a joint linear three-return head, input-only instance
normalization, and replication padding for histories not divisible by patch or
downsampling stride. Unused calendar embedding, decomposition, unused multiscale
layers, and deployment kernel-merging utilities are omitted. Training retains
both kernel branches; checkpoint weights are not fused. This is an adaptation,
not binary-compatible with official forecast checkpoints.
For a final training minibatch with only one example and one remaining temporal
patch, depthwise-branch BatchNorm uses its running statistics. This avoids the
undefined single-value batch variance without dropping the example.

## LiT

Paper-derived implementation of [LiT: limit order book transformer, Section 4](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full).
No reliable official implementation is assumed.

Preserves two price/volume channels, side-specific full-depth temporal patches,
linear projection, **concatenated** learned position embeddings, self-attention,
then LSTM. L20 becomes ten levels per side. Four-snapshot patches are retained as
the default temporal patch size; oldest snapshot is replicated on the left for a
partial initial patch. Bid and ask tokens remain distinct, with position labels;
their encoded vectors are concatenated at each temporal position before LSTM.
The classification output becomes an unactivated three-return linear head.

Embedding 64 (48 content + 16 position), four heads, two encoders, FFN 128 and
LSTM 64 are explicit, untuned adaptation defaults. They are **not claimed to be
the paper's exact undocumented hyperparameters**. `--history-rows 64` supports
the paper-style sample count. The default uses the same seconds-based history as
the other models. Loss, horizons and optimizer follow this task's common trainer.

## OF/OFI-LSTM and E0

The prompt does not identify a particular OFI-LSTM paper or architecture. The
implementation uses standard level-wise bid/ask order-flow case equations with
20-channel OF by default; optionally subtracts ask from bid to obtain 10-channel
OFI. A two-layer hidden-64 LSTM and linear head are untuned defaults. It is not
claimed to reproduce an unspecified paper's hyperparameters. E0 returns zeros.

## ModernTCN license

MIT License

Copyright (c) 2024 luodhhh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
