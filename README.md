# BTC L10 direct return forecasting

Prepared implementations of E0, OF/OFI-LSTM, HFformer, PatchTST, ModernTCN and
LiT adapted to L10. Every output is `[return_60s, return_120s, return_180s]`, where
`return_h = log(mid[target_h] / mid[origin])`. E0 always returns zero.

**No training epoch has been run.** Real-CSV checks and bounded CPU forward/backward
checks are in [reports/validation.json](reports/validation.json); the complete
handoff is [reports/IMPLEMENTATION.md](reports/IMPLEMENTATION.md).

## Setup and safe checks

Python 3.11+ recommended. Validation here used a local `.venv` with Python 3.14.7.
For the RTX 4090, install the appropriate CUDA-enabled PyTorch build first, then
the remaining requirements. Do not replace it with a CPU-only wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The CSV is not copied or modified. In this workspace it was found at the path
below. On another machine, set `LOB_CSV` to the dataset location.

```bash
export LOB_CSV=/Users/son/Projects/P0_LOB/data/processed/BTCUSDT_L10_oct2023.csv
python -m compileall -q src train.py scripts tests
python -m pytest -q
python scripts/validate.py --csv "$LOB_CSV" --backward
python train.py --model ofi_lstm --csv "$LOB_CSV" --prepare-only --output runs/inspection
python train.py --model lit --csv "$LOB_CSV" --smoke-test --backward
```

`--smoke-test` reads the CSV and constructs valid datasets, then takes at most two
samples in one CPU batch. `--backward` adds exactly one backward call and no
optimizer step. Smoke/prepare modes return before any training or batch probe,
even if the config includes `auto_batch_size=true`. The validation script checks
all six models; each learned model can receive one backward call. Test checkpoints
are temporary and contain untrained weights.

## Data contract

- Required fields: `timestamp_ms` plus `bid_price_1..10`, `bid_qty_1..10`,
  `ask_price_1..10`, `ask_qty_1..10`. `segment_id` is used if present. Column aliases
  and timestamp units can be configured in JSON. Missing/nonfinite data, invalid
  prices, crossed books, unordered levels, and non-increasing timestamps fail
  explicitly. No sorting, deduplication, interpolation, or forward fill occurs.
- Timestamps are UTC integer nanoseconds. Default splits are 70%/15%/15% of
  elapsed timestamp duration, not row fractions. Override both `train_end` and
  `validation_end` with ISO timestamps. All history and targets must remain inside
  their own split; training labels never reach validation.
- Cadence is the median of continuous train-split intervals. History rows use
  `ceil(history_seconds / median_dt)` for 60/120/180 seconds. The first history row
  through the origin are included; this is an approximate fixed-row history, not
  resampling. Optional `history_rows=64` overrides the count.
- Default stride seconds are history/6 (10/20/30s), converted to nearest row count.
  Override `stride_seconds` independently. Stride uses an origin grid anchored at
  each split's first full history; gaps filter that grid without re-anchoring it.
- `bad_edge = (dt > 2s) OR segment_changed`. A prefix sum rejects any bad edge
  between history start and the last target in O(1). An edge of exactly 2s is valid.
  `max_gap_seconds` is configurable, default 2.
- Each target is `searchsorted(timestamps, origin_time+horizon, side="left")`.
  Overshoot must be in `[0, target_tolerance_seconds]`, default 2s. A missing or
  invalid target drops the entire three-target sample. Lookup uses no future
  information to build input features; future prices are used only for labels.
- Feature arrays are contiguous float32 after float64 feature calculations and
  train-only standardization. Each dataset holds one shared array plus origin and
  target indices, and slices in `__getitem__`; overlapping windows are never saved.
  No pandas operations occur in workers. Input rows and metadata are O(N), not
  O(N × history). Targets remain unscaled natural log returns.
- Flow and lag-return features reset to zero at gaps, segment changes, and split
  boundaries. Window-local instance normalization in PatchTST/ModernTCN uses only
  that sample's historical input, not future labels or corpus statistics.

## Features and architectures

| Model | Input per snapshot | Default architecture |
|---|---|---|
| E0 | Same origins/labels; input ignored | Constant `[0,0,0]` |
| OFI-LSTM | 20 bid/ask OF channels; optional 10 OFI | 2-layer LSTM, hidden 64, dropout .1, linear 3 |
| HFformer | 36 L1–L9 price/qty + lag return + weighted mid | d=36, 6 heads, 2 post-norm encoders, FFN 64, spiking PReLU, dropout .3, linear decoder |
| PatchTST | 40 raw L10 fields | HF channel-independent backbone, temporal patch 16/stride 8, d=128, 16 heads, 3 layers, FFN 256, dropout .2, joint regression head |
| ModernTCN | 40 raw L10 fields | Patch 16/stride 8; 4 stages of d=256; 1 block/stage; large kernels 31/29/27/13, small 5; ConvFFN expansion 2 |
| LiT | `[side=2, depth=10, price/qty=2]` | Full-side patches of 4 snapshots; projection 48 + learned position 16; 2 encoders, 4 heads, FFN 128; LSTM 64; linear 3 |

Raw feature order for PatchTST/ModernTCN is all bid prices, all bid quantities,
all ask prices, all ask quantities. HFformer uses that order over nine levels,
then `log(mid_t/mid_(t-1))`, then
`(ask_price_1*bid_qty_1 + bid_price_1*ask_qty_1)/(bid_qty_1+ask_qty_1)`.
Zero total L1 quantity falls back to ordinary mid. OF input order is bid OF1..10,
then ask OF1..10; OFI subtracts ask flow from bid flow at each level.

All backbone parameters are initialized from scratch. Architecture settings live
in `src/models/__init__.py`; data/trainer settings live in `src/config.py`.
Use `--config configs/default.json` and/or CLI overrides. For example,
`--model-kwargs '{"hidden_size":128}'` changes an LSTM setting explicitly.
Unknown model settings are rejected. No hyperparameter search is implemented.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source revisions, retained components,
intentional corrections, and paper/code ambiguities.

## Future training commands — not executed

After activating the environment and setting `LOB_CSV`, these start real training:

```bash
python train.py --model ofi_lstm --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name of_60s
python train.py --model hfformer --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name hf_60s
python train.py --model patchtst --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name patch_60s
python train.py --model moderntcn --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name tcn_60s
python train.py --model lit --csv "$LOB_CSV" --history-seconds 60 --device cuda --run-name lit_60s
```

Use `--history-seconds 120` or `180` with a different run name for other histories;
`--history-rows 64` selects the explicit LiT paper-style count. E0 needs no training:

```bash
python train.py --model e0 --csv "$LOB_CSV" --evaluate validation --num-workers 0
```

Trainer defaults: AdamW, lr 1e-4, decay 1e-4, 30 epochs, cosine scheduler, MSE,
gradient norm clip 1. Optional quantile loss is configured through JSON. Best
checkpoint selection always uses validation mean MSE. Test evaluation is explicit.
The common optimizer settings are untuned task defaults, not paper reproductions.

RTX 4090 preparation: auto BF16 where supported, otherwise FP16 with GradScaler;
TF32; pinned memory; persistent workers; prefetch factor; nonblocking transfers;
configurable gradient accumulation; optional `--compile`. CPU uses FP32. Linux
fork workers share the contiguous data arrays; spawn-based workers may duplicate
arrays, so account for host RAM or use `--num-workers 0` on a laptop.

`--auto-batch-size` is opt-in: probes 128, 256, 512, ... up to 4096 by default on
a cloned model, one forward/backward/optimizer step per candidate; checks free
VRAM after optimizer allocation and reserves at least 2 GiB by default. It restores
RNG and does not alter the real model. If 128 does not fit, choose a smaller batch
manually. The probe does not measure throughput or guarantee compile workspace
size. No CUDA probing or benchmarking has been performed here.

## Checkpoints and fine-tuning

Each `checkpoints/<model>/<run_name>/{best,last}/` contains:

```text
model.safetensors         config.json
optimizer.pt             scheduler.pt
trainer_state.pt          experiment.json
preprocessing.json       feature_schema.json
target_config.json       split_manifest.json
data_stats.json          environment.json
```

Both best and last contain full resume state. Saves stage a complete directory
before replacement; a retained `.previous` directory identifies an interrupted
replacement and is not silently discarded. Training refuses to overwrite an
existing run without `--resume`. Resume is **epoch-boundary**, not mid-epoch:
restores optimizer, scheduler, AMP scaler, best score, global step, model/freeze
mode, Python/NumPy/Torch/CUDA RNG and independent training-sampler RNG.
Dataset fingerprint, splits, origin hashes, preprocessing and optimization
contract are checked before resume; changing the epoch budget is a new fine-tune.

```bash
# Resume the same experiment (epoch budget and batch contract remain fixed).
python train.py --resume checkpoints/lit/lit_60s/last --csv "$LOB_CSV" --device cuda

# Full fine-tuning: model + original preprocessing, new optimizer/scheduler/run.
python train.py --init-from checkpoints/lit/lit_60s/best --csv "$LOB_CSV" --device cuda --run-name lit_finetune

# Freeze backbone parameters, BatchNorm statistics and backbone dropout.
python train.py --init-from checkpoints/lit/lit_60s/best --csv "$LOB_CSV" --device cuda --head-only --run-name lit_head

# Explicit held-out evaluation after training.
python train.py --init-from checkpoints/lit/lit_60s/best --csv "$LOB_CSV" --evaluate test --device cuda
```

Every Transformer exposes real Linear `q_proj`, `k_proj`, `v_proj`, `out_proj`
modules and named FFN layers. Future LoRA tooling can inject adapters into these;
this phase implements no LoRA training or adapter serialization. Default full
checkpoints are unmerged base weights. `model.freeze_backbone()` and
`model.unfreeze()` also support Python-driven fine-tuning.

```python
from src.training.checkpoint import from_pretrained
model = from_pretrained("checkpoints/lit/lit_60s/best")
metadata = model.checkpoint_metadata
model.save_pretrained("export/lit", metadata)
```

These are local `save_pretrained`/`from_pretrained` helpers, not registered
Transformers `AutoModel` classes. The resulting folder is ready to upload through
Hugging Face Hub together with this source package/requirements; loading still
requires this project. No Hub upload is performed. Saved normalizer statistics,
feature ordering, histories, target definition and split fingerprint accompany
weights. A new dataset fine-tune reuses the base normalizer and fixed input shape.

## Verification limits

All implemented models and full default shapes were checked on CPU, with no epochs
or optimizer steps. Unit tests cover data and checkpoint edge cases. CUDA AMP,
`torch.compile`, batch probing, throughput, and long-run optimizer/scheduler behavior
need checking on the target GPU when training is authorized. No validation tuning,
forecasting-quality claim, or trading performance claim is made.
