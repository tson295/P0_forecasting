# BTC L10 direct return forecasting

Prepared implementations of E0, OF/OFI-LSTM, HFformer, PatchTST, ModernTCN and
LiT adapted to L10. Every output is `[return_60s, return_120s, return_180s]`, where
`return_h = log(mid[target_h] / mid[origin])`. E0 always returns zero.

This repository holds the **frozen base experiment**: 60-second history, 49 history
rows, stride 8 rows, batch 128, 30 epochs, seed 42, no hyperparameter tuning and no
pretrained weights. Every learned model is randomly initialised and trained from
scratch on the BTC L10 train split. Frozen run configs live in `configs/`; the
pre-training gate is `scripts/check_contracts.py`.

## Setup and safe checks

Python 3.12. For the RTX 4090 install a CUDA-enabled PyTorch build first, then the
remaining requirements. Do not replace it with a CPU-only wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
pip install -r requirements.txt
```

The CSV is never copied or modified. Set `LOB_CSV` to the dataset location; it must
hash to `e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab`.

```bash
export LOB_CSV=$PWD/BTCUSDT_L10_oct2023.csv
sha256sum "$LOB_CSV"
python -m compileall -q src train.py scripts tests
python -m pytest -q
python scripts/check_contracts.py --csv "$LOB_CSV"     # hard gate, exits non-zero on any violation
python scripts/validate.py --csv "$LOB_CSV" --backward
python scripts/profile_gpu.py --mode all               # hardware, CUDA smoke, throughput, compile, concurrency
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
| HFformer | 38 raw-scale: 36 L1–L9 price/qty + lag return + weighted mid | d=36, 6 heads, 2 post-norm encoders, FFN 64, spiking PReLU, dropout .3, causal, no position encoding, linear decoder |
| PatchTST | 40 raw L10 fields | HF channel-independent backbone, temporal patch 16/stride 8, d=128, 16 heads, 3 layers, FFN 256, dropout .2, joint regression head |
| ModernTCN | 40 raw L10 fields | Patch 16/stride 8; 4 stages of d=256; 1 block/stage; large kernels 31/29/27/13, small 5; ConvFFN expansion 2 |
| LiT | `[side=2, depth=10, price/qty=2]` | Full-side patches of 4 snapshots (80 numbers/token, 26 tokens); projection 96 concatenated with learned position 32; 4 encoders, 8 heads, FFN 256; bid+ask concat 256 into LSTM 128; linear 3 |

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

## Second experiment: Gate one-year capture (branch `gate-1y`)

A second frozen run on `BTC_L10_gate_1y.csv`, a 365-day L10 capture
(2025-09-17 -> 2026-09-16) sampled on an exact 10-second grid. The EDA is
[reports/EDA_GATE_1Y.md](reports/EDA_GATE_1Y.md); the results are
[reports/FINAL_REPORT_GATE1Y.md](reports/FINAL_REPORT_GATE1Y.md).

What the cadence forced, and nothing else:

- The file arrives out of chronological order (a 7.46-day block prepended, filling
  the main block's only multi-day hole) with nine duplicated timestamps whose
  payloads differ. `data.sort_by_timestamp` and `data.duplicate_timestamp_policy`
  repair this. Both default to off/`error`, so a book is still never reordered or
  deduplicated silently, and what was repaired is recorded in
  `split_manifest.source.row_repairs` of every checkpoint.
- `max_gap_seconds` and `target_tolerance_seconds` move 2.0s -> 10.0s. At 10s
  cadence the 2.0s rule marks every edge in the file as a gap.
- `history_seconds` moves 60 -> 490 so it re-resolves to the same
  `history_rows = 49` and `stride_rows = 8`. 60s would give 6 rows, below
  PatchTST's and ModernTCN's `patch_length = 16`. Because the window shape is
  unchanged, all five parameter counts are unchanged.

Architecture, optimizer, loss, batch size, epochs and seed are identical to the
first run; only the data scales (274,215 / 59,016 / 58,963 samples).

Every tool takes `--suite {oct2023,gate1y}`:

```bash
python scripts/eda.py --csv BTC_L10_gate_1y.csv --history-seconds 490 --output reports/eda_gate_1y
python scripts/check_contracts.py --csv BTC_L10_gate_1y.csv --expected contracts/gate1y.json \
                                  --output reports/contract_check_gate1y.json
python scripts/profile_gpu.py --mode all --suite gate1y --output-dir reports/vast_gate1y
python scripts/run_training.py --suite gate1y --csv BTC_L10_gate_1y.csv
python scripts/upload_hf.py --suite gate1y
```

### Metrics are reported in raw price

Training is unchanged: the target is still `log(mid[t+h]/mid[t])` and the loss is
still MSE on that log return. Only the reported space changed. Every
`*_metrics.json` from this run carries RMSE, MAE, standard R2, RMSE_E0 and RMSE
gain vs E0 measured on the mid price in quote currency at the top level, computed
from the exported columns via `pred_mid = origin_mid * exp(pred_return)`, with the
previous log-return family nested under `log_return` so the two runs stay
comparable. E0 predicts that the price does not move, so `RMSE_E0 =
sqrt(mean((target_mid - origin_mid)^2))`.

Standard R2 in price space is near 1 for every model including E0, because
sigma(target mid) is about 16,600 USD while every error is around 50 USD. It is
reported for contract completeness; **RMSE gain vs E0 is the column to read**.

## Base experiment runbook

Run every step from the repository root, in this order. Each step is a hard gate
for the next one.

```bash
# 1. Gates
python -m compileall -q src train.py scripts tests
python -m pytest -q
python scripts/check_contracts.py --csv "$LOB_CSV"
python scripts/validate.py --csv "$LOB_CSV" --backward

# 2. Systems profiling (never touches a real checkpoint, never calls fit)
python scripts/profile_gpu.py --mode all

# 3. Commit the training source, then train from exactly that commit
git rev-parse HEAD
python scripts/run_training.py --csv "$LOB_CSV"          # schedule comes from the benchmark

# 4. Predictions and metrics for train/validation/test, from the best checkpoint
for run in e0_60s ofi_lstm_60s_base hfformer_60s_base patchtst_60s_base \
           moderntcn_60s_base lit_60s_base; do
  python scripts/export_predictions.py --config "configs/$run.json" --csv "$LOB_CSV"
done

# 5. Publish and report
python scripts/upload_hf.py --dry-run
python scripts/upload_hf.py
python scripts/final_report.py --hf-repo "<user>/Pretrain_Model" --training-commit "<sha>"
```

`run_training.py` reads the concurrency decision from
`reports/vast/concurrency_benchmark.json`, launches each frozen run as its own
process on GPU 0 with `batch_size=128` unchanged, samples the GPU every three
seconds into `reports/vast/gpu_usage.csv`, and resumes a crashed job from its
`last` checkpoint instead of restarting at epoch 0. Concurrency decides only
which frozen runs execute at the same time; it never changes batch size,
architecture, learning rate, epochs, stride or split.

The six frozen runs are `e0_60s`, `ofi_lstm_60s_base`, `hfformer_60s_base`,
`patchtst_60s_base`, `moderntcn_60s_base` and `lit_60s_base`. E0 is never
trained: it predicts return 0 at every horizon on exactly the same origins and
targets as the learned models.

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

## Metrics and prediction artifacts

Exactly four metric families are reported, per horizon 1m/2m/3m:

```text
rmse   mae   r2 (standard)   rmse_e0   rmse_gain_vs_e0
```

`rmse_e0 = sqrt(mean(y^2))` because E0 predicts return 0, and
`rmse_gain_vs_e0 = 1 - rmse_model/rmse_e0`, so a positive gain beats E0, zero
equals E0 and negative is worse. E0's own gain is 0 by construction. R2 is the
standard coefficient of determination, never R2_OS.

`scripts/export_predictions.py` loads the `best` checkpoint, verifies that both
`best` and `last` reload to identical predictions, runs deterministic FP32
inference and writes, per run:

```text
artifacts/<model>/<run_name>/
    train_predictions.csv.gz   validation_predictions.csv.gz   test_predictions.csv.gz
    train_metrics.json         validation_metrics.json         test_metrics.json
    training_history.jsonl     run_summary.json
```

Every prediction table carries `origin_index`, `origin_timestamp_ns`,
`origin_mid`, and for each horizon `target_index`, `target_timestamp_ns`,
`target_mid`, `true_return`, `pred_return` and
`pred_mid = origin_mid * exp(pred_return)`, ordered by origin timestamp. Every
actual-vs-predicted, error, scatter and horizon-comparison figure can therefore be
rebuilt without running a model again. Test inference happens only after training
and best-checkpoint selection are complete; the test split is never used to choose
a checkpoint, an epoch or an architecture.

## Verification limits

Checkpoints, prediction tables and the raw CSV stay out of Git and live locally and
on the Hugging Face repo only. No validation tuning, forecasting-quality claim, or
trading performance claim is made.
