# Hand-written sections spliced into the generated report by scripts/cls_report.py

<!-- section: contract -->
The experiment keeps the WF3 contract and changes only the prediction target, the output heads and the loss.

| Item | Value |
|---|---|
| Data | BTCUSDT L10 (10 bid + 10 ask levels, price and quantity), nominal 10 s grid, `BTC_L10_gate_1y.csv` |
| History / stride | 490 s = 49 rows / 20 s = 2 rows (WF3 `prepare_data`: ceil(490 / median train dt), round(20 / dt)) |
| Horizons | h1 = 60 s, h2 = 120 s, h3 = 180 s; target = first timestamp ≥ origin + h, tolerance 10 s |
| Split | WF3 walk-forward: last 15 % of elapsed time is the test split; the rest cut into 4 equal time blocks; fold k trains on blocks [0, k) and validates on block k |
| Embargo | 180 s purge + embargo between consecutive splits (WF3 `target_limits`); audit floor max(embargo, longest horizon) |
| Test split | rows [2667637, 3139597), 235,848 samples, identical for every architecture, fold, binning method and formulation |
| Features | OFI-LSTM: 20 order-flow channels; ModernTCN and Transformer: the 40 raw L10 columns; all with the WF3 train-only per-field z-score, refitted per fold on that fold's train rows |
| Target | delta_h(t) = target_mid(t+h) − origin_mid(t) in USD (the displacement from E0), mapped to a frozen class interval |
| Class definitions | fitted on FOLD 1 TRAIN only, frozen for every split of every fold (`labeling/*.json`) |
| Primary binning | equal-width: K = 32 finite bins on [−q_h, +q_h], q_h = p99(abs(delta_h)) on F1 train, plus 2 overflow classes (34 classes) |
| Comparison binning | quantile: 34 equal-frequency classes at the F1-train quantiles of delta_h (33 at 60 s after merging a duplicate edge at the point mass delta = 0) |
| Loss | multi-horizon: (CE_60 + CE_120 + CE_180) / 3; single-horizon: CE_h. Plain CrossEntropy: no class weights, focal, ordinal, distance-aware or auxiliary regression terms |
| Decoding (official) | argmax class → interval midpoint (a + b)/2; equal-width overflow → −q_h − w_h/2 and +q_h + w_h/2; quantile outer classes → F1-train median of the class's own samples; pred_mid = origin_mid + decoded delta |
| Recipe | the WF3 recipe: AdamW (lr 1e-4, weight decay 1e-4), CosineAnnealingLR (T_max = 30), 30 epochs, batch 128, gradient clip 1.0, bf16 autocast + TF32, seed 42 |
| Selection | best epoch = lowest validation mean CrossEntropy; test is read only by the export after training and selection |
| Official metrics | RMSE and MAE of the decoded future mid (USD), R² gain vs E0 = 1 − SSE_model/SSE_E0 (WF3 definition), directional accuracy (DA) |
| Matrix | 2 binning methods × 3 architectures × 3 folds × (multi + 3 single-horizon) = 18 + 54 = 72 trained models |

Directional accuracy is defined once for every run: samples whose true displacement is exactly 0 are excluded; on the rest, a hit is sign(pred_delta) = sign(true_delta), and a prediction of exactly 0 counts as a miss. E0 therefore scores DA = 0 (it never calls a direction); 50 % is the coin-flip reference. DA is only an evaluation metric; the task stays multi-class interval prediction.

<!-- section: dataset -->
| Item | Value |
|---|---|
| Source | private Hugging Face dataset `Tson29/btc-l10-gate-1y`, file `BTC_L10_gate_1y.csv` (1,147,814,349 bytes), downloaded with the authenticated account; not copied into the experiment repo or into Git |
| SHA256 | `6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df` (computed locally, exact match) |
| Raw rows | 3,139,606 (verified) |
| Repair (WF3 procedure) | stable sort by timestamp (2,531,566 rows moved), duplicate timestamps keep_first (9 rows dropped) → 3,139,597 rows (verified) |
| Range | 2025-09-17T00:00:00Z → 2026-09-16T23:59:50Z (verified) |
| Cadence | median dt 10.0 s, p99 10.0 s, max 7,210 s; 31 gaps > 10 s; no segment transitions |
| Samples | F1 333,177 / 332,849; F2 666,059 / 332,433; F3 998,525 / 334,350 (train / validation); test 235,848 — identical to the WF3 run |

<!-- section: provenance -->
| Item | Value |
|---|---|
| GitHub repository / branch | `tson295/P0_forecasting`, branch `classification` (new; no existing branch modified, no force push) |
| Base commit | `fdbb369946cc07eae9e615c968943036507be129`, the WF3 training-source commit |
| Where the WF3 code lives | `fdbb369` is not in `tson295/P0_forecasting` (none of its branches — `main`, `OB`, `run/ml-lstm-expanded`, `tfm_autots` — contains WF3 code); it is on `tson295/Pretrain_Model`, branch `walkforward-3fold`. The `classification` branch was therefore created from `fdbb369` itself (fetched from `Pretrain_Model`) and pushed to `P0_forecasting`, so this work is based on the verified WF3 codebase and not on an unrelated P0_forecasting branch |
| Ported from the WF3 report commit `124b1e9` | `scripts/check_leakage.py` (audit floor max(embargo, longest horizon)); the WF3 report and audit are kept under `reports/wf3_reference/` |
| WF3 prompt | the root `prompt.md` of `fdbb369` moved to `reports/wf3_reference/prompt_wf3.md`; the root `prompt.md` is this experiment's prompt, verbatim |
| Hugging Face | experiment repo `Tson29/LOB_Classification_WF3` (private, new); WF3 regression repo `Tson29/Pretrain_Model_WF3` read only; dataset `Tson29/btc-l10-gate-1y` read only |
| Hardware | NVIDIA GeForce RTX 4090 reporting 47.4 GiB (49,140 MiB) of VRAM — the prompt assumed 24 GB; the concurrency policy was measured on the real card — AMD EPYC 7502 (32 cores), 78 GiB RAM |
| Software | Python 3.12.3, torch 2.8.0+cu128 (the WF3 version), numpy 2.5.3, pandas 3.0.6, safetensors 0.8.0 |

<!-- section: distribution -->
Every split is described with the same frozen fold-1-train definitions. Two facts shape everything that follows. First, the displacement distribution is sharply peaked at zero and heavy-tailed: at 60 s, 6.2 % of F1-train samples have delta exactly 0 (the mid is on a 0.05 USD grid) and the largest move is 6,172 USD. Second, the market calms down over the year: RMSE_E0 at 60 s falls from 68.0 USD on F1 train to 36.0 USD on the test split, so under the F1-fitted equal-width bins the two central bins hold ~31 % of F1-train samples but ~55 % of test samples, and the overflow classes shrink from ~1 % to ~0.2 %. The class definitions are deliberately not refitted (the F1 → F2 → F3 comparison needs one fixed target), so later folds and the test split see a more concentrated label distribution than the one the bins were designed on.

<!-- section: equal_width -->
Displacements are first snapped to 1e-6 USD (`numpy.round(delta, 6)`): mids sit on a 0.05 USD grid but their float64 difference carries ~1e-11 USD of rounding noise, so without snapping one USD value (e.g. −0.10) has several binary forms and an edge landing on it would split identical moves across two classes. Every fit, every assignment and every exported `true_delta` uses the snapped value; metrics use the mids themselves. q_h is `numpy.percentile(abs(delta_h), 99)` (linear interpolation) over the 333,177 fold-1 train samples. K = 32 was chosen after inspecting the F1-train distributions (see `labeling/label_distribution.json`): it keeps ≥ 378 F1-train samples in the rarest finite bin, puts ~28–32 % of the F1-train mass in the two central bins, and gives an argmax half-width w_h/2 of 6.9 / 9.7 / 11.9 USD — K = 20 would have made the central half-width 11–19 USD, larger than a third of the test RMSE_E0 at 60 s. 0 USD is exactly an edge (edges are w·k for k = −16..16 with the ends set to ±q exactly); every finite class is [a, b) except the last, [q − w, q], and only delta < −q / delta > +q are overflow.

<!-- section: quantile -->
Edges are `numpy.quantile(delta_h, j/34)` for j = 1..33 on the same (snapped) fold-1 train samples, themselves snapped to 1e-6 USD, so an edge that lands on a price-grid point mass is exactly that grid value and the whole mass goes to the class above it. At 60 s, 6.2 % of samples sit exactly at delta = 0, so two neighbouring quantiles coincide at 0 and are merged: 33 classes at 60 s, 34 at 120 s and 180 s. Interior classes decode to their midpoint; the two unbounded outer classes decode to the fold-1-train median of their own samples (never to anything computed on validation or test).

<!-- section: models -->
- **OFI-LSTM**: the WF3 `OFILSTM` class with the WF3 configuration (20 order-flow channels, hidden 384, 3 layers, dropout 0.1, eager as in WF3); the regression `Linear(384, 3)` is replaced by one `Linear(384, C_h)` per predicted horizon on the last hidden state.
- **ModernTCN**: the WF3 `ModernTCN` class with the WF3 configuration (dims 4×256, large kernels 31/29/27/13, small 5, FFN ratio 2, patch 16/8, downsample 2, dropout 0.05, instance norm), compiled as in WF3; the regression `Linear(10240, 3)` after flatten + dropout is replaced by one `Linear(10240, C_h)` per horizon. Its forward runs through `src/cls/fast_moderntcn.py`, which evaluates the same modules with the same parameters as batched matmuls (cuDNN otherwise launches one GEMM per group, ~2,000 kernels per step): float64 outputs match the WF3 forward to 1e-15 and gradients to 5e-14 (`tests/test_cls.py`), and training is ~4× faster.
- **Transformer** (new baseline): per-window instance normalization (the same one ModernTCN applies) → `Linear(40, 256)` → fixed sinusoidal positional encoding → dropout → 4-layer pre-norm `nn.TransformerEncoder` (d_model 256, 8 heads, FFN 1024, GELU, dropout 0.1, final LayerNorm) → mean over the 49 time steps → one `Linear(256, C_h)` per horizon. No FlashAttention/RoPE experiments; PyTorch's default attention kernels. Compiled with CUDA graphs.

Training-step implementation choices that do not change the arithmetic: fused AdamW (one kernel per step instead of a multi-tensor loop), `torch.compile` of model + loss for ModernTCN, CUDA graphs for the Transformer, and a GPU-resident feature matrix from which each batch of windows is gathered (the WF3 `LOBDataset.__getitem__` windows exactly, in the WF3 `RandomSampler` order).

<!-- section: benchmark -->
Every benchmark worker is a real training process on fold 1 (equal-width, multi-horizon) running exactly the trainer's step — same batch 128, bf16, compile mode, fused AdamW — measured for 45 s after warm-up, with all workers of a configuration released together at a file barrier while NVML (GPU utilization, memory) and psutil (CPU, RAM, disk I/O) are sampled every 0.5 s. `benchmarks/concurrency_benchmark_raw.json` holds every measurement.

Before the sweep, the per-step cost was cut without touching any model or hyperparameter: (1) the WF3 ModernTCN forward spends its time in ~2,000 tiny cuDNN grouped-GEMM launches per step, so it was re-expressed as batched matmuls over the same parameters (2.2k → 11.0k samples/s, identical outputs); (2) the feature matrix lives on the GPU and batches are gathered there (no DataLoader workers; the same windows and shuffle order as WF3); (3) fused AdamW; (4) CUDA graphs for the Transformer (9.4k → 44k samples/s). An earlier benchmark taken before these changes is kept in `benchmarks/superseded/`.

Findings: ModernTCN and the Transformer each keep the GPU ~95–97 % busy on their own, and any second heavy job only lowers aggregate throughput (TT, FF, TF ≈ 0.91× serial). OFI-LSTM is latency-bound alone (a 49-step, 3-layer recurrence at batch 128: 46 % GPU utilization, one CPU core) and scales to 2.82× with 4 concurrent jobs, flat beyond (5, 6, 8 jobs: 2.83×, 2.82×, 2.82×). Mixed configurations show higher *normalized* throughput (TLL 2.11×, TFLL 1.80×), but they get it by slowing the heavy job by 40–60 %; measured against time-slicing the best exclusive groups (one heavy job alone, or four OFI-LSTMs), every mixed configuration is 2–10 % less efficient. The schedule therefore runs one ModernTCN or one Transformer alone, or up to four OFI-LSTMs together, longest job first, and a job that has finished its GPU work and is only writing prediction files releases its slot. VRAM never binds (the card reports 47.4 GiB; the largest job reserves ~2.2 GiB), so no batch-size or memory measure was needed. CPU: 32 cores, at most ~4 busy; RAM ≤ ~20 GiB; disk I/O negligible (data resident in memory).
