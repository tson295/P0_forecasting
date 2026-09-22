You are starting from a completely NEW Claude Code session on a Vast.ai machine with:

* **1× NVIDIA RTX 4090 24GB**
* Ubuntu/Linux
* GitHub and Hugging Face credentials expected to already be configured

You have NO previous conversation context.

Your job is to autonomously:

1. clone and set up the repository;
2. create a new Git branch;
3. save THIS ENTIRE PROMPT verbatim into `prompt.md`;
4. download and verify the required dataset from Hugging Face;
5. create a new Hugging Face experiment repository;
6. implement the classification experiment below;
7. benchmark and maximize safe GPU concurrency;
8. run the complete initial experiment;
9. analyze results;
10. deepen the SAME classification method with controlled follow-up experiments;
11. generate reports/artifacts;
12. push code to GitHub;
13. push model artifacts/results to Hugging Face;
14. verify both destinations before declaring completion.

Do NOT merely prepare code or commands. Actually execute the work.

Do not ask for confirmation for normal implementation decisions. Inspect the existing project, make reasonable choices, record them, and continue.

---

# 0. Authentication

First verify GitHub:

```bash
ssh -T git@github.com
```

Set up Hugging Face environment:

```bash
export HF_HOME="$HOME/.hf_home"
mkdir -p "$HF_HOME"
hf auth whoami
```

Ensure all later shells/tmux jobs inherit:

```bash
export HF_HOME="$HOME/.hf_home"
```

If GitHub or Hugging Face authentication is genuinely unavailable, do not fabricate credentials. Report the exact missing authentication and stop.

Otherwise continue automatically.

---

# 1. Clone the project

Work under `$HOME`.

Repository:

```text
git@github.com:tson295/P0_forecasting.git
```

If it is not present:

```bash
cd ~
git clone git@github.com:tson295/P0_forecasting.git
cd P0_forecasting
```

If already present:

```bash
cd ~/P0_forecasting
git fetch --all --tags
```

The existing WF3 experiment used training-source commit:

```text
fdbb369946cc07eae9e615c968943036507be129
```

Verify this commit exists.

Inspect the current repository and the existing private Hugging Face repo:

```text
Tson29/Pretrain_Model_WF3
```

Read at minimum:

```text
README.md
reports/FINAL_REPORT_WF3.md
```

Use the existing WF3 implementation as the source of truth for:

* preprocessing;
* features;
* walk-forward split;
* sample eligibility;
* target timestamp lookup;
* embargo;
* E0;
* metric implementation;
* artifact conventions.

Do not accidentally base this work on an unrelated experimental branch.

---

# 2. Create a separate Git branch

This experiment must NOT modify the existing WF3 branch.

From the verified WF3 codebase create:

```bash
git checkout -b classification
```

If `classification` already exists locally/remotely, inspect it first. Continue it only if it clearly belongs to this experiment; otherwise avoid overwriting unrelated work.

All new code must be pushed to:

```text
GitHub:
tson295/P0_forecasting

Branch:
classification
```

Never force-push unless absolutely unavoidable.

---

# 3. SAVE THIS PROMPT INTO THE REPOSITORY

Immediately after cloning and switching to the `classification` branch:

Create:

```text
prompt.md
```

at the repository root.

**Copy the ENTIRE content of this prompt verbatim into `prompt.md`, from the first sentence through the final instruction.**

Do not summarize it.
Do not rewrite it.
Do not omit sections.
Do not replace it with a link.

`prompt.md` is part of experiment provenance and must be committed and pushed to GitHub with the implementation.

Do this before substantial implementation begins.

---

# 4. Dataset — download directly from Hugging Face

The dataset has already been uploaded.

Private Hugging Face DATASET repository:

```text
Tson29/btc-l10-gate-1y
```

Required file:

```text
BTC_L10_gate_1y.csv
```

The repository contains the full raw CSV, approximately 1.15 GB.

Download it directly using the authenticated Hugging Face account.

For example, use `hf download` or `huggingface_hub.hf_hub_download`.

Do NOT ask the user to upload the dataset manually.

Do NOT substitute another dataset.

After download, verify:

```text
Expected raw rows:
3,139,606
```

Expected date range:

```text
2025-09-17T00:00:00Z
→
2026-09-16T23:59:50Z
```

Expected SHA256:

```text
6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df
```

Compute SHA256 locally and require an exact match.

If the hash does not match, stop the experiment and report the mismatch.

The raw file is intentionally NOT sorted chronologically.

Reuse the existing WF3 repair procedure:

* sort by timestamp;
* duplicate timestamp policy = keep first;
* 9 duplicate rows were previously removed.

Expected usable rows after repair:

```text
3,139,597
```

Do NOT push this raw CSV to GitHub.

Do NOT upload another duplicate copy into the experiment model repository.

The raw data stays in:

```text
Tson29/btc-l10-gate-1y
```

---

# 5. Create a new Hugging Face experiment repository

Do NOT overwrite:

```text
Tson29/Pretrain_Model_WF3
```

Create a new PRIVATE Hugging Face MODEL repository:

```text
Tson29/LOB_Classification_WF3
```

If it already exists from a partial attempt, inspect it and resume safely rather than destroying valid artifacts.

This new repo will contain:

* checkpoints;
* configs;
* label definitions;
* predictions;
* metrics;
* training histories;
* reports;
* leakage audit;
* hardware/concurrency benchmarks;
* experiment manifests.

---

# 6. Existing WF3 contract — KEEP IT

Dataset:

* BTCUSDT L10;
* 10 bid levels + 10 ask levels;
* price and quantity;
* nominal 10-second grid.

Preserve:

```text
History = 490 s = 49 rows
Stride = 20 s = 2 rows
```

Horizons:

```text
h1 = 60 s
h2 = 120 s
h3 = 180 s
```

Walk-forward:

```text
3 expanding folds
```

Final held-out test:

```text
last 15% of elapsed time
```

Embargo:

```text
180 s
```

The final test set must remain identical for every:

* architecture;
* fold;
* binning method;
* multi/single formulation.

Reuse the exact existing WF3:

* history validity;
* gap handling;
* target timestamp selection;
* tolerance;
* leakage control;
* preprocessing;
* train-only normalization.

Do NOT redesign the split.

---

# 7. Motivation

Existing regression results showed:

* E0 is extremely strong.
* E0 predicts future mid = current/origin mid.
* ModernTCN improved substantially as training history increased from F1 → F2 → F3.
* ModernTCN F3 beat E0 on all h1/h2/h3 in squared-error metrics.
* OFI-LSTM showed positive test behavior particularly at h3 for F2/F3.
* h3 appears to contain clearer signal than h1/h2.
* one possible explanation is stronger microstructure noise at shorter horizons.

The new experiment tests whether direct continuous regression can be replaced by:

```text
future price displacement
→ discrete price-change class
→ classification
→ decode class back into price
```

This discretized classification formulation is the CORE METHOD.

Do not abandon it later because an initial result is bad.

---

# 8. Classification target

For every valid origin `t` and horizon `h`:

```text
delta_h(t) = target_mid(t+h) - origin_mid(t)
```

where:

```text
h ∈ {60s, 120s, 180s}
```

The target is therefore USD mid-price displacement from E0.

Do NOT implement:

* binary direction prediction;
* up/down;
* down/flat/up.

The task is multi-class prediction of a price-change interval.

---

# 9. Freeze class definitions using F1 TRAIN only

This is critical.

All class boundaries must be fitted from:

```text
FOLD 1 TRAIN ONLY
```

Then freeze them.

Exactly the same boundaries must be applied to:

```text
F1 train/validation/test
F2 train/validation/test
F3 train/validation/test
```

Never recalculate boundaries for F2/F3.

Reason:

The experiment explicitly studies whether increasing training data F1 → F2 → F3 improves learning.

If class definitions change between folds, that comparison becomes confounded.

Never use:

* validation;
* test;
* future folds

to define boundaries.

Save all boundaries/statistics as explicit JSON artifacts.

---

# 10. METHOD A — PRIMARY: equal-width p99 classification

Using F1 TRAIN only, calculate separately:

```text
q_60  = p99(abs(delta_60))
q_120 = p99(abs(delta_120))
q_180 = p99(abs(delta_180))
```

For horizon `h`, the main finite classification range is:

```text
[-q_h, +q_h]
```

Split this into equal-width intervals.

Use the SAME number of finite bins for all horizons.

The actual USD bin width may differ because q60/q120/q180 differ.

Choose a reasonable moderate initial number of bins after inspecting the F1-train distributions.

Use an EVEN number of finite bins so:

```text
0 USD
```

is exactly a boundary.

No ordinary bin should cross zero.

Example structure:

```text
delta < -q_h          lower overflow

[-q_h, ...)
...
[-w, 0)
[0, w)
...
(..., +q_h]

delta > +q_h          upper overflow
```

There are:

```text
K finite bins + 2 overflow classes
```

### Meaning of overflow

p99 intentionally covers the central ~99% absolute movement region.

Movements outside it must NOT be clipped into the edge bins.

Instead:

```text
delta < -q_h
```

gets the lower overflow class.

And:

```text
delta > +q_h
```

gets the upper overflow class.

For every horizon save:

* q_h;
* finite min/max;
* K;
* bin width;
* exact edges;
* number of samples per class;
* percentage in lower overflow;
* percentage in upper overflow.

Do not automatically modify the method simply because class frequencies are imbalanced.

Initial training uses ordinary CrossEntropy.

No class weighting initially.

---

# 11. METHOD B — quantile classification

Also implement a second required binning method:

```text
quantile bins
```

Fit quantile boundaries using F1 TRAIN ONLY.

Freeze them for all folds.

Use approximately the same total number of classes as Method A so comparisons are meaningful.

Do not treat this as the primary formulation.

It is a required comparison/ablation.

Primary method remains:

```text
equal-width ±p99
```

---

# 12. Models

Run exactly:

```text
1. OFI-LSTM
2. ModernTCN
3. Custom Transformer
```

## OFI-LSTM

Reuse the existing WF3 implementation:

* existing input features;
* existing normalization;
* existing backbone architecture/config as closely as possible.

Change the output formulation from regression to classification.

Do not unnecessarily redesign it.

## ModernTCN

Reuse:

* existing WF3 features;
* preprocessing;
* backbone/config.

Replace regression output with classification head(s).

Do not change the core ModernTCN architecture unless required to support classification output.

## Custom Transformer

Add a conventional Transformer encoder baseline:

```text
input sequence
→ Linear projection
→ positional encoding
→ TransformerEncoder
→ sequence representation
→ classification head(s)
```

Keep it simple.

Do not add:

* FlashAttention experiments;
* RoPE experiments;
* pretrained foundation models;
* unrelated architectural novelty.

Aim for a sensible few-million-parameter model.

Record exact parameter count.

---

# 13. STAGE 1 — multi-horizon classification

RUN THIS FIRST.

Each architecture uses ONE shared backbone and three output heads:

```text
Backbone
├── h1 head: 60s
├── h2 head: 120s
└── h3 head: 180s
```

Each head outputs class logits.

If a true displacement lies inside interval:

```text
[a, b)
```

its true class is exactly that interval.

Example:

```text
[20,30) = class 4
[30,40) = class 5

delta = 36.7
→ class 5
```

This is deterministic interval assignment.

It is NOT a nearest-neighbor heuristic.

### Loss

For each horizon use standard multi-class CrossEntropy:

```text
CE_60
CE_120
CE_180
```

Overall multi-horizon loss:

```text
Loss = (CE_60 + CE_120 + CE_180) / 3
```

For the initial benchmark do NOT add:

* focal loss;
* class weighting;
* ordinal loss;
* distance-aware loss;
* auxiliary MSE;
* auxiliary regression target.

Keep the first comparison clean.

---

# 14. STAGE 2 — single-horizon classification

Only after the complete multi-horizon stage is evaluated, run independent models:

```text
model_h1 → 60s only
model_h2 → 120s only
model_h3 → 180s only
```

Do this for all three architectures and both binning methods.

The purpose is to test whether:

* shared representation helps;
* or noisy h1/h2 hurt h3.

Keep all settings as comparable as possible with the multi-horizon run.

---

# 15. Decode classification back into price

Primary inference is:

```text
predicted_class = argmax(logits)
```

For an ordinary finite interval:

```text
[a,b)
```

decode its displacement using midpoint:

```text
decoded_delta = (a+b)/2
```

Then:

```text
pred_mid = origin_mid + decoded_delta
```

### Equal-width overflow decoding

Use:

```text
lower overflow:
-q_h - width_h/2

upper overflow:
+q_h + width_h/2
```

and document it.

### Quantile outer-class decoding

Use a deterministic representative derived from F1 TRAIN only.

Never derive decoding representatives from validation/test.

### Probability diagnostic

Softmax probabilities will naturally be available.

You MAY also compute:

```text
expected_delta = Σ p_k * class_representative_k
```

as a secondary diagnostic.

However:

```text
argmax class → representative
```

remains the official primary classification decoding.

Do not replace the main method silently.

---

# 16. EXACTLY FOUR official metrics

The official benchmark contains ONLY:

```text
RMSE
MAE
R² gain vs E0
Directional Accuracy (DA)
```

Do NOT add official:

* classification accuracy;
* F1;
* balanced accuracy;
* bin accuracy;
* top-k accuracy;
* ±1-bin accuracy;
* bin-distance error.

### RMSE

Evaluate decoded future MID PRICE.

### MAE

Evaluate decoded future MID PRICE.

### R² gain vs E0

Reuse the existing WF3 definition:

```text
R2_gain_vs_E0
=
1 - SSE_model / SSE_E0
```

E0:

```text
pred_mid = origin_mid
```

Interpretation:

```text
positive → beats E0
zero     → equals E0
negative → worse than E0
```

### Directional Accuracy

Use:

```text
pred_delta = pred_mid - origin_mid
true_delta = target_mid - origin_mid
```

DA measures sign agreement.

Define exact-zero handling once, document it, and use the same convention for every run.

DA is only an evaluation metric.

Do NOT change the task into direction classification.

---

# 17. Initial experiment matrix

The required initial experiment contains:

```text
2 binning methods
×
3 architectures
×
3 folds
×
multi + single
```

Multi-horizon:

```text
2 × 3 × 3 = 18 trained models
```

Single-horizon:

```text
2 × 3 × 3 folds × 3 horizons
= 54 trained models
```

Total initial trained jobs:

```text
72
```

Do not silently skip runs.

Maintain a machine-readable run registry containing:

* planned;
* running;
* completed;
* failed;
* retried.

---

# 18. Optimize execution on RTX 4090 24GB

Do NOT simply run 72 jobs serially.

Before the full sweep, benchmark representative short runs for:

```text
OFI-LSTM
ModernTCN
Transformer
```

Record:

* VRAM allocated;
* VRAM reserved;
* GPU utilization;
* samples/sec;
* step time;
* CPU utilization if relevant.

Then experimentally compare safe concurrency such as:

```text
1 concurrent job
2 concurrent jobs
3 concurrent jobs
4 concurrent jobs
```

when possible.

Measure TOTAL throughput.

Do not maximize concurrency merely for its own sake.

A configuration where 4 jobs each become extremely slow is not better than 2 efficient jobs.

Prefer mixed workloads when useful, e.g.:

```text
ModernTCN
+
Transformer
+
lightweight OFI-LSTM
```

if that gives better GPU utilization and aggregate throughput.

Monitor:

* GPU compute;
* VRAM;
* CPU;
* RAM;
* disk I/O;
* dataloader contention.

### If OOM happens

Adjustment priority:

```text
1. reduce concurrent jobs
2. adjust batch size if necessary
3. only then consider implementation-level memory optimizations
```

Do NOT arbitrarily shrink the model architecture simply because several concurrent jobs do not fit.

Save:

```text
concurrency_benchmark.json
training_schedule.json
```

or equivalent machine-readable artifacts.

---

# 19. Use tmux

Long-running execution must use tmux.

Use a clearly named session such as:

```bash
tmux new -s classification
```

or detached windows/jobs.

Requirements:

* persistent logs;
* independent job status;
* recoverability;
* failed jobs rerunnable;
* automatic transition from training → evaluation → report → upload.

Do not leave the workflow after training with evaluation/upload unfinished.

---

# 20. Leakage audit

Extend the existing WF3 leakage audit.

Verify explicitly:

1. chronological split unchanged;
2. test split identical for all experiments;
3. no history window crosses forbidden split boundaries;
4. targets remain isolated correctly;
5. train-only normalization remains train-only;
6. p99 is fitted ONLY on F1 TRAIN;
7. quantiles are fitted ONLY on F1 TRAIN;
8. class boundaries never use validation/test;
9. class representatives never use validation/test;
10. all folds use identical frozen class definitions;
11. prediction/test sample fingerprints match across models where expected.

If leakage audit fails, do NOT treat the result as valid.

Save the audit report.

---

# 21. Artifacts per run

Persist enough to reproduce each result.

At minimum:

```text
config
model config
feature schema
preprocessing metadata
label definition
split manifest
environment
git commit
training history
best checkpoint
validation metrics
test metrics
test predictions
run summary
```

Prediction CSVs must contain enough information to independently recompute:

```text
RMSE
MAE
R² gain vs E0
DA
```

---

# 22. Analyze the complete initial sweep

Do NOT start arbitrary tuning before the required baseline matrix is complete.

After all initial runs finish, answer:

1. Does discretized classification beat E0?
2. Does it improve over the previous regression behavior?
3. Equal-width or quantile: which works better?
4. Does F1 → F2 → F3 improve as training data increases?
5. Is h3 stronger than h1/h2?
6. Does single-horizon improve h3?
7. Does multi-horizon sharing help any horizon?
8. Which architecture benefits most?
9. Does DA agree with RMSE/R² gain?
10. Which positive results repeat across folds instead of appearing as one isolated cell?

Do not overclaim extremely small improvements.

---

# 23. REQUIRED follow-up: deepen THIS SAME method

The work does NOT stop after the initial 72 jobs.

Whether the results are strong or weak, continue investigating the SAME classification formulation.

The core must remain:

```text
continuous future price displacement
→ fixed discretization
→ multi-class prediction
→ decode to price
```

Do NOT abandon this and invent another task.

Allowed follow-up directions include controlled tests of:

```text
number of bins
p99 range sensitivity
p98 / p99 / p99.5 range
bin width
overflow representation
argmax vs expected-value decoding
multi vs single horizon
shared vs separate heads
capacity changes inside the SAME three architectures
```

Every follow-up experiment must follow:

```text
Hypothesis
→ ONE controlled change
→ Result
→ Conclusion
```

Do not change multiple major factors simultaneously.

### If initial results are bad

Diagnose why THIS classification method fails and modify it carefully.

Do NOT switch to:

* another unrelated forecasting task;
* RL;
* diffusion;
* arbitrary ensembles;
* unrelated new architectures;
* binary direction prediction.

### If initial results are good

Identify what part of the current formulation causes the improvement and refine it systematically.

The core research contribution must remain this discretized classification approach.

---

# 24. Reporting

Produce a final report with sections:

```text
1. Experiment contract
2. Dataset provenance
3. Git/HF provenance
4. Leakage audit
5. Label-distribution analysis
6. Equal-width p99 definition
7. Quantile definition
8. Model architectures / parameter counts
9. GPU benchmark / concurrency strategy
10. Multi-horizon results
11. Single-horizon results
12. F1 → F2 → F3 scaling
13. h1 vs h2 vs h3
14. Equal-width vs quantile
15. Follow-up experiments
16. Interpretation
17. Limitations
18. Final conclusions
```

Official benchmark tables must show ONLY:

```text
RMSE
MAE
R² gain vs E0
DA
```

Make comparisons easy across:

* fold;
* horizon;
* architecture;
* binning method;
* multi vs single.

---

# 25. Hugging Face artifact structure

Push to:

```text
Tson29/LOB_Classification_WF3
```

Use a structured layout similar to:

```text
equal_width/
    multi/
        ofi_lstm/
        moderntcn/
        transformer/
    single/
        ofi_lstm/
        moderntcn/
        transformer/

quantile/
    multi/
        ofi_lstm/
        moderntcn/
        transformer/
    single/
        ofi_lstm/
        moderntcn/
        transformer/

labeling/
reports/
audits/
benchmarks/
```

Upload:

* best checkpoints;
* configs;
* label boundaries;
* metrics;
* test predictions;
* histories;
* reports;
* leakage audit;
* run registry;
* concurrency benchmark;
* environment/provenance.

Do NOT duplicate the raw 1.15GB source CSV here.

Dataset source remains:

```text
Tson29/btc-l10-gate-1y
```

After uploading, READ BACK representative files from Hugging Face to verify the upload is actually usable.

---

# 26. GitHub publication

Commit:

* `prompt.md`;
* source code;
* configs;
* orchestration scripts;
* evaluation scripts;
* audit code;
* reports;
* lightweight experiment metadata.

Push to:

```text
origin/classification
```

Do NOT push:

* raw BTC CSV;
* huge checkpoints already stored on Hugging Face.

Do not unnecessarily git-ignore useful reports/configs/metadata.

---

# 27. Final verification

Do not declare completion before all of the following are checked.

## Dataset

Verify:

```text
Tson29/btc-l10-gate-1y
```

was successfully downloaded.

Verify local:

```text
BTC_L10_gate_1y.csv
```

matches SHA256:

```text
6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df
```

## GitHub

Verify:

* remote branch `classification` exists;
* `prompt.md` exists on that branch;
* final code commit is pushed;
* important changes are not left uncommitted.

## Hugging Face

Verify:

```text
Tson29/LOB_Classification_WF3
```

exists and representative:

* reports;
* metrics;
* predictions;
* checkpoints;
* labels;
* audits

can be read back.

## Experiment

Report:

* total planned baseline jobs;
* successful jobs;
* failed jobs;
* retried jobs;
* follow-up jobs;
* leakage audit status.

---

# 28. Final response

At the end report concisely:

1. final Git commit hash;
2. GitHub branch;
3. Hugging Face dataset source;
4. Hugging Face experiment repo;
5. p99 values for h1/h2/h3;
6. chosen initial bin count and widths;
7. best multi-horizon results;
8. best single-horizon results;
9. equal-width vs quantile conclusion;
10. OFI-LSTM vs ModernTCN vs Transformer;
11. F1 → F2 → F3 scaling conclusion;
12. h1/h2/h3 conclusion;
13. follow-up experiments performed;
14. leakage audit result;
15. any remaining limitations.

Remember:

**The core of this work is discretized multi-class forecasting of future price displacement.**

Good initial results should lead to deeper investigation of this same method.

Bad initial results should also lead to diagnosis and controlled refinement of this same method.

Do not independently abandon or replace the research direction.

And ensure this entire prompt itself is preserved verbatim in:

```text
prompt.md
```

and committed to the `classification` Git branch.
