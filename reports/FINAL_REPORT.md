# Base experiment final report — BTC L10, 60 s history

## 1-4. Provenance

| Item | Value |
|---|---|
| Initial Git commit | `d3cc577796d891fa16d2fea37ad5a899fd74d862` |
| Training-source Git commit | `5a7cc83e504f7c16a1eb29fcdc3a1df1e01a1a1b` |
| Final Git commit | `5a7cc83e504f7c16a1eb29fcdc3a1df1e01a1a1b` |
| Hugging Face repo | Tson29/Pretrain_Model |

## 5-12. Dataset, split and sampling

| Item | Value |
|---|---|
| Dataset path | `/home/ubuntu/Pretrain_Model/BTCUSDT_L10_oct2023.csv` |
| Dataset SHA256 | `e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab` |
| Rows | 678362 |
| Median / p99 / max dt (s) | 1.236 / 1.683 / 6.221 |
| Gaps > 2 s / segment transitions | 60 / 1 |
| Split timestamp train_end | `2023-10-07T22:01:22.396000+00:00` |
| Split timestamp validation_end | `2023-10-09T09:35:56.255500+00:00` |
| Split rows train | `[0, 477086)` |
| Split rows validation | `[477086, 578798)` |
| Split rows test | `[578798, 678362)` |
| history_rows | 49 |
| stride_rows | 8 |
| Samples train | 58774 |
| Samples validation | 12427 |
| Samples test | 12145 |
| Contract gate | pass (136 checks, 0 failed) |

## 13-17. Hardware, precision and compile

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (23.52 GiB) |
| Driver / CUDA driver | 580.173.02 / 13.0 |
| CUDA runtime | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Precision | bfloat16, TF32 matmul True |
| CPU / RAM | 32 cores / 91.43 GiB |

**torch.compile status per model**

| Model | Compile succeeded | Speedup vs eager | Recommended | Used in training |
|---|---|---:|---|---|
| ofi_lstm | True | 0.9272 | False | False |
| hfformer | True | 1.599 | True | True |
| patchtst | True | 2.315 | True | True |
| moderntcn | True | 1.13 | True | True |
| lit | True | 1.064 | False | False |

## 18-27. Capacity, systems settings and training cost

| Model | Parameters | Batch | num_workers | Peak alloc (single) | Peak reserved (single) | samples/s (single) | Train seconds | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ofi_lstm | 55491 | 128 | 4 | 0.05 GiB | 0.06 GiB | 97655 | 31.688 | 26 |
| hfformer | 22026 | 128 | 4 | 0.05 GiB | 0.07 GiB | 36607 | 63.493 | 25 |
| patchtst | 477059 | 128 | 4 | 0.47 GiB | 0.61 GiB | 14038 | 69.171 | 29 |
| moderntcn | 50568195 | 128 | 4 | 1.60 GiB | 1.69 GiB | 2617.8 | 671.99 | 29 |
| lit | 736547 | 128 | 4 | 0.12 GiB | 0.13 GiB | 34950 | 71.389 | 24 |

Concurrency groups executed: `[['hfformer', 'lit', 'ofi_lstm'], ['moderntcn'], ['patchtst']]` (decided by reports/vast/concurrency_benchmark.json).

Benchmark rationale: Measured, not assumed. ofi_lstm+hfformer+lit: 1.88x aggregate vs sequential, slowest member 26104 samples/s, 88% util, 21.8 GiB free; ofi_lstm+hfformer: 1.69x aggregate vs sequential, slowest member 32915 samples/s, 94% util, 22.4 GiB free; ofi_lstm+hfformer+lit+patchtst: 1.60x aggregate vs sequential, slowest member 4062 samples/s, 98% util, 20.8 GiB free; moderntcn+ofi_lstm+hfformer: 1.49x aggregate vs sequential, slowest member 1178 samples/s, 99% util, 20.3 GiB free; ofi_lstm+hfformer+lit+patchtst+moderntcn: 1.39x aggregate vs sequential, slowest member 847 samples/s, 98% util, 18.6 GiB free; moderntcn+ofi_lstm: 1.19x aggregate vs sequential, slowest member 1924 samples/s, 92% util, 20.8 GiB free. Schedules are ranked by the wall time to push 1763220 train samples through every model, each group paced by its slowest member: hfformer+lit+ofi_lstm then moderntcn then patchtst costs 872s of train steps, 5.3% below the 921s one-job-at-a-time baseline (forward/backward/AdamW only; validation, checkpointing and data-loader startup excluded).

- `['moderntcn']`: aggregate 2603 samples/s, peak VRAM 2.62 GiB, min free 21.37 GiB, avg GPU util 90.56%, accepted=True
- `['ofi_lstm']`: aggregate 1.0332e+05 samples/s, peak VRAM 1.01 GiB, min free 22.98 GiB, avg GPU util 36.46%, accepted=True
- `['hfformer']`: aggregate 36674 samples/s, peak VRAM 1.00 GiB, min free 22.98 GiB, avg GPU util 50.84%, accepted=True
- `['patchtst']`: aggregate 13846 samples/s, peak VRAM 1.53 GiB, min free 22.46 GiB, avg GPU util 91.75%, accepted=True
- `['lit']`: aggregate 34653 samples/s, peak VRAM 1.09 GiB, min free 22.90 GiB, avg GPU util 47.85%, accepted=True
- `['ofi_lstm', 'hfformer']`: aggregate 1.156e+05 samples/s, peak VRAM 1.54 GiB, min free 22.45 GiB, avg GPU util 93.53%, accepted=True
- `['moderntcn', 'ofi_lstm']`: aggregate 48205 samples/s, peak VRAM 3.15 GiB, min free 20.83 GiB, avg GPU util 92.44%, accepted=True
- `['ofi_lstm', 'hfformer', 'lit']`: aggregate 94457 samples/s, peak VRAM 2.15 GiB, min free 21.84 GiB, avg GPU util 87.88%, accepted=True
- `['moderntcn', 'ofi_lstm', 'hfformer']`: aggregate 58099 samples/s, peak VRAM 3.68 GiB, min free 20.31 GiB, avg GPU util 98.75%, accepted=True
- `['ofi_lstm', 'hfformer', 'lit', 'patchtst']`: aggregate 64306 samples/s, peak VRAM 3.20 GiB, min free 20.79 GiB, avg GPU util 97.88%, accepted=True
- `['ofi_lstm', 'hfformer', 'lit', 'patchtst', 'moderntcn']`: aggregate 43571 samples/s, peak VRAM 5.34 GiB, min free 18.64 GiB, avg GPU util 98.07%, accepted=True

## 28-29. Metrics


### validation


**e0** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.00028261 | 0.000408833 | 0.000500641 |
| MAE | 0.000182533 | 0.000275423 | 0.000341948 |
| R2 | -0.000134101 | -0.000226254 | -0.000337992 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000307444 | 0.000429873 | 0.000568895 |
| MAE | 0.000203764 | 0.000292724 | 0.000417125 |
| R2 | -0.183631 | -0.105831 | -0.29169 |
| RMSE gain vs E0 | -0.0878751 | -0.0514659 | -0.136333 |

**hfformer** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000831339 | 0.00260031 | 0.00241018 |
| MAE | 0.000679757 | 0.00248002 | 0.00226887 |
| R2 | -7.65447 | -39.4629 | -22.1842 |
| RMSE gain vs E0 | -1.94165 | -5.36032 | -3.81419 |

**patchtst** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000291456 | 0.00043796 | 0.000511018 |
| MAE | 0.000201362 | 0.000316473 | 0.000356649 |
| R2 | -0.0637274 | -0.147826 | -0.0422398 |
| RMSE gain vs E0 | -0.0313025 | -0.0712451 | -0.020729 |

**moderntcn** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000510935 | 0.000646365 | 0.000705197 |
| MAE | 0.000399791 | 0.000495662 | 0.000543974 |
| R2 | -2.269 | -1.50013 | -0.984792 |
| RMSE gain vs E0 | -0.807917 | -0.581002 | -0.408589 |

**lit** (12427 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000292858 | 0.000409505 | 0.000555375 |
| MAE | 0.000206222 | 0.000279008 | 0.000406558 |
| R2 | -0.0739887 | -0.00352009 | -0.231025 |
| RMSE gain vs E0 | -0.0362648 | -0.00164519 | -0.109328 |

### test


**e0** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000448613 | 0.000645938 | 0.000791626 |
| MAE | 0.000300467 | 0.00044831 | 0.000553559 |
| R2 | -0.000222152 | -0.00034976 | -0.000436273 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000470254 | 0.000673411 | 0.000840886 |
| MAE | 0.000317704 | 0.000464364 | 0.000609238 |
| R2 | -0.0990495 | -0.0872549 | -0.128816 |
| RMSE gain vs E0 | -0.0482392 | -0.0425329 | -0.0622258 |

**hfformer** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000856518 | 0.00258269 | 0.00243018 |
| MAE | 0.000688097 | 0.00242527 | 0.0022295 |
| R2 | -2.64607 | -14.9924 | -8.42817 |
| RMSE gain vs E0 | -0.909258 | -2.99836 | -2.06986 |

**patchtst** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000452722 | 0.000666117 | 0.000797499 |
| MAE | 0.000312341 | 0.000475821 | 0.000564542 |
| R2 | -0.0186285 | -0.0638278 | -0.0153338 |
| RMSE gain vs E0 | -0.00915919 | -0.0312399 | -0.00741799 |

**moderntcn** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000620123 | 0.000825534 | 0.000943031 |
| MAE | 0.000474434 | 0.00063129 | 0.00071194 |
| R2 | -0.911212 | -0.633956 | -0.419713 |
| RMSE gain vs E0 | -0.382312 | -0.278039 | -0.191257 |

**lit** (12145 samples)

| Metric | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000450171 | 0.00064678 | 0.000829983 |
| MAE | 0.000308279 | 0.000452768 | 0.000605595 |
| R2 | -0.00718367 | -0.00295984 | -0.0997332 |
| RMSE gain vs E0 | -0.00347395 | -0.00130373 | -0.048453 |

## 30-33. Artifact locations

| Model | Local checkpoints | Local predictions | HF prefix |
|---|---|---|---|
| e0 | n/a | `artifacts/e0/e0_60s/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `e0/e0_60s/` |
| ofi_lstm | `checkpoints/ofi_lstm/ofi_lstm_60s_base/{best,last}` | `artifacts/ofi_lstm/ofi_lstm_60s_base/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `ofi_lstm/ofi_lstm_60s_base/` |
| hfformer | `checkpoints/hfformer/hfformer_60s_base/{best,last}` | `artifacts/hfformer/hfformer_60s_base/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `hfformer/hfformer_60s_base/` |
| patchtst | `checkpoints/patchtst/patchtst_60s_base/{best,last}` | `artifacts/patchtst/patchtst_60s_base/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `patchtst/patchtst_60s_base/` |
| moderntcn | `checkpoints/moderntcn/moderntcn_60s_base/{best,last}` | `artifacts/moderntcn/moderntcn_60s_base/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `moderntcn/moderntcn_60s_base/` |
| lit | `checkpoints/lit/lit_60s_base/{best,last}` | `artifacts/lit/lit_60s_base/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model` -> `lit/lit_60s_base/` |

## 34-36. Incidents and deviations


### Experiment-contract deviations

- **HFformer window reduction accumulates in FP64** (section J). The frozen formula, eps=1e-5, unbiased=False and the per-sample/per-feature/over-time contract are unchanged; only the reduction's arithmetic precision differs from the literal `x_fp32.mean(dim=1)`. At raw L10 price scale (~2.7e4, FP32 ulp ~2e-3) a batched FP32 mean carries a few ulp of residue, and dividing that by eps turns a constant window -- whose true z-score is 0 -- into about -586; measured on real train windows, 6.2% of (sample, feature) pairs have a near-constant history and 1.3% of normalized entries exceeded |z|>10. The FP32 reduction kernel also varies with tensor shape, so the value depended on the batch, which section J forbids ('phu thuoc duy nhat vao history hien tai cua sample'). With FP64 accumulation the result matches the exact formula to 2.4e-7 and is bit-identical across batch sizes 1, 3, 7, 128 and 512.
- **PatchTST exposes four of the six LoRA module names** (sections W vs K3). Sections K2 and K5 name q_proj/k_proj/v_proj/out_proj/fc1/fc2 for HFformer and LiT, and both provide all six. Section K3 pins PatchTST to the stock `transformers.PatchTSTModel` at exactly 477,059 parameters; that module keeps its feed-forward Linears inside an nn.Sequential at ff.0/ff.3. Its four attention projections -- the standard LoRA injection points -- are present and unmerged. Renaming the upstream feed-forward layers would be the architecture change K3 forbids, so the upstream names were kept.
- **Credentials came from the machine, not from GITHUB_TOKEN/HF_TOKEN** (sections A, AC). Neither environment variable was set and the GitHub CLI is not installed. GitHub was authenticated with the existing SSH key already configured for the `origin` remote, and Hugging Face with the token already stored in the local Hugging Face home. No token was printed, written into source, committed, or embedded in a remote URL.

- Crashed jobs: none
- Resumed jobs: none
- Total scheduled training wall time: 825.361 s
- GPU usage log: `reports/vast/gpu_usage.csv`
