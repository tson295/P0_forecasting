# Gate experiment final report — BTC L10 1 year at 10 s, 490 s history

## 1-4. Provenance

| Item | Value |
|---|---|
| Initial Git commit | `2d046baf6ba58b630b3182edc8e7a9ad230ae57e` |
| Training-source Git commit | `5d924df374c87406f75b501766552ff271313357` |
| Final Git commit | `128af48aad51c5719d05e93d82e9c89682d0071d` |
| Hugging Face repo | Tson29/Pretrain_Model_Gate_1Y |

## 5-12. Dataset, split and sampling

| Item | Value |
|---|---|
| Dataset path | `/home/ubuntu/Pretrain_Model/BTC_L10_gate_1y.csv` |
| Dataset SHA256 | `6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df` |
| Rows | 3139597 |
| Median / p99 / max dt (s) | 10.0 / 10.0 / 7210.0 |
| Gaps > 10 s / segment transitions | 31 / 0 |
| Split timestamp train_end | `2026-05-30T11:59:53+00:00` |
| Split timestamp validation_end | `2026-07-24T05:59:51.500000+00:00` |
| Split rows train | `[0, 2195317)` |
| Split rows validation | `[2195317, 2667637)` |
| Split rows test | `[2667637, 3139597)` |
| history_rows | 49 |
| stride_rows | 8 |
| Samples train | 274215 |
| Samples validation | 59016 |
| Samples test | 58963 |
| Contract gate | pass on `BTC_L10_gate_1y.csv` (136 checks, 0 failed) |

### Data repair

| Item | Value |
|---|---|
| rows_in_file | 3139606 |
| rows_used | 3139597 |
| sorted_by_timestamp | True |
| rows_moved_by_sort | 2531566 |
| duplicate_timestamp_policy | keep_first |
| duplicate_rows_dropped | 9 |

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
| ofi_lstm | True | 0.9861 | False | False |
| hfformer | True | 1.65 | True | True |
| patchtst | True | 2.295 | True | True |
| moderntcn | True | 1.129 | True | True |
| lit | True | 1.097 | False | False |

## 18-27. Capacity, systems settings and training cost

| Model | Parameters | Batch | num_workers | Peak alloc (single) | Peak reserved (single) | samples/s (single) | Train seconds | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ofi_lstm | 55491 | 128 | 4 | 0.05 GiB | 0.06 GiB | 1.0608e+05 | 158.42 | 16 |
| hfformer | 22026 | 128 | 4 | 0.05 GiB | 0.06 GiB | 37017 | 254.04 | 24 |
| patchtst | 477059 | 128 | 4 | 0.47 GiB | 0.59 GiB | 14065 | 311.59 | 27 |
| moderntcn | 50568195 | 128 | 4 | 1.60 GiB | 1.69 GiB | 2627.5 | 3048 | 29 |
| lit | 736547 | 128 | 4 | 0.12 GiB | 0.15 GiB | 35542 | 329.03 | 13 |

Concurrency groups executed: `[['hfformer', 'lit', 'ofi_lstm'], ['moderntcn'], ['patchtst']]` (decided by reports/vast_gate1y/concurrency_benchmark.json).

Benchmark rationale: Measured, not assumed. ofi_lstm+hfformer+lit: 1.85x aggregate vs sequential, slowest member 26143 samples/s, 89% util, 21.8 GiB free; ofi_lstm+hfformer: 1.68x aggregate vs sequential, slowest member 33445 samples/s, 89% util, 22.4 GiB free; ofi_lstm+hfformer+lit+patchtst: 1.59x aggregate vs sequential, slowest member 4020 samples/s, 90% util, 20.8 GiB free; ofi_lstm+hfformer+lit+patchtst+moderntcn: 1.36x aggregate vs sequential, slowest member 850 samples/s, 98% util, 18.6 GiB free. Schedules are ranked by the wall time to push 8226450 train samples through every model, each group paced by its slowest member: hfformer+lit+ofi_lstm then moderntcn then patchtst costs 4045s of train steps, 5.2% below the 4267s one-job-at-a-time baseline (forward/backward/AdamW only; validation, checkpointing and data-loader startup excluded).

- `['moderntcn']`: aggregate 2618.8 samples/s, peak VRAM 2.62 GiB, min free 21.37 GiB, avg GPU util 97.6%, accepted=True
- `['ofi_lstm']`: aggregate 1.0503e+05 samples/s, peak VRAM 1.01 GiB, min free 22.98 GiB, avg GPU util 35%, accepted=True
- `['hfformer']`: aggregate 36781 samples/s, peak VRAM 1.00 GiB, min free 22.98 GiB, avg GPU util 47.48%, accepted=True
- `['patchtst']`: aggregate 13964 samples/s, peak VRAM 1.53 GiB, min free 22.46 GiB, avg GPU util 89.42%, accepted=True
- `['lit']`: aggregate 35000 samples/s, peak VRAM 1.09 GiB, min free 22.90 GiB, avg GPU util 53.56%, accepted=True
- `['ofi_lstm', 'hfformer']`: aggregate 1.1463e+05 samples/s, peak VRAM 1.54 GiB, min free 22.45 GiB, avg GPU util 89.01%, accepted=True
- `['moderntcn', 'ofi_lstm']`: aggregate 48058 samples/s, peak VRAM 3.15 GiB, min free 20.83 GiB, avg GPU util 95.85%, accepted=False
- `['ofi_lstm', 'hfformer', 'lit']`: aggregate 92168 samples/s, peak VRAM 2.15 GiB, min free 21.84 GiB, avg GPU util 89.4%, accepted=True
- `['moderntcn', 'ofi_lstm', 'hfformer']`: aggregate 58472 samples/s, peak VRAM 3.68 GiB, min free 20.31 GiB, avg GPU util 94.37%, accepted=False
- `['ofi_lstm', 'hfformer', 'lit', 'patchtst']`: aggregate 64305 samples/s, peak VRAM 3.20 GiB, min free 20.79 GiB, avg GPU util 89.53%, accepted=True
- `['ofi_lstm', 'hfformer', 'lit', 'patchtst', 'moderntcn']`: aggregate 43115 samples/s, peak VRAM 5.34 GiB, min free 18.64 GiB, avg GPU util 98.11%, accepted=True

## 28-29. Metrics


**Metric space.** Training is unchanged: the target is still `log(mid[t+h]/mid[t])` and the loss is still MSE on that log return. Only the reported space changed -- the same predictions are measured on the mid price via `pred_mid = origin_mid*exp(pred_return)`, so RMSE and MAE below are in quote currency (USD). E0 predicts that the price does not move, so its predicted mid is the origin mid and `RMSE_E0 = sqrt(mean((target_mid-origin_mid)^2))`. Standard R2 is near 1 for every model including E0, because sigma(target mid) is about 16,579 USD while the E0 error measured on this file is 50.4 / 71.7 / 87.9 USD at 1m/2m/3m; it is reported for contract completeness only, and **RMSE gain vs E0** is the column to read. Each price table is followed by the same four families in log-return units, so both spaces stay comparable.


### validation


**e0** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.5071 | 55.1287 | 67.1092 |
| MAE (USD) | 24.4286 | 35.4803 | 43.6322 |
| R2 | 0.999825 | 0.999641 | 0.999468 |
| RMSE gain vs E0 | 1.88738e-15 | -8.88178e-15 | -8.43769e-15 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000612192 | 0.000876277 | 0.00106636 |
| MAE | 0.000385979 | 0.00056043 | 0.00068897 |
| R2 | -3.89957e-06 | -2.54647e-05 | -1.88063e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.5339 | 55.1925 | 68.0446 |
| MAE (USD) | 24.5513 | 35.5778 | 44.9885 |
| R2 | 0.999825 | 0.99964 | 0.999453 |
| RMSE gain vs E0 | -0.000696071 | -0.00115814 | -0.013939 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.00061267 | 0.00087735 | 0.00108111 |
| MAE | 0.000387932 | 0.000561989 | 0.000710268 |
| R2 | -0.00156627 | -0.00247491 | -0.0278655 |
| RMSE gain vs E0 | -0.000780876 | -0.00122394 | -0.0138275 |

**hfformer** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.9379 | 61.1648 | 69.0206 |
| MAE (USD) | 25.2447 | 43.7817 | 46.3618 |
| R2 | 0.999821 | 0.999558 | 0.999438 |
| RMSE gain vs E0 | -0.0111865 | -0.109491 | -0.0284813 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000618882 | 0.000969304 | 0.00109558 |
| MAE | 0.000398726 | 0.000689849 | 0.000731426 |
| R2 | -0.0219798 | -0.223625 | -0.0555609 |
| RMSE gain vs E0 | -0.0109282 | -0.106162 | -0.0273953 |

**patchtst** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.505 | 55.1329 | 67.1122 |
| MAE (USD) | 24.4616 | 35.501 | 43.6415 |
| R2 | 0.999825 | 0.999641 | 0.999468 |
| RMSE gain vs E0 | 5.42803e-05 | -7.70433e-05 | -4.43781e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000612163 | 0.000876348 | 0.00106642 |
| MAE | 0.000386497 | 0.000560752 | 0.000689116 |
| R2 | 9.18742e-05 | -0.000187317 | -0.00012829 |
| RMSE gain vs E0 | 4.78879e-05 | -8.09209e-05 | -5.47394e-05 |

**moderntcn** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.6045 | 55.2513 | 67.2444 |
| MAE (USD) | 24.6773 | 35.704 | 43.8621 |
| R2 | 0.999824 | 0.99964 | 0.999466 |
| RMSE gain vs E0 | -0.00252987 | -0.00222375 | -0.00201419 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.00061374 | 0.000878231 | 0.00106857 |
| MAE | 0.000389876 | 0.000563948 | 0.0006926 |
| R2 | -0.00506722 | -0.00449065 | -0.00415862 |
| RMSE gain vs E0 | -0.00252845 | -0.00223005 | -0.00206773 |

**lit** (59016 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.5247 | 55.1351 | 67.1241 |
| MAE (USD) | 24.504 | 35.5075 | 43.6644 |
| R2 | 0.999825 | 0.999641 | 0.999468 |
| RMSE gain vs E0 | -0.0004566 | -0.000115956 | -0.000222234 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000612486 | 0.000876397 | 0.00106664 |
| MAE | 0.000387163 | 0.000560864 | 0.000689489 |
| R2 | -0.000965225 | -0.000298423 | -0.000534986 |
| RMSE gain vs E0 | -0.000480545 | -0.000136466 | -0.000258052 |

### test


**e0** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.3957 | 51.173 | 62.5648 |
| MAE (USD) | 20.7301 | 30.2982 | 37.4449 |
| R2 | 0.999974 | 0.999948 | 0.999922 |
| RMSE gain vs E0 | 4.88498e-15 | 1.44329e-15 | -1.13243e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000491153 | 0.000689845 | 0.000844683 |
| MAE | 0.000283889 | 0.000415799 | 0.000514424 |
| R2 | -2.59098e-05 | -2.47865e-05 | -3.95718e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.3748 | 51.1825 | 63.99 |
| MAE (USD) | 20.8384 | 30.3512 | 39.8182 |
| R2 | 0.999974 | 0.999948 | 0.999918 |
| RMSE gain vs E0 | 0.000573445 | -0.000184554 | -0.0227786 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000490733 | 0.000689897 | 0.00086521 |
| MAE | 0.000285428 | 0.000416522 | 0.000548614 |
| R2 | 0.00168228 | -0.000174325 | -0.0492371 |
| RMSE gain vs E0 | 0.00085444 | -7.47648e-05 | -0.0243025 |

**hfformer** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 37.031 | 58.8968 | 64.9998 |
| MAE (USD) | 22.2326 | 42.1241 | 41.8901 |
| R2 | 0.999973 | 0.999931 | 0.999916 |
| RMSE gain vs E0 | -0.0174547 | -0.150935 | -0.0389191 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000500445 | 0.00080108 | 0.000879907 |
| MAE | 0.000305717 | 0.000585073 | 0.000578492 |
| R2 | -0.0382238 | -0.348525 | -0.085185 |
| RMSE gain vs E0 | -0.0189195 | -0.161246 | -0.0417015 |

**patchtst** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.406 | 51.1826 | 62.5713 |
| MAE (USD) | 20.8132 | 30.3505 | 37.4769 |
| R2 | 0.999974 | 0.999948 | 0.999922 |
| RMSE gain vs E0 | -0.000284248 | -0.000186302 | -0.000102935 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000491261 | 0.000689968 | 0.000844766 |
| MAE | 0.000285087 | 0.000416565 | 0.000514887 |
| R2 | -0.000467041 | -0.000379715 | -0.000237542 |
| RMSE gain vs E0 | -0.000220535 | -0.000177444 | -9.89764e-05 |

**moderntcn** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.5505 | 51.3057 | 62.7391 |
| MAE (USD) | 21.2643 | 30.7326 | 37.8902 |
| R2 | 0.999973 | 0.999948 | 0.999922 |
| RMSE gain vs E0 | -0.0042537 | -0.00259199 | -0.00278581 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000493152 | 0.000691716 | 0.000847182 |
| MAE | 0.000291536 | 0.000422102 | 0.000520853 |
| R2 | -0.00818336 | -0.00545687 | -0.00596638 |
| RMSE gain vs E0 | -0.00407033 | -0.0027123 | -0.00295891 |

**lit** (58963 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.4342 | 51.1959 | 62.6095 |
| MAE (USD) | 20.9322 | 30.3919 | 37.5542 |
| R2 | 0.999974 | 0.999948 | 0.999922 |
| RMSE gain vs E0 | -0.00105816 | -0.000446845 | -0.000714242 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000491689 | 0.000690181 | 0.000845333 |
| MAE | 0.000286822 | 0.000417195 | 0.000516035 |
| R2 | -0.00221213 | -0.001 | -0.00158066 |
| RMSE gain vs E0 | -0.00109249 | -0.000487478 | -0.000770218 |

## 30-33. Artifact locations

| Model | Local checkpoints | Local predictions | HF prefix |
|---|---|---|---|
| e0 | n/a | `artifacts/e0/e0_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `e0/e0_490s_gate1y/` |
| ofi_lstm | `checkpoints/ofi_lstm/ofi_lstm_490s_gate1y/{best,last}` | `artifacts/ofi_lstm/ofi_lstm_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `ofi_lstm/ofi_lstm_490s_gate1y/` |
| hfformer | `checkpoints/hfformer/hfformer_490s_gate1y/{best,last}` | `artifacts/hfformer/hfformer_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `hfformer/hfformer_490s_gate1y/` |
| patchtst | `checkpoints/patchtst/patchtst_490s_gate1y/{best,last}` | `artifacts/patchtst/patchtst_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `patchtst/patchtst_490s_gate1y/` |
| moderntcn | `checkpoints/moderntcn/moderntcn_490s_gate1y/{best,last}` | `artifacts/moderntcn/moderntcn_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `moderntcn/moderntcn_490s_gate1y/` |
| lit | `checkpoints/lit/lit_490s_gate1y/{best,last}` | `artifacts/lit/lit_490s_gate1y/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_Gate_1Y` -> `lit/lit_490s_gate1y/` |

## 34-36. Incidents and deviations


### Experiment-contract deviations

- **The file needed an explicit, recorded row repair before anything read it** (sections E, F). `BTC_L10_gate_1y.csv` arrives out of order: rows 0..64079 are a 7.46-day block (2026-07-01..07-08) prepended in front of the main 2025-09-17..2026-09-16 block, and 9 timestamps carry two rows with different payloads, so as delivered the file is neither monotone nor unique. The Oct-2023 contract forbids reordering or dropping rows silently, so the repair is opt-in and recorded rather than automatic: `data.sort_by_timestamp` still defaults to false and `data.duplicate_timestamp_policy` still defaults to `error`, which leaves the Oct-2023 run bit-identical and still fails loudly on an unexpected file. This run sets them to true and `keep_first`; the sort is stable, so rows sharing a timestamp keep their file order, and exactly what was done is written to `split_manifest.source.row_repairs` in every checkpoint and shown in the Data repair table above.
- **`max_gap_seconds` and `target_tolerance_seconds` rescaled 2.0 s -> 10.0 s.** The new file sits on a perfect 10 s grid: min dt = median dt = 10.0 s, every dt a multiple of it, and 31 gaps longer than one slot. The Oct-2023 2.0 s rule measures cleanliness in units of that file's ~1.2 s cadence; applied here it marks all 3,139,596 edges bad and would leave zero usable samples. Both knobs moved together to the measured cadence, so the rule itself -- history, origin and every target inside one uninterrupted stretch -- is unchanged; only its unit follows the file.
- **`history_seconds` 60 -> 490, chosen so the architectures stay identical** (section K). At 10 s cadence 60 s resolves to history_rows=6. PatchTST then refuses to build at all (its history must be >= patch_length=16), while HFformer and LiT do build but at a different capacity (22,026 -> 21,897 and 736,547 -> 735,843), so the frozen parameter contract breaks either way. 490 s re-resolves to history_rows=49 and stride_rows=8 -- exactly the Oct-2023 values -- so every model sees the same input shape and all five parameter counts are unchanged (ofi_lstm 55,491; hfformer 22,026; patchtst 477,059; moderntcn 50,568,195; lit 736,547). Only the wall-clock span of one history window differs, because a row is now 10 s instead of ~1.2 s.
- **R2 is reported in price space although it is uninformative there** (sections 28-29). The four metric families are frozen, so R2 is reported for completeness; but on the mid price sigma(target mid) is about 16,579 USD while the E0 error measured on this file is 50.4 / 71.7 / 87.9 USD at 1m/2m/3m, which pins R2 near 0.9999 for all six models including E0. It is kept as-is rather than replaced by an invented metric: the log-return table under each price table preserves the Oct-2023 comparison, and RMSE gain vs E0 is what separates the models.

- Crashed jobs: none
- Resumed jobs: none
- Total scheduled training wall time: 3716.67 s
- GPU usage log: `reports/vast_gate1y/gpu_usage.csv`
