# Walk-forward final report — BTC L10 1 year at 10 s, 3 expanding folds, 20 s stride

## 1-4. Provenance

| Item | Value |
|---|---|
| Initial Git commit | `194749cf1dcbcab058ef826db50a740350503db3` |
| Training-source Git commit | `fdbb369946cc07eae9e615c968943036507be129` |
| Final Git commit | `fdbb369946cc07eae9e615c968943036507be129` |
| Hugging Face repo | Tson29/Pretrain_Model_WF3 |

## 5-12. Dataset, split and sampling

| Item | Value |
|---|---|
| Dataset path | `/home/ubuntu/Pretrain_Model/BTC_L10_gate_1y.csv` |
| Dataset SHA256 | `6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df` |
| Rows | 3139597 |
| Median / p99 / max dt (s) | 10.0 / 10.0 / 7210.0 |
| Gaps > 10 s / segment transitions | 31 / 0 |
| Split scheme | walk_forward, folds=3, test_fraction=0.15, embargo_seconds=180 (per-fold layout under Walk-forward and leakage) |
| history_rows | 49 |
| stride_rows | 2 |
| Contract gate | pass on `BTC_L10_gate_1y.csv` (465 checks, 0 failed) |

### Data repair

| Item | Value |
|---|---|
| rows_in_file | 3139606 |
| rows_used | 3139597 |
| sorted_by_timestamp | True |
| rows_moved_by_sort | 2531566 |
| duplicate_timestamp_policy | keep_first |
| duplicate_rows_dropped | 9 |

## Walk-forward and leakage

The last `test_fraction` of elapsed time is held out as the test split and is never trained on and never used to pick a checkpoint; the rest is cut into `folds+1` equal blocks of elapsed time, and fold k trains on blocks `[0, k)` and validates on block k, so every fold validates strictly after everything it trained on. **The test split is identical for every model and every fold** -- same rows, same samples -- so the test tables compare folds, not datasets. Each fold's `best_epoch` is chosen on that fold's validation block alone.

| Item | Value |
|---|---|
| Split scheme | walk_forward |
| Folds / this report | 3 / 1, 2, 3 |
| test_fraction | 0.15 |
| embargo_seconds | 180 |
| Longest target horizon (s) | 180 |
| Test split rows | `[2667637, 3139597)` |
| Test samples | 235848 |
| Test split identical across folds | True |

**Fold blocks** (equal spans of elapsed time; the last row is the held-out tail)

| Block | Rows | Starts (UTC) |
|---|---|---|
| 1 | `[0, 666900)` | `2025-09-17T00:00:00+00:00` |
| 2 | `[666900, 1333126)` | `2025-12-03T13:29:57.875000+00:00` |
| 3 | `[1333126, 1998587)` | `2026-02-19T02:59:55.750000+00:00` |
| 4 | `[1998587, 2667637)` | `2026-05-07T16:29:53.625000+00:00` |
| test | `[2667637, 3139597)` | `2026-07-24T05:59:51.500000+00:00` |

**Purge and embargo per fold.** Separation is the wall-clock distance between the last raw record one split's samples touch and the first record the next split's samples touch; it must exceed the 180 s longest horizon, otherwise the two splits could share a target.

| Fold | Train rows | Validation rows | Train samples | Validation samples | train->val (s) | val->test (s) | audit min gap (s) | > 180 s |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `[0, 666900)` | `[666900, 1333126)` | 333177 | 332849 | 200 | 1.3403e+07 | 200 | True |
| 2 | `[0, 1333126)` | `[1333126, 1998587)` | 666059 | 332433 | 200 | 6.7014e+06 | 200 | True |
| 3 | `[0, 1998587)` | `[1998587, 2667637)` | 998525 | 334350 | 190 | 200 | 190 | True |

**Independent leakage audit.** `scripts/check_leakage.py` re-derives the spans, the separations, the standardizer fit and the test fingerprint from the raw timestamps, so a bug in the splitter cannot certify itself.

| Item | Value |
|---|---|
| Audit report | `reports/leakage_audit_wf3.json` |
| Audited file | `BTC_L10_gate_1y.csv` (the file the artifacts name: True) |
| Runs audited | 18/18 |
| Status | pass |
| Checks passed | 307/307 |
| Failed checks | none |
| Embargo declared / enforced by the checks (s) | 180 / 180 |
| Test split identical across every model and fold | True |

## 13-17. Hardware, precision and compile

| Item | Value |
|---|---|
| GPU | n/a (n/a) |
| Driver / CUDA driver | n/a / n/a |
| CUDA runtime | n/a |
| PyTorch | n/a |
| Precision | n/a, TF32 matmul n/a |
| CPU / RAM | n/a cores / n/a |

**torch.compile status per model**

| Model | Compile succeeded | Speedup vs eager | Recommended | Used in training (f1/f2/f3) |
|---|---|---:|---|---|
| ofi_lstm | True | 0.9973 | False | False/False/False |
| hfformer | True | 1.535 | True | True/True/True |
| patchtst | True | 1.526 | True | True/True/True |
| moderntcn | True | 1.13 | True | True/True/True |
| lit | True | 1.194 | True | True/True/True |

## 18-27. Capacity, systems settings and training cost

**Capacity this run vs the frozen baseline of the first two experiments**

| Model | Parameters (this run) | Baseline (oct2023/gate1y) | Multiplier |
|---|---:|---:|---:|
| e0 | 0 | 0 | n/a |
| ofi_lstm | 2990211 | 55491 | 53.9x |
| hfformer | 4677278 | 22026 | 212.4x |
| patchtst | 14434563 | 477059 | 30.3x |
| moderntcn | 50568195 | 50568195 | 1.0x |
| lit | 15995235 | 736547 | 21.7x |

| Model | Fold | Parameters | Batch | num_workers | Peak alloc (single) | Peak reserved (single) | samples/s (single) | Train seconds | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ofi_lstm | 1 | 2990211 | 128 | 4 | 0.19 GiB | 0.27 GiB | 38583 | 2390.6 | 29 |
| ofi_lstm | 2 | 2990211 | 128 | 4 | 0.19 GiB | 0.27 GiB | 38583 | 4288.7 | 26 |
| ofi_lstm | 3 | 2990211 | 128 | 4 | 0.19 GiB | 0.27 GiB | 38583 | 6203.8 | 24 |
| hfformer | 1 | 4677278 | 128 | 4 | 0.64 GiB | 0.73 GiB | 13406 | 4941.3 | 29 |
| hfformer | 2 | 4677278 | 128 | 4 | 0.64 GiB | 0.73 GiB | 13406 | 8717.7 | 27 |
| hfformer | 3 | 4677278 | 128 | 4 | 0.64 GiB | 0.73 GiB | 13406 | 12482 | 27 |
| patchtst | 1 | 14434563 | 128 | 4 | 4.01 GiB | 4.25 GiB | 2077.1 | 34281 | 29 |
| patchtst | 2 | 14434563 | 128 | 4 | 4.01 GiB | 4.25 GiB | 2077.1 | 56832 | 29 |
| patchtst | 3 | 14434563 | 128 | 4 | 4.01 GiB | 4.25 GiB | 2077.1 | 63274 | 29 |
| moderntcn | 1 | 50568195 | 128 | 4 | 1.60 GiB | 1.69 GiB | 2614.6 | 25296 | 29 |
| moderntcn | 2 | 50568195 | 128 | 4 | 1.60 GiB | 1.69 GiB | 2614.6 | 47392 | 29 |
| moderntcn | 3 | 50568195 | 128 | 4 | 1.60 GiB | 1.69 GiB | 2614.6 | 61947 | 29 |
| lit | 1 | 15995235 | 128 | 4 | 0.81 GiB | 0.85 GiB | 15183 | 5731.7 | 28 |
| lit | 2 | 15995235 | 128 | 4 | 0.81 GiB | 0.85 GiB | 15183 | 10529 | 24 |
| lit | 3 | 15995235 | 128 | 4 | 0.81 GiB | 0.85 GiB | 15183 | 16182 | 22 |

Concurrency groups executed: `n/a` (decided by n/a).


## 28-29. Metrics


**Metric space.** Training is unchanged: the target is still `log(mid[t+h]/mid[t])` and the loss is still MSE on that log return. Only the reported space changed -- the same predictions are measured on the mid price via `pred_mid = origin_mid*exp(pred_return)`, so RMSE and MAE below are in quote currency (USD). E0 predicts that the price does not move, so its predicted mid is the origin mid and `RMSE_E0 = sqrt(mean((target_mid-origin_mid)^2))`. **R2 gain vs E0** = `1 - SSE_model/SSE_E0` is reported instead of a mean-based R2 on the price: sigma(target mid) is three orders of magnitude larger than any model's error here, which pins a mean-based price R2 near 0.9999 for every model including E0 and separates nothing. The gain is 0 when a model equals E0, positive when it beats E0, and is the squared-error twin of RMSE gain vs E0 (`r2_gain = 1-(1-rmse_gain)^2`). Each price table is followed by the same four families in log-return units, where the mean of the target is near zero and R2 is meaningful, so both spaces stay comparable.


### Fold 1 — validation


**e0** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 56.0091 | 79.6118 | 97.5238 |
| MAE (USD) | 33.7286 | 49.1678 | 60.8751 |
| R2 gain vs E0 | -1.11022e-15 | -2.24265e-14 | -6.66134e-16 |
| RMSE gain vs E0 | -6.66134e-16 | -1.13243e-14 | -4.44089e-16 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.0007027 | 0.00099814 | 0.00122289 |
| MAE | 0.000410915 | 0.000598359 | 0.000740766 |
| R2 | -1.33176e-05 | -2.59573e-05 | -3.86258e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 56.0935 | 80.0402 | 97.6362 |
| MAE (USD) | 34.1987 | 50.138 | 61.3003 |
| R2 gain vs E0 | -0.00301803 | -0.0107908 | -0.00230593 |
| RMSE gain vs E0 | -0.00150788 | -0.00538095 | -0.0011523 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000703623 | 0.00100277 | 0.00122409 |
| MAE | 0.000416213 | 0.000609324 | 0.000745643 |
| R2 | -0.00264179 | -0.00932854 | -0.00200225 |
| RMSE gain vs E0 | -0.00131335 | -0.0046404 | -0.000981294 |

**hfformer** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 56.0088 | 79.6112 | 97.5222 |
| MAE (USD) | 33.7393 | 49.1753 | 60.8847 |
| R2 gain vs E0 | 9.7269e-06 | 1.67517e-05 | 3.2987e-05 |
| RMSE gain vs E0 | 4.86346e-06 | 8.37589e-06 | 1.64937e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000702697 | 0.000998133 | 0.00122287 |
| MAE | 0.000411035 | 0.000598444 | 0.000740875 |
| R2 | -4.48777e-06 | -1.08451e-05 | -8.1879e-06 |
| RMSE gain vs E0 | 4.41484e-06 | 7.55591e-06 | 1.52185e-05 |

**patchtst** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 56.0035 | 79.6098 | 97.526 |
| MAE (USD) | 33.8131 | 49.2249 | 60.9209 |
| R2 gain vs E0 | 0.000199951 | 5.01745e-05 | -4.60856e-05 |
| RMSE gain vs E0 | 9.99803e-05 | 2.50876e-05 | -2.30425e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000702633 | 0.00099811 | 0.0012229 |
| MAE | 0.00041186 | 0.000599001 | 0.000741285 |
| R2 | 0.000178707 | 3.43466e-05 | -5.90887e-05 |
| RMSE gain vs E0 | 9.60157e-05 | 3.01516e-05 | -1.0231e-05 |

**moderntcn** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 56.0173 | 79.6786 | 97.6172 |
| MAE (USD) | 34.1236 | 49.5195 | 61.1754 |
| R2 gain vs E0 | -0.000295067 | -0.00167919 | -0.00191708 |
| RMSE gain vs E0 | -0.000147523 | -0.000839245 | -0.000958082 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000702779 | 0.000998876 | 0.00122391 |
| MAE | 0.000415364 | 0.000602336 | 0.000744169 |
| R2 | -0.000237632 | -0.00149997 | -0.00171449 |
| RMSE gain vs E0 | -0.00011215 | -0.000736716 | -0.000837549 |

**lit** (332849 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 59.8782 | 79.74 | 98.9328 |
| MAE (USD) | 40.7165 | 49.4996 | 63.7523 |
| R2 gain vs E0 | -0.142934 | -0.00322141 | -0.0291046 |
| RMSE gain vs E0 | -0.0690807 | -0.00160941 | -0.0144479 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000744819 | 0.000999542 | 0.00123806 |
| MAE | 0.000490624 | 0.000602086 | 0.000773583 |
| R2 | -0.123485 | -0.00283572 | -0.0250125 |
| RMSE gain vs E0 | -0.0599388 | -0.00140386 | -0.0124095 |

### Fold 2 — validation


**e0** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 44.8153 | 63.6534 | 77.7652 |
| MAE (USD) | 28.7381 | 41.4497 | 50.91 |
| R2 gain vs E0 | -5.37348e-14 | -1.26565e-14 | -4.996e-14 |
| RMSE gain vs E0 | -2.68674e-14 | -6.43929e-15 | -2.5091e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000635412 | 0.000902881 | 0.00110337 |
| MAE | 0.000404711 | 0.000583824 | 0.000717132 |
| R2 | -6.56329e-06 | -1.30663e-05 | -1.98533e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 44.8369 | 63.8702 | 77.7634 |
| MAE (USD) | 28.8919 | 41.8327 | 50.9458 |
| R2 gain vs E0 | -0.000964174 | -0.00682418 | 4.67981e-05 |
| RMSE gain vs E0 | -0.000481971 | -0.00340629 | 2.33993e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000635675 | 0.000905839 | 0.00110332 |
| MAE | 0.000406825 | 0.00058914 | 0.000717614 |
| R2 | -0.000837226 | -0.00657725 | 6.62056e-05 |
| RMSE gain vs E0 | -0.000415242 | -0.00327668 | 4.30295e-05 |

**hfformer** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 44.8155 | 63.6535 | 77.7669 |
| MAE (USD) | 28.75 | 41.4501 | 50.9131 |
| R2 gain vs E0 | -7.94174e-06 | -3.14145e-06 | -4.19099e-05 |
| RMSE gain vs E0 | -3.97086e-06 | -1.57072e-06 | -2.09547e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000635413 | 0.000902882 | 0.00110339 |
| MAE | 0.000404875 | 0.000583831 | 0.000717175 |
| R2 | -1.25034e-05 | -1.62386e-05 | -6.19933e-05 |
| RMSE gain vs E0 | -2.97004e-06 | -1.58614e-06 | -2.10694e-05 |

**patchtst** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 44.8182 | 63.6696 | 77.7811 |
| MAE (USD) | 28.7883 | 41.4921 | 50.9358 |
| R2 gain vs E0 | -0.000128768 | -0.000509907 | -0.000408436 |
| RMSE gain vs E0 | -6.43819e-05 | -0.000254921 | -0.000204197 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000635443 | 0.000903098 | 0.00110359 |
| MAE | 0.000405407 | 0.000584414 | 0.000717491 |
| R2 | -0.000105815 | -0.000493391 | -0.000417043 |
| RMSE gain vs E0 | -4.96244e-05 | -0.00024013 | -0.000198571 |

**moderntcn** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 44.7324 | 63.6244 | 77.7546 |
| MAE (USD) | 28.7939 | 41.5025 | 50.9376 |
| R2 gain vs E0 | 0.00369763 | 0.000910448 | 0.00027247 |
| RMSE gain vs E0 | 0.00185053 | 0.000455328 | 0.000136245 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000634224 | 0.000902452 | 0.00110322 |
| MAE | 0.000405461 | 0.000584556 | 0.000717521 |
| R2 | 0.00372723 | 0.000935918 | 0.000245462 |
| RMSE gain vs E0 | 0.00186863 | 0.000474599 | 0.000132664 |

**lit** (332433 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 46.8655 | 66.5604 | 77.8183 |
| MAE (USD) | 31.9239 | 45.5753 | 50.9895 |
| R2 gain vs E0 | -0.0935908 | -0.093424 | -0.00136502 |
| RMSE gain vs E0 | -0.0457489 | -0.0456692 | -0.000682279 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000663573 | 0.000942812 | 0.0011041 |
| MAE | 0.000448978 | 0.000641157 | 0.000718244 |
| R2 | -0.0906104 | -0.0904233 | -0.00135381 |
| RMSE gain vs E0 | -0.0443195 | -0.0442265 | -0.000666742 |

### Fold 3 — validation


**e0** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.8355 | 52.6241 | 64.4122 |
| MAE (USD) | 23.294 | 34.0279 | 41.9954 |
| R2 gain vs E0 | -4.52971e-14 | -6.43929e-15 | 2.73115e-14 |
| RMSE gain vs E0 | -2.26485e-14 | -3.33067e-15 | 1.34337e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000562506 | 0.000802758 | 0.000982247 |
| MAE | 0.00035018 | 0.000510979 | 0.000630295 |
| R2 | -1.0391e-05 | -2.02994e-05 | -3.06092e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.843 | 52.7094 | 64.3894 |
| MAE (USD) | 23.4095 | 34.2307 | 41.9985 |
| R2 gain vs E0 | -0.000411727 | -0.00324448 | 0.000707376 |
| RMSE gain vs E0 | -0.000205842 | -0.00162093 | 0.00035375 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000562674 | 0.000804143 | 0.000982017 |
| MAE | 0.000351889 | 0.000514 | 0.000630401 |
| R2 | -0.000607695 | -0.00347556 | 0.000438818 |
| RMSE gain vs E0 | -0.000298604 | -0.0017261 | 0.000234734 |

**hfformer** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.8368 | 52.6233 | 64.4108 |
| MAE (USD) | 23.3226 | 34.0305 | 41.9999 |
| R2 gain vs E0 | -7.39337e-05 | 2.9913e-05 | 4.38232e-05 |
| RMSE gain vs E0 | -3.69662e-05 | 1.49566e-05 | 2.19118e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.00056253 | 0.00080275 | 0.000982237 |
| MAE | 0.00035059 | 0.000511015 | 0.00063036 |
| R2 | -9.54036e-05 | -5.01998e-08 | -9.74964e-06 |
| RMSE gain vs E0 | -4.25049e-05 | 1.01245e-05 | 1.04295e-05 |

**patchtst** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.83 | 52.6226 | 64.4166 |
| MAE (USD) | 23.3276 | 34.0443 | 42.0119 |
| R2 gain vs E0 | 0.000295357 | 5.67334e-05 | -0.000134841 |
| RMSE gain vs E0 | 0.000147689 | 2.83671e-05 | -6.74183e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000562433 | 0.000802751 | 0.000982329 |
| MAE | 0.000350662 | 0.000511216 | 0.000630532 |
| R2 | 0.000249326 | -4.35431e-06 | -0.000196684 |
| RMSE gain vs E0 | 0.000129865 | 7.97243e-06 | -8.30316e-05 |

**moderntcn** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.7304 | 52.5556 | 64.3699 |
| MAE (USD) | 23.27 | 33.9879 | 41.974 |
| R2 gain vs E0 | 0.00569789 | 0.00260088 | 0.0013148 |
| RMSE gain vs E0 | 0.00285302 | 0.00130129 | 0.000657616 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000561102 | 0.000801855 | 0.000981684 |
| MAE | 0.000349842 | 0.000510425 | 0.000630005 |
| R2 | 0.00497338 | 0.0022274 | 0.00111703 |
| RMSE gain vs E0 | 0.00249497 | 0.00112446 | 0.000573968 |

**lit** (334350 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.8293 | 53.6614 | 64.9692 |
| MAE (USD) | 23.3096 | 35.6215 | 42.8247 |
| R2 gain vs E0 | 0.000336664 | -0.0398115 | -0.0173694 |
| RMSE gain vs E0 | 0.000168346 | -0.0197115 | -0.0086473 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000562426 | 0.000817576 | 0.000990402 |
| MAE | 0.000350396 | 0.000534116 | 0.000642455 |
| R2 | 0.000272653 | -0.0372808 | -0.0167043 |
| RMSE gain vs E0 | 0.000141531 | -0.0184595 | -0.00830215 |

### Test — the same held-out tail split for every fold

Each fold's own best checkpoint, scored on identical samples, so the differences below are the folds and nothing else.

#### Fold 1


**e0** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9526 | 51.2282 | 62.5951 |
| MAE (USD) | 20.6645 | 30.2886 | 37.3973 |
| R2 gain vs E0 | -3.33067e-14 | -1.11022e-14 | -2.24265e-14 |
| RMSE gain vs E0 | -1.66533e-14 | -5.55112e-15 | -1.13243e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484755 | 0.000691121 | 0.000844698 |
| MAE | 0.000282988 | 0.000415692 | 0.000513848 |
| R2 | -1.60221e-05 | -3.14725e-05 | -4.72816e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.0409 | 51.6425 | 62.7913 |
| MAE (USD) | 21.1201 | 31.3345 | 37.7854 |
| R2 gain vs E0 | -0.00491625 | -0.0162388 | -0.00627845 |
| RMSE gain vs E0 | -0.00245511 | -0.00808669 | -0.00313431 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000485915 | 0.000696962 | 0.000847498 |
| MAE | 0.000289573 | 0.00043078 | 0.000519479 |
| R2 | -0.00480743 | -0.0170073 | -0.00668708 |
| RMSE gain vs E0 | -0.0023928 | -0.00845193 | -0.00331425 |

**hfformer** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.953 | 51.229 | 62.5974 |
| MAE (USD) | 20.6728 | 30.2941 | 37.4047 |
| R2 gain vs E0 | -2.45747e-05 | -3.11738e-05 | -7.21445e-05 |
| RMSE gain vs E0 | -1.22873e-05 | -1.55868e-05 | -3.60716e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484762 | 0.000691133 | 0.000844734 |
| MAE | 0.00028311 | 0.000415774 | 0.000513962 |
| R2 | -4.37613e-05 | -6.73133e-05 | -0.000131328 |
| RMSE gain vs E0 | -1.38693e-05 | -1.79197e-05 | -4.20205e-05 |

**patchtst** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.959 | 51.2401 | 62.6078 |
| MAE (USD) | 20.7478 | 30.3482 | 37.4479 |
| R2 gain vs E0 | -0.000354188 | -0.00046394 | -0.00040396 |
| RMSE gain vs E0 | -0.000177078 | -0.000231943 | -0.000201959 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484835 | 0.000691277 | 0.000844875 |
| MAE | 0.000284198 | 0.000416563 | 0.000514587 |
| R2 | -0.000342713 | -0.000483033 | -0.00046649 |
| RMSE gain vs E0 | -0.00016333 | -0.000225748 | -0.000209572 |

**moderntcn** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 36.0628 | 51.3673 | 62.7493 |
| MAE (USD) | 21.1001 | 30.6711 | 37.777 |
| R2 gain vs E0 | -0.006138 | -0.00543759 | -0.00493193 |
| RMSE gain vs E0 | -0.0030643 | -0.00271511 | -0.00246293 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000486187 | 0.000693045 | 0.000846876 |
| MAE | 0.000289262 | 0.000421231 | 0.000519344 |
| R2 | -0.00593286 | -0.00560909 | -0.00521131 |
| RMSE gain vs E0 | -0.00295401 | -0.00278484 | -0.00257857 |

**lit** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 39.9298 | 51.3354 | 64.258 |
| MAE (USD) | 27.4654 | 30.6691 | 40.0848 |
| R2 gain vs E0 | -0.233486 | -0.00418753 | -0.053836 |
| RMSE gain vs E0 | -0.110624 | -0.00209158 | -0.0265652 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000542064 | 0.000692632 | 0.000868874 |
| MAE | 0.000380653 | 0.000421211 | 0.000552705 |
| R2 | -0.250438 | -0.00441083 | -0.0581094 |
| RMSE gain vs E0 | -0.118221 | -0.00218722 | -0.0286201 |

#### Fold 2


**e0** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9526 | 51.2282 | 62.5951 |
| MAE (USD) | 20.6645 | 30.2886 | 37.3973 |
| R2 gain vs E0 | -3.33067e-14 | -1.11022e-14 | -2.24265e-14 |
| RMSE gain vs E0 | -1.66533e-14 | -5.55112e-15 | -1.13243e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484755 | 0.000691121 | 0.000844698 |
| MAE | 0.000282988 | 0.000415692 | 0.000513848 |
| R2 | -1.60221e-05 | -3.14725e-05 | -4.72816e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9808 | 51.4684 | 62.5735 |
| MAE (USD) | 20.9127 | 30.8236 | 37.4391 |
| R2 gain vs E0 | -0.00157264 | -0.00939851 | 0.000689691 |
| RMSE gain vs E0 | -0.000786011 | -0.00468827 | 0.000344905 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000485055 | 0.000694559 | 0.000844295 |
| MAE | 0.000286569 | 0.000423486 | 0.00051441 |
| R2 | -0.00125367 | -0.0100072 | 0.000906006 |
| RMSE gain vs E0 | -0.000618621 | -0.00497532 | 0.000476735 |

**hfformer** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9527 | 51.2284 | 62.5973 |
| MAE (USD) | 20.6942 | 30.2897 | 37.4046 |
| R2 gain vs E0 | -5.16101e-06 | -6.33328e-06 | -7.03002e-05 |
| RMSE gain vs E0 | -2.5805e-06 | -3.16664e-06 | -3.51495e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484755 | 0.000691123 | 0.000844733 |
| MAE | 0.000283425 | 0.000415708 | 0.00051396 |
| R2 | -1.56361e-05 | -3.89676e-05 | -0.000128626 |
| RMSE gain vs E0 | 1.92984e-07 | -3.74743e-06 | -4.06692e-05 |

**patchtst** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9599 | 51.2431 | 62.5959 |
| MAE (USD) | 20.7505 | 30.3504 | 37.4316 |
| R2 gain vs E0 | -0.000404737 | -0.000581739 | -2.37526e-05 |
| RMSE gain vs E0 | -0.000202348 | -0.000290827 | -1.18762e-05 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484856 | 0.000691336 | 0.000844726 |
| MAE | 0.000284241 | 0.000416606 | 0.000514364 |
| R2 | -0.000430488 | -0.000653928 | -0.000113889 |
| RMSE gain vs E0 | -0.000207208 | -0.000311169 | -3.33016e-05 |

**moderntcn** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9862 | 51.2817 | 62.6303 |
| MAE (USD) | 20.969 | 30.5231 | 37.5612 |
| R2 gain vs E0 | -0.00186902 | -0.00209012 | -0.00112343 |
| RMSE gain vs E0 | -0.000934072 | -0.00104452 | -0.00056156 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000485085 | 0.000691812 | 0.000845168 |
| MAE | 0.00028737 | 0.000419102 | 0.000516242 |
| R2 | -0.00137697 | -0.00203323 | -0.00115981 |
| RMSE gain vs E0 | -0.00068023 | -0.00100035 | -0.000556082 |

**lit** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 38.4718 | 54.7923 | 62.6625 |
| MAE (USD) | 25.1397 | 35.928 | 37.5394 |
| R2 gain vs E0 | -0.145049 | -0.143986 | -0.00215274 |
| RMSE gain vs E0 | -0.0700697 | -0.0695727 | -0.00107579 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000521237 | 0.00074275 | 0.000845697 |
| MAE | 0.000347509 | 0.000496962 | 0.000515951 |
| R2 | -0.156198 | -0.155026 | -0.00241271 |
| RMSE gain vs E0 | -0.0752579 | -0.0747042 | -0.00118196 |

#### Fold 3


**e0** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9526 | 51.2282 | 62.5951 |
| MAE (USD) | 20.6645 | 30.2886 | 37.3973 |
| R2 gain vs E0 | -3.33067e-14 | -1.11022e-14 | -2.24265e-14 |
| RMSE gain vs E0 | -1.66533e-14 | -5.55112e-15 | -1.13243e-14 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484755 | 0.000691121 | 0.000844698 |
| MAE | 0.000282988 | 0.000415692 | 0.000513848 |
| R2 | -1.60221e-05 | -3.14725e-05 | -4.72816e-05 |
| RMSE gain vs E0 | 0 | 0 | 0 |

**ofi_lstm** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9586 | 51.3641 | 62.5684 |
| MAE (USD) | 20.9023 | 30.6265 | 37.3962 |
| R2 gain vs E0 | -0.0003339 | -0.00531309 | 0.000855491 |
| RMSE gain vs E0 | -0.000166936 | -0.00265303 | 0.000427837 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484735 | 0.000693004 | 0.000844241 |
| MAE | 0.00028642 | 0.000420598 | 0.000513801 |
| R2 | 6.86845e-05 | -0.00549032 | 0.0010355 |
| RMSE gain vs E0 | 4.23535e-05 | -0.00272562 | 0.00054151 |

**hfformer** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9581 | 51.2296 | 62.6017 |
| MAE (USD) | 20.7236 | 30.2985 | 37.4172 |
| R2 gain vs E0 | -0.000306731 | -5.43551e-05 | -0.000209748 |
| RMSE gain vs E0 | -0.000153354 | -2.71772e-05 | -0.000104869 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484838 | 0.000691143 | 0.000844799 |
| MAE | 0.000283858 | 0.000415842 | 0.000514149 |
| R2 | -0.000356698 | -9.53192e-05 | -0.000285119 |
| RMSE gain vs E0 | -0.000170321 | -3.19219e-05 | -0.000118906 |

**patchtst** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.952 | 51.2348 | 62.6099 |
| MAE (USD) | 20.7332 | 30.3339 | 37.4484 |
| R2 gain vs E0 | 3.22252e-05 | -0.000257173 | -0.000470808 |
| RMSE gain vs E0 | 1.61127e-05 | -0.000128578 | -0.000235377 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484748 | 0.000691215 | 0.000844921 |
| MAE | 0.000283997 | 0.000416363 | 0.000514606 |
| R2 | 1.29723e-05 | -0.000305562 | -0.000574321 |
| RMSE gain vs E0 | 1.4497e-05 | -0.000137031 | -0.000263472 |

**moderntcn** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9257 | 51.2133 | 62.5843 |
| MAE (USD) | 20.8625 | 30.3987 | 37.4807 |
| R2 gain vs E0 | 0.00149263 | 0.000582114 | 0.000345031 |
| RMSE gain vs E0 | 0.000746592 | 0.000291099 | 0.00017253 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484211 | 0.000690831 | 0.000844475 |
| MAE | 0.000285822 | 0.000417289 | 0.000515062 |
| R2 | 0.00223103 | 0.000806695 | 0.000480934 |
| RMSE gain vs E0 | 0.00112414 | 0.000419158 | 0.00026413 |

**lit** (235848 samples)

| price (USD) | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE (USD) | 35.9505 | 52.4363 | 63.2809 |
| MAE (USD) | 20.6959 | 32.4657 | 38.596 |
| R2 gain vs E0 | 0.000113147 | -0.0477185 | -0.0220309 |
| RMSE gain vs E0 | 5.65751e-05 | -0.0235812 | -0.0109555 |

| log return | 1m | 2m | 3m |
|---|---:|---:|---:|
| RMSE | 0.000484703 | 0.000708782 | 0.000854859 |
| MAE | 0.000283438 | 0.000447359 | 0.000531396 |
| R2 | 0.000200171 | -0.0517948 | -0.0242509 |
| RMSE gain vs E0 | 0.0001081 | -0.0255543 | -0.0120289 |

### Test stability across folds

RMSE gain vs E0 on the test split: per model and horizon, every fold, their mean and the max-min spread. Spread is the cost of picking one fold's checkpoint blind.

| Model | Horizon | fold 1 | fold 2 | fold 3 | mean | spread |
|---|---|---:|---:|---:|---:|---:|
| e0 | 1m | -1.665e-14 | -1.665e-14 | -1.665e-14 | -1.665e-14 | 0 |
| e0 | 2m | -5.551e-15 | -5.551e-15 | -5.551e-15 | -5.551e-15 | 0 |
| e0 | 3m | -1.132e-14 | -1.132e-14 | -1.132e-14 | -1.132e-14 | 0 |
| ofi_lstm | 1m | -0.002455 | -0.000786 | -0.0001669 | -0.001136 | 0.002288 |
| ofi_lstm | 2m | -0.008087 | -0.004688 | -0.002653 | -0.005143 | 0.005434 |
| ofi_lstm | 3m | -0.003134 | 0.0003449 | 0.0004278 | -0.0007872 | 0.003562 |
| hfformer | 1m | -1.229e-05 | -2.581e-06 | -0.0001534 | -5.607e-05 | 0.0001508 |
| hfformer | 2m | -1.559e-05 | -3.167e-06 | -2.718e-05 | -1.531e-05 | 2.401e-05 |
| hfformer | 3m | -3.607e-05 | -3.515e-05 | -0.0001049 | -5.87e-05 | 6.972e-05 |
| patchtst | 1m | -0.0001771 | -0.0002023 | 1.611e-05 | -0.0001211 | 0.0002185 |
| patchtst | 2m | -0.0002319 | -0.0002908 | -0.0001286 | -0.0002171 | 0.0001622 |
| patchtst | 3m | -0.000202 | -1.188e-05 | -0.0002354 | -0.0001497 | 0.0002235 |
| moderntcn | 1m | -0.003064 | -0.0009341 | 0.0007466 | -0.001084 | 0.003811 |
| moderntcn | 2m | -0.002715 | -0.001045 | 0.0002911 | -0.001156 | 0.003006 |
| moderntcn | 3m | -0.002463 | -0.0005616 | 0.0001725 | -0.0009507 | 0.002635 |
| lit | 1m | -0.1106 | -0.07007 | 5.658e-05 | -0.06021 | 0.1107 |
| lit | 2m | -0.002092 | -0.06957 | -0.02358 | -0.03175 | 0.06748 |
| lit | 3m | -0.02657 | -0.001076 | -0.01096 | -0.01287 | 0.02549 |

## 30-33. Artifact locations

| Model | Local checkpoints | Local predictions | HF prefix |
|---|---|---|---|
| e0 | n/a | `artifacts/e0/e0_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `e0/e0_wf3_f1/` |
| ofi_lstm | `checkpoints/ofi_lstm/ofi_lstm_wf3_f1/{best,last}` | `artifacts/ofi_lstm/ofi_lstm_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `ofi_lstm/ofi_lstm_wf3_f1/` |
| hfformer | `checkpoints/hfformer/hfformer_wf3_f1/{best,last}` | `artifacts/hfformer/hfformer_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `hfformer/hfformer_wf3_f1/` |
| patchtst | `checkpoints/patchtst/patchtst_wf3_f1/{best,last}` | `artifacts/patchtst/patchtst_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `patchtst/patchtst_wf3_f1/` |
| moderntcn | `checkpoints/moderntcn/moderntcn_wf3_f1/{best,last}` | `artifacts/moderntcn/moderntcn_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `moderntcn/moderntcn_wf3_f1/` |
| lit | `checkpoints/lit/lit_wf3_f1/{best,last}` | `artifacts/lit/lit_wf3_f1/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `lit/lit_wf3_f1/` |
| e0 | n/a | `artifacts/e0/e0_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `e0/e0_wf3_f2/` |
| ofi_lstm | `checkpoints/ofi_lstm/ofi_lstm_wf3_f2/{best,last}` | `artifacts/ofi_lstm/ofi_lstm_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `ofi_lstm/ofi_lstm_wf3_f2/` |
| hfformer | `checkpoints/hfformer/hfformer_wf3_f2/{best,last}` | `artifacts/hfformer/hfformer_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `hfformer/hfformer_wf3_f2/` |
| patchtst | `checkpoints/patchtst/patchtst_wf3_f2/{best,last}` | `artifacts/patchtst/patchtst_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `patchtst/patchtst_wf3_f2/` |
| moderntcn | `checkpoints/moderntcn/moderntcn_wf3_f2/{best,last}` | `artifacts/moderntcn/moderntcn_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `moderntcn/moderntcn_wf3_f2/` |
| lit | `checkpoints/lit/lit_wf3_f2/{best,last}` | `artifacts/lit/lit_wf3_f2/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `lit/lit_wf3_f2/` |
| e0 | n/a | `artifacts/e0/e0_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `e0/e0_wf3_f3/` |
| ofi_lstm | `checkpoints/ofi_lstm/ofi_lstm_wf3_f3/{best,last}` | `artifacts/ofi_lstm/ofi_lstm_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `ofi_lstm/ofi_lstm_wf3_f3/` |
| hfformer | `checkpoints/hfformer/hfformer_wf3_f3/{best,last}` | `artifacts/hfformer/hfformer_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `hfformer/hfformer_wf3_f3/` |
| patchtst | `checkpoints/patchtst/patchtst_wf3_f3/{best,last}` | `artifacts/patchtst/patchtst_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `patchtst/patchtst_wf3_f3/` |
| moderntcn | `checkpoints/moderntcn/moderntcn_wf3_f3/{best,last}` | `artifacts/moderntcn/moderntcn_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `moderntcn/moderntcn_wf3_f3/` |
| lit | `checkpoints/lit/lit_wf3_f3/{best,last}` | `artifacts/lit/lit_wf3_f3/{train,validation,test}_predictions.csv.gz` | `Tson29/Pretrain_Model_WF3` -> `lit/lit_wf3_f3/` |

## 34-36. Incidents and deviations


### Experiment-contract deviations

- **Stride 8 -> 2 rows and capacity scaled 20-200x, so this run is not parameter-comparable to the first two** (sections H, K). `stride_seconds` 20.0 resolves to stride_rows=2 on the same 10 s grid, which roughly quadruples the windows per unit of time (fold 3 trains on 998,525 samples against the Gate run's 274,215), and each architecture is widened inside its own family through the config's `model_kwargs`: ofi_lstm 55,491 -> 2,990,211, hfformer 22,026 -> 4,677,278, patchtst 477,059 -> 14,434,563, lit 736,547 -> 15,995,235; ModernTCN stays at 50,568,195 because it is already the GPU bottleneck. `src/models` DEFAULTS are untouched, so the Oct-2023 and Gate suites still build their frozen sizes and still gate on them. The Capacity table in 18-27 states both counts; no metric here is comparable one-to-one with the earlier two reports.
- **Mean-based R2 on the price is gone, replaced by R2 gain vs E0** (sections 28-29; this also supersedes the prompt's `do not report R2_OS` line). On the mid price the mean is not a baseline anyone would use -- sigma(target mid) is thousands of USD against a ~50 USD E0 error -- so the old price R2 read ~0.9999 for all six models including E0 and ranked nothing. At the user's instruction `price_metrics` now returns `r2_gain_vs_e0 = 1 - SSE_model/SSE_E0` and no `r2` key at all, so 0 means 'exactly E0' and positive means 'beats E0'; the identity `r2_gain = 1-(1-rmse_gain)^2` ties it to the RMSE column. The nested log-return table keeps its own mean-based R2, which stays informative because the mean log return is near zero.
- **An explicit 180 s embargo now drops samples at every split boundary**, which the first two runs did not have. `target_limits()` hands each split an exclusive ns bound -- the next split's first timestamp minus `embargo_seconds=180` -- and `LOBDataset` drops any sample whose targets reach it, so no training window can read a record inside the embargo. The earlier runs bounded each target by its own split's last row and nothing further (`target < hi`), so a train sample's 3-minute target could sit one row short of the validation split's first record -- adjacent, never overlapping; their sample counts include those boundary-adjacent windows and this run's do not. The measured separations are in the Walk-forward and leakage section and `scripts/check_leakage.py` re-derives every one of them from the raw timestamps instead of trusting the manifest.
- **Three folds mean three trained models per architecture**, so there is no single 'the model'. `best_epoch`, `total_seconds`, every checkpoint and every metric in this report are per fold, selected on that fold's validation block alone. The test section then scores each fold's own best checkpoint on the one held-out tail split, which is identical for all 18 runs, and the stability table is the honest summary: fold-to-fold spread, not a single number.

- Crashed jobs: n/a
- Resumed jobs: n/a
- Total scheduled training wall time: n/a s
- GPU usage log: `reports/vast_wf3/gpu_usage.csv`
