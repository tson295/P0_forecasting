# TimesFM / AutoTS phase

All configured feature-search and final-representative stages completed.
See tfm_autots_per_fold_horizon.csv and tfm_autots_summary.csv for results.
LoRA uses a FIT prefix and inner early stopping; residual heads use a later held-out FIT suffix.
AutoTS WR/MR rolling predictions are batched across independent origins.
AutoTS feature-search folds use CPU caches prepared before calibration and candidate fits.
MR later recursive steps and native final template bake-off retain their required preprocessing.
Native forecast cache timing and real batch timing are stored separately from fit timing.
No smoke/tests/probe fits/warmup/benchmark pass or other model families were run by this command.
The final TEST holdout was not evaluated in this phase.

Code hash: 03d7a4a8e4b34ff25f89cc51f397e898604e5eee0cf42fe96c99532e42c8cec2
Config hash: 822935ae5e8f
