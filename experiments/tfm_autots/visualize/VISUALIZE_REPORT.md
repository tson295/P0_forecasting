# TimesFM / AutoTS visualization

- Output: `experiments/tfm_autots/visualize`
- Run/config provenance: phase config hash `822935ae5e8f`, phase code hash `03d7a4a8e4b34ff25f89cc51f397e898604e5eee0cf42fe96c99532e42c8cec2`, visualizer hash `4082579396c3e2633977b6bee412b46368178aea37c63da9d91fdc5bbdca72ab`.
- Inputs: `data/BTC_1m_2y.csv`, final representative `wins/tfm*.json/.npz`, `wins/autots*.json/.npz`, and saved summary tables.
- Forecast paths: 30 PNG overlays across 5 real VAL folds; seed `8587` only, selected deterministically from shared valid origins without inspecting errors.
- Heatmaps: five PNGs. Gain/R² OS use the saved phase cells; RMSE/MAE/standard R² are calculated from the saved prediction arrays over all valid origins and then averaged over eval seeds.
- No model, checkpoint, feature store, training, refit, inference, test holdout, latency replay, smoke, probe, or benchmark was run. MAE is a post-hoc metric from saved predictions; latency remains unavailable.
