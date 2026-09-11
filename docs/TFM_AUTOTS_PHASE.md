# Phase tfm_autots — OHLCV, TimesFM và AutoTS

Branch `tfm_autots`. Chỉ code; chưa chạy test, smoke, training, inference hoặc benchmark.
Luồng này dùng dữ liệu OHLCV cũ, độc lập với `src_OB` và goal Order Book trên Vast.
Data phase: `BTC_1m_2y.csv`, 2024-09-03 16:29 → 2026-09-03 16:29 UTC theo manifest.
File 5m đi kèm chỉ phục vụ features. Rà soát trước Vast: `TFM_AUTOTS_VAST_REVIEW.md`.

## Chạy thật trên Vast

Prompt `/goal` có sẵn tại `docs/VAST_GOAL.txt`, runbook clone/setup/push tại `docs/VAST_SESSION_PROMPT.md`.
Sau khi có `.venv` CUDA, dùng `bash scripts/vast_tfm_autots_run.sh` trong tmux để lưu env/log,
chống process trùng và tự chọn `--resume`. Lệnh Python bên dưới là entrypoint mà launcher gọi.

Chuẩn bị CUDA-enabled PyTorch và LightGBM GPU/CUDA phù hợp image, rồi cài
`requirements-tfm-autots.txt`. Không dùng bootstrap/probe fit cũ; không cần JAX, CatBoost hoặc pytest.
Tải các CSV canonical và artifact `prev_run_dir` đã có bằng Git LFS hoặc từ bản sao được cấp quyền.
Không chạy lệnh sau ở local; chỉ trên Vast đã có CUDA:

```bash
export P0_TFM_AUTOTS_VAST=1
export CUDA_VISIBLE_DEVICES=0
python run.py tfm-autots --config configs/tfm_autots.json
```

Lệnh tự chuẩn bị S0 từ artifact cũ rồi chạy `loop tfm → tfm-final → loop autots_wr → loop autots_mr
→ autots-search → summary`. WR/MR là hai nhánh nội bộ của AutoTS, không thêm family thứ ba.
Giữ feature search, folds, seed và template validation của phase OHLCV cũ. Không chạy standalone LightGBM
để chấm candidate, champion/ensemble, các family khác hay evaluation TEST holdout.
`models.lgbm`/`models.xgb` trong config chỉ cấu hình backend bên trong AutoTS; chúng không có trong model_order.

Output mới: `experiments/tfm_autots/`; không trộn artifact cũ vào run này.
`--resume` tiếp tục feature-search progress và bỏ qua stage đã hoàn tất theo phase_progress.json.
Phải giữ cùng code/config/data. Resume giữa template bake-off có thể chạy lại bake-off chưa hoàn tất;
không tuyên bố có checkpoint optimizer hoặc recovery tới từng bước thư viện AutoTS.
CLI không tự stop/destroy instance, không gọi agent và không chạy probe/test/latency pass riêng.

## TimesFM dự báo trước, XReg học residual sau

`models_tfm_residual.py` thay model `tfm` qua factory; class cũ vẫn giữ cho references/helper của OB.
Không gọi `forecast_with_covariates()` trong đường mới.

Trong mỗi FIT, dành 5 ngày cuối cho residual calibration, 5 ngày trước đó cho inner early stopping,
với purge 60 phút giữa các phần. LoRA train trên prefix còn lại. Outer ES không dùng để chọn adapter
cho run này, tránh nhìn về tương lai của tập residual calibration. Outer VAL và TEST không train model.
Các khoảng này được khai báo trong config; thiếu data thì fail, không giảm ngầm hoặc lấy nhãn VAL bù vào.

Một adapter được train theo fold/seed/epoch mode rồi freeze; baseline native và các candidate dùng cùng
adapter. TimesFM nhận context r1 gốc, forecast theo batch rồi cộng dồn thành vector dự báo y1/y2/y3.
Cache dự báo trong RAM có giới hạn và trên disk theo hash data/origins/adapter/pretrained/forecast contract.
Thay candidate hay permutation covariates không chạy lại TimesFM khi forecast cache đã có.

Residual labels = cumulative log-return thật - vector TimesFM trên calibration chưa dùng để fit/chọn LoRA.
Input hồi quy = features tại origin + cả vector forecast; không dùng future covariates/actual future labels
khi predict. Ba horizon có hệ số riêng, giải chung nhiều RHS trên CUDA bằng torch float64 pinv; scaler chỉ
fit trên calibration, ridge=0 và pinv_rtol=1e-10 khai báo rõ. Prediction cuối = TimesFM + residual prediction.
Baseline không có covariates vẫn chỉ dùng native TimesFM. Các metrics giá của harness giữ conversion cuối.

Đây là thay đổi phương pháp hiệu chỉnh, không phải chỉ đổi mode `timesfm + xreg` của thư viện. Mode thư viện
dùng residual backcast trong context, còn ở đây học lỗi forecast theo horizon trên các origins đã biết nhãn
ở training-side. Không refit beta theo mỗi origin VAL và không pooled fit qua các origins tương lai.

## AutoTS: CPU cache trước feature search

TimesFM cũng chuẩn bị pool covariates FIT-only trước search, dùng lại hạ tầng trong `autots_cache.py`
nhưng lưu riêng ở `series_preprocess/tfm/`. Chỉ lấy các cột đã chọn khi fit residual head; adapter/forecast
cache vẫn độc lập. LoRA cộng loss trên device và đọc scalar cuối epoch/ES, có log từng epoch.

`loop autots_wr` và `loop autots_mr` tự gọi `autots_cache.prepare_cache()` trước calibration và candidate
đầu tiên. Mỗi nhánh chuẩn bị toàn bộ pool S0 + các cột của 163 candidates theo cấu hình, cho mọi fold/seed.
Không cần chạy thêm lệnh prepare riêng.

```text
CPU: extract feature pool → scaler FIT-only → training X/Y + prediction contexts
                                      │
                                CACHE READY
                                      │
                 candidate → chọn cột/seed rows → GPU fit → batch predict → metric
```

- Cache dùng `.npy` mở read-only bằng memory map trong `autots_preprocess/autots_wr|autots_mr/`.
  Scaler, covariates theo phút và masks FIT/ES/VAL được chuẩn bị một lần mỗi fold; candidate không dựng
  lại full-grid feature matrix hoặc chuẩn hoá lại. Calibration, add-one, PI, pruning và confirmation dùng chung.
- WR cache windows, targets và bootstrap row selectors theo seed, giữ cách lấy mẫu có hoàn lại của
  AutoTS 1.0.4, kể cả khi `max_windows` lớn hơn số windows. Khi fit chỉ lấy đúng rows/columns đã chọn.
- MR cache rolling training features, next-return targets và thống kê lọc cột. Mask cuối vẫn xét đúng
  tập cột và thứ tự của candidate; không loại toàn cục một cột chỉ vì nó trùng với cột ngoài candidate.
- Cache VAL histories cho cả WR/MR và rolling features bước đầu cho MR. Bước 2–3 của MR vẫn phải cập nhật
  features từ prediction của chính candidate đó; không thể dùng chung chúng giữa các model đã fit khác nhau.
- Chỉ publish `READY.json` sau khi tất cả fold đã chuẩn bị xong. Key gồm code, data, dependency versions,
  preprocessing recipe, feature pool, origin masks và seeds. Giữ thư mục `.building.*` khi bị ngắt; không coi
  cache dở là ready. Worker mở cache trên disk, không nhận bản sao toàn pool qua mỗi task.

Cache này phục vụ feature search WR/MR. Bước `autots-search` cuối vẫn dùng native template bake-off và
refit trên FIT+ES với các internal validation windows riêng. Nó không dùng nhầm cache FIT của search.
AutoTS-final tái sử dụng selection predictions khi confirmation dùng cùng seed, không fit lại cell trùng.
CPU vẫn thực hiện chọn/copy cột, chuyển dữ liệu vào estimator, metrics và recursive features cần thiết;
GPU-only áp dụng cho training, không có CPU training fallback.

## AutoTS predict theo batch

Đã đọc source wheel AutoTS 1.0.4 từ PyPI, không cài/import/chạy thư viện trong phiên sửa code.
Feature search fit estimator native từ ma trận đã cache; template bake-off vẫn gọi framework AutoTS.
Rolling outer prediction dùng đường thực thi theo batch:

- WR: dựng các windows và origin covariates thành ma trận; estimator predict cả batch cho ba bước.
- MR: mỗi origin là một cột history riêng khi gọi rolling feature generator. Chuyển feature cuối thành
  một row/origin, gọi estimator theo batch; append predicted return riêng từng origin, lặp ba steps.
  Giữ đúng min_threshold/tail, EWM/rolling, covariate-at-origin và preprocessing mask/scaler native.
- Guard version và các tùy chọn không hỗ trợ (datepart, polynomial/cointegration, transforms cross-series);
  không âm thầm trộn origins hoặc fallback implementation khác.
- Real fits qua GPU wrapper: cả leaf estimators của MultiOutputRegressor và native XGBoost multioutput.
  CUDA input cho XGBoost, backend GPU tường minh, fail khi fallback. Native AutoTS không nuốt lỗi GPU
  rồi coi đó là một template thông thường bị loại.

## Artifact và thời gian

- `lora/`: adapter, forecast_cache, residual_heads có coefficients/scaler/features/partition metadata,
  runtime records tách adapter/cache, calibration forecast/cache, residual fit và prediction/cache.
- `autots_fits/`: estimator và inference state đã fit; không serialize lại toàn bộ FIT/X/Y cho mỗi candidate.
  JSON ghi cache path, frame-building time, fit time, batched prediction time và trace các batches thật.
  Với cache, `prepared_fit` tách `selection_seconds` và `gpu_fit_seconds` (thời gian gọi fit thật, gồm
  transfer/backend setup; không phải CUDA kernel time thuần). `READY.json` ghi preprocessing time và dung lượng.
- `wins/`, feature-search logs và predictions dùng định dạng cũ; hai bảng `tfm_autots_*` tổng hợp hai family.
  RMSE gain dùng mean-seed RMSE; R² OS dùng mean của seed-specific MSE chia cùng E0 MSE, không bình phương
  mean-seed RMSE rồi gọi là mean MSE. Aggregation được ghi rõ, không gọi mean-fold là pooled.
- Timing cache hit không phải native inference latency. Trace batch có số origins; không gọi batch time
  hay batch time/origin là p95 single-request. Đã tắt warmup/latency replay cũ cho phase này.

Chưa có số đo mới để khẳng định mức tăng tốc hoặc API tương thích runtime. Các lỗi runtime phát hiện trong
lượt training thật cần được sửa có provenance; không chạy smoke/test để xác nhận trong phase này.
