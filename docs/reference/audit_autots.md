# AUDIT — AutoTS (cho plan §2.2 #6)

Ngày: 2026-08-29 · researcher · Trạng thái: **chưa cài package, chưa chạy AutoTS** (TRAINING: LOCKED).
Phương pháp: metadata PyPI + đọc **source thật** của wheel `autots-1.0.4-py3-none-any.whl` (tải bằng `pip download --no-deps` vào thư mục tạm, KHÔNG cài vào env) + GitHub `winedarksea/AutoTS@master`. Số dòng trích dẫn là của `autots/models/sklearn.py` trong wheel 1.0.4.
Mọi claim gắn với **autots 1.0.4**; không suy rộng sang version khác.

## 1. Version chốt

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Package | `autots` **1.0.4**, wheel `py3-none-any` (pure python) | PyPI JSON API |
| requires-python | `>=3.9` → **OK với Python 3.12.10** | PyPI metadata |
| Deps bắt buộc | `numpy>=1.21.6`, `pandas>=1.1.0`, **`statsmodels>=0.13.0`**, `scikit-learn>=1.0.0` | PyPI metadata |
| lightgbm / xgboost | **không** phải dep bắt buộc — nằm trong extra `additional`; import lười bên trong `retrieve_regressor` | source |
| Lệnh cài | `pip install autots==1.0.4` (kèm `statsmodels`; lightgbm/xgboost đã có sẵn) | — |

- **`statsmodels` chưa có trong env local** → `import autots` sẽ fail cho tới khi cài (`autots/__init__.py` import chuỗi dài: AutoTS → evaluator → models statsmodels/cassandra…).
- **pandas 3.0.3 / numpy 2.4.4: CHƯA XÁC MINH runtime.** Không có upper-pin trong metadata; đọc source không thấy API đã bị xoá ở pandas 3.0 (`applymap`, `iteritems`, `DataFrame.append`, `is_categorical_dtype`, alias freq `'T'/'H'` hard-code) trong các đường đi ta dùng; `autots.tools.shaping.infer_frequency` chỉ bọc `pd.infer_freq` (trả `'min'` đúng trên pandas 3.0.3, đã test). **Nhưng phải smoke-test sau khi cài** (xem §9).

## 2. Hai model của plan — import path và signature thật

Cả hai đều là tên hợp lệ trong `autots/models/model_list.py` và là class trong `autots/models/sklearn.py`:

```python
from autots.models.sklearn import WindowRegression, MultivariateRegression
```

```python
# dòng 2144
class WindowRegression(ModelObject):
    def __init__(self, name="WindowRegression", frequency='infer', prediction_interval=0.9,
        holiday_country='US', random_seed=2022, verbose=0,
        window_size=10,
        regression_model: dict = {"model": 'RandomForest', "model_params": {}},
        input_dim='univariate', output_dim='forecast_length', normalize_window=False,
        shuffle=False, forecast_length=1, max_windows=5000,
        fourier_encoding_components=None, scale=False, datepart_method=None,
        regression_type=None, n_jobs=-1, **kwargs)
    def fit(self, df, future_regressor=None, static_regressor=None)
    def fit_data(self, df, future_regressor=None)      # KHÔNG refit: chỉ basic_profile + last_window
    def predict(self, forecast_length=None, future_regressor=None, just_point_forecast=False, df=None)
```

```python
# dòng 2822
class MultivariateRegression(ModelObject):
    def __init__(self, name="MultivariateRegression", frequency='infer', prediction_interval=0.9,
        regression_type=None, holiday_country='US', verbose=0, random_seed=2020,
        forecast_length=28,
        regression_model: dict = {"model": 'RandomForest', "model_params": {}},
        holiday=False, mean_rolling_periods=30, macd_periods=None, std_rolling_periods=7,
        max_rolling_periods=7, min_rolling_periods=7, ewm_var_alpha=None, ewm_alpha=0.5,
        additional_lag_periods=None, window=5, datepart_method=None, polynomial_degree=None,
        probabilistic=False, transformation_dict=None, n_jobs=-1, ... )
    def fit(self, df, future_regressor=None, static_regressor=None, regressor_per_series=None)
    def fit_data(self, df, future_regressor=None, static_regressor=None, regressor_per_series=None)
    def predict(self, forecast_length=None, future_regressor=None, just_point_forecast=False,
                df=None, regressor_per_series=None)
```

- `df` là **wide format**: DatetimeIndex + 1 cột mỗi series. Ta có 1 series (`r1`) → `df` shape `(n, 1)`.
- `regression_model` đúng như plan giả định: `{"model": <tên>, "model_params": {...}}`.
- **Tên regressor hợp lệ** (đọc `retrieve_regressor`, dòng 389): `'LightGBM'`, `'LightGBMRegressorChain'`, `'xgboost'` hoặc `'XGBRegressor'`, `'HistGradientBoost'`, `'ExtraTrees'`, `'RandomForest'`, `'DecisionTree'`, `'KNN'`, `'MLP'`, `'ElasticNet'`, `'Ridge'`, `'SVM'`… **Lưu ý viết thường `'xgboost'`** (không phải `'XGBoost'`).

## 3. Chạy CHỈ 2 model cố định, không search

**Dùng thẳng class, KHÔNG dùng `AutoTS(...)`.** Lý do (source `autots/evaluator/auto_ts.py`, `__init__` dòng 189):

- `AutoTS(...)` mặc định `max_generations=25`, `initial_template='General+Random'`, `transformer_list="auto"`, `transformer_max_depth=6`, `num_validations="auto"`, `models_mode="random"`, `ensemble=None` nhưng vẫn sinh template ngẫu nhiên → **là search**, plan cấm ("không AutoTS tự search", §2.2 và §9).
- `AutoTS` tự chia validation theo `validation_method='backwards'` → **không phải fold §1.2** của ta, và không cho điều khiển rolling-origin theo từng origin.
- Ngay cả `max_generations=0` vẫn chạy template ban đầu + transformer search + metric riêng của AutoTS (`metric_weighting` smape/spl/contour…) — không phải MedianGain trên giá của plan §0.

→ **Kết luận**: `from autots.models.sklearn import WindowRegression, MultivariateRegression` rồi tự `fit` / `fit_data` / `predict` trong harness của ta. Cách này cũng đồng nghĩa **không có transformer nào được áp** (đúng ý plan: "transformer cố định tối thiểu, không search"), và fold/metric/seed hoàn toàn do harness kiểm soát.

## 4. Rolling-origin: KHÔNG phải refit 1.400 lần

**Có `fit_data()` ở cả hai class** — chỉ cập nhật cửa sổ dữ liệu, **không train lại**:

- `WindowRegression.fit_data(df)` → `basic_profile(df)` + `self.last_window = df.tail(window_size)`.
- `MultivariateRegression.fit_data(df)` → `basic_profile(df)` + `self.sktraindata = df.tail(self.min_threshold)` (`min_threshold >= 90` với config mặc định).
- `basic_profile` rẻ: chỉ ghi shape/columns/`train_last_date`, và **chỉ suy luận frequency lần đầu** (sau đó `self.frequency` đã là chuỗi nên bỏ qua).
- `create_forecast_index` sinh index `t+1..t+forecast_length` từ `train_last_date` → **origin do ta điều khiển bằng df truyền vào `fit_data`**.

Mẫu dùng cho mỗi fold:

```python
m = WindowRegression(forecast_length=3, window_size=W, output_dim='forecast_length',
                     regression_type='User', regression_model={"model": "LightGBM",
                     "model_params": {...GPU..., "random_state": seed}}, n_jobs=1,
                     max_windows=200_000, random_seed=seed, frequency='min')
m.fit(df_fit, future_regressor=R_fit)                 # 1 lần / fold / candidate
for t in idx_val:                                     # ~1.437 origin
    m.fit_data(df.loc[:t_ts].tail(W + 8))             # KHÔNG refit; chỉ cần đủ window
    f = m.predict(forecast_length=3, future_regressor=R_pred_row_for_t,
                  just_point_forecast=True)           # f: (3, 1) r_hat cho t+1..t+3
```

- `just_point_forecast=True` bỏ qua `Point_to_Probability` ở `WindowRegression` (rẻ hơn). Ở `MultivariateRegression`, `Point_to_Probability` **vẫn được tính** trước khi return (source), nhưng chỉ trên `sktraindata` (~90 dòng) nên rẻ.
- `MultivariateRegression` là model **1 bước**: `Y = df[1:]`, `X` từ `df[:-1]`; `predict` lặp `forecast_length` lần và nối dự báo vào `current_x` (đệ quy). Hợp với plan §0 ("model dự báo one-step return phải cộng dồn").
- `WindowRegression` với `output_dim='forecast_length'` là **multi-output trực tiếp**: một `MultiOutputRegressor` ra 3 giá trị `r_{t+1..t+3}` từ cửa sổ `window_size` giá trị r1 cuối. Cộng dồn → `y_h`.

## 5. External regressor: cách truyền và CĂN THỜI GIAN (điểm dễ leak nhất)

Bật bằng `regression_type='User'` (bắt buộc, nếu không `future_regressor` bị bỏ qua). `future_regressor` là DataFrame DatetimeIndex, cột = feature (ta dùng B0\* + candidate). **Cả hai model đều cần regressor phủ cả các timestamp tương lai t+1..t+3** → khớp với plan ("dự báo 3 bước từ t giữ nguyên giá trị tại t"). Nhưng **hai model căn thời gian KHÁC NHAU**:

| | Lúc `fit` | Lúc `predict` | Recipe để vừa causal (<= s-1) vừa khớp train/serve |
|---|---|---|---|
| `MultivariateRegression` | `cut_regr = regressor_train[1:]` gán vào index của `base = df[:-1]` → hàng feature dự báo bar `s` lấy **regressor tại đúng `s`** (thời điểm target) | `base_regr = concat([regressor_train, future_regressor])` rồi dịch index lùi 1 → cùng quy ước | dựng `R` sao cho **`R.loc[s] = f(s-1)`**; ba hàng tương lai `R.loc[t+1] = R.loc[t+2] = R.loc[t+3] = f(t)` |
| `WindowRegression` | `window_maker` lấy chỉ số `x[:, 0]` = **vị trí ĐẦU cửa sổ** → regressor tại `origin - window_size + 1` (cũ hơn origin `window_size-1` bar) | `future_regressor.tail(1)` — **chỉ hàng cuối**, broadcast | dựng `R` sao cho **`R.loc[s] = f(s + window_size - 1)`** (khi đó cặp train là "f tại origin → target origin+1..+3", causal), và khi predict truyền 1 hàng `= f(t)` |

- **Nếu không dịch** (`R.loc[s] = f(s)`) thì với `MultivariateRegression`, feature dùng để dự báo bar `s` là `f(s)`, mà `f(s)` được tính từ `C_s` — tức chứa chính target `r1_s` → **LEAKAGE**. Phải chặn trong code và test §6.4.
- Với `WindowRegression`, nếu truyền `R.loc[s] = f(s)` thì không leak nhưng regressor **cũ hơn origin `window_size-1` bar** và lệch train/serve so với `.tail(1)`; recipe ở bảng vừa causal vừa khớp (hàng cuối của `R` mà `window_maker` thực sự dùng chỉ tới `i <= n - window_size - forecast_length`, nên `f(i + window_size - 1)` luôn nằm trong df, không chạm ES/VAL).
- `window_maker` lấy mẫu ngẫu nhiên khi số cửa sổ > `max_windows` (mặc định **5000**) — với FIT 10k–16k origin nghĩa là **vứt phần lớn dữ liệu**. Đặt `max_windows` đủ lớn (hoặc `None`) và ghi vào config.

## 6. BUG trong autots 1.0.4 phải vá ở phía ta

`autots/models/sklearn.py:3337` (`MultivariateRegression.fit_data`):

```python
self.regressor_train = future_regressor.reindex(df)      # df là DataFrame, không phải index
```

→ đã test trên pandas 3.0.3: `ValueError: Index data must be 1-dimensional` (sai cả trên pandas 2.x). Nghĩa là **không gọi được `fit_data(df, future_regressor=...)`** khi `regression_type='User'`.

Cách đi vòng (không sửa thư viện): gọi `m.fit_data(df_slice)` **không truyền regressor**, rồi tự gán đúng như `fit` làm:

```python
m.fit_data(df_slice)                                  # cập nhật last_window / sktraindata + train_last_date
m.regressor_train = R.reindex(df_slice.index)         # đúng ngữ nghĩa của fit()
fc = m.predict(forecast_length=3, future_regressor=R_future_3rows, just_point_forecast=True)
```

Lợi ích phụ: giữ `regressor_train` **chỉ ở phần đuôi** (không phải cả FIT 16k x 307) tránh `pd.concat` một frame lớn ở **mỗi** origin trong `MultivariateRegression.predict` — nếu không sẽ là điểm nghẽn tốc độ chính.

## 7. GPU cho `regression_model`

`retrieve_regressor` truyền thẳng `model_params` vào constructor → **mọi tham số GPU đều đi qua được**:

```python
# LightGBM (dòng 471): LGBMRegressor(verbose=-1, random_state=random_seed, n_jobs=..., **model_params)
{"model": "LightGBM", "model_params": {"device_type": "gpu", "n_estimators": 400,
                                       "learning_rate": 0.03, "max_depth": 6}}
# xgboost (dòng 564): xgb.XGBRegressor(verbosity=0, **model_params, n_jobs=...)
{"model": "xgboost", "model_params": {"device": "cuda", "tree_method": "hist",
                                      "n_estimators": 400, "learning_rate": 0.03,
                                      "max_depth": 6, "random_state": 8586}}
```

Ba lưu ý:
1. **Seed**: nhánh LightGBM tự set `random_state=random_seed` (từ `random_seed` của model AutoTS); nhánh **xgboost KHÔNG set** → phải tự truyền `random_state` trong `model_params` (cần cho 3 seed §2.1b).
2. **`device_type="gpu"` của LightGBM cần build OpenCL** (wheel pip mặc định không có) — đúng ràng buộc đã ghi trong MEMORY cho B0; phải dùng cùng build GPU trên Vast.
3. `WindowRegression` với `forecast_length=3` là multi-output → LightGBM bị bọc `MultiOutputRegressor(LGBMRegressor(..., n_jobs=1), n_jobs=n_jobs)`. **Đặt `n_jobs=1` ở model AutoTS** để 3 model con chạy tuần tự trên GPU (tránh 3 process tranh GPU). XGBoost multi-output là native (không bọc) nên không có vấn đề này.

## 8. Khả thi theo plan §2.2 #6

| Mục plan | Verdict | Căn cứ |
|---|---|---|
| Tên `WindowRegression`, `MultivariateRegression` | **ĐÚNG**, `autots.models.sklearn` | source + `model_list.py` |
| `regression_model` LightGBM / XGBoost | **KHẢ THI** (`'LightGBM'`, `'xgboost'`) | `retrieve_regressor` |
| Truyền tham số GPU | **KHẢ THI** (pass-through), lưu ý seed của xgboost và build OpenCL của LightGBM | §7 |
| 2 model cố định, không search | **KHẢ THI** — gọi thẳng class, không dùng `AutoTS(...)` | §3 |
| target r1, `forecast_length=3` → cộng dồn → giá | **KHẢ THI** | §4 |
| base regressor = B0\* + candidate làm regressor | **KHẢ THI** qua `regression_type='User'` + `future_regressor` | §5 |
| Regressor "chỉ từ dữ liệu <= s-1" | **KHẢ THI NHƯNG PHẢI TỰ DỊCH**; dùng nguyên trạng là **LEAKAGE** với `MultivariateRegression` | §5 |
| Rolling-origin không refit 1.400 lần | **KHẢ THI** (`fit_data` + `predict`), nhưng phải vá bug dòng 3337 | §4, §6 |
| Không ép dương | **KHÔNG có** cờ ép dương ở 2 class này (`no_negatives` chỉ có ở `AutoTS(...)` — mà ta không dùng) → không cần làm gì | source |
| Tổng hợp `F*_A1 ∪ F*_A2` sau khi cả hai vòng lặp xong | **KHẢ THI**, thuần harness | — |
| pandas 3.0 / numpy 2.4 | **CHƯA XÁC MINH** | §1, §9 |

## 9. Chi phí (ƯỚC LƯỢNG, chưa đo) và kiến nghị về lưới origin

| Việc | Ước lượng | Ghi chú |
|---|---|---|
| `fit` 1 fold, `WindowRegression` + LightGBM GPU | vài giây – 1 phút | X = (n_windows, window_size + n_regressor) |
| `fit` 1 fold, `MultivariateRegression` + XGBoost GPU | vài giây – 1 phút | X = (n_fit, ~20 feature nội bộ + n_regressor) |
| `fit_data` + `predict` **một origin**, WindowRegression | ~2–6 ms | pandas overhead chi phối; 3 lần `LGBMRegressor.predict` trên 1 dòng |
| `fit_data` + `predict` **một origin**, MultivariateRegression | ~10–30 ms | 3 bước đệ quy, mỗi bước sinh rolling feature trên ~90 dòng |
| 1 candidate = 5 fold x 1.437 origin, WindowRegression | **~1–3 phút** | |
| 1 candidate = 5 fold x 1.437 origin, MultivariateRegression | **~3–10 phút** | với điều kiện đã cắt `regressor_train` (§6) |
| 39 candidate + base, mỗi model | **~1–4 h** | plan ước 2–4 h/model → **khớp hoặc rẻ hơn** |

**Kiến nghị methodology (thuộc §3 + §6.6)**: plan dự tính chấm AutoTS trên "lưới origin thưa mỗi 5 phút". Nhưng §3 so win_m với champion bằng `Gain = 1 - RMSE_win / RMSE_champion` **trên cùng 15 ô** — nếu AutoTS chấm trên tập origin khác champion thì hai bảng RMSE không so được. Theo ước lượng trên, **chạy đủ 7.185 origin là khả thi** → đề xuất bỏ lưới thưa, dùng đúng tập origin của mọi model. Nếu vì lý do thời gian phải dùng lưới thưa, thì **bắt buộc tính lại RMSE của champion trên đúng lưới thưa đó** trước khi so; ghi rõ trong `champion_log.csv`. Đây là ràng buộc của plan hiện có, không phải đề xuất mở rộng.

## 10. Rủi ro cài đặt và điểm CHƯA XÁC MINH

1. **pandas 3.0.3 / numpy 2.4.4 chưa được kiểm chứng runtime.** AutoTS 1.0.4 không pin trần; đọc source không thấy API đã bị xoá ở pandas 3.0 trong đường đi ta dùng, nhưng `autots/__init__.py` import rất rộng (statsmodels, cassandra, transform, mcp…) nên rủi ro nằm ở **import time**, không phải ở 2 class ta dùng. Smoke test tối thiểu sau khi user cho phép cài:
   ```
   import autots; print(autots.__version__)
   from autots.models.sklearn import WindowRegression, MultivariateRegression
   # df 1 series, 500 bar, freq 'min'; regression_type='User', regressor 3 cột
   # fit → fit_data → predict(forecast_length=3, just_point_forecast=True) → shape (3,1)
   ```
   Nếu import fail vì pandas 3.0 → báo cho user, **không tự hạ pandas** (pandas 3.0.3 là môi trường đã chốt của B0/harness).
2. `statsmodels` là dep bắt buộc, sẽ được cài thêm.
3. Bug `fit_data` dòng 3337 (§6) — vá ở phía ta, **không sửa thư viện** (sửa site-packages là không tái tạo được).
4. `n_jobs=-1` mặc định + LightGBM GPU → nhiều process tranh GPU; đặt `n_jobs=1` (§7).
5. `max_windows=5000` mặc định cắt dữ liệu train của `WindowRegression` (§5).
6. Chưa đo tốc độ thật; §9 là ước lượng.
7. `create_forecast_index` / `infer_frequency` với freq `'min'`: `pd.infer_freq` trả `'min'` đúng trên pandas 3.0.3 (đã test tay), nhưng **chuỗi phải liên tục** — dữ liệu ta đã kiểm không gap (§1.1), và df truyền vào `fit_data` phải là lát liên tục kết thúc tại t.

## 11. Việc kế tiếp

- **coder**: implement `models_autots.py` thay `models_pending.pending("autots_wr"/"autots_mr")`:
  - batch-object mang `df` (wide, index datetime, 1 cột `r1`), ma trận regressor `R` **đã dịch đúng theo từng model** (§5), và `rv60`/`TargetTransform` để trả `pred_z` (giống TimesFM: `pred_z = ((cumsum(r_hat) / denom) - mean) / scale`).
  - vá bug `fit_data` theo §6; cắt `regressor_train` về phần đuôi.
  - config cố định (không search, không transformer), `n_jobs=1`, `max_windows` lớn, seed truyền qua `random_seed` + `model_params['random_state']` (xgboost).
- **checker**: test §6.4 riêng cho AutoTS — với candidate `f`, thay đổi giá trị `f` tại các bar `> t` **không được** đổi prediction tại origin `t`; và kiểm tra `R.loc[s]` đúng bằng `f(s-1)` (MR) / `f(s + window_size - 1)` (WR).
- **main-controller**: quyết 2 việc — (a) bỏ lưới origin thưa cho AutoTS (khuyến nghị: bỏ, xem §9); (b) cho phép cài `autots==1.0.4` + `statsmodels` để chạy smoke import trên pandas 3.0.3.

---

# §12 — Framework search `AutoTS(...)` với feature set đã FREEZE (audit 2026-08-31)

Ngày: 2026-08-31 · researcher · **chưa cài package, chưa chạy AutoTS** (TRAINING: LOCKED).
Phương pháp: đọc **source thật** của wheel `autots-1.0.4-py3-none-any.whl` (`pip download --no-deps` vào thư mục tạm, KHÔNG cài vào env). Mọi số dòng là của wheel 1.0.4 (`autots/evaluator/auto_ts.py`, `auto_model.py`, `validation.py`, `autots/models/sklearn.py`, `model_list.py`, `autots/tools/transform.py`). Mọi claim gắn với **autots 1.0.4**.

**Bối cảnh**: thiết kế mới (user chốt 2026-08-31) — WR/MR cố định chỉ là *probe* tìm feature (add-one → prune → F_WR, F_MR, union → F_WR_best / F_MR_best); sau đó **freeze feature set** và chạy `AutoTS(...)` search riêng cho từng frozen set; mỗi fold search **chỉ trên training-side**, freeze template, rolling predict outer VAL, chấm bằng metric của project. Lưu ý: plan rev 9b (dòng 204, 429 của `docs/RESEARCH_PLAN.md`) **đang cấm** "AutoTS tự search" — mục này là audit tính khả thi, không phải đã sửa plan.

## 12.1 Chữ ký thật của `AutoTS.__init__` / `fit` (auto_ts.py:189, :1249)

```python
AutoTS(forecast_length=14, frequency='infer', prediction_interval=0.9, max_generations=25,
       no_negatives=False, constraint=None, ensemble=None, initial_template='General+Random',
       random_seed=2022, holiday_country='US', subset=None, aggfunc='first', na_tolerance=1,
       metric_weighting={...13 khoa...}, drop_most_recent=0, drop_data_older_than_periods=None,
       model_list='scalable',            # <- MAC DINH 1.0.4 la 'scalable', KHONG phai 'default'
       transformer_list="auto", transformer_max_depth=6, models_mode="random",
       num_validations="auto", models_to_validate=0.15, max_per_model_class=None,
       skip_slow_models_seconds=None, validation_method='backwards', min_allowed_train_percent=0.5,
       remove_leading_zeroes=False, prefill_na=None, introduce_na=None, preclean=None,
       model_interrupt="stop", generation_timeout=None, current_model_file=None, force_gc=False,
       horizontal_ensemble_validation=True, custom_metric=None, verbose=1, n_jobs=0.5)

AutoTS.fit(df, date_col=None, value_col=None, id_col=None, future_regressor=None,
           weights={}, result_file=None, grouping_ids=None, validation_indexes=None)
AutoTS.fit_data(df, date_col=None, value_col=None, id_col=None, future_regressor=None, weights={})
AutoTS.predict(forecast_length="self", prediction_interval='self', future_regressor=None,
               hierarchy=None, just_point_forecast=False, fail_on_forecast_nan=True,
               verbose='self', df=None)
```

- `initial_template` **chấp nhận `pd.DataFrame`** (auto_ts.py:374 → `self.import_template(df, method='only')`) — cửa duy nhất để nạp template do ta soạn. Sau đó `__init__` lọc theo `model_list` (:417) và **cắt bớt transformer** không thuộc `transformer_list` / vượt `transformer_max_depth` (:426–466) — chỉ cắt bớt, không bao giờ thêm. Template phải có đủ khoá `{"fillna":..., "transformations":{}, "transformation_params":{}}` trong `TransformationParameters` (:440 raise nếu thiếu).
- `metric_weighting` chỉ nhận khoá trong `all_valid_weightings` (auto_model.py:3171) — `{'rmse_weighting': 1}` hợp lệ.
- `no_negatives=False` mặc định (KHÔNG ép dương) → đúng cho signed log-return; giữ `False`, `constraint=None`.
- Phải khai báo tường minh `drop_most_recent=0`, `introduce_na=False`, `prefill_na=None`, `preclean=None`: `introduce_na=None` + NaN ở 2 hàng cuối sẽ **tự chèn NaN** vào đuôi train của validation (auto_ts.py:2280–2290).
- `n_jobs=0.5` = một nửa số core (`tools/cpu_count.py:60`). **`TemplateWizard` chạy các model TUẦN TỰ** (`for row in template_dict`, auto_model.py:~2190) → không song song giữa model; `n_jobs` chỉ đi vào trong model.

### `future_regressor` ở tầng `AutoTS`: truyền được, nhưng KHÔNG bảo đảm được dùng

- `fit(future_regressor=R)` → `fit_data` lưu `self.future_regressor_train` (:1204). Nếu `R.shape[0] != df.shape[0]` thì **in cảnh báo rồi `reindex(..., fill_value=0)`** (:1193–1200) — im lặng biến thành 0 → phải truyền `R` đúng bằng index của `df`.
- `R` sau đó được `reindex` theo index train/test của **từng validation split** (:1383–1388, :2269–2274) rồi bơm vào `TemplateWizard`.
- **Chỉ model có `regression_type='User'` trong `ModelParameters` mới thực sự dùng nó.** `regression_type` **do search sinh ngẫu nhiên**: `WindowRegression.get_new_params` (sklearn.py:2517) → `None` 80% / `'User'` 20%; `MultivariateRegression.get_new_params` (sklearn.py:3683) → `None` 70% / `'User'` 30%. Ngoài ra phần lớn model trong `model_list` mặc định (`'scalable'`) **không nhận regressor** (danh sách nhận: `model_list.py:395`).
- **Knob duy nhất ép được**: `models_mode='regressor'` → cả hai class kiểm tra `if "regressor" in method: regression_type = "User"` (sklearn.py:2500–2501, :3680–3681). Không có mode nào vừa ép `'User'` vừa ép họ regressor (`generate_regressor_params` so khớp `method` bằng `==`, sklearn.py:1108).

## 12.2 Leakage ra ngoài `df`: KHÔNG. Mọi validation nội bộ nằm TRONG `df` được truyền vào

- `fit` gọi `fit_data(df)` → `self.df_wide_numeric`; toàn bộ index validation sinh bởi `generate_validation_indices` (`evaluator/validation.py:91`) và **mọi nhánh đều là lát của `df_wide_numeric.index`**: `backwards` → `idx[0 : n-(y+1)*fl]` (:180); `even` → `idx[0 : size*(y+1)+fl]` (:189); `seasonal n` → `idx[0:val_per]` (:202); `similarity` → `df.index[df.index <= indx[-1]]` (:151); `seasonal` → `df.index[0:x[-1]]` (:163); `mixed_length` → tuple lát của `idx` (:204–229). `_run_validations` chỉ `df_wide_numeric.reindex(cval_idx)` (:2216–2220). **Không có tham số nào nạp thêm dữ liệu ngoài `df`.**
- Ngoại lệ duy nhất "dữ liệu ngoài": `holiday_country='US'` (package `holidays`) và `datepart_method` — lịch/thời gian, không phải giá tương lai. Khuyến nghị `holiday_country=None` (`tools/seasonal.py:137 date_part(holiday_country=None)` chấp nhận None).
- **Kết luận Q2**: truyền `df` = training-side của fold (kết thúc cuối ES, trước purge 60') là ĐỦ để chặn leakage outer VAL. Với fold §1.2, holdout nội bộ của AutoTS rơi vào **cuối ES**, cách VAL ≥ 60' → hợp purge.
- **CẢNH BÁO methodology (không phải leakage)**: với `forecast_length=3` + `validation_method='backwards'`, mỗi holdout nội bộ chỉ dài **3 bar**. `num_validations='auto'` → 3 (validation.py:72) ⇒ AutoTS chọn template dựa trên **4 × 3 = 12 điểm**. `num_validations='max'` → tối đa 50 ⇒ 153 điểm nhưng nhân số fit lên ~51×. `validation_method='custom'` với tuple `(train_idx, test_idx)` dài hơn sẽ khiến `_run_template` đặt `forecast_length=len(df_test)` (:2087) — tức dự báo một mạch 1.437 bước, KHÔNG phải rolling 3 bước → **không dùng được** để bắt chước protocol của ta.

## 12.3 Freeze template → rolling predict outer VAL (Q3)

### Artifact và thuộc tính sau `fit`
`best_model` (DataFrame 1 dòng: `ID, Model, ModelParameters, TransformationParameters, Ensemble`), `best_model_name`, `best_model_params` (dict), `best_model_transformation_params` (dict), `best_model_id`, `regressor_used` (auto_ts.py:2004–2029 `parse_best_model`). `export_template(filename, models='best'|'all', n, include_results)` (:2576) ghi `.csv`/`.json` (`save_template`, :2746). `import_best_model(path_or_df)` (:2904) nạp lại + `parse_best_model`. `current_model_file="<path>"` ghi `<path>.json` cho model đang chạy (auto_model.py:875–896).

### KHÔNG dùng đường `AutoTS.fit_data()/predict()` cho rolling — 4 lý do từ source
1. `_predict` chỉ đi đường "không refit" khi `best_model_name in update_fit` (`model_list.py:461` = `MultivariateRegression, DatepartRegression, GluonTS, WindowRegression, Cassandra, PreprocessingRegression`). Ngoài danh sách đó → `model_forecast(...)` = **refit đầy đủ mỗi origin** (auto_ts.py:2398–2422).
2. Đường "không refit" gọi `ModelPrediction.fit_data(use_data, future_regressor=use_regr_train)` (:2394) → `MultivariateRegression.fit_data` → **bug `sklearn.py:3337` `future_regressor.reindex(df)`** → `ValueError: Index data must be 1-dimensional`. MR + `regression_type='User'` **crash ngay**.
3. `ModelPrediction.fit_data` (auto_model.py:1028–1030) **KHÔNG áp transformer**: `self.model.fit_data(df, future_regressor)` với `df` **thô**, trong khi model bên trong đã fit trên `self.transformer_object._fit(df)` = dữ liệu **đã transform** (:898, :906). ⇒ khi template có transformer khác rỗng, đường update nhanh cho dữ liệu **sai thang**. Chỉ an toàn khi `transformations == {}`.
4. `AutoTS.fit_data` mỗi origin còn chạy `df_cleanup` + `NumericTransformer` + `profile_time_series` + `validate_num_validations` + `generate_validation_indices` (:1087–1232) — vô ích, ~10–50 ms × 14.370 lời gọi.

### Pseudo-code CHUẨN: (a) search → (b) freeze → (c) rolling predict

```python
# ---------- (a) SEARCH: chi tren training-side cua fold (FIT + ES, ket thuc truoc purge 60') ----------
from autots import AutoTS
df_tr = wide(r1, bars[: end_ES])                    # 1 cot 'r1', DatetimeIndex freq 'min', KHONG cham purge/VAL
R_tr  = regressor_frame(F_frozen, df_tr.index, shift=SHIFT)   # §5: MR R.loc[s]=f(s-1); WR R.loc[s]=f(s+W-1)

auto = AutoTS(
    forecast_length=3, frequency='min',
    model_list=['WindowRegression', 'MultivariateRegression'],  # chi 2 ho da audit alignment (§5)
    models_mode='regressor',              # EP regression_type='User' o MOI candidate (sklearn.py:2500, 3680)
    initial_template='Random',            # 'General' chi co 1/42 dong dung regressor -> vo dung o day
    max_generations=G, generation_timeout=T_min, skip_slow_models_seconds=S,
    num_validations='auto', validation_method='backwards',
    metric_weighting={'rmse_weighting': 1},
    ensemble=None, no_negatives=False, constraint=None, drop_most_recent=0, subset=None,
    introduce_na=False, prefill_na=None, preclean=None, holiday_country=None,
    transformer_list=[], transformer_max_depth=0,        # xem RUI RO T (12.4)
    random_seed=selection_seed, n_jobs=1, verbose=0,
    current_model_file=f"{out}/autots_cur_f{f}_{setname}",
)
auto.fit(df_tr, future_regressor=R_tr)     # R_tr phai TRUNG index df_tr (lech -> fill_value=0 am tham)

# ---------- (b) FREEZE ----------
auto.export_template(f"{out}/tmpl_best_f{f}_{setname}.csv", models='best', n=1)
auto.export_template(f"{out}/tmpl_all_f{f}_{setname}.csv",  models='all')   # log toan bo candidate da thu
name, params = auto.best_model_name, auto.best_model_params
trans        = auto.best_model_transformation_params
assert str(params.get('regression_type')).lower() == 'user'   # neu khong -> template BO QUA F_frozen -> loai
assert not trans.get('transformations')                        # co transformer -> khong dung duoc duong nhanh
assert params.get('datepart_method') in (None, 'None')         # datepart = them cot NGOAI F_frozen (12.4)
params['regression_model']['model_params'].update(GPU_KW)      # ep GPU thu cong sau khi search (12.5)

# ---------- (c) ROLLING PREDICT tren outer VAL, template da freeze, KHONG refit ----------
from autots.evaluator.auto_model import ModelMonster
W = int(params['window_size']) if name == 'WindowRegression' else None
R = regressor_frame(F_frozen, full_index, shift=(W - 1) if W else -1)   # DUNG LAI theo window_size DA SEARCH
m = ModelMonster(name, parameters=params, frequency='min', forecast_length=3,
                 prediction_interval=0.9, holiday_country=None,
                 random_seed=seed, verbose=0, n_jobs=1)
m.fit(df_tr, future_regressor=R.reindex(df_tr.index))          # MOT LAN / fold
for t in val_origins:                                          # 1.437 origin
    sl = df.loc[:t].tail(TAIL_BARS)                            # chi tau <= t
    m.fit_data(sl)                                             # cap nhat cua so, KHONG train lai
    if name == 'MultivariateRegression':
        m.regressor_train = R.reindex(sl.index)                # va bug sklearn.py:3337 (dung ngu nghia fit())
    fc = m.predict(forecast_length=3, future_regressor=three_rows_of(f_at(t)),
                   just_point_forecast=True)
    y1, y2, y3 = np.cumsum(np.asarray(fc)[:, 0])               # one-step r_hat -> y_h; P_hat = C_t*exp(y_h)
```

- **Không refit lần nào trong vòng lặp**: `WindowRegression.fit_data` = `basic_profile` + `last_window = df.tail(window_size)` (sklearn.py:2311–2314); `MultivariateRegression.fit_data` = `basic_profile` + `sktraindata = df.tail(min_threshold)` (:3326–3342). Đây đúng bằng vòng lặp đang có trong `src/p0/models_autots.py` — **delta code duy nhất là `_make()` lấy params từ template đã search thay vì hằng số**, cộng việc dựng lại `R` theo `window_size` mới.
- `output_dim` do search chọn đều chạy được với 3 hàng future regressor giá trị `f(t)`: nhánh `'forecast_length'` dùng `future_regressor.tail(1)` (sklearn.py:2434), nhánh `'1step'` dùng `future_regressor.reindex(index).iloc[x]` (:2380) — vì cả 3 hàng đều bằng `f(t)` nên hai nhánh cho cùng giá trị.
- Chi phí rolling: như §9 (WR ~2–6 ms/origin, MR ~10–30 ms/origin) → 5 fold × 1.437 origin ≈ **2–10 phút/frozen set**. Không phải nút thắt.

## 12.4 GPU-only (Q4) — VERDICT

**a. Model nhận `regression_model={"model":..., "model_params":{...}}`** (⇒ ép được LightGBM GPU / XGBoost `device='cuda'`): `RollingRegression` (sklearn.py:1732 — **đã bị comment khỏi mọi `model_list`**, model_list.py:399,431), `WindowRegression` (:2163), `DatepartRegression` (:2602), `MultivariateRegression` (:2852), `PreprocessingRegression` (:4023). `model_lists['regressions'] = ['WindowRegression','DatepartRegression','MultivariateRegression','PreprocessingRegression']` (model_list.py:430). Không có `UnivariateRegression`; `MultivariateMotif` **không** nhận `regression_model`. Hai **transformer** cũng dựng regressor sklearn riêng: `DatepartRegressionTransformer` (transform.py:1310), `BTCD` (:3240).

**b. Có ép được MỌI candidate trong search dùng regressor GPU không → KHÔNG.** Bằng chứng dứt khoát:
1. `generate_regressor_params` (sklearn.py:1081+) **không bao giờ sinh khoá `device` / `device_type`**. Toàn bộ `model_params` ngẫu nhiên là mặc định CPU (LightGBM CPU; `xgb.XGBRegressor` không `device` → CPU).
2. Danh sách regressor được bốc (`sklearn_model_dict` :761, `multivariate_model_dict` :785) gồm `ElasticNet, MLP, DecisionTree, KNN, Adaboost, SVM, BayesianRidge, HistGradientBoost, ExtraTrees, RadiusNeighbors, PoissonRegresssion, RANSAC, Ridge, RandomForest` — **sklearn thuần, không có đường GPU nào**. Chỉ `LightGBM`/`xgboost` (xác suất ~0,06–0,15) mới có khái niệm GPU.
3. Biến `gpu` ở sklearn.py:876 là `['Transformer','KerasRNN','MLP','ElasticNetwork']` (= "no dnn"), **không liên quan tree-GPU**; `model_lists['gpu']` (model_list.py:252) là danh sách model DNN. Cả hai không giúp gì.
4. Kể cả khi nạp `initial_template` GPU, `max_generations >= 1` sẽ **phá** params: `MultivariateRegression` nằm trong `recombination_approved` (model_list.py:344) → `NewGeneticTemplate` dùng `dict_recombination(a, b)` với `b` có thể là `ModelMonster(...).get_new_params(...)` mới (auto_model.py:2945–2965) → khoá `regression_model` (kèm `device`) bị ghi đè bởi params CPU. `WindowRegression` **không** trong `recombination_approved` (:340 đã comment) → rơi vào `random.choice([c0, get_new_params()])` (auto_model.py:2990–2992) → **50% mỗi con là params CPU ngẫu nhiên**.
5. `models_mode`: `'gradient_boosting'` thu hẹp về `{xgboost, HistGradientBoost, LightGBM, LightGBMRegressorChain}` (sklearn.py:877, 1102) — **vẫn CPU** (HistGradientBoost CPU-only) và **loại trừ** `models_mode='regressor'`. `model_interrupt` chỉ xử lý Ctrl+C. Không có tham số nào giới hạn param space của `regression_model`.
6. Transformer search có thể kéo `DatepartRegressionTransformer`/`BTCD` → thêm model sklearn CPU ngay trong pipeline transform.

**Kết luận GPU-only**: **KHÔNG thể giới hạn search space của AutoTS 1.0.4 thành GPU-compatible một cách sạch.** Cấu hình duy nhất 100% GPU là `initial_template = <DataFrame do ta soạn, mọi dòng model_params GPU>` + `max_generations=0` + `model_list=['WindowRegression','MultivariateRegression']` + `transformer_list=[]`. Khi đó AutoTS chỉ còn là **"bake-off template do ta liệt kê + validation nội bộ + chấm/chọn best của AutoTS"**, **không còn genetic search**; gọi là "AutoTS framework search" là không đúng.

**c. Nếu chấp nhận search thật thì cái gì chạy CPU** (theo trọng số trong source, `model_list=['WindowRegression','MultivariateRegression']`, `models_mode='regressor'`):
- `regression_model` bốc từ `sklearn_model_dict`/`multivariate_model_dict`: P(LightGBM) ≈ 0,19/0,11, P(xgboost) ≈ 0,10/0,02 → **~70–85% candidate dùng regressor sklearn CPU-only**; và ngay cả nhánh LightGBM/xgboost cũng **chạy CPU** vì không có khoá `device` ⇒ thực tế **~100% số fit trong search chạy CPU**, trừ khi vá params sau mỗi lần sinh (monkey-patch, không sạch).
- Quy mô: `df` 1 series × 9,9k–17,1k dòng; `X` = (n_window ≤ max_windows, window_size ≤ 90 + k) với k = |F_frozen| ≈ 40–310 cột (MR thêm ~20 cột rolling nội bộ) → ma trận ~15.000 × 100–400 float64 ≈ 12–48 MB; vừa RAM, nhưng SVM/KNN/RandomForest/RadiusNeighbors trên cỡ này là hàng phút tới hàng chục phút mỗi fit.
- Nếu mở `model_list` rộng hơn (`'scalable'`, `'fast'`, `'default'`): thêm statsmodels ETS/GLM/VAR/ARDL/UnobservedComponents, Prophet… — CPU 100% **và** phần lớn **không nhận regressor** ⇒ bỏ qua `F_frozen`, thí nghiệm mất ý nghĩa.

**d. Lựa chọn để user quyết** (không tự chọn hộ):

| # | Cấu hình | GPU-only? | Còn là "search"? | Feature set thật sự được dùng? |
|---|---|---|---|---|
| A | `initial_template=<ta soạn, GPU>`, `max_generations=0`, `model_list=['WindowRegression','MultivariateRegression']`, `transformer_list=[]` | **CÓ** | Không (bake-off template cố định + validation của AutoTS) | Có (ta ép `regression_type='User'`, `datepart_method=None`) |
| B | `models_mode='regressor'`, `model_list` 2 model, `max_generations=1–3`, `transformer_list=[]` | **KHÔNG** (≈100% fit trên CPU) | Có (genetic thật) | Có `regression_type='User'`, **nhưng** `datepart_method` ngẫu nhiên thêm cột (WR 10%, MR **80%**) và `holiday=True` (MR 10%) |
| C | `model_list='scalable'` mặc định | KHÔNG | Có | **Không** — đa số model bỏ qua regressor |
| D | Không chạy framework search; giữ plan §2.2 #6 (2 model cố định) | CÓ | — | Có |

→ **A và D là hai phương án duy nhất không vi phạm invariant "training chỉ trên GPU, cấm CPU training"** (`.claude/CLAUDE.md`; plan §0 dòng 38). B/C cần user **nới invariant GPU-only** một cách tường minh.

**RỦI RO T (feature set không thật sự freeze)** — kể cả phương án B:
- `datepart_method` khác None → `date_part(df.index, ...)` được **concat vào `future_regressor`** ở cả fit và predict (sklearn.py:2231–2249 cho WR); MR có `add_date_part_choice` xác suất **0,8** (sklearn.py:3663) ⇒ **thêm cột ngoài `F_frozen`**.
- MR còn có `holiday=True` (10%), `transformation_dict` nội bộ (25%), `scale_full_X`, `frac_slice` (bỏ bớt dữ liệu train).
- WR có `fourier_encoding_components` (10%) → biến đổi ngẫu nhiên toàn bộ X; `scale=True` (70%); `max_windows` bốc `5000` (p≈0,15) → **cắt còn 5.000 cửa sổ** trong ~15k; `window_size` tự giảm nếu quá dài (sklearn.py:2259–2265).
⇒ Muốn "feature set TUYỆT ĐỐI không đổi" thì phải **assert sau khi freeze** (`datepart_method is None`, `holiday is False`) và loại template vi phạm — loại xong thì search còn rất ít, quay về phương án A.

## 12.5 Ép GPU cho template ĐÃ FREEZE (hợp lệ, không đụng search)

`retrieve_regressor` truyền thẳng `model_params` vào constructor (LightGBM sklearn.py:471; xgboost :564) → sau khi có `best_model_params`, ta **update** `params['regression_model']['model_params']`:

```python
GPU_KW = {"LightGBM": {"device_type": "gpu"},
          "xgboost":  {"device": "cuda", "tree_method": "hist", "random_state": seed}}  # nhanh xgboost KHONG tu set seed
```

Nếu `params['regression_model']['model']` **không** thuộc `{LightGBM, xgboost, XGBRegressor}` (≈70–85% khả năng, 12.4c) thì **template thắng không có phiên bản GPU** → refit outer VAL của nó là **CPU training** (vi phạm invariant). Đây là điểm chết của phương án B: không chỉ search chạy CPU, mà **model cuối cùng cũng có thể CPU-only**. Giữ `n_jobs=1` để `MultiOutputRegressor(LGBMRegressor(n_jobs=1), n_jobs=n_jobs)` (sklearn.py:471) không cho 3 model con tranh GPU (đã ghi §7).

## 12.6 Chi phí ước lượng (Q5) — ƯỚC LƯỢNG, chưa đo

Số lần fit trong **một** `AutoTS.fit` với `model_list` 2 model, `initial_template='Random'`:
- initial template = `RandomTemplate(len(model_list) * 12)` = **24 dòng** (auto_ts.py:380);
- mỗi generation: `top_n = num_mod_types * max_per_model_class_g` = 2 × 5 = **10** (auto_ts.py:1451–1455);
- validation: `models_to_validate=0.15` × số kết quả, × `num_validations` (auto = 3) lần (auto_ts.py:1704–1757).

| max_generations | số fit / search | thời gian / search (giả định 2 phút/fit) |
|---|---|---|
| 0 | 24 + 3·⌈0,15·24⌉ = **36** | ~1,2 h |
| 1 | 34 + 3·⌈0,15·34⌉ = **52** | ~1,7 h |
| 3 | 54 + 3·⌈0,15·54⌉ = **81** | ~2,7 h |

"2 phút/fit" = trung vị giả định cho X ≈ 15.000 × (10–90 + 40–310) trên CPU với `n_jobs=1`; **đuôi rất nặng** (`n_estimators` bốc tới 1794 ở `lightgbmp1`, sklearn.py:933; SVM/KNN/RandomForest trên 15k×300 có thể 10–30 phút/fit) → phải dùng `skip_slow_models_seconds` và `generation_timeout`.

**Tổng cho 5 fold × 2 frozen set** (nếu `F_WR_best == F_MR_best` thì dedup còn 1 set → chia đôi):

| max_generations | 5 fold × 2 set | + rolling predict | Tổng |
|---|---|---|---|
| 0 | ~12 h | ~0,3 h | **≈ 12–15 h** |
| 1 | ~17 h | ~0,3 h | **≈ 17–20 h** |
| 3 | ~27 h | ~0,3 h | **≈ 27–35 h** |

So sánh: ngân sách của **cả plan** hiện là ≈ 12–25 h máy (MEMORY → Pitfalls). Thêm framework search = **nhân đôi tới nhân ba** tổng chi phí, và phần lớn thời gian đó là CPU (trả tiền giờ GPU của Vast để GPU idle).
**3 seed §2.1b**: nếu search lại cho từng seed thì ×3 (36–105 h). Khuyến nghị đúng theo §1.3: **search một lần ở `selection_seed`, freeze template, 3 evaluation seed chỉ refit template đã freeze** — search là một bước *selection*, không phải bước đo nhiễu.

## 12.7 Artifact (Q6)

- `auto.export_template(f, models='best', n=1)` → 1 dòng `[ID, Model, ModelParameters, TransformationParameters, Ensemble]`; `models='all'` → mọi candidate đã thử; `include_results=True` kèm metric nội bộ. `.csv` hoặc `.json` (`save_template`, auto_ts.py:2746–2762).
- Nạp lại: `AutoTS(...).import_best_model(path)` (:2904) hoặc `import_template(path, method='only')` (:2822). Với đường chạy ở 12.3 chỉ cần `json.loads(row['ModelParameters'])` → `ModelMonster(name, parameters=params, ...)`.
- Ghi thêm cho mỗi (fold, frozen set): `random_seed`, `models_mode`, `model_list`, `max_generations`, `num_validations`, `metric_weighting`, số candidate thử/thất bại (`auto.failure_rate()`), và **`regression_type` + `datepart_method` của template thắng** (bằng chứng feature set có thật sự được dùng và không bị nới).
- `auto.results('validation')` / `auto.initial_results.model_results` để log điểm nội bộ — **chỉ diagnostic**, không được dùng thay MedianGain (plan §0).

## 12.8 pandas 3.0.3 (Q7)

Quét toàn bộ `autots/` trong wheel 1.0.4: **không** có `.applymap(`, `.iteritems(`, `is_categorical_dtype`, `is_sparse`, `is_datetime64tz_dtype`, hay `DataFrame.append` (mọi `.append(` đều là list). Alias freq cũ (`'M'/'H'/'T'`) chỉ xuất hiện ở `autots/models/gluonts.py:194` (`gluon_freq = "M"`) — model không nằm trong đường ta dùng và `gluonts` không cài. ⇒ Đường `AutoTS(...)` **không có API pandas đã bị xoá ở 3.0** theo grep tĩnh.
Rủi ro còn lại vẫn là **import-time** (`autots/__init__.py` import `datasets, auto_ts, event_forecasting, transform, shaping, regressor, mlflow, auto_model, anomaly_detector, cassandra, impute, feature_detector`) và **runtime của statsmodels/sklearn** khi search chạm model ngoài họ regression. **CHƯA XÁC MINH bằng chạy thật** — cần smoke import + 1 search tí hon (df 500 bar, `max_generations=0`, `model_list=['WindowRegression']`) sau khi user cho phép cài `autots==1.0.4` + `statsmodels`.

## 12.9 Việc kế tiếp

- **main-controller / user quyết**: (a) có nới invariant "training chỉ GPU" để chạy search thật (phương án B) hay không; (b) nếu không nới → chọn A (bake-off template GPU, `max_generations=0`) hay D (giữ plan §2.2 #6); (c) ngân sách giờ Vast (+12–35 h so với tổng 12–25 h hiện tại); (d) sửa `docs/RESEARCH_PLAN.md` dòng 204/429 nếu bỏ lệnh cấm "AutoTS tự search".
- **coder** (chỉ khi user chốt): thêm `AutoTSSearchModel` dùng lại nguyên vòng lặp rolling của `src/p0/models_autots.py`, `_make()` lấy từ template đã freeze; dựng lại `R` theo `window_size` đã search; assert `regression_type=='User'`, `transformations=={}`, `datepart_method is None`, `holiday is False`.
- **checker**: test §6.4 cho đường search — (i) đổi giá trị feature ở bar `> t` không đổi prediction tại `t`; (ii) `df` truyền vào `AutoTS.fit` có `max(index) < VAL_start − 60'`; (iii) template thắng phải có `regression_type='User'`; (iv) prediction từ `ModelMonster(template)` khớp với `auto._predict` trên 1 origin (kiểm tra freeze đúng).

---

## §12.10 — PHƯƠNG ÁN A: nạp `initial_template` GPU (API chính xác) [audit 2026-08-31]

(User chốt phương án A ở §12.4d. Đánh số §12.10 vì §12.7 đã dùng cho Artifact.) Vẫn đọc source wheel 1.0.4, chưa cài, chưa chạy.

### A1. `import_template` — chữ ký thật (auto_ts.py:2822)

```python
AutoTS.import_template(filename, method="add_on", enforce_model_list=True,
                       include_ensemble=False, include_horizontal=False, force_validation=False)
```
- `filename` nhận **str path (`.csv` / `.json`) HOẶC thẳng `pd.DataFrame`** — `load_template` (:2764) có nhánh `isinstance(filename, pd.DataFrame) → filename.copy()`.
- `method`: `'add_on'|'add on'|'addon'` → **merge** vào `self.initial_template` đã sinh trong `__init__` (giữ cả template ngẫu nhiên); `'only'|'user only'|'user_only'|'import_only'` → **thay hẳn** `self.initial_template = import_template` (:2885). Giá trị khác → `return ValueError(...)` (trả về, không raise — bẫy im lặng).
- `enforce_model_list=True` → `_enforce_model_list` (:2788) bỏ dòng có `Model` ngoài `self.model_list`; **raise ValueError nếu còn 0 dòng** khi `method='only'`.
- `include_ensemble=False` → gọi `unpack_ensemble_models(..., keep_ensemble=False)`; với `Ensemble=0` là no-op (auto_model.py:1238).
- `force_validation=True` → mọi dòng import được đẩy thẳng vào cross-validation bất kể điểm vòng đầu (`self.validate_import`, :2890 → dùng ở :1785).
- Docstring dòng 2832: **"Must be done before the AutoTS object is .fit()"** — đúng, `fit` đọc `self.initial_template` ở :1396.
- **Đơn giản hơn**: truyền thẳng DataFrame vào `initial_template=` của constructor — `__init__:374` tự gọi `self.import_template(df, method='only')`. Dùng cách này thì **không** có template ngẫu nhiên nào được sinh.

### A2. Schema template (bắt buộc)

- Cột: `['Model', 'ModelParameters', 'TransformationParameters', 'Ensemble']` (`self.template_cols`, auto_ts.py:291). `ID` **tuỳ chọn** — `load_template` thử `template_cols_id` trước, KeyError thì lùi về `template_cols` (:2776–2785). Nên tự sinh bằng `create_model_id` (auto_model.py:138) để log/round-trip khớp.
- `ModelParameters` và `TransformationParameters` phải là **chuỗi JSON**, không phải dict: `TemplateWizard` gọi `json.loads(row['ModelParameters'])` / `json.loads(row['TransformationParameters'])` (auto_model.py:2199–2200); `__init__` cũng `json.loads(row['TransformationParameters'])` khi cắt transformer (:435).
- `Ensemble` = `0` (int).
- Khoá hợp lệ trong `ModelParameters` = **đúng khoá của `get_params()`** của class: WR (sklearn.py:2567) và MR (sklearn.py:3766). `ModelMonster` truyền `**parameters` cùng với `frequency, prediction_interval, holiday_country, random_seed, verbose, forecast_length, n_jobs` (auto_model.py:274–285, 369–380) ⇒ **KHÔNG được** đặt 7 khoá này trong `ModelParameters` (trùng kwargs → TypeError). Khoá thiếu sẽ lấy default của class (cả hai class có `**kwargs`), nên **nên ghi đầy đủ** để artifact tái tạo được.

### A3. Snippet ĐÚNG (a) template 2 dòng GPU → (b) AutoTS → (c) fit → (d) best + export

```python
import json, pandas as pd
from autots import AutoTS
from autots.evaluator.auto_model import create_model_id

WR_GPU = {                                  # khoa = WindowRegression.get_params() (sklearn.py:2567)
    "window_size": 60, "input_dim": "univariate", "output_dim": "forecast_length",
    "normalize_window": False, "max_windows": 200000, "fourier_encoding_components": None,
    "scale": False, "datepart_method": None, "regression_type": "User",
    "regression_model": {"model": "LightGBM",
                         "model_params": {"device_type": "gpu", "n_estimators": 400,
                                          "learning_rate": 0.03, "max_depth": 6,
                                          "num_leaves": 31}},
                         # SỬA 2026-08-31 (canary package thật): BỎ "verbose": -1 khỏi model_params —
                         # retrieve_regressor (dòng 471) đã truyền verbose/random_state/n_jobs rồi, đặt lại
                         # → TypeError "got multiple values for keyword argument 'verbose'", và AutoTS nuốt
                         # thành "Template Eval Error" nên template LightGBM bị bỏ im lặng.
}
MR_GPU = {                                  # khoa = MultivariateRegression.get_params() (sklearn.py:3766)
    "regression_model": {"model": "xgboost",
                         "model_params": {"device": "cuda", "tree_method": "hist",
                                          "n_estimators": 400, "learning_rate": 0.03,
                                          "max_depth": 6, "random_state": 8587}},
    "mean_rolling_periods": 30, "macd_periods": None, "std_rolling_periods": 7,
    "max_rolling_periods": 7, "min_rolling_periods": 7,
    "quantile90_rolling_periods": None, "quantile10_rolling_periods": None,
    "ewm_alpha": 0.5, "ewm_var_alpha": None, "additional_lag_periods": None,
    "abs_energy": False, "rolling_autocorr_periods": None, "nonzero_last_n": None,
    "datepart_method": None, "polynomial_degree": None, "regression_type": "User",
    "window": 5, "holiday": False, "probabilistic": False, "scale_full_X": False,
    "cointegration": None, "cointegration_lag": 1, "series_hash": False,
    "frac_slice": None, "discard_data": None, "transformation_dict": None,
    "synthetic_boundary_ratio": 0.0, "rolling_skew_periods": None,
    "diff_periods": None, "rolling_range_periods": None,
}
NO_TRANS = {"fillna": None, "transformations": {}, "transformation_params": {}}

def tmpl_row(model, params):
    return {"ID": create_model_id(model, params, NO_TRANS), "Model": model,
            "ModelParameters": json.dumps(params),           # PHAI la chuoi JSON
            "TransformationParameters": json.dumps(NO_TRANS),
            "Ensemble": 0}

tmpl = pd.DataFrame([tmpl_row("WindowRegression", WR_GPU),
                     tmpl_row("MultivariateRegression", MR_GPU)])   # + cac bien the window_size / n_estimators

auto = AutoTS(
    forecast_length=3, frequency='min',
    model_list=['WindowRegression', 'MultivariateRegression'],
    initial_template=tmpl,          # DataFrame -> __init__:374 goi import_template(method='only')
    max_generations=0,              # KHONG sinh the he moi
    num_validations=10, validation_method='backwards',
    models_to_validate=0.99,        # 0.99 -> ep 100% (auto_ts.py:1706)
    max_per_model_class=99,         # KHONG de None: mac dinh = ceil(N/len(model_list))+1 se CAT BOT dong
    metric_weighting={'rmse_weighting': 1},
    ensemble=None, no_negatives=False, constraint=None, drop_most_recent=0, subset=None,
    introduce_na=False, prefill_na=None, preclean=None, holiday_country=None,
    transformer_list='superfast', transformer_max_depth=0,   # xem A6
    skip_slow_models_seconds=None, random_seed=selection_seed, n_jobs=1, verbose=1,
    current_model_file=f"{out}/cur_f{fold}_{setname}",
)
auto.fit(df_tr, future_regressor=R_tr)          # df_tr = FIT+ES; R_tr TRUNG index df_tr
auto.export_template(f"{out}/tmpl_best_f{fold}_{setname}.csv", models='best', n=1)
auto.export_template(f"{out}/tmpl_all_f{fold}_{setname}.csv",  models='all', include_results=True)
name, params, trans = auto.best_model_name, auto.best_model_params, auto.best_model_transformation_params
# -> ModelMonster(name, parameters=params, ...) + vong lap rolling cua §12.3(c)
```

**Bẫy phải tránh**: nếu template có nhiều dòng cùng một `Model` (ví dụ 8 WR + 2 MR) mà để `max_per_model_class=None` thì `_construct_validation_template` (:1710–1713) đặt `max_per_model_class = ceil(N/2)+1` rồi `.groupby('Model').head(...)` ⇒ **một số dòng WR không bao giờ được validate** và do đó không bao giờ được chọn (`_best_non_horizontal` lọc `Runs >= num_validations+1`, :1954).

### A4. Xác nhận `max_generations=0` (từ source)

- `fit` chạy `_run_template(self.initial_template, ...)` **một lần** (auto_ts.py:1409) → `TemplateWizard` lặp `for row in template_dict` **tuần tự**, 1 fit/dòng (auto_model.py:2195+).
- Vòng thế hệ: `while current_generation < self.max_generations and ...` (:1430) → với `max_generations=0` **không vào vòng** ⇒ **không thêm dòng nào**. `NewGeneticTemplate` không được gọi. **Không có chỗ nào ép tối thiểu 1 generation.**
- `ensemble=None` → `self.ensemble = []` (:337) ⇒ bỏ qua toàn bộ khối ensemble (:1500, :1546) và khối horizontal/mosaic (:1610).
- Vẫn chạy: `_construct_validation_template` (:1533) → `_run_validations` nếu `num_validations > 0` (:1536) → `validation_agg` → `_set_best_model` (:1689). Best = min `Score` với `Score` = `generate_score(metric_weighting)`; mọi trọng số không khai báo mặc định 0 (auto_model.py:3220–3243) ⇒ `{'rmse_weighting': 1}` = chỉ RMSE (RMSE trên **log-return**, không phải trên giá — chỉ dùng để chọn template, KHÔNG thay MedianGain, plan §0).
- Metric gộp qua các vòng validation bằng **mean** (`validation_aggregation`, auto_model.py:3072: `'rmse': 'mean'`, `'Runs': 'sum'`).

### A5. Số fit thực tế và số điểm chọn best

Công thức (chỉ đúng cho `max_generations=0`, `ensemble=None`):

```
so_fit = N + M * num_validations
  N = so dong template (moi dong 1 fit o vong danh gia dau, validation_round = 0)
  M = so dong qua duoc vong dau (Exceptions.isna()) va duoc chon vao validation_template
      = N khi models_to_validate >= 0.99 VA max_per_model_class du lon
so_diem_chon_best = forecast_length * (num_validations + 1) = 3 * (num_validations + 1)
```

`num_validations='auto'` → 3 (validation.py:71–72, vì `max_possible = n/3 >= 4`); `'max'` → **50** (`50 if max_possible > 51`, :74).
Với `validation_method='backwards'`, holdout thứ y là **3 bar cuối** của lát `idx[0 : n-(y+1)*3]` (validation.py:180) ⇒ toàn bộ điểm chấm nằm trong **3·(num_validations+1) bar cuối của ES**.

| `num_validations` | số lát | số fit (N=10) | số fit (N=12) | điểm chọn best | ước lượng/search (30 s/fit, GPU) | ×10 search (5 fold × 2 set) |
|---|---|---|---|---|---|---|
| 2 | 3 | 30 | 36 | 9 | ~15 phút | ~2,5 h |
| `'auto'` = 3 | 4 | 40 | 48 | 12 | ~20 phút | ~3,3 h |
| 5 | 6 | 60 | 72 | 18 | ~30 phút | ~5,0 h |
| **10** | 11 | **110** | 132 | **33** | ~55 phút | **~9,2 h** |
| 20 | 21 | 210 | 252 | 63 | ~105 phút | ~17,5 h |
| `'max'` = 50 | 51 | 510 | 612 | 153 | ~4,3 h | ~43 h (vượt ngân sách) |

"30 s/fit" = trung vị giả định cho LightGBM GPU / XGBoost GPU trên X ≈ 15.000 × (60–90 + k), k = |F_frozen| ≈ 40–310 (§9: "vài giây – 1 phút"). **ƯỚC LƯỢNG, chưa đo.**

**Khuyến nghị: `num_validations=10`, `validation_method='backwards'`** — 33 điểm, ≈9 h cho 5 fold × 2 frozen set (vừa ngân sách 12–15 h, còn chỗ cho rolling predict ~0,3 h và cho hao hụt). Nếu cần rẻ hơn: `num_validations=5` (18 điểm, ~5 h). **Không dùng `'max'`.**
Cảnh báo methodology (giữ nguyên từ §12.2): 33 điểm nằm gọn trong **33 phút cuối của ES** ⇒ tín hiệu chọn best rất yếu, template thắng có thể khác nhau giữa 5 fold. Đây là giới hạn của AutoTS (điểm chấm luôn là đuôi `forecast_length` của mỗi lát), không sửa được bằng tham số. Phương án thay thế `validation_method='even'` trải đều 3 bar cuối của (num_validations+1) tiền tố cách đều (validation.py:189) — phủ rộng hơn nhưng lát đầu chỉ có ~n/(nv+1) bar train nên các fold không cùng cỡ train. **Quyết định cuối vẫn là MedianGain trên outer VAL (plan §0); điểm nội bộ của AutoTS chỉ là diagnostic.**

### A6. `transformer_list=[]` là BẪY — dùng `'superfast'` + `transformer_max_depth=0`

`transformer_list_to_dict` (transform.py:8980): `if not transformer_list or transformer_list == "all": transformer_list = transformer_dict` ⇒ **`[]` (falsy) được hiểu là "all"**, ngược hoàn toàn với ý định. Trong phương án A điều này **vô hại** vì (i) `RandomTransform` chỉ được gọi khi sinh template ngẫu nhiên — mà `initial_template` là DataFrame nên không sinh; (ii) `NewGeneticTemplate` không chạy (`max_generations=0`); (iii) vòng cắt transformer ở `__init__:426–466` chỉ **bỏ bớt**, và `transformer_max_depth=0` cắt `list(transformations.items())[:0]` = `{}`. Nhưng để đọc code không hiểu nhầm, dùng `transformer_list='superfast'` (alias hợp lệ) + `transformer_max_depth=0`.

**Xác nhận `TransformationParameters` rỗng được chấp nhận và không có transformer mặc định**: `ModelPrediction.__init__` dựng `GeneralTransformer(**{"fillna": None, "transformations": {}, "transformation_params": {}}, ...)` (auto_model.py:846) — signature `GeneralTransformer(fillna=None, transformations={}, transformation_params={}, ...)` (transform.py:8323). `_fit` = `_first_fit` + vòng `for i in sorted(self.transformations.keys())` = **không lặp lần nào** (transform.py:8598–8610); `_first_fit` chỉ gọi `fill_na`, mà `fill_na` là no-op khi không có NaN (`nan_flag` False, :8390+). ⇒ **biến đổi đồng nhất, AutoTS không tự chèn transformer nào.** Bắt buộc phải có đủ 3 khoá trong JSON, nếu thiếu `transformations`/`transformation_params` thì `__init__:440` raise `ValueError("initial_template is missing transformation parameters...")`.

### A7. Round-trip export/import

- `export_template(f, models='best', n=1)` với `include_results=False` trả về **thẳng `self.best_model`** (auto_ts.py:2610–2611) = 1 dòng với cột `template_cols_id` = `['ID','Model','ModelParameters','TransformationParameters','Ensemble']`.
- `models='all'` → `self.initial_results.model_results[template_cols_id].drop_duplicates()` (:2606) — mọi candidate đã thử. `include_results=True` kèm cột metric nội bộ.
- **Không mất thông tin**: `ModelParameters` được lưu/đọc nguyên văn chuỗi JSON đã dùng lúc fit (`TemplateWizard` chỉ `json.loads`, không viết lại), nên `regression_model` lồng nhau (`{"model": ..., "model_params": {...}}`) giữ nguyên **kể cả khoá GPU `device_type`/`device`**. Round-trip `json.dumps → csv → read_csv → json.loads` là bit-exact về nội dung dict.
- Lưu ý CSV: chuỗi JSON chứa dấu `,` và `"` → `to_csv`/`read_csv` xử lý được nhưng **nên dùng `.json`** (`to_json(orient='columns')`, :2756) nếu muốn tránh mọi rủi ro quoting; `load_template` đọc cả hai.
- `import_best_model(path_or_df)` (:2904) nạp lại + `parse_best_model` → `best_model_name/params/transformation_params`. Với đường chạy §12.3(c) chỉ cần `json.loads(row['ModelParameters'])`.
- `ID` = `md5(model_str + json.dumps(params) + json.dumps(trans))` sau khi bỏ hết whitespace (auto_model.py:138–154) ⇒ ID ổn định giữa các lần chạy nếu thứ tự khoá dict giữ nguyên.

### A8. Ba lưu ý vận hành

1. `AutoTS.__init__` gọi `random.seed(random_seed)` (:302) và `fit` gọi `random.seed` + `np.random.seed` (:1323–1324) ⇒ **thay đổi RNG toàn cục**; harness phải tự set lại seed sau khi gọi (ảnh hưởng bước sau).
2. `verbose <= 0` khiến `fit` chạy `warnings.filterwarnings("ignore")` **toàn cục** (:1309–1312). Dùng `verbose=1` nếu không muốn mất warning của harness.
3. `n_jobs=1` (giữ như §7): `MultiOutputRegressor(LGBMRegressor(n_jobs=1), n_jobs=n_jobs)` (sklearn.py:471) — `n_jobs>1` sẽ cho 3 model con tranh GPU.
