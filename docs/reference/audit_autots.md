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
