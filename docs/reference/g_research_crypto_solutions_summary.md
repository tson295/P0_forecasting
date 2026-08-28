# Tổng hợp các lời giải G-Research Crypto Forecasting

> **STATUS — LITERATURE/FEATURE REFERENCE ONLY.** Tài liệu này khảo sát một competition có target/metric/scope khác và không có quyền thay đổi canonical roadmap. Các model/ensemble/metric ngoài scope bên dưới không phải project candidates. Chỉ các feature causal từ OHLCV/amount/multi-scale mới có thể vào danh sách candidate Bước 2 của `docs/RESEARCH_PLAN.md` (thử lần lượt từng cột); `docs/RESEARCH_PLAN.md` và `.claude/CLAUDE.md` luôn thắng khi xung đột.

> Mục tiêu của tài liệu: gom các ý tưởng chính từ những lời giải được thảo luận thành một tài liệu thống nhất, tập trung vào cách xử lý dữ liệu, feature engineering, validation, training, ensemble và những bài học có thể tái sử dụng cho bài toán time-series tài chính.

---

## 0. Bối cảnh cuộc thi

### Dữ liệu

Dữ liệu minute-level cho 14 crypto asset, gồm các cột chính:

- `timestamp`
- `Asset_ID`
- `Count`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `VWAP`
- `Target`

Mỗi hàng tương ứng với một asset tại một phút.

### Target

Target là một dạng **15-minute forward residualized return**, tức không đơn giản chỉ là:

\[
r_{t,t+15}=\frac{P_{t+15}-P_t}{P_t}
\]

mà đã cố gắng loại bỏ một phần chuyển động thị trường chung.

Điều này khiến các feature kiểu:

- relative return,
- cross-asset movement,
- market-average movement,
- beta,
- asset-vs-market strength

trở nên đặc biệt hợp lý.

### Metric

Competition dùng **weighted Pearson correlation**.

Do đó mục tiêu không phải chỉ giảm MAE/MSE mà là tạo prediction có tương quan tốt với target theo đúng thứ tự và hướng biến động.

---

# 1. Nathaniel Maddux — 2nd Place

## 1.1 Triết lý chính

Solution này không dựa vào kiến trúc deep learning phức tạp.

Trọng tâm là:

\[
\boxed{
\text{Reliable validation}
\rightarrow
\text{Feature engineering}
\rightarrow
\text{Ablation}
\rightarrow
\text{LightGBM}
}
\]

Điểm đáng học nhất không nằm ở model mà ở **quy trình nghiên cứu feature**.

## 1.2 Model

Model chính:

\[
\boxed{\text{LightGBM regression}}
\]

Loss cơ bản là squared-error regression.

Không có ensemble phức tạp giữa nhiều architecture. Bản thân gradient boosting đã là một dạng ensemble của nhiều cây.

## 1.3 Hyperparameter tuning

Các hyperparameter chính được quan tâm gồm:

- `n_estimators`
- `num_leaves`
- `learning_rate`

Một số thử nghiệm khác như:

- regularization bổ sung,
- augmentation,
- feature neutralization

không đem lại cải thiện CV đủ ổn định nên không được giữ lại.

### Bài học

Không nên thêm kỹ thuật chỉ vì chúng phổ biến.

Giữ feature/model component khi:

\[
CV(\text{base}+\text{component})
>
CV(\text{base})
\]

một cách đủ nhất quán qua các fold.

## 1.4 Validation

Một trong những phần mạnh nhất của solution.

Sử dụng dạng:

\[
\boxed{\text{walk-forward grouped cross-validation}}
\]

Theo thời gian, thay vì random KFold.

Khoảng:

- ~40 tuần train
- gap
- ~40 tuần validation
- nhiều fold dịch dần về phía trước

Minh họa:

```text
Fold 1
TRAIN ─────────────────────────
                               GAP
                                   VALID ─────────────────────

Fold 2
          TRAIN ─────────────────────────
                                           GAP
                                               VALID ─────────

Fold 3
                    TRAIN ─────────────────────────
```

### Vì sao phải có gap?

Target sử dụng tương lai.

Hai sample rất gần nhau có future window overlap mạnh.

Ví dụ:

```text
train origin      : t
validation origin : t+1
```

Target của chúng có thể dùng gần như cùng một đoạn tương lai.

Nếu không purge/gap:

\[
\text{CV score}
\]

có thể bị optimistic.

## 1.5 Dùng dữ liệu cũ hay chỉ recent data?

Nathaniel thử nghiệm việc giới hạn training vào recent history.

Kết quả CV cho thấy sử dụng nhiều dữ liệu hơn vẫn có lợi.

Do đó final model dùng lượng lịch sử lớn thay vì chỉ recent regime.

### Ý nghĩa

Financial data non-stationary không đồng nghĩa:

\[
\text{old data}=\text{useless}
\]

Phải kiểm tra bằng validation.

## 1.6 Feature engineering

Chi tiết toàn bộ alpha feature không được công khai.

Những điều quan trọng có thể rút ra:

- feature engineering là trọng tâm;
- dùng feature importance để định hướng experiment;
- tập trung đào sâu các nhóm feature đã có signal;
- tối ưu pipeline feature generation;
- dùng Numba để tăng tốc;
- chú ý memory/copy overhead.

### Không nên làm

Không nên gán cho solution này một danh sách indicator cụ thể nếu tác giả không công khai.

## 1.7 Điểm mạnh

- Validation rất nghiêm túc.
- Ít phụ thuộc vào fancy model.
- Quy trình ablation rõ.
- Tối ưu feature pipeline tốt.
- Hợp với dữ liệu low signal/high noise.

## 1.8 Điểm yếu

- Feature alpha quan trọng không được public đầy đủ.
- Khó tái tạo chính xác final score chỉ từ writeup.

## 1.9 Bài học quan trọng

\[
\boxed{
\text{Feature quality + trustworthy CV}
\gg
\text{model complexity}
}
\]

---

# 2. Patrick Yam — 7th Place

## 2.1 Triết lý chính

Gần như đối lập với 2nd place.

Patrick tập trung chủ yếu vào:

\[
\boxed{\text{representation learning / sequence modeling}}
\]

thay vì engineer hàng trăm technical features.

Theo writeup, phần lớn effort là modeling.

Feature bổ sung nổi bật:

\[
\boxed{\text{time of day}}
\]

## 2.2 Input

Thay vì biến dữ liệu thành hàng trăm rolling indicators trước khi train, model nhận một đoạn sequence lịch sử.

\[
X_t=[x_{t-L+1},...,x_t]
\]

với \(L\) là sequence length.

Model tự học temporal representation.

## 2.3 Time-of-day feature

Crypto giao dịch 24/7 nhưng distribution không nhất thiết giống nhau ở mọi giờ.

Có thể encode:

\[
hour_t
\]

hoặc tốt hơn:

\[
\sin(2\pi h/24)
\]

\[
\cos(2\pi h/24)
\]

để biểu diễn tính chu kỳ.

Model lúc này học:

\[
p(y_t\mid X_t,\text{time-of-day})
\]

## 2.4 Sequence-length ensemble

Final submission ensemble nhiều model với sequence length khác nhau.

Ý tưởng:

\[
L_1<L_2<L_3<L_4
\]

Ví dụ trực giác:

```text
short context  -> micro momentum
medium context -> local trend
long context   -> regime/background
```

Final prediction:

\[
\hat y=\frac{1}{M}\sum_{m=1}^{M}\hat y^{(m)}
\]

### Tại sao ensemble theo context length có ích?

Mỗi receptive field có bias khác nhau.

Một model nhìn rất ngắn có thể phản ứng nhanh nhưng noisy.

Một model nhìn dài ổn định hơn nhưng có thể lag.

Average nhiều context giúp giảm variance.

## 2.5 Feature engineering

Rất ít traditional technical indicator.

Đây là bằng chứng rằng:

\[
\boxed{\text{MA/MACD/RSI không phải điều bắt buộc}}
\]

nếu sequence model đủ khả năng học representation từ raw temporal history.

## 2.6 Điểm mạnh

- Hạn chế hand-crafted bias.
- Model tự học temporal dependencies.
- Ensemble context length hợp lý.
- Có tính đối trọng với các solution feature-heavy.

## 2.7 Điểm yếu

- Training/inference phức tạp hơn LightGBM.
- Dễ overfit nếu CV không đủ tốt.
- Khó giải thích feature importance.
- Cần nhiều compute hơn tabular GBDT.

## 2.8 Bài học

Có hai hướng hợp lệ:

```text
Raw sequence -> neural model
```

hoặc:

```text
Raw sequence -> engineered temporal features -> tabular model
```

Không có hướng nào luôn thắng.

---

# 3. bturan19 — 9th Place

## 3.1 Triết lý chính

Solution này quay trở lại hướng:

\[
\boxed{
\text{technical feature engineering}
+
\text{LightGBM}
+
\text{market-regime experts}
}
\]

Điểm nổi bật nhất:

\[
\boxed{\text{Hull Moving Average}}
\]

được báo cáo là feature rất quan trọng.

## 3.2 Hull Moving Average

HMA cố gắng giảm lag của moving average truyền thống.

\[
HMA_n=
WMA_{\sqrt n}
\left(
2WMA_{n/2}(P)-WMA_n(P)
\right)
\]

Trong đó WMA là Weighted Moving Average.

### Ý tưởng

\[
2\times\text{fast trend}-\text{slow trend}
\]

tạo estimate phản ứng nhanh hơn, sau đó lại smoothing bằng WMA ngắn hơn.

Mục tiêu:

\[
\boxed{\text{smoothness}+\text{lower lag}}
\]

## 3.3 Multi-scale windows

Các window được sử dụng:

\[
55,\ 210,\ 340,\ 890,\ 3750
\]

Với minute-level data, chúng tương ứng nhiều temporal scale từ dưới một giờ đến vài ngày.

### Bài học quan trọng

Không nên hiểu đây là:

> Fibonacci có sức mạnh dự báo đặc biệt.

Ý nghĩa thực sự là:

\[
\boxed{\text{multi-scale temporal representation}}
\]

Model nhìn cùng lúc nhiều horizon.

## 3.4 Regime-specific models

Một ý tưởng rất hay:

- model cho up market;
- model cho down market;
- model cho stable market.

Ký hiệu:

\[
M_{up},M_{down},M_{stable}
\]

Final prediction có thể average:

\[
\hat y=
\frac{\hat y_{up}+\hat y_{down}+\hat y_{stable}}{3}
\]

### Vì sao hợp lý?

Mối quan hệ feature-target có thể phụ thuộc regime:

\[
f_{bull}(X)\neq f_{bear}(X)
\]

Ví dụ cùng một positive momentum:

- bull regime -> continuation;
- bear regime -> temporary rebound.

Một single global function có thể khó học tốt cả hai.

## 3.5 Model

Các expert chủ yếu là LightGBM.

Đây không nhất thiết là:

\[
14\ assets\times 3\ regimes=42\ independent\ asset-specific\ models
\]

mà tư tưởng chính là regime experts dùng chung logic prediction cho tập asset.

## 3.6 Điểm mạnh

- HMA giảm lag tốt hơn simple MA.
- Multi-scale temporal features.
- Regime specialization.
- LightGBM inference nhanh.
- Feature importance dễ phân tích.

## 3.7 Điểm yếu

- Handcrafted regime definition có thể không ổn định.
- Fibonacci-style windows không nên copy mù quáng.
- Các feature derived từ price có redundancy cao.

## 3.8 Bài học

Moving-average-derived features hoàn toàn có thể hữu ích trong ML nếu:

- được thiết kế multi-scale;
- normalize hợp lý;
- đánh giá bằng walk-forward CV;
- kết hợp với model phi tuyến.

---

# 4. Tom Forbes — 13th Place

## 4.1 Triết lý chính

Hybrid:

\[
\boxed{
\text{feature engineering}
+
\text{LightGBM}
+
\text{neural network}
+
\text{target engineering}
}
\]

## 4.2 Feature set

Số feature không quá lớn.

Các nhóm chính:

- raw features;
- historical returns;
- historical volatility;
- EMA;
- timestamp features.

Một số horizon được nhắc tới gồm:

\[
return_{30}
\]

\[
return_{120}
\]

và EMA:

\[
EMA_{21}, EMA_{35}, EMA_{80}, EMA_{250}
\]

## 4.3 Vì sao multi-scale EMA?

```text
EMA 21  -> rất ngắn
EMA 35  -> ngắn
EMA 80  -> trung bình
EMA 250 -> dài hơn
```

Model có thể học các interaction như:

\[
EMA_{21}>EMA_{80}
\]

nhưng:

\[
return_{30}<0
\]

Tức medium trend vẫn bullish nhưng short-term đang pullback.

## 4.4 Historical volatility

Ví dụ:

\[
RV_n=
\sqrt{\frac1n\sum_{i=0}^{n-1}r_{t-i}^2}
\]

Hai đoạn thời gian có cùng trend nhưng volatility khác nhau có conditional distribution rất khác.

Do đó:

\[
p(y\mid trend,vol)
\]

hợp lý hơn chỉ:

\[
p(y\mid trend)
\]

## 4.5 Target engineering

Solution này không chỉ engineer input mà còn suy nghĩ về cấu trúc target.

Có các khái niệm dạng:

\[
TargetZero
\]

và:

\[
TargetBeta
\]

để phân tách:

- raw/forward return component;
- market/beta-related component.

### Ý nghĩa

Official target đã residualize market movement.

Do đó có thể học riêng:

\[
\text{asset-specific return}
\]

và:

\[
\text{market-related component}
\]

rồi kết hợp.

## 4.6 Ensemble

Dùng cả:

\[
\boxed{\text{LightGBM}}
\]

và:

\[
\boxed{\text{Keras neural network}}
\]

sau đó average prediction.

Nếu error của hai model không hoàn toàn tương quan:

\[
corr(\epsilon_1,\epsilon_2)<1
\]

thì average có thể giảm variance.

## 4.7 Extra crypto data

Một phần pipeline có sử dụng thêm cryptocurrency ngoài 14 asset competition.

Ý tưởng:

\[
\text{more assets}
\rightarrow
\text{more examples of generic crypto dynamics}
\]

Nhưng cần cẩn thận domain mismatch.

## 4.8 Điểm mạnh

- Feature set nhỏ nhưng có chủ đích.
- Multi-scale EMA/returns/volatility.
- Ensemble model family khác nhau.
- Có target engineering phù hợp metric.

## 4.9 Điểm yếu

- External asset có thể gây distribution mismatch.
- Ensemble phức tạp hơn single LightGBM.
- Target decomposition có thể nhạy với estimation assumptions.

## 4.10 Bài học

Không nhất thiết phải engineer hàng trăm feature.

Một feature set nhỏ nhưng có:

\[
\text{return}+\text{trend}+\text{volatility}+\text{time}
\]

có thể rất mạnh.

---

# 5. T. Morimura — 14th Place

## 5.1 Triết lý chính

Đây là một pipeline feature-heavy và thực dụng:

\[
\boxed{
\text{many causal technical/statistical features}
+
\text{LightGBM ensemble}
+
\text{asset-specific fallback/model selection}
}
\]

## 5.2 Các nhóm feature

Feature vector gồm nhiều nhóm:

- base OHLC-related features;
- lagged returns;
- realized volatility;
- rolling statistics;
- beta/market-relative features;
- RSI-like features;
- MACD-like features;
- market aggregate features;
- volume features;
- technical lag features.

## 5.3 MACD-like features

Điểm quan trọng: không đơn giản gọi textbook:

\[
MACD=EMA_{12}-EMA_{26}
\]

Pipeline tạo custom multi-scale MACD-style statistics.

Có sampling theo khoảng như:

\[
15\text{ minutes}
\]

và:

\[
60\text{ minutes}
\]

Sau đó tính difference giữa short-window average và long-window average.

Dạng khái quát:

\[
MACD^{(15)}=mean(short\ samples)-mean(long\ samples)
\]

và:

\[
MACD^{(60)}=mean(short\ samples)-mean(long\ samples)
\]

### Bài học

Không cần giữ nguyên indicator textbook.

Có thể lấy **ý tưởng toán học** của indicator rồi điều chỉnh temporal scale phù hợp target.

## 5.4 RSI-like features

RSI được tính ở nhiều temporal scale, ví dụ:

\[
RSI_{15}
\]

và:

\[
RSI_{240}
\]

Tức model có thể so sánh momentum rất ngắn với momentum vài giờ.

## 5.5 Realized volatility

Các feature volatility giúp model nhận biết current market regime.

Ví dụ:

\[
RV_{15}, RV_{60}, RV_{240}
\]

và có thể tạo ratio:

\[
\frac{RV_{15}}{RV_{240}}
\]

để đo hiện tại có đang volatility spike hay không.

## 5.6 Market-relative features

Rất hợp với official target.

Ví dụ khái niệm:

\[
r^{asset}_{k}-r^{market}_{k}
\]

hoặc beta-adjusted behavior.

Một coin tăng 1% không có ý nghĩa giống nhau nếu market +0.9% hay market -1.0%.

Điều quan trọng có thể là **relative strength**.

## 5.7 Volume features

Không chỉ price.

Có thể sử dụng:

- raw volume;
- lagged volume;
- rolling mean;
- relative volume;
- price × volume relationships.

Điều này bổ sung information mà pure price-history indicator không có.

## 5.8 Model

Chủ yếu là LightGBM.

Một configuration có các thành phần regularization tương đối mạnh như:

- minimum data per leaf;
- row subsampling;
- column subsampling;
- L1;
- L2;
- moderate learning rate;
- vài trăm boosting rounds.

`Asset_ID` được dùng như categorical/context feature trong một số training path.

## 5.9 LightGBM ensemble

Load nhiều LightGBM checkpoint/model khác nhau:

\[
M_1,M_2,M_3
\]

sau đó:

\[
\hat y_{LGBM}=
\frac{\hat y_1+\hat y_2+\hat y_3}{3}
\]

Đây là variance reduction đơn giản nhưng hiệu quả.

## 5.10 Ridge / asset-specific model selection

Pipeline còn có Ridge model theo asset/context.

Một số asset có thể dùng prediction từ LightGBM ensemble, một số trường hợp có thể switch sang Ridge.

Ý tưởng:

\[
\boxed{\text{different assets may prefer different inductive biases}}
\]

Không bắt buộc một architecture phải tốt nhất cho tất cả asset.

## 5.11 Missing values

Một chiến lược inference được dùng:

```text
NaN -> sentinel
Inf -> sentinel
```

Ví dụ:

\[
NaN\rightarrow -999
\]

Tree model có thể học sentinel như missing state riêng.

Điều này khác với việc bắt buộc forward-fill mọi giá trị.

## 5.12 Data filtering

Pipeline có nhiều switch/filter để:

- tránh leakage;
- giới hạn khoảng năm;
- kiểm soát supplemental data;
- resample;
- giảm memory.

### Bài học

Data pipeline là một phần của model.

Không chỉ:

```python
pd.read_csv()
model.fit()
```

mà phải quản lý:

\[
\text{time range}
+
\text{causality}
+
\text{missingness}
+
\text{memory}
+
\text{online parity}
\]

## 5.13 Điểm mạnh

- Feature rất đa dạng.
- Có market-relative information.
- Multi-scale RSI/MACD/volatility.
- LightGBM ensemble dễ deploy.
- Có asset-specific model selection.
- Pipeline gần với production inference.

## 5.14 Điểm yếu

- Feature count cao -> redundancy.
- Technical indicators dễ overfit nếu validation yếu.
- Sentinel missing values cần train/inference consistent.
- Pipeline phức tạp, khó ablate hơn.

## 5.15 Bài học

Technical indicator hữu ích khi được xem như:

\[
\boxed{\text{engineered statistical summaries}}
\]

chứ không phải quy tắc trading cứng.

---

# 6. liyang.chen — 23rd Place

## 6.1 Tình trạng thông tin

Writeup được biết với tiêu đề thiên về việc không nên hoảng khi live score tạm thời âm.

Tuy nhiên phần body chi tiết về:

- exact feature set;
- model architecture;
- CV;
- hyperparameters;
- ensemble

không có đủ dữ liệu được xác minh trong phần tổng hợp hiện tại.

Do đó không nên tự suy đoán.

## 6.2 Bài học có thể rút ra chắc chắn

Competition có:

\[
\boxed{\text{extremely low signal-to-noise ratio}}
\]

Pearson correlation trên một đoạn live ngắn có variance rất lớn.

Một model có expected correlation dương vẫn có thể tạm thời quan sát:

\[
\hat\rho<0
\]

trên một short window.

Khi số sample tăng:

\[
Var(\hat\rho)\downarrow
\]

nên leaderboard/live estimate ổn định dần hơn.

### Bài học

Không nên chọn model dựa trên:

- một short live period;
- một fold;
- một seed;
- một asset.

Phải đánh giá robustness.

---

# 7. Kirderf — 37th Place

## 7.1 Tình trạng thông tin

Không đủ phần body đã xác minh để mô tả chắc chắn:

- exact model;
- exact feature set;
- exact CV;
- exact training pipeline.

Vì vậy tài liệu này không tự bịa chi tiết.

## 7.2 Điều có thể giữ lại

Solution vẫn đạt thứ hạng tốt trong một competition có lượng đội lớn.

Điều này nhắc lại rằng:

\[
\text{small but robust predictive correlation}
\]

đã đủ để tạo khác biệt lớn trong financial forecasting.

---

# 8. So sánh toàn bộ các hướng

| Rank | Team | Philosophy | Feature engineering | Main model | Ensemble / specialization |
|---:|---|---|---|---|---|
| 2 | Nathaniel Maddux | CV + feature research | Rất quan trọng, alpha không public đầy đủ | LightGBM | GBDT itself, không fancy ensemble |
| 7 | Patrick Yam | Sequence representation learning | Rất ít | Sequence/attention model | Nhiều sequence lengths |
| 9 | bturan19 | Technical + regime experts | HMA multi-scale | LightGBM | Up/down/stable experts |
| 13 | Tom Forbes | Compact feature set + target engineering | Return, vol, EMA, time | LGBM + Keras | Cross-model averaging |
| 14 | T. Morimura | Feature-heavy quantitative pipeline | RSI, MACD-like, vol, market, volume | LightGBM + Ridge | 3×LGBM + asset selection |
| 23 | liyang.chen | Chưa đủ body để xác minh | — | — | — |
| 37 | Kirderf | Chưa đủ body để xác minh | — | — | — |

---

# 9. Những pattern chung nổi bật

## 9.1 Không có indicator thần thánh

Ta thấy ba extreme:

### 7th

Gần như không dùng traditional indicator.

### 9th

Hull Moving Average rất quan trọng.

### 14th

Có RSI, MACD-like, volatility và nhiều statistical features.

Vậy kết luận đúng là:

\[
\boxed{
\text{Indicator usefulness is conditional on model + target + CV}
}
\]

Không thể nói:

```text
MACD luôn tốt
```

hay:

```text
RSI luôn vô dụng
```

---

# 10. Feature engineering dưới góc signal processing

## 10.1 Moving Average

\[
MA_n(t)=\frac1n\sum_{i=0}^{n-1}P_{t-i}
\]

Là smoother / low-pass filter.

Giúp:

- giảm noise;
- biểu diễn local trend.

Nhược điểm:

- lag.

## 10.2 EMA

\[
EMA_t=\alpha P_t+(1-\alpha)EMA_{t-1}
\]

Trọng số dữ liệu mới lớn hơn, phản ứng nhanh hơn SMA.

## 10.3 MACD

\[
MACD=EMA_{fast}-EMA_{slow}
\]

Đo difference giữa fast trend và slow trend.

Không tạo ra information mới nếu model đã có full price history.

Nó cung cấp một **inductive bias / compressed representation**.

## 10.4 HMA

Thiết kế để smooth nhưng giảm lag so với SMA/EMA truyền thống.

9th place cho thấy nó có thể hữu ích thực tế.

## 10.5 VWAP

\[
VWAP=\frac{\sum_iP_iV_i}{\sum_iV_i}
\]

Khác MA/MACD vì có volume information.

Feature đáng thử:

\[
\log\frac{Close}{VWAP}
\]

thay vì raw VWAP.

---

# 11. Tại sao raw indicator thường không tối ưu?

Ví dụ:

\[
MA_{60}=60000
\]

không stationary khi price level thay đổi.

Nên normalize:

\[
\boxed{\frac{Close-MA_{60}}{MA_{60}}}
\]

hoặc:

\[
\boxed{\log\frac{Close}{MA_{60}}}
\]

Tương tự MACD:

\[
MACD^{norm}=\frac{EMA_{fast}-EMA_{slow}}{Close}
\]

hoặc:

\[
MACD^{volnorm}=\frac{EMA_{fast}-EMA_{slow}}{\sigma_{rolling}}
\]

Những representation này scale-free hơn.

---

# 12. Một feature pipeline tổng hợp từ tất cả các solution

## Layer 0 — Raw data

\[
O,H,L,C,V,VWAP,Count
\]

## Layer 1 — Price geometry

\[
\log(C/O)
\]

\[
\log(H/L)
\]

\[
\frac{C-L}{H-L+\epsilon}
\]

\[
\frac{H-C}{H-L+\epsilon}
\]

## Layer 2 — Returns

\[
r_k(t)=\log\frac{C_t}{C_{t-k}}
\]

Test nhiều horizon:

\[
k\in\{1,3,5,15,30,60,120,240,900\}
\]

## Layer 3 — Volatility

\[
RV_k=\sqrt{\frac1k\sum r^2}
\]

với:

\[
k\in\{15,60,240,900\}
\]

Thêm ratio:

\[
\frac{RV_{15}}{RV_{240}}
\]

## Layer 4 — Volume

\[
\log(Volume_t)
\]

\[
relativeVolume_t=\frac{Volume_t}{mean_k(Volume)}
\]

\[
zVolume_t=\frac{Volume_t-\mu_k}{\sigma_k}
\]

## Layer 5 — VWAP relationships

\[
\boxed{vwapGap_t=\log\frac{Close_t}{VWAP_t}}
\]

Thêm:

- \(\Delta VWAP\)
- rolling mean của VWAP gap
- rolling std của VWAP gap
- VWAP-gap z-score

## Layer 6 — MA / EMA / HMA

Ví dụ:

\[
EMA_{15},EMA_{30},EMA_{60},EMA_{240}
\]

Nhưng dùng normalized relationship:

\[
\log\frac{Close}{EMA_k}
\]

\[
\log\frac{EMA_{15}}{EMA_{60}}
\]

Test thêm HMA theo multi-scale windows.

## Layer 7 — MACD-like

Không nhất thiết dùng 12/26 textbook.

Tạo generalized MACD:

\[
MACD_{a,b}=\frac{EMA_a-EMA_b}{Close}
\]

với nhiều pair:

\[
(a,b)\in\{(5,30),(15,60),(60,240)\}
\]

## Layer 8 — RSI / momentum

Multi-scale:

\[
RSI_{15},RSI_{60},RSI_{240}
\]

Dùng như numerical feature, không dùng rule:

```text
RSI > 70 => sell
```

## Layer 9 — Cross-asset / market features

Đây là layer cực quan trọng.

Tạo market-average return:

\[
r^{market}_k=\sum_jw_jr^{(j)}_k
\]

Sau đó:

\[
relativeReturn^{asset}_k=r^{asset}_k-r^{market}_k
\]

Thêm:

- relative volatility;
- relative volume;
- rolling beta;
- cross-asset dispersion.

## Layer 10 — Time/context

- minute of hour;
- hour of day;
- day of week.

Encode cyclical:

\[
\sin(2\pi h/24),\quad \cos(2\pi h/24)
\]

---

# 13. Một model strategy tổng hợp từ các đội

## Family A — LightGBM feature model

Input:

\[
X_t^{engineered}
\]

Model:

\[
\hat y_A=LGBM(X_t)
\]

Ưu điểm:

- nhanh;
- mạnh trên tabular feature;
- dễ analyze importance;
- CPU inference tốt.

## Family B — Sequence model

Input:

\[
X_{t-L+1:t}
\]

Model có thể là:

- TCN;
- 1D CNN;
- Transformer/attention.

Train nhiều context:

\[
L=60,240,900
\]

ensemble:

\[
\hat y_B=mean(\hat y_{60},\hat y_{240},\hat y_{900})
\]

Học từ ý tưởng của Patrick Yam.

## Family C — Regime experts

Xác định regime dựa trên market:

\[
z_t\in\{up,down,stable,highVol\}
\]

Train expert:

\[
M_z
\]

Có thể:

- route prediction theo current regime;
- hoặc ensemble tất cả experts.

Học từ 9th place.

---

# 14. Ensemble cuối

Một cấu trúc hợp lý:

\[
\hat y=w_A\hat y_{LGBM}+w_B\hat y_{sequence}+w_C\hat y_{regime}
\]

Không chọn weight dựa trên leaderboard.

Tối ưu weight trên OOF/walk-forward predictions.

Ví dụ constraint:

\[
w_A+w_B+w_C=1
\]

\[
w_i\ge0
\]

---

# 15. Validation framework nên dùng

Đây là phần phải học từ 2nd place.

Không random split.

Dùng:

\[
\boxed{\text{walk-forward validation}}
\]

Ví dụ:

```text
Fold 1
Train: 2018 -------- 2019
Gap
Valid: 2020 H1

Fold 2
Train: 2018 ------------- 2020 H1
Gap
Valid: 2020 H2

Fold 3
Train: 2018 ---------------------- 2020
Gap
Valid: 2021 H1
```

Có purge/gap để tránh overlapping future target windows.

---

# 16. Feature selection / ablation

Mỗi feature family phải test riêng.

Ví dụ:

```text
F0 = returns
F1 = F0 + volatility
F2 = F1 + VWAP
F3 = F2 + EMA/HMA
F4 = F3 + MACD/RSI
F5 = F4 + cross-asset
```

Đo:

\[
\Delta CV_i=CV(F_i)-CV(F_{i-1})
\]

Không chỉ nhìn một mean score.

Cần xem:

- median fold improvement;
- worst fold;
- per-asset improvement;
- per-time-regime improvement.

---

# 17. Feature correlation

Nếu hai feature gần như cùng thông tin:

\[
corr(f_i,f_j)\approx1
\]

thì không nhất thiết phải giữ cả hai.

Ví dụ:

\[
EMA_{60}
\]

và:

\[
SMA_{60}
\]

có thể rất correlated.

MACD cũng thường overlap với EMA ratios.

Nên đánh giá incremental signal.

---

# 18. Feature-target correlation

Không nên chỉ dùng Pearson correlation raw để quyết định.

Một feature có:

\[
corr(f,y)\approx0
\]

vẫn có thể hữu ích nếu relationship phi tuyến.

Ví dụ:

\[
y\propto f^2
\]

thì Pearson correlation có thể gần zero.

Do đó dùng cả:

- univariate correlation;
- LightGBM gain;
- permutation importance;
- feature ablation;
- temporal robustness.

---

# 19. Temporal robustness

Financial feature dễ chết theo thời gian.

Một feature chỉ tốt ở bull market nhưng không tốt ở crash regime thì nguy hiểm.

Nên report:

\[
score_{fold1},score_{fold2},...,score_{foldN}
\]

thay vì chỉ:

\[
mean(score)
\]

---

# 20. Những điều không nên copy máy móc

## Không copy Fibonacci windows chỉ vì 9th dùng

Test chúng, nhưng cũng test horizon dựa trên domain:

\[
15,60,240,1440
\]

## Không add 50 indicator TA cùng lúc

Vì:

\[
\text{redundancy}+\text{multiple testing}+\text{overfit}
\]

## Không random KFold

Sẽ phá temporal causality.

## Không dùng raw price-level indicator nếu không cần

Ưu tiên stationary/relative representation.

## Không tin leaderboard short-window quá mức

Noise lớn.

## Không tin feature importance một cách tuyệt đối

Tree gain importance có bias.

Phải xác minh bằng ablation/permutation.

---

# 21. Những ý tưởng đáng học nhất từ từng đội

## 2nd — Nathaniel Maddux

\[
\boxed{\text{Validation + feature research discipline}}
\]

Đây là lesson số 1.

## 7th — Patrick Yam

\[
\boxed{\text{Sequence model có thể thay hand-crafted indicators}}
\]

và ensemble nhiều context length.

## 9th — bturan19

\[
\boxed{\text{HMA + multi-scale + regime experts}}
\]

## 13th — Tom Forbes

\[
\boxed{\text{compact feature set + target engineering + model diversity}}
\]

## 14th — T. Morimura

\[
\boxed{\text{rich causal features + market-relative context + practical ensemble}}
\]

## 23rd

\[
\boxed{\text{live correlation rất noisy; cần robustness}}
\]

## 37th

Không đủ detail xác minh để rút ra exact technical recipe.

---

# 22. Kết luận chung

Từ toàn bộ các solution, pattern mạnh nhất là:

\[
\boxed{
\text{Causal raw data}
\rightarrow
\text{multi-scale representations}
\rightarrow
\text{market-relative features}
\rightarrow
\text{walk-forward CV}
\rightarrow
\text{feature ablation}
\rightarrow
\text{robust ensemble}
}
\]

Không có evidence rằng MACD hay RSI tự thân là alpha.

Chúng chỉ là cách nén quá khứ.

Điều quyết định là:

\[
\boxed{CV(Base+Feature)-CV(Base)}
\]

có dương và ổn định hay không.

Trong G-Research, các feature đáng ưu tiên nhất về mặt nghiên cứu là:

1. multi-horizon returns;
2. realized volatility;
3. cross-asset / market-relative returns;
4. VWAP-price relationship;
5. relative volume;
6. EMA/HMA multi-scale;
7. MACD/RSI normalized;
8. time/context;
9. regime features.

Và lesson quan trọng nhất từ leaderboard:

\[
\boxed{\text{Signal + validation}\gg\text{fancy architecture}}
\]

Một LightGBM với feature đúng và CV đúng có thể đánh bại một deep model phức tạp.

---

# 23. Pipeline đề xuất nếu triển khai lại

```text
                    RAW MINUTE DATA
                          │
             ┌────────────┼────────────┐
             │            │            │
          Returns      Volume       OHLC/VWAP
             │            │            │
             └───────┬────┴────┬───────┘
                     │         │
                 Rolling    Market-wide
                 features     features
                     │         │
                     └────┬────┘
                          │
             ┌────────────┼──────────────┐
             │            │              │
         LGBM model   Sequence model  Regime experts
             │            │              │
             └────────────┼──────────────┘
                          │
                     OOF ensemble
                          │
                          ▼
                       Prediction
```

---

# 24. Recommended experiment order

```text
E0  Persistence / zero-return baseline
E1  Basic returns
E2  + rolling volatility
E3  + volume features
E4  + VWAP-relative features
E5  + market-relative returns
E6  + EMA/HMA
E7  + MACD/RSI
E8  + regime features
E9  + sequence model
E10 ensemble
```

Sau mỗi bước:

\[
\Delta CV
\]

phải được đo và lưu lại.

Đây là cách biến các ý tưởng của nhiều đội thành một **research pipeline có kiểm soát**, thay vì copy tất cả feature vào cùng một model.
