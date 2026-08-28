# TimesFM cho bài toán Distribution Forecasting BTC với dữ liệu OHLCV

> **STATUS — LEGACY RESEARCH REFERENCE, SUPERSEDED 2026-08-24.**
> Tài liệu này được giữ để bảo toàn lịch sử nghiên cứu R0→R6, không còn là canonical objective/metric protocol. Quyết định user mới đặt **POINT FORECASTING** làm objective duy nhất; CRPS/pinball/coverage/sharpness và Track B bên dưới không thuộc canonical evaluation. Trong plan đơn giản hóa 2026-08-27 chỉ còn **TFM-POINT** (native point, zero-shot, cộng dồn one-step return thành y_h) làm benchmark ở Bước 4 của `docs/RESEARCH_PLAN.md`; ladder R1–R6 không dùng (LoRA chỉ xét nếu TFM-POINT có skill). Khi có xung đột, RESEARCH_PLAN và `.claude/CLAUDE.md` thắng.

## 0. Phạm vi và giả định

Tài liệu này tổng hợp toàn bộ hướng tiếp cận đã thảo luận cho bài toán sử dụng **TimesFM** để dự báo phân phối tương lai của BTC, từ baseline đơn giản đến **R6 — LoRA fine-tuning**.

### Dữ liệu thực tế có sẵn

Chỉ sử dụng:

- **Open**
- **High**
- **Low**
- **Close**
- **Volume**

Không sử dụng và không giả định có:

- Order book
- Book ticker
- Aggregate trades
- Order-flow imbalance
- Bid/ask imbalance
- Funding rate
- Liquidation
- Open Interest
- Spot/Futures microstructure features
- VWAP nếu dữ liệu gốc không cung cấp trực tiếp

Do đó, toàn bộ pipeline phải được thiết kế sao cho **TimesFM + các phép biến đổi từ OHLCV** là đủ để chạy end-to-end.

---

# 1. Mục tiêu bài toán

Bài toán không chỉ là dự đoán một giá trị duy nhất:

\[
\hat{y}_{t+h}
\]

mà là dự đoán **phân phối xác suất** của tương lai:

\[
p(y_{t+1:t+H}\mid \mathcal{H}_t)
\]

với:

- \(t\): thời điểm hiện tại
- \(H\): forecast horizon
- \(\mathcal{H}_t\): toàn bộ thông tin OHLCV có thể quan sát tới thời điểm \(t\)

Output cuối cùng mong muốn là một ensemble gồm nhiều trajectory:

\[
\left\{
Y^{(1)},Y^{(2)},\ldots,Y^{(N)}
\right\}
\]

trong đó:

\[
Y^{(n)}
=
\left[
y_{t+1}^{(n)},
\ldots,
y_{t+H}^{(n)}
\right]
\]

Ví dụ:

\[
N=100
\]

future paths.

Các paths này sẽ được sử dụng để đánh giá bằng probabilistic metric, đặc biệt là **CRPS**.

---

# 2. Tại sao không chỉ dùng point forecast?

Một point forecast chỉ trả về:

\[
\hat{y}_{t+h}
\]

Ví dụ:

\[
\hat{r}_{t+10}=0.02\%
\]

Nhưng hai forecast sau có thể cùng mean:

### Forecast A

\[
E[r]=0.02\%
\]

với uncertainty nhỏ.

### Forecast B

\[
E[r]=0.02\%
\]

nhưng uncertainty rất lớn.

Nếu chỉ dùng MSE/MAE thì hai trường hợp trên có thể trông giống nhau.

Trong distribution forecasting, model phải trả lời cả hai câu hỏi:

1. **Trung tâm của tương lai nằm ở đâu?**
2. **Model bất định đến mức nào?**

Do đó:

\[
\boxed{
\text{Point forecast}
\neq
\text{Distribution forecast}
}
\]

---

# 3. Vai trò của TimesFM

TimesFM được dùng như một **pretrained time-series forecasting prior**.

Nó học các pattern thời gian tổng quát từ quá trình pretraining và có thể sử dụng zero-shot hoặc fine-tune nhẹ trên chuỗi BTC.

Trong bài toán này, TimesFM không được xem là toàn bộ hệ thống cuối cùng.

Vai trò của nó là cung cấp:

\[
\boxed{
\text{forecast center}
+
\text{native quantiles}
}
\]

sau đó ta xây dựng một predictive distribution hoàn chỉnh hơn ở bên ngoài TimesFM.

---

# 4. Input nào nên đưa vào TimesFM?

Vì dữ liệu chỉ có OHLCV, biến quan trọng nhất cần quyết định là chuỗi target.

## 4.1. Không nên ưu tiên raw Close nếu mục tiêu là short-horizon dynamics

Raw Close:

\[
C_t
\]

có absolute price level lớn và non-stationary.

Ví dụ:

\[
60000,\ 60010,\ 60030,\ldots
\]

TimesFM có thể forecast trực tiếp price, nhưng với bài toán tài chính ngắn hạn, lựa chọn tự nhiên hơn là **log-return**.

## 4.2. Log-return từ Close

Định nghĩa:

\[
r_t
=
\log C_t-\log C_{t-1}
\]

hay:

\[
r_t
=
\log\left(\frac{C_t}{C_{t-1}}\right)
\]

TimesFM nhận:

\[
r_{t-L+1:t}
\]

và forecast:

\[
r_{t+1:t+H}
\]

Sau đó reconstruct lại giá:

\[
\log C_{t+h}
=
\log C_t
+
\sum_{i=1}^{h}r_{t+i}
\]

nên:

\[
C_{t+h}
=
C_t
\exp
\left(
\sum_{i=1}^{h}r_{t+i}
\right)
\]

### Lợi ích

- Giảm phụ thuộc absolute price level.
- Gần với đối tượng thực sự cần dự báo: price movement.
- Thuận tiện hơn cho distribution modeling.
- Tail và volatility có ý nghĩa rõ ràng hơn.

---

# 5. OHLCV còn dùng được gì ngoài Close?

Mặc dù TimesFM core có thể chỉ forecast một target series, OHLCV vẫn có thể tạo ra nhiều biến lịch sử hữu ích.

## 5.1. Candle return

\[
r_t=\log(C_t/C_{t-1})
\]

## 5.2. Intrabar range

\[
Range_t
=
\frac{H_t-L_t}{C_{t-1}}
\]

hoặc log-range.

Nó phản ánh mức độ biến động bên trong candle.

## 5.3. Candle body

\[
Body_t
=
\frac{C_t-O_t}{O_t}
\]

## 5.4. Upper wick

\[
UpperWick_t
=
\frac{
H_t-\max(O_t,C_t)
}{
O_t
}
\]

## 5.5. Lower wick

\[
LowerWick_t
=
\frac{
\min(O_t,C_t)-L_t
}{
O_t
}
\]

## 5.6. Volume

Có thể dùng:

\[
\log(1+V_t)
\]

hoặc volume normalized bằng rolling statistics.

## 5.7. Realized-volatility proxy

Ví dụ:

\[
RV_{t,k}
=
\sqrt{
\sum_{i=t-k+1}^{t}r_i^2
}
\]

với nhiều window:

\[
k\in\{5,10,30,60,\ldots\}
\]

## 5.8. Rolling return

\[
R_{t,k}
=
\sum_{i=t-k+1}^{t}r_i
\]

## 5.9. Volume change

\[
\Delta \log V_t
=
\log(1+V_t)-\log(1+V_{t-1})
\]

---

# 6. Lưu ý quan trọng về OHLCV-derived features

Không phải cứ có OHLCV-derived feature là có thể nhét trực tiếp tất cả vào TimesFM.

Nếu sử dụng covariate yêu cầu giá trị của feature trong cả future horizon thì các biến như:

- future volume
- future High
- future Low
- future candle range

đều **không biết trước tại inference time**.

Do đó, hướng an toàn nhất là:

### Nhánh TimesFM

Dùng target history:

\[
r_{t-L+1:t}
\]

để sinh point forecast + quantiles.

### Optional historical calibration/correction model

Dùng OHLCV-derived features chỉ từ:

\[
\tau\le t
\]

để hiệu chỉnh output TimesFM.

Không được sử dụng feature chứa thông tin sau prediction timestamp.

---

# 7. Output quan trọng của TimesFM

TimesFM có thể trả:

\[
\mu_h
\]

và các quantile:

\[
q_{0.1,h},
q_{0.2,h},
\ldots,
q_{0.9,h}
\]

cho mỗi horizon:

\[
h=1,\ldots,H
\]

Ví dụ tại một horizon:

\[
q_{0.1}=-0.05\%
\]

\[
q_{0.5}=0.01\%
\]

\[
q_{0.9}=0.08\%
\]

Ta có một mô tả ban đầu của uncertainty.

Nhưng đây mới chỉ là **marginal uncertainty**.

---

# 8. Marginal distribution và joint trajectory

TimesFM có thể cho:

\[
F_1(y),F_2(y),\ldots,F_H(y)
\]

với:

\[
F_h(y)
=
P(Y_{t+h}\le y)
\]

Nhưng bài toán trajectory cần:

\[
p(
Y_{t+1},
Y_{t+2},
\ldots,
Y_{t+H}
)
\]

Hai thứ này không giống nhau.

Có đúng marginal ở từng horizon không đảm bảo ta có realistic paths.

Ví dụ sample độc lập:

\[
u_h\overset{iid}{\sim}U(0,1)
\]

rồi:

\[
Y_h=F_h^{-1}(u_h)
\]

có thể sinh trajectory nhảy lên xuống thiếu temporal coherence.

Do đó cần phân biệt:

\[
\boxed{
\text{Marginal calibration}
}
\]

và:

\[
\boxed{
\text{Temporal dependence}
}
\]

---

# 9. Adaptation Ladder

Pipeline được xây theo kiểu ablation ladder:

\[
R0
\rightarrow
R1
\rightarrow
R2
\rightarrow
R3
\rightarrow
R4
\rightarrow
R5
\rightarrow
R6
\]

Mỗi rung chỉ thêm một ý tưởng chính.

Mục tiêu là xác định:

> Improvement thực sự đến từ component nào?

Không thay đổi nhiều thứ cùng lúc.

---

# 10. R0 — Point Forecast Replication

## Ý tưởng

Lấy point forecast của TimesFM:

\[
\hat{Y}
=
[
\hat{y}_1,\ldots,\hat{y}_H
]
\]

và copy nó thành \(N\) paths:

\[
Y^{(1)}
=
Y^{(2)}
=
\cdots
=
Y^{(N)}
=
\hat{Y}
\]

Ví dụ:

```text
Path 1: [0.01, 0.02, 0.02, ...]
Path 2: [0.01, 0.02, 0.02, ...]
...
Path N: [0.01, 0.02, 0.02, ...]
```

## Distribution tương ứng

Variance bằng 0:

\[
Var(Y)=0
\]

Model đang thể hiện certainty gần như tuyệt đối.

Có thể hình dung predictive distribution như Dirac delta:

\[
p(Y)=\delta(Y-\hat{Y})
\]

## Mục tiêu của R0

R0 không được kỳ vọng là probabilistic model tốt.

Nó là baseline để trả lời:

> Chỉ dùng TimesFM point forecast thì CRPS như thế nào?

---

# 11. R1 — Gaussian Noise

## Ý tưởng

Thêm noise quanh point forecast:

\[
y_h^{(n)}
=
\hat{y}_h
+
\epsilon_h^{(n)}
\]

với:

\[
\epsilon_h^{(n)}
\sim
\mathcal{N}(0,\sigma_h^2)
\]

Phiên bản baseline đơn giản nhất có thể dùng:

\[
\sigma_h=\sigma
\]

cho tất cả horizon.

## Lợi ích

- Có uncertainty.
- Dễ triển khai.
- Không cần fine-tune TimesFM.
- Có thể tạo ensemble paths ngay.

## Hạn chế

Gaussian noise tự đặt không tận dụng uncertainty mà TimesFM đã học.

Nếu \(\sigma\) cố định:

- Không thích ứng volatility regime.
- Không phản ánh horizon-specific uncertainty.
- Không phản ánh asymmetric distribution.
- Không phản ánh heavy tails.
- Không phản ánh BTC-specific distribution shape.

R1 chủ yếu có giá trị như một baseline trung gian.

---

# 12. R2 — Native Quantile Sampling

Đây là rung quan trọng nhất về mặt chuyển đổi từ point forecast sang distribution forecast.

## 12.1. Dùng trực tiếp TimesFM quantiles

Với mỗi horizon \(h\), TimesFM cho:

\[
Q_h(0.1),
Q_h(0.2),
\ldots,
Q_h(0.9)
\]

Ta xem:

\[
Q_h(\tau)=F_h^{-1}(\tau)
\]

là quantile function tại horizon \(h\).

## 12.2. Nội suy quantile

Giả sử:

\[
Q_h(0.7)=0.02
\]

\[
Q_h(0.8)=0.04
\]

và sample:

\[
u=0.73
\]

Linear interpolation:

\[
Q_h(0.73)
=
Q_h(0.7)
+
\frac{0.73-0.7}{0.8-0.7}
[
Q_h(0.8)-Q_h(0.7)
]
\]

## 12.3. Sampling

Draw:

\[
u\sim U(0,1)
\]

rồi:

\[
y_h=Q_h(u)
\]

Lặp lại để tạo nhiều samples.

## Lợi ích

Uncertainty bây giờ đến từ chính TimesFM:

\[
\boxed{
\text{native learned uncertainty}
}
\]

thay vì Gaussian noise tự đặt.

Quantile spread có thể thay đổi theo:

- history
- horizon
- local volatility
- current regime mà model suy ra từ target history

## Kết quả từ experiment gốc

Trong tài liệu thí nghiệm, Native Quantile Sampling là bước đem lại cải thiện lớn nhất trong adaptation ladder, với CRPS giảm khoảng 19% so với point-only baseline.

---

# 13. R2.5 — Temporal Dependence Layer (khuyến nghị bổ sung)

Đây không phải rung R0–R6 gốc, nhưng là một bổ sung quan trọng nếu metric đánh giá **whole trajectory**.

## Vấn đề

Nếu sample độc lập:

\[
u_1,\ldots,u_H
\overset{iid}{\sim}
U(0,1)
\]

thì:

\[
Y_h=Q_h(u_h)
\]

có đúng marginal nhưng temporal dependence có thể sai.

## Hướng 1 — Gaussian Copula

Ước lượng correlation matrix:

\[
R\in\mathbb{R}^{H\times H}
\]

từ historical forecast residuals.

Sample:

\[
z
\sim
\mathcal{N}(0,R)
\]

sau đó:

\[
u_h=\Phi(z_h)
\]

và:

\[
y_h=Q_h(u_h)
\]

Kết quả:

- Mỗi horizon vẫn giữ marginal từ TimesFM.
- Các horizon không còn độc lập.
- Path có temporal coherence tốt hơn.

## Hướng 2 — Residual block sampling

Từ historical residual trajectories:

\[
e_{t,1:H}
\]

sample cả một residual block thay vì sample từng timestep riêng lẻ.

Lợi ích:

- Giữ dependence thực nghiệm.
- Không cần giả định Gaussian copula.

## Quyết định thực nghiệm

Không đưa bước này vào R0–R6 nếu muốn giữ đúng ladder gốc.

Có thể chạy như một ablation riêng:

\[
R2
\rightarrow
R2+\text{Dependence}
\]

để xem metric trajectory có cải thiện không.

---

# 14. R3 — Tail Completion

## Vấn đề

TimesFM native quantiles chỉ mô tả phần giữa:

\[
q_{0.1},\ldots,q_{0.9}
\]

Ta chưa có trực tiếp:

\[
q_{0.01},
q_{0.05},
q_{0.95},
q_{0.99}
\]

Do đó hai tails:

\[
u<0.1
\]

và:

\[
u>0.9
\]

chưa được mô hình hóa đầy đủ.

Với BTC, đây là vấn đề đáng kể vì return có thể heavy-tailed.

## Ý tưởng

Giữ nguyên phần trung tâm từ TimesFM:

\[
0.1\le u\le0.9
\]

và nối thêm tails bằng **Student-t distribution**.

Pipeline:

```text
TimesFM q10 ... q90
        │
        ├── giữ nguyên body
        │
        └── fit/attach Student-t tails
                    │
                    ▼
           Full inverse CDF
```

## Tại sao Student-t?

Student-t có tail dày hơn Gaussian.

Degree of freedom:

\[
\nu
\]

điều khiển tail thickness.

- \(\nu\) nhỏ → tail dày
- \(\nu\to\infty\) → gần Gaussian

## Nguyên tắc

Không thay phần TimesFM đã dự báo tốt.

Chỉ extrapolate ngoài:

\[
q_{10}
\]

và:

\[
q_{90}
\]

## Cần kiểm tra

- Continuity tại \(q_{10}\), \(q_{90}\)
- Tail scale
- Degree of freedom
- Extreme coverage
- CRPS / tail-sensitive diagnostics

---

# 15. R4 — Drift Re-centering

## Vấn đề

TimesFM có thể extrapolate recent drift quá mạnh.

Ví dụ chuỗi vừa tăng:

\[
\uparrow\uparrow\uparrow
\]

forecast tiếp tục:

\[
\uparrow\uparrow\uparrow
\]

Trong short-horizon financial returns, directional drift có thể rất yếu và dễ overfit.

## Ý tưởng

Giả sử ensemble có center:

\[
\mu_h
\]

Ta chuyển:

\[
Y_h'
=
Y_h-\mu_h+\mu_h^*
\]

Trong phiên bản neutral:

\[
\mu_h^*=0
\]

hoặc có thể dùng historical calibration để estimate target center.

## Điều quan trọng

Re-centering không thay spread:

\[
Var(Y_h')=Var(Y_h)
\]

Nó chỉ thay center.

Do đó:

\[
\boxed{
\text{mean correction}
\neq
\text{uncertainty correction}
}
\]

## Mục tiêu

Giảm directional bias mà không phá uncertainty structure.

---

# 16. R5 — Per-Asset / BTC-Specific Tuning

Trong experiment gốc, R5 là per-asset tuning.

Nếu pipeline hiện tại chỉ tập trung BTC thì R5 nên được hiểu là:

\[
\boxed{
\text{BTC-specific distribution calibration}
}
\]

thay vì phải có nhiều asset.

Có thể tune:

- Student-t degree of freedom
- tail scale
- recentering strength
- quantile interpolation method
- horizon-specific scale
- dependence parameters
- rolling calibration window

## Ví dụ

\[
\nu_{\text{BTC}}
\]

cho Student-t tail.

Hoặc:

\[
\lambda_h
\]

cho recentering:

\[
\mu_h'
=
(1-\lambda_h)\mu_h
\]

## Ý nghĩa thực nghiệm

Experiment gốc cho thấy per-asset tuning không mang lại cải thiện lớn.

Điều này gợi ý:

> Các structural changes như native quantile sampling, tail completion và bias correction quan trọng hơn manual tuning.

## Với bài toán chỉ có BTC

R5 vẫn nên giữ để kiểm chứng:

> Sau khi distribution architecture đã đúng, BTC-specific calibration có còn đem lại lợi ích đáng kể không?

---

# 17. R6 — Distribution LoRA Fine-tuning

R0–R5 chủ yếu thay đổi cách **sử dụng output của TimesFM**.

R6 mới thay đổi model parameters.

## Ý tưởng

Không fine-tune toàn bộ TimesFM.

Dùng LoRA:

\[
W'
=
W+\Delta W
\]

với:

\[
\Delta W=BA
\]

trong đó rank:

\[
r\ll d
\]

Số parameter cần train nhỏ hơn nhiều so với full fine-tuning.

## Mục tiêu

Adapt TimesFM từ general time-series prior sang:

\[
\text{BTC OHLCV / return domain}
\]

## Training data

Input:

\[
r_{t-L+1:t}
\]

Target:

\[
r_{t+1:t+H}
\]

Nếu fine-tune quantile behavior thì loss cần được thiết kế phù hợp với output head / training API thực tế.

## Tại sao LoRA đặt cuối?

Nếu distribution construction đang sai thì fine-tune model chưa chắc giải quyết được.

Ví dụ:

- Chỉ dùng point forecast
- Không có tails
- Samples độc lập temporal
- Distribution miscalibrated

thì tăng model accuracy chưa chắc tối ưu CRPS.

Do đó logic:

\[
\boxed{
\text{Build distribution correctly first}
}
\]

rồi mới:

\[
\boxed{
\text{adapt model weights}
}
\]

## Kết luận từ experiment gốc

Fine-tuning vẫn cải thiện kết quả, nhưng mức gain nhỏ hơn bước chuyển:

\[
\text{Point Forecast}
\rightarrow
\text{Native Distribution Forecast}
\]

---

# 18. Pipeline cuối cùng đề xuất

```text
Historical OHLCV
      │
      ├── Close → log-return
      │
      ├── range / body / wick
      │
      ├── volume transforms
      │
      └── rolling return / RV
      │
      ▼
Target return history
      │
      ▼
   TimesFM
      │
      ├── Point forecast
      │
      └── q10 ... q90
                │
                ▼
       Quantile interpolation
                │
                ▼
       Native marginal CDFs
                │
                ├── optional temporal dependence
                │      Copula / residual blocks
                │
                ▼
        Student-t tail completion
                │
                ▼
          Drift re-centering
                │
                ▼
       BTC-specific calibration
                │
                ▼
         LoRA-adapted TimesFM
                │
                ▼
        N future return paths
                │
                ▼
      Reconstruct BTC prices
                │
                ▼
        CRPS + calibration
```

---

# 19. Một cách tổ chức R0–R6 rõ ràng

| Run | Thành phần mới | Mục đích |
|---|---|---|
| **R0** | TimesFM point copied thành N paths | Baseline |
| **R1** | + Gaussian noise | Có uncertainty sơ cấp |
| **R2** | + Native quantile sampling | Dùng uncertainty thật của TimesFM |
| **R3** | + Student-t tail completion | Mô hình hóa extreme events |
| **R4** | + Drift re-centering | Giảm directional bias |
| **R5** | + BTC-specific tuning | Calibration riêng cho BTC |
| **R6** | + LoRA fine-tuning | Adapt TimesFM weights cho BTC |

Ablation phụ nên cân nhắc:

| Run phụ | Thành phần |
|---|---|
| **R2-D** | R2 + temporal copula |
| **R3-D** | R3 + temporal copula |
| **R4-D** | R4 + temporal copula |

Điều này giúp kiểm tra dependence modeling thực sự có cần thiết cho metric hay không.

---

# 20. Evaluation

## 20.1. CRPS

Với ensemble samples:

\[
y_1,\ldots,y_N
\]

và observation:

\[
x
\]

empirical CRPS:

\[
CRPS
=
\frac{1}{N}
\sum_{n=1}^{N}
|y_n-x|
-
\frac{1}{2N^2}
\sum_{n=1}^{N}
\sum_{m=1}^{N}
|y_n-y_m|
\]

Term đầu:

\[
\frac{1}{N}\sum|y_n-x|
\]

đánh giá ensemble gần truth tới đâu.

Term hai:

\[
\frac{1}{2N^2}\sum|y_n-y_m|
\]

phản ánh ensemble dispersion.

Do đó CRPS đồng thời đánh giá:

- location accuracy
- spread
- calibration

Lower is better.

---

# 21. Không chỉ nhìn CRPS

Nên theo dõi thêm:

## Point metrics

- MAE
- RMSE
- directional accuracy nếu cần

## Quantile metrics

Pinball loss:

\[
L_\tau(y,q)
=
\begin{cases}
\tau(y-q), & y\ge q\\
(1-\tau)(q-y), & y<q
\end{cases}
\]

## Coverage

Ví dụ interval:

\[
[q_{0.1},q_{0.9}]
\]

theoretical coverage:

\[
80\%
\]

Kiểm tra empirical coverage có gần:

\[
0.8
\]

hay không.

## Sharpness

Hai model đều calibrated nhưng model có interval hẹp hơn hợp lý sẽ tốt hơn.

## Tail diagnostics

Kiểm tra riêng:

- extreme positive returns
- extreme negative returns
- high-volatility regime

## Horizon-wise metrics

Không chỉ aggregate:

\[
h=1,\ldots,H
\]

mà phải xem lỗi từng horizon.

---

# 22. Validation protocol

Vì đây là time series:

\[
\boxed{
\text{Không random shuffle train/validation}
}
\]

Nên dùng walk-forward / expanding-window.

Ví dụ:

```text
Train 1  ────────────
Val 1                 ───

Train 2  ─────────────────
Val 2                      ───

Train 3  ─────────────────────
Val 3                           ───
```

Mọi transform cần fit chỉ trên training history.

Ví dụ:

- normalization
- volatility scale
- tail parameters
- copula correlation
- recentering coefficients

Không được nhìn validation/test future.

---

# 23. Data leakage cần tránh

Prediction timestamp:

\[
t
\]

chỉ được sử dụng data:

\[
\tau\le t
\]

Các rolling features phải kết thúc tại \(t\).

Không được vô tình dùng:

- \(H_{t+1}\)
- \(L_{t+1}\)
- \(V_{t+1}\)
- candle tương lai chưa đóng
- normalization statistics tính từ toàn dataset

---

# 24. Vai trò của Volume khi không có order book

Vì không có microstructure data, Volume trở thành nguồn information bổ sung quan trọng nhất ngoài price.

Ví dụ cùng một return:

\[
r_t=+0.2\%
\]

nhưng:

### Case A

Volume thấp.

### Case B

Volume cao gấp 10 lần rolling median.

Hai move có thể mang ý nghĩa khác nhau.

Có thể tạo:

\[
VolumeZ_t
=
\frac{
\log(1+V_t)-\mu_{V,t}
}{
\sigma_{V,t}
}
\]

với rolling statistics chỉ dùng quá khứ.

Volume đặc biệt hữu ích để mô tả:

- activity regime
- volatility regime
- unusual price moves

Tuy nhiên không nên coi:

\[
Volume\uparrow
\Rightarrow
Price\uparrow
\]

Volume không có direction nếu chỉ có aggregate candle volume.

---

# 25. Những gì không thể suy ra từ OHLCV

Với OHLCV đơn thuần, không thể biết trực tiếp:

- Buyer hay seller là aggressor
- Bid liquidity / ask liquidity
- Order book imbalance
- Order-flow imbalance
- Quote cancellation
- Spread
- Market depth
- Microprice
- Liquidation flow
- Futures positioning
- Funding
- Open Interest

Do đó không nên tạo các feature giả định thay thế chúng rồi gọi cùng tên.

Pipeline phải chấp nhận rằng một phần microstructure information là **không quan sát được**.

---

# 26. Điều đó ảnh hưởng TimesFM như thế nào?

Với dữ liệu OHLCV-only, TimesFM trở nên quan trọng hơn như một temporal prior vì ta không có realtime microstructure branch để sửa forecast.

Tuy nhiên cũng cần kỳ vọng thực tế:

\[
\boxed{
\text{OHLCV-only}
}
\]

có information set yếu hơn:

\[
\boxed{
\text{OHLCV + order book + trades}
}
\]

đặc biệt ở ultra-short horizon.

Vì vậy nghiên cứu nên tập trung vào:

1. Forecast representation tốt.
2. Distribution construction tốt.
3. Tail calibration.
4. Temporal dependence.
5. BTC-specific fine-tuning.
6. OHLCV-derived regime features.

---

# 27. Optional OHLCV Correction Head

Nếu chỉ TimesFM vẫn chưa đủ, có thể xây một side model rất nhẹ từ historical OHLCV-derived features.

Ví dụ input:

\[
x_t=
[
r_t,
Range_t,
Body_t,
UpperWick_t,
LowerWick_t,
\log(1+V_t),
RV_{5},
RV_{30},
R_{5},
R_{30},
...
]
\]

Model:

- LightGBM
- small MLP
- TCN
- GRU

Output:

\[
\Delta\mu_h
\]

và:

\[
s_h>0
\]

Sau đó hiệu chỉnh TimesFM quantiles:

\[
Q_h^*(\tau)
=
\mu_h^{TFM}
+
\Delta\mu_h
+
s_h
\left[
Q_h^{TFM}(\tau)-\mu_h^{TFM}
\right]
\]

Trong đó:

### \(\Delta\mu_h\)

sửa center/direction.

### \(s_h\)

sửa spread/volatility.

Đây là extension sau R0–R6, không bắt buộc cho baseline TimesFM experiment.

---

# 28. Các câu hỏi thí nghiệm cần trả lời

## Q1. TimesFM point forecast có thực sự tốt hơn baseline hiện tại không?

So sánh R0 với baseline model.

## Q2. Gain đến từ TimesFM forecasting hay chỉ từ việc thêm uncertainty?

So sánh:

\[
R0
\leftrightarrow
R1
\leftrightarrow
R2
\]

## Q3. Native quantiles có calibrated không?

Kiểm tra coverage.

## Q4. Tail completion có thực sự cần?

So sánh:

\[
R2
\leftrightarrow
R3
\]

đặc biệt trên extreme events.

## Q5. Drift của TimesFM có bias?

So sánh:

\[
R3
\leftrightarrow
R4
\]

và kiểm tra signed error.

## Q6. BTC-specific manual tuning đáng giá không?

\[
R4
\leftrightarrow
R5
\]

## Q7. LoRA có mang lại improvement đủ lớn so với complexity?

\[
R5
\leftrightarrow
R6
\]

## Q8. Temporal dependence có quan trọng?

So sánh independent sampling với copula/residual block sampling.

---

# 29. Thứ tự triển khai khuyến nghị

## Phase 1 — Data

- Load OHLCV.
- Sort timestamp.
- Check gaps.
- Check duplicate timestamps.
- Xử lý missing data.
- Tạo Close log-return.
- Tạo walk-forward splits.

## Phase 2 — TimesFM baseline

- R0 point forecast.
- Cache predictions.
- Đánh giá point metrics + CRPS.

## Phase 3 — Distribution

- R1 Gaussian.
- R2 native quantiles.
- Quantile interpolation.
- Calibration diagnostics.

## Phase 4 — Dependence

- Independent sampling baseline.
- Gaussian copula.
- Residual block sampling nếu phù hợp.

## Phase 5 — Structural corrections

- R3 tail completion.
- R4 recentering.
- R5 BTC calibration.

## Phase 6 — Training

- R6 LoRA fine-tuning.

## Phase 7 — Final ablation

Chạy cùng split, cùng seed policy và cùng metric.

---

# 30. Bảng ablation cuối cùng

| Run | TimesFM | Quantile | Tail | Recenter | BTC Tune | LoRA | Dependence |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | ✓ |  |  |  |  |  |  |
| R1 | ✓ | Gaussian |  |  |  |  | optional |
| R2 | ✓ | Native |  |  |  |  | independent |
| R2-D | ✓ | Native |  |  |  |  | Copula |
| R3 | ✓ | Native | ✓ |  |  |  | chosen |
| R4 | ✓ | Native | ✓ | ✓ |  |  | chosen |
| R5 | ✓ | Native | ✓ | ✓ | ✓ |  | chosen |
| R6 | ✓ | Native | ✓ | ✓ | ✓ | ✓ | chosen |

---

# 31. Nguyên tắc ra quyết định

Không chọn model dựa trên một metric aggregate duy nhất.

Một run chỉ được coi là tốt hơn nếu:

1. CRPS cải thiện ổn định trên walk-forward folds.
2. Calibration không xấu đi rõ rệt.
3. Tail performance không bị phá.
4. Improvement xuất hiện ở nhiều horizon.
5. Không phụ thuộc duy nhất vào một market regime.
6. Complexity tăng có lý do tương xứng với gain.

---

# 32. Kết luận

Toàn bộ hướng nghiên cứu có thể cô đọng thành:

\[
\boxed{
\text{OHLCV}
\rightarrow
\text{returns}
\rightarrow
\text{TimesFM}
\rightarrow
\text{native quantiles}
\rightarrow
\text{distribution construction}
\rightarrow
\text{trajectory sampling}
\rightarrow
\text{CRPS}
}
\]

Adaptation ladder:

\[
\boxed{
R0
\rightarrow
R1
\rightarrow
R2
\rightarrow
R3
\rightarrow
R4
\rightarrow
R5
\rightarrow
R6
}
\]

với ý nghĩa:

\[
\text{Point}
\rightarrow
\text{Noise}
\rightarrow
\text{Native Quantiles}
\rightarrow
\text{Tail}
\rightarrow
\text{Bias Correction}
\rightarrow
\text{BTC Calibration}
\rightarrow
\text{LoRA}
\]

Bài học trung tâm của experiment là:

\[
\boxed{
\text{Xây đúng predictive distribution trước khi cố fine-tune model}
}
\]

Trong setup hiện tại, tuyệt đối không dựa vào order book, trades hay các microstructure features vì chúng không tồn tại trong data.

Nguồn ý tưởng ban đầu: tài liệu nội bộ “Experiment: Thử nghiệm TimesFM” và các trao đổi tiếp theo về TimesFM, distribution forecasting, quantile sampling, tail completion, drift re-centering, temporal dependence và LoRA.
