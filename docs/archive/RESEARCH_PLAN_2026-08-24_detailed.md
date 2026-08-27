> **ARCHIVED 2026-08-27 — không còn hiệu lực.** Bản này được thay bằng plan 4 bước đơn giản hóa trong `docs/RESEARCH_PLAN.md` / `.claude/CLAUDE.md`. Giữ chỉ để tham khảo lịch sử.

# RESEARCH PLAN — BTC 1-phút point forecasting

Cập nhật: 2026-08-24. Đây là bản sửa in-place của roadmap cũ theo quyết định user mới nhất và audit read-only đã hoàn tất. Quyết định trong bản này override mọi scope, Track B, holdout hoặc metric assumption cũ nếu xung đột.

Nhãn trạng thái:

- **[FROZEN]** — quyết định user canonical; không tự thay đổi.
- **[PLAN]** — thiết kế chưa đủ evidence để freeze; phải được pre-register trước khi dùng.
- **[FUTURE]** — chỉ thực hiện sau khi user unlock training.
- **[OPTIONAL]** — không thuộc pipeline mặc định; cần phê duyệt riêng nếu kích hoạt.
- **[OBSOLETE]** — quyết định cũ đã bị quyết định user mới override.

---

## 1. Objective, target và model scope

### 1.1 Primary research objective [FROZEN]

Toàn project có một objective chính:

~~~text
POINT FORECASTING
~~~

Canonical target:

\[
y_h(t)=\log\frac{C[t+h]}{C[t]},
\qquad h\in\{1,2,3\}\text{ phút}.
\]

Mọi candidate cuối cùng phải sinh ba scalar prediction:

\[
\hat y_1,\hat y_2,\hat y_3.
\]

Không còn Track B với distribution forecasting là objective độc lập. Quantile, trajectory hoặc distribution machinery của TimesFM chỉ là internal mechanism nếu nó giúp tạo point estimate tốt hơn theo cùng point-metric contract.

### 1.2 Model scope [FROZEN]

**Core point-forecast model tracks**

1. **Tree models**
   - frozen LightGBM B0;
   - LightGBM candidate/B*;
   - XGBoost;
   - CatBoost;
   - tree point ensemble nếu đủ điều kiện.

2. **Foundation time-series model**
   - TimesFM original/native point forecast;
   - TimesFM custom point-oriented variants;
   - TimesFM LoRA/adaptation nếu version/API thực tế hỗ trợ.

3. **LSTM research exception**
   - đúng một track có thiết kế rõ ràng, proposal ban đầu là **LSTM-DMH-512**.

Việc mở LSTM không mở GRU, TCN, generic Transformer, PatchTST, MLP sequence model, Chronos, TimeGPT hoặc DL family khác. Không thêm Random Forest, Extra Trees, SGD hoặc model từ paper chỉ để mở rộng bảng benchmark.

### 1.3 Auxiliary research roles [FROZEN]

- **yfinance**: optional auxiliary data-source / audit tool; không phải model và không thay Binance/Hyperliquid canonical data.
- **AutoTS**: optional constrained research harness; không phải model candidate độc lập và không được tự search model zoo ngoài scope.
- **arXiv:2407.18334v1**: research/reference only; không có quyền mở model scope.

---

## 2. Training state và P0 data-integrity gate

### 2.1 Current state [FROZEN]

~~~text
TRAINING_LOCKED
~~~

Trong trạng thái này được phép: research, sửa docs/governance, thiết kế config/schema/test, static validation và unit test nhẹ không cần full data hoặc model nặng.

Không được: train B0/candidate/LSTM, chạy real fold/holdout, load TimesFM checkpoint nặng, tải lại data, cài package hoặc sinh experiment result giả.

### 2.2 Audit finding: local snapshot không canonical [BLOCKER]

Manifest mô tả BTC 1 phút có 289.320 bars, nhưng audit local snapshot ghi nhận:

| Asset 1 phút | Expected theo manifest | Actual data rows local |
|---|---:|---:|
| BTC | 289.320 | 21.917 |
| ETH | 289.320 | 22.999 |
| SOL | 289.315 | 24.579 |
| XRP | 289.315 | 21.618 |

Các file LF tương ứng cũng thiếu lớn; nhiều file dừng ở kích thước cố định và không kết thúc bằng newline. Manifest là intended contract, chưa phải bằng chứng rằng bytes local hiện đầy đủ.

Không được dùng snapshot truncate này làm canonical dataset, không được tạo checksum từ nó rồi gọi là canonical, và không được chạy real experiment trước khi P0 PASS.

### 2.3 P0 gate bắt buộc [FROZEN]

Trước mọi real experiment:

1. khôi phục canonical CSV đầy đủ bằng quy trình được user cho phép;
2. actual row count khớp expected contract;
3. timestamp range được xác minh;
4. timezone UTC được xác minh;
5. grid 60s/300s, duplicate và gap được kiểm tra;
6. schema được chuẩn hóa;
7. tạo **data/data_checksums.json** cho snapshot đã được xác minh;
8. freeze data version và checksum trong config/log.

**data/data_checksums.json hiện chưa tồn tại.** Không tạo file đó trong task governance này.

### 2.4 Canonical ingestion/schema adapter [PLAN]

Raw CSV dùng:

~~~text
open, high, low, close, volume
~~~

trong khi frozen B0 nhận:

~~~text
Open, High, Low, Close, Volume
~~~

Ingestion/harness phải có schema adapter explicit, deterministic và có unit test. Không sửa **Baseline_LGBM.py** để accommodate raw schema. Không trộn các file **\*_close.csv** vào canonical OHLCV snapshot.

---

## 3. Partition, walk-forward và final holdouts

### 3.1 Target-availability partition contract [FROZEN]

Partition dùng half-open interval \([T_{start},T_{end})\). Một origin \(t\) chỉ thuộc partition nếu toàn bộ target của nó thuộc cùng partition:

\[
t\ge T_{start}
\quad\text{và}\quad
t+h_{max}\cdot60s<T_{end},
\qquad h_{max}=3.
\]

Không đủ nếu chỉ assert \(t<T_{end}\). Quy tắc này áp dụng cho FIT, ES, VAL, HOLDOUT-NEAR và HOLDOUT-FAR. Origins sát biên không đủ target phải bị loại, không được mượn future bar từ partition kế tiếp.

### 3.2 Walk-forward development protocol [FROZEN structure; timestamps PLAN]

Giữ cấu trúc nếu canonical data sau restoration cho phép:

- 5 walk-forward folds;
- 3 ngày VAL mỗi fold;
- rolling 45-day initial budget;
- 5 ngày cuối train-region là ES;
- purge 60 phút giữa ES và VAL;
- budget development ban đầu 60 ngày;
- expanding chỉ ở data-scaling phase.

Semantics:

~~~text
FIT  → học model parameters
ES   → chọn training duration / best_iteration
VAL  → candidate/model/config selection; prediction tại VAL là OOF
HOLDOUT → final locked generalization only
~~~

FIT và ES cũng phải tuân thủ target-boundary rule. ES không phải VAL và không được dùng như một VAL bổ sung.

Implementation truth của B0: objective là Huber và early stopping hiện monitor Huber. B0 giữ nguyên. Nếu nghiên cứu alignment Huber objective/ES với RMSE-based selection, phải tạo explicit candidate/variant; không silently đổi B0 và không tự phát minh ES metric.

### 3.3 HOLDOUT-NEAR [FROZEN concept; timestamps PLAN]

Mục tiêu: đo generalization khi deployment period tương đối gần development.

Sau Dev phải có embargo rõ ràng:

\[
gap_{Dev\rightarrow Near}
\ge
\max(\text{target-boundary requirement},60\text{ phút}).
\]

Near không bắt đầu tại bar kế tiếp của Dev.

### 3.4 HOLDOUT-FAR [FROZEN concept; timestamps PLAN]

Mục tiêu: đo temporal/regime robustness khi evaluation period nằm xa development hơn. Far phải có temporal separation đáng kể; exact gap và timestamps chỉ được đề xuất sau khi canonical full range được phục hồi và phải có rationale.

Không hard-code chia 10/10/10 hoặc timestamp dựa trên snapshot truncate hiện tại.

### 3.5 Near/Far one-shot protocol [FROZEN]

~~~text
LOCK ALL RESEARCH DECISIONS
          ↓
REFIT FINAL MODELS
          ↓
FINAL EVALUATION
       ↙       ↘
     NEAR      FAR
~~~

- Cả Near và Far là final holdout.
- Cùng frozen model parameters, pipeline và pre-registered inference policy.
- Chỉ dùng causal observed history có sẵn tại mỗi origin; không dùng holdout labels để update model.
- Final runner không được expose kết quả Near trước khi Far hoàn tất.
- Không inspect, plot, tune, feature-select, calibrate hoặc sửa pipeline sau khi nhìn một trong hai kết quả.
- Report Near và Far riêng; không tạo composite metric mới.

Holdout 30 ngày liền ngay sau Dev trong roadmap cũ là **[OBSOLETE]**.

---

## 4. Frozen B0, metric contract và flat-forecast hypothesis

### 4.1 Baseline B0 [FROZEN]

**Baseline_LGBM.py** là frozen baseline:

- 306 features: 22 fine × 8 lags + 16 coarse × 8 lags + rv60/log_rv60;
- max historical offset khoảng 504 phút;
- ba LightGBM độc lập cho \(h=1,2,3\);
- train-only volatility-normalized TargetTransform;
- Huber objective, alpha 0,90;
- early stopping monitor Huber khi ES được truyền vào.

Không sửa file, không ablate lại 306 feature gốc và không silently đổi hyperparameter. Schema adapter, split và metric handling thuộc ingestion/harness.

Runtime invariant cũ vẫn giữ: real experiment chạy GPU generic trên Vast sau unlock; không hard-code GPU model; CPU chỉ cho static/unit/synthetic test nhẹ.

### 4.2 Canonical metrics [FROZEN]

Primary per horizon × fold:

\[
Gain_{h,f}
=
1-
\frac{RMSE_{cand}(h,f)}
     {RMSE_{base}(h,f)}.
\]

\(RMSE_{base}\) là frozen point baseline đã đăng ký cho comparison đó, ban đầu là B0; identity của baseline phải được ghi trong config/log và không đổi giữa chừng trong một experiment.

Primary summaries:

- MedianGain;
- WinRate;
- P10Gain / WorstGain.

Secondary hiện có:

- RMSE;
- MAE;
- Pearson \(r\);
- directional accuracy;
- feature importance/diagnostics đã tồn tại nếu applicable.

E0, với \(\hat y=0\), luôn được log làm mốc gốc. Daily-block paired analysis đã có trong protocol có thể giữ vì target \(h=2,3\) chồng lấp; không coi per-bar errors là iid.

**No-new-metric invariant:** không tự thêm CRPS, pinball, coverage, sharpness, energy score, variogram score, \(R^2\), MAPE, sMAPE, Sharpe, IC, calibration slope, variance/IQR ratio, signed-bias metric hoặc metric canonical khác. Auto-discovery chỉ áp dụng cho feature/hypothesis, không áp dụng cho metric.

### 4.3 OOF terminology [FROZEN]

OOF = Out-Of-Fold prediction. Trong mỗi walk-forward fold, model fit trên FIT/ES và prediction trên corresponding VAL block là OOF prediction. Primary evidence vẫn là:

\[
5\ folds\times3\ horizons=15\ cells.
\]

Không tồn tại một “OOF metric” mới.

### 4.4 Noise normalization và KEEP/DROP [PLAN]

B0 × 3 seeds trên cùng folds để estimate seed noise; ngưỡng ban đầu giữ từ plan cũ:

\[
\epsilon=\max(0.005\text{ pp},1\sigma_{seed}).
\]

- KEEP nếu MedianGain ≥ \(+\epsilon\) và WinRate ≥ 60%.
- DROP nếu MedianGain < \(-\epsilon\), hoặc WinRate < 40%, hoặc P10Gain < \(-3\epsilon\).
- Vùng giữa là neutral; giữ record và không diễn giải quá mức.
- Ca sát ngưỡng: thêm seed đã pre-register, không đổi metric.

### 4.5 Flat forecast là hypothesis [FROZEN framing]

Chưa có real run chứng minh B0 hoặc TimesFM đang forecast phẳng. Roadmap phải phân biệt:

1. conditional mean thực sự gần zero;
2. model có signal nhưng point prediction bị shrink;
3. data/schema/target/decode/pipeline bị lỗi.

Không ép forecast “rung hơn” để đẹp plot. Các diagnostic mới được audit gợi ý chỉ là potential investigation notes và không trở thành canonical metrics nếu user chưa phê duyệt. P0 assertions, target reconstruction tests và existing metrics là evidence ban đầu.

---

## 5. Feature discovery và tree branch

### 5.1 FEATURE DISCOVERY trong TRAINING_LOCKED [FROZEN]

Workflow:

~~~text
TRAINING_LOCKED
      ↓
literature / indicator research
      ↓
candidate family design
      ↓
causality + redundancy review
      ↓
freeze Wave-1 ladder
      ↓
unlock training
      ↓
walk-forward evaluation
~~~

X1–X5 là user-seeded families, không phải exhaustive whitelist. Research được phép đề xuất \(D1\ldots Dk\), với \(D\) là discovered family; \(k\) không hard-code trước research.

Nguồn discovery hợp lệ: finance/technical-analysis literature, statistical time-series features, OHLCV/amount/volume interaction, cross-asset, multi-timescale và causal market-microstructure proxies có thể reconstruct trung thực từ available data.

Mỗi family phải document trước real VAL:

1. family name;
2. economic/statistical hypothesis;
3. lý do phù hợp horizon 1–3 phút;
4. exact mathematical definition;
5. lookback;
6. causality;
7. redundancy với B0;
8. redundancy với family khác;
9. data availability;
10. compute/memory expectation;
11. leakage risks.

Không import toàn bộ TA-Lib rồi ném hàng trăm feature vào model. Các nhóm được phép nghiên cứu gồm RSI, MACD, ATR, Bollinger, ADX, stochastic-type oscillator, momentum, mean-reversion, trend strength, candle geometry, range estimators, realized/range volatility, vol-of-vol, compression/breakout, volume-price interaction, relative volume, jump proxy, lead-lag, cross-asset strength, regime và multi-scale interaction. Việc được nghiên cứu không có nghĩa tự động được vào Wave-1.

### 5.2 User-seeded X1–X5 [PLAN]

| Seed | Family | Nội dung ban đầu |
|---|---|---|
| X1 | xasset | ETH/SOL/XRP returns và relative strength/market mean; HYPE loại khỏi initial long-history set vì canonical OHLCV hiện dự kiến quá ngắn |
| X2 | vwap_amt_proxy | rolling volume-weighted typical-price proxy; avg_price_proxy từ amount/volume; amount/volume transforms, tên luôn có proxy khi không phải true VWAP |
| X3 | htf5 | causal BTC 5-minute closed bars; backward/as-of alignment |
| X4 | longterm | multi-hour/day returns, volatility ratios, EMA-relative, drawdown/run-up ngoài max-history B0 |
| X5 | regime | volatility, trend, compression và calendar interactions causal |

Exact formulas/lookbacks của X1–X5 phải được đưa qua cùng family dossier như D families trước khi Wave-1 freeze.

### 5.3 Wave-1, confirmation và second wave [PLAN]

- Freeze một ladder hữu hạn \(X1\ldots X5,D1\ldots Dk\) trước unlock.
- Ablate tuần tự; mỗi run thay đúng một family.
- Pruning chỉ chạm cột đã thêm, không chạm 306 feature B0.
- Winning set phải có confirmation riêng trên seed/config đã pre-register.
- Không vô hạn nghĩ feature mới chỉ vì candidate vừa thua VAL.
- Second-wave discovery, nếu được phép sau diagnostics, phải mang nhãn **EXPLORATORY**, có hypothesis và confirmation riêng; không silently nhập vào original ladder.

### 5.4 Data scaling [PLAN]

Sau khi feature information set B* được freeze:

~~~text
45d → 90d → 135d → expanding
~~~

Không scale data đồng thời với feature discovery nếu làm attribution không rõ. Canonical data phải PASS P0 trước.

### 5.5 Tree-family comparison và ensemble [PLAN]

Trên cùng B*, budget thắng cuộc, folds, origins và TargetTransform:

- LightGBM candidate;
- XGBoost;
- CatBoost.

Nếu ít nhất hai model có Gain dương và stability đủ điều kiện, thử tree point ensemble bằng equal-weight và inverse-OOF-MSE đã có trong plan cũ. Không mở model family khác và không để model-family comparison contaminate feature attribution.

Mọi nghiên cứu objective, ES alignment, regularization, point calibration hoặc affine correction phải là explicit named candidate; B0 không đổi.

---

## 6. TimesFM point-oriented branch

### 6.1 Version/API audit trước implementation [PLAN]

Trước code phụ thuộc TimesFM phải pin và record:

- package version;
- exact checkpoint/revision;
- backend;
- native point API;
- quantile API nếu dùng;
- context limit;
- LoRA/adaptation support;
- signed-return behavior;
- **infer_is_positive** hoặc equivalent nếu API đó có;
- deterministic repeated-inference smoke test;
- backend parity nếu practical.

Không generalize behavior từ version cũ sang version mới. GitHub issue về flatline chỉ là research note, không phải bằng chứng TimesFM chắc chắn lỗi. Trong TRAINING_LOCKED không load checkpoint nặng.

### 6.2 TFM-POINT native benchmark [FROZEN benchmark; implementation PLAN]

**TFM-POINT** là original/native point output của checkpoint, không noise, recenter, BTC calibration hoặc LoRA.

Input return context kết thúc đúng tại \(t\). Nếu API trả one-step:

\[
r_{t+1},r_{t+2},r_{t+3},
\]

phải reconstruct:

\[
y_1=r_{t+1},
\quad
y_2=r_{t+1}+r_{t+2},
\quad
y_3=r_{t+1}+r_{t+2}+r_{t+3}.
\]

Không so one-step \(r\) với cumulative \(y_h\). TFM-POINT dùng cùng origins, folds, Near/Far và point metrics như tree/LSTM.

### 6.3 Mapping legacy R0→R6 sang point ladder [FROZEN mapping rule; candidates PLAN]

Tài liệu **timesfm_ohlcv_distribution_forecasting_R0_R6.md** là legacy research reference. Không giữ từng rung máy móc:

| Legacy rung | Point relevance và mapping mới |
|---|---|
| R0 point-copy | Nếu point extraction vẫn là native point thì đây là no-op control, không phải candidate có point gain |
| R1 zero-mean Gaussian noise | Với expectation, \(E[\hat y+\epsilon]=\hat y\); loại khỏi default point ladder trừ khi có hypothesis point-specific được pre-register |
| R2 native quantiles | Có thể tạo **TFM-QMEAN** hoặc một q50 summary nếu output khác native point và extraction được pre-register |
| R2-D dependence | Chỉ giữ nếu reconstruct path trước rồi point extraction thực sự thay point estimate; không giữ chỉ vì distribution theory |
| R3 Student-t tails | Chỉ giữ nếu tail construction thay canonical expectation; nếu center/mean không đổi thì no-op cho objective hiện tại |
| R4 recenter | Giữ như **TFM-RECENTER**, vì trực tiếp thay location |
| R5 BTC calibration | Giữ component tác động center/point như **TFM-BTC-CAL**; bỏ distribution-only tuning khỏi canonical ladder |
| R6 LoRA | Giữ **TFM-LORA**, vì parameter adaptation có thể thay point skill |

Point-oriented ladder ban đầu:

~~~text
TFM-POINT
→ TFM-QMEAN / optional pre-registered Q50
→ TFM-RECENTER
→ TFM-BTC-CAL
→ TFM-LORA
~~~

Exact candidates phụ thuộc version/API audit. Không sweep hàng chục quantile để tìm quantile có RMSE tốt nhất trên VAL.

### 6.4 Point extraction và path reconstruction [FROZEN]

Nếu có trajectories \(y_h^{(1)},\ldots,y_h^{(N)}\), canonical RMSE-oriented extraction là:

\[
\hat y_h=\frac{1}{N}\sum_n y_h^{(n)}.
\]

Median/q50 chỉ là explicit pre-registered point variant có rationale. q95/q99 không mặc định là point predictor.

Nếu model sinh one-step paths \(r_{t+1:t+3}^{(n)}\), phải reconstruct **từng path**:

\[
y_1^{(n)}=r_{t+1}^{(n)},
\]

\[
y_2^{(n)}=r_{t+1}^{(n)}+r_{t+2}^{(n)},
\]

\[
y_3^{(n)}=r_{t+1}^{(n)}+r_{t+2}^{(n)}+r_{t+3}^{(n)},
\]

rồi mới lấy mean/median. Không cộng marginal quantiles và gọi đó là cumulative quantile. Bắt buộc có unit test cho ordering này.

Mọi TimesFM variant cuối cùng sinh \(\hat y_1,\hat y_2,\hat y_3\) và dùng metric contract §4.2. Không có CRPS/pinball/coverage/sharpness trong canonical evaluation.

---

## 7. LSTM research exception

### 7.1 Initial proposal: LSTM-DMH-512 [PLAN]

Research question:

> Dense sequential representation của cùng information set có cải thiện point skill so với sparse fixed-lag B0 hay không?

Proposal ban đầu:

- context khoảng 512 phút;
- unidirectional, sequence kết thúc đúng tại origin \(t\);
- không bidirectional, không future bars;
- không dùng raw absolute Close nếu representation hiện tại tránh nonstationary level;
- initial information set tương thích B0;
- một LSTM layer nhỏ;
- initial hidden size khoảng 64;
- final hidden state → linear 3-output head;
- direct multi-horizon \(\hat z_1,\hat z_2,\hat z_3\), không recursive;
- train-only target transform tương thích B0 nếu implementation cho phép fair comparison;
- initial Huber multi-horizon loss tương thích implementation truth;
- ít nhất 3 seeds sau unlock.

Đây là proposal, không phải frozen architecture. Có thể refine kỹ thuật sau review nhưng không biến thành hyperparameter sweep lớn và không mở thêm DL family.

### 7.2 Fair comparison và feature sequencing [FROZEN]

LSTM dùng cùng target, origins, folds, VAL, Near/Far và point metrics.

Attribution order:

1. establish E0/B0;
2. feature discovery/ablation ở tree track;
3. freeze B*;
4. đánh giá LSTM trước trên fair/base information set;
5. chỉ sau đó mới cân nhắc LSTM + B* như explicit question.

Không KEEP vì plot dao động hơn. Không thêm probabilistic objective, dropout/noise distribution hoặc metric mới. KEEP/DROP theo §4.2–4.4.

---

## 8. Vai trò của yfinance, AutoTS và paper

### 8.1 yfinance [OPTIONAL AUXILIARY DATA AUDIT]

- Không phải model.
- Không thay Binance/Hyperliquid canonical data.
- Không dùng Yahoo BTC-USD để bù missing Binance rows.
- Không coi Yahoo volume/microstructure tương đương Binance.
- Mọi future use làm feature cần explicit user decision thay đổi data contract, version pin, timestamp/missingness audit và snapshot checksum riêng.
- Không download trong TRAINING_LOCKED task hiện tại.

### 8.2 AutoTS [OPTIONAL AUXILIARY HARNESS — NOT CURRENT MODEL CANDIDATE]

Nếu future probe được user approve:

- explicit whitelist chỉ component đã nằm trong scope;
- project target/split/metric contract override AutoTS defaults;
- không sMAPE/default metric;
- không custom validation split;
- không uncontrolled model search;
- không silent target transform;
- không tự thêm ensemble/model family vì AutoTS hỗ trợ.

AutoTS prediction interval/distribution output không tạo objective hoặc metric mới cho project.

### 8.3 arXiv:2407.18334v1 [REFERENCE ONLY]

Paper benchmark 41 models nhưng không có LSTM và dùng horizon/data/evaluation/trading objective khác. Chỉ giữ các principle phù hợp như log-return representation, chronological evaluation, rolling-window sensitivity và tách model selection khỏi final forward evaluation.

Không đưa rankings Random Forest/SGD hoặc model khác của paper vào roadmap.

---

## 9. Research questions Q1–Q15

Q1. E0 so với frozen B0 trên 15 VAL cells như thế nào?

Q2. Nếu point forecast gần zero, existing metrics và pipeline assertions ủng hộ hypothesis conditional mean yếu, shrinkage hay implementation failure?

Q3. Trong Wave-1, family nào trong X1–X5 và D1–Dk cải thiện frozen baseline?

Q4. Winning feature set có survive confirmation mà không phụ thuộc một fold/horizon không?

Q5. Sau khi B* freeze, 45d/90d/135d/expanding budget nào tốt nhất?

Q6. LightGBM candidate, XGBoost hay CatBoost thắng trên cùng B*, budget và folds?

Q7. Tree ensemble đủ điều kiện có cải thiện best single tree không?

Q8. TFM-POINT native tốt đến đâu so với E0, B0 và best tree?

Q9. Pre-registered TimesFM quantile/path point extraction có cải thiện native point không?

Q10. TimesFM recenter/BTC point calibration có cải thiện point skill đủ ổn định không?

Q11. TimesFM LoRA có cải thiện point forecasting đủ để biện minh complexity không?

Q12. LSTM-DMH trên fair/base information set có thêm point skill so với B0 không?

Q13. LSTM-DMH + frozen B* có cải thiện LSTM base và best tree trong comparison công bằng không?

Q14. Best tree, TFM-POINT/best custom TimesFM, TFM-LoRA và LSTM-DMH khác nhau thế nào trên VAL theo metric hiện có?

Q15. Sau khi mọi quyết định đã lock, generalization của các survivor khác nhau thế nào giữa HOLDOUT-NEAR và HOLDOUT-FAR?

---

## 10. Experiment tracking [PLAN]

Mỗi run:

- exp id;
- timestamp;
- git commit/diff state;
- config hash;
- canonical data version/checksum;
- model;
- feature family/set;
- seed;
- fold/partition;
- per-horizon existing metrics;
- best iteration/epoch nếu applicable;
- GPU/environment;
- runtime;
- decision.

TimesFM thêm:

- package/checkpoint/revision;
- backend;
- context;
- native point API;
- quantile/path config nếu dùng nội bộ;
- point extraction method;
- LoRA config;
- cumulative reconstruction rule.

LSTM thêm:

- architecture config;
- context length;
- hidden size;
- number of layers;
- target transform;
- optimizer/training config;
- seed;
- best epoch;
- feature information set.

Không có CRPS-specific tracking field. Re-run phải reproducible từ frozen config. Chưa có experiment finding cho tới khi user unlock và run thật hoàn tất.

---

## 11. Leakage/static test suite [PLAN]

Bắt buộc trước real run:

1. target không cross partition boundary;
2. explicit \(t+h_{max}\) assertion;
3. FIT → ES target separation;
4. ES → purge 60 phút → VAL correctness;
5. cross-asset causality/staleness;
6. 5-minute bar phải đã đóng tại origin;
7. all feature-discovery implementations causal;
8. TimesFM context cutoff đúng tại \(t\);
9. TimesFM one-step → cumulative reconstruction;
10. path reconstruction trước point extraction;
11. cấm sum-of-marginal-quantiles shortcut;
12. LSTM sequence kết thúc đúng tại \(t\);
13. normalization/TargetTransform train-only;
14. point calibration train-only;
15. schema adapter deterministic và không sửa B0;
16. Near isolation;
17. Far isolation;
18. holdout inaccessible với research runner.

Quy tắc chung: tại origin \(t\), feature/context chỉ dùng \(\tau\le t\), và target chỉ dùng future timestamps nằm trong chính partition.

---

## 12. Execution order

### 12.1 Current TRAINING_LOCKED stage

- P0 restoration plan và data-integrity/schema test design;
- feature/indicator literature research;
- family dossiers và causality/redundancy review;
- freeze Wave-1 ladder;
- TimesFM version/API research không load checkpoint nặng;
- LSTM architecture review;
- config schema, leakage/unit/static tests;
- Near/Far timestamp proposal chỉ sau khi full canonical range đã được xác minh.

### 12.2 Sau explicit training unlock [FUTURE]

~~~text
P0 Data integrity PASS
↓
E0 / frozen B0
↓
execute pre-frozen Feature Discovery Wave-1
↓
X1...X5 + D1...Dk ablation
↓
confirmation
↓
data scaling
↓
tree family comparison
↓
TFM native point
↓
TimesFM point-oriented custom ladder
↓
LSTM research track
↓
allowed point ensembles
↓
lock everything
↓
Near + Far final evaluation
~~~

Exact ordering giữa TimesFM và LSTM còn **[PLAN]**, nhưng feature attribution phải hoàn thành trước model-family comparison.

---

## 13. Final evaluation [FUTURE]

Nếu survive research, final locked comparison gồm:

- E0;
- frozen B0;
- B*;
- best tree;
- eligible tree ensemble;
- TimesFM native point;
- best custom TimesFM point;
- TimesFM LoRA winner;
- LSTM-DMH winner;
- explicitly approved point ensemble nếu có.

Tất cả chạy trong cùng final stage trên:

1. HOLDOUT-NEAR;
2. HOLDOUT-FAR.

Dùng cùng metric §4.2, report riêng từng holdout, không composite Near/Far metric và không sửa bất kỳ quyết định nào sau khi kết quả được reveal.

---

## 14. Remaining [PLAN] và obsolete decisions

Chưa freeze:

- exact Near/Far timestamps;
- Far temporal gap;
- exact restored data snapshot/version;
- exact D1–Dk families và Wave-1 ordering;
- second-wave discovery;
- final LSTM architecture;
- exact TimesFM package/checkpoint/backend/API behavior;
- custom TimesFM candidates sống sót sau point-relevance audit;
- optional AutoTS/yfinance use;
- exact final refit/inference schedule, miễn không update từ holdout labels.

Đã obsolete:

- ML tree-only / DL TimesFM-only tuyệt đối;
- LSTM bị forbidden tuyệt đối;
- Track B distribution objective;
- CRPS/pinball/coverage/sharpness/tail metric contract;
- one-shot 30-day holdout liền ngay sau Dev;
- giữ R0→R6 chỉ vì distribution attribution;
- model zoo expansion từ AutoTS hoặc paper.
