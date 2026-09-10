# Report 2 — ML + DL hiện tại với order book

> Đã được thay thế bởi yêu cầu triển khai ngày 2026-09-10 và [src_OB/README.md](../src_OB/README.md).
> Thiết kế mới dùng Direct cho cả ML và DL (một model/horizon), không MIMO; metric chính RMSE/MAE/R² trên raw mid-price.
> AutoTS native và TimesFM zero-shot/LoRA nằm trong pipeline mới. Không áp dụng feature search/ablation và trading-metric plan dưới đây.
> Cập nhật cuối: baseline OF/OFI 10 level, cộng flow trước khi drop cùng mid, distance tùy chọn; historical freeze [2024-09-10, 2026-09-10) UTC. Không có DeepLOB model. AutoTS search model/tham số trong GPU allowlist, không fix sẵn LightGBM; xem README cho cấu hình đang triển khai.

Ngày: 2026-09-10  
Trạng thái: kiểm tra và đề xuất, chưa code  
Nguồn ý tưởng: [`docs/IDEA.md`](IDEA.md)  
Model trong phạm vi: LightGBM, XGBoost, CatBoost, XGB-RF và LSTM  
Ngoài phạm vi: AutoTS và TimesFM cũ, đã mô tả trong [`REPORT_1_AUTOTS_TFM_CURRENT.md`](REPORT_1_AUTOTS_TFM_CURRENT.md)  
Kiểm thử trong phiên: không chạy smoke test, canary, training, inference hoặc test suite

## 1. Kết luận đề xuất

Thiết kế order-book mới nên ưu tiên dữ liệu và representation trước architecture:

1. Replay order book đúng sequence và khóa causal bằng `available_ts`.
2. Chuyển raw LOB thành order flow/order-flow imbalance, queue imbalance, spread, microprice và elapsed-time feature.
3. Dùng cùng representation cho bốn tree model hiện có và plain LSTM.
4. Tree model giữ Direct strategy: một model riêng cho mỗi horizon 1/2/3 phút.
5. LSTM giữ MIMO: một head output đồng thời vector ba horizon.
6. Dùng context order book ngắn; không mặc định mang context OHLCV 512 phút sang microstructure.
7. Ablation theo nhóm representation, không search từng level/window thành hàng trăm candidate.
8. Ngoài RMSE/MAE, phải đánh giá signal sau spread, fee và latency.

Ưu tiên chạy model nên là LightGBM → XGBoost → LSTM → CatBoost → XGB-RF. Hai tree model đầu kiểm tra signal nhanh; LSTM là DL chính vì IDEA cho thấy plain LSTM + order flow thường đủ mạnh. CatBoost và XGB-RF đóng vai trò kiểm tra độ bền, không cần sweep architecture.

## 2. Các ý từ IDEA.md được áp dụng

| Ý trong IDEA | Cách áp dụng vào project |
|---|---|
| Order flow tốt hơn raw LOB; representation quan trọng hơn architecture | OF/OFI là input chính. Raw book chỉ là nguồn để dựng representation và một ablation tham chiếu nhỏ. |
| Plain LSTM + OF mạnh; CNN/deep LSTM không đáng thêm mặc định | Giữ LSTM một lớp hiện tại, thay input và context trước khi thay architecture. |
| Window dài không mặc định tốt; signal LOB decay nhanh | Dùng event/time window ngắn và đo riêng h=1; không dùng context 512 phút cho LOB chỉ vì code cũ dùng 512. |
| Event index thiếu thông tin thời gian; elapsed time có ích | Mỗi event/bin có `delta_t`, age, update intensity và time-since-price-change. |
| Transformer cần nhiều data; linear decoder tạo phần lớn cải thiện | Chưa thêm HFformer/Transformer mới. Chỉ xem xét khi dữ liệu book đủ lớn và baseline OF + LSTM đã được khóa. |
| MIMO giữ dependency giữa các horizon | LSTM output `[y_1,y_2,y_3]` cùng lúc như code hiện tại. |
| Direct horizon có thể bền hơn khi misspecification/non-stationarity | LightGBM/XGBoost/CatBoost/XGB-RF tiếp tục train ba model riêng theo horizon. |
| Forecast tốt chưa chắc trade được | Thêm metric net-PnL, coverage, adverse selection và transaction-level correctness. |

## 3. Data contract order book

Repo hiện chưa có order-book file, loader hoặc schema. Không nên code model trước khi chốt contract này.

### 3.1 Raw schema tối thiểu

- `symbol`, `venue`.
- `event_ts`: timestamp do sàn cung cấp.
- `available_ts`: timestamp record có thể thực sự được dùng bởi hệ thống.
- `sequence_id` hoặc `update_id`.
- `event_type`: snapshot/delta/trade nếu nguồn có.
- `bid_price_l`, `bid_size_l`, `ask_price_l`, `ask_size_l` cho `l=1..L`.
- Tick size, lot size và metadata thay đổi symbol.

Nếu dữ liệu là snapshot định kỳ, feature add/cancel/trade phải bị loại. Không được suy diễn event type từ snapshot thưa rồi coi như full order flow.

### 3.2 Replay và chất lượng book

Mỗi partition phải được replay theo sequence độc lập:

1. Bắt đầu từ snapshot hợp lệ.
2. Áp delta theo đúng sequence.
3. Phát hiện duplicate, out-of-order và sequence gap.
4. Kiểm tra bid/ask tăng giảm đúng theo level, size không âm và `best_bid < best_ask`.
5. Khi có gap, đánh dấu đoạn book không đáng tin cho tới snapshot phục hồi.

Không forward-fill im lặng qua gap. Mỗi sample phải mang `book_age_ms`, `gap_flag` và số update hợp lệ.

### 3.3 Causal time

Tại decision cutoff `t`, input chỉ chứa record có:

```text
available_ts <= t
```

Không chỉ dựa vào `event_ts`, vì record có thể đến muộn. Timestamp của OHLCV hiện phải được xác nhận là nhãn đầu hay cuối phút trước khi join.

Target chính vẫn giữ để so với pipeline cũ:

```text
y_h(t) = log(C[t+h] / C[t]), h = 1,2,3 phút
```

Feature book kết thúc tại cutoff `t`; target bắt đầu sau cutoff. Nếu order-book coverage ngắn hơn OHLCV hai năm, split phải dựng lại trên phần giao dữ liệu. Không được giữ nguyên fold hai năm và điền book bị thiếu.

## 4. Representation ưu tiên

### 4.1 Order flow theo level

Với mỗi lần book thay đổi và mỗi level `l`, dựng signed flow từ biến động price/size:

- Bid price tốt lên hoặc size tăng tại cùng price: flow bid dương.
- Bid price xấu đi hoặc size giảm: flow bid âm.
- Ask price tốt xuống hoặc ask liquidity giảm theo hướng mua chủ động: quy đổi dấu nhất quán để positive flow biểu diễn áp lực tăng giá.
- Chuẩn hóa theo depth hoặc rolling scale chỉ dùng quá khứ.

Giữ riêng `OF_bid_l`, `OF_ask_l` nếu cần, đồng thời tạo `OFI_l` và cumulative `OFI_1:L`. Công thức dấu phải được khóa trong data contract và kiểm tra bằng các event ví dụ viết tay trước khi training.

### 4.2 State feature

- `spread_bps`.
- `microprice_gap_bps = 1e4 × (microprice-mid)/mid`.
- Queue imbalance L1, cumulative L5 và L10.
- Log bid/ask depth ratio L5 và L10.
- Depth slope/convexity và chênh lệch hai phía.
- Top-level concentration.

### 4.3 Flow và event-time feature

- OFI top-1/top-5/top-10 trong các window ngắn.
- Add/cancel imbalance và signed trade flow nếu event type đáng tin.
- Delta imbalance, replenishment và depletion.
- `delta_t` từ event trước.
- Time since last mid-price change.
- Update count/rate, mean/max inter-arrival time.
- Book age, gap/stale flags.

### 4.4 Hai dạng input dùng chung một nguồn

Tree model nhận snapshot feature tại origin và các aggregation causal, ví dụ 1s/5s/15s/60s. LSTM nhận sequence ngắn của event hoặc time-bin gồm OF/OFI + state + elapsed time.

Không flatten raw `price × size × level × hàng trăm event` trực tiếp vào tree model. Không đưa absolute BTC price/size chưa chuẩn hóa vào sequence; ưu tiên distance-to-mid theo tick/bps và size theo tỷ lệ depth.

## 5. Thiết kế cho từng model hiện có

### 5.1 LightGBM

Code hiện tại train ba regressor độc lập, mỗi horizon một model, Huber objective và early stopping. Với order book:

- Giữ Direct `h=1,2,3`.
- Input là feature state + OF/OFI + elapsed-time aggregation tại origin.
- Bắt đầu từ order-book representation nhỏ, không ghép toàn bộ 72 B0 + ext cũ.
- Dùng LightGBM làm model phát hiện signal đầu tiên và kiểm feature importance theo nhóm.

LightGBM là ưu tiên số một vì fit nhanh, chịu missing flag tốt và thích hợp để biết representation có signal trước khi chạy DL.

### 5.2 XGBoost

Code hiện tại dùng `reg:pseudohubererror`, `hist`, GPU và ba model Direct. Giữ nguyên output contract, thay input bằng cùng representation của LightGBM.

XGBoost là phép kiểm tra signal thứ hai: nếu OF/OFI chỉ thắng ở LightGBM nhưng không tái hiện ở XGBoost hoặc LSTM, cần xem lại overfit theo tree split hoặc leakage.

### 5.3 CatBoost

CatBoost hiện train ba model Direct với Huber trên GPU. Dùng cùng bundle đã khóa sau hai model đầu. Không chạy lại feature search lớn.

Giá trị của CatBoost ở đây là kiểm độ bền với cách chia cây và regularization khác, không phải tạo thêm một pool feature riêng.

### 5.4 XGB-RF

XGB-RF hiện dùng 500 cây song song, squared error, một boosting round và ba horizon riêng. Nó phù hợp làm bagging-style robustness baseline trên order-book aggregation.

Chỉ chạy sau khi bundle OF đã khóa. Nếu XGB-RF không hơn E0 nhưng boosting model có gain, kết quả cho thấy signal cần interaction có cấu trúc hoặc boosting tuần tự; không cần tăng số candidate.

### 5.5 Plain LSTM MIMO

LSTM hiện tại đã gần với kết luận IDEA về architecture:

- Một lớp LSTM, hidden 64.
- Linear head output đồng thời ba horizon.
- MIMO loss trên vector `[y_1,y_2,y_3]`.

Phần cần đổi là input và time scale:

```text
old: 512 bước × feature OHLCV theo phút
new: sequence ngắn × [OF/OFI theo level, state, delta_t, age, quality flags]
```

Hai representation có thể chốt trước outer VAL:

- Event sequence: N update cuối trước cutoff, mỗi update có `delta_t`.
- Time-bin sequence: bin 100 ms hoặc 1 s trên một window ngắn nếu event rate quá lớn/không đều.

Không mặc định dùng stacked LSTM, CNN front-end hoặc attention. Đầu tiên giữ nguyên một lớp và linear decoder để đo giá trị của representation. Context được chọn từ một lưới nhỏ dựa trên thời gian thật, không chỉ số event, vì elapsed time là biến quan trọng.

LSTM tiếp tục output direct cumulative target `[y_1,y_2,y_3]`, không recursively gọi lại prediction h=1 để tạo h=2/h=3.

## 6. Ma trận ablation ưu tiên representation

Không chạy 163 add-one candidate. Dùng các bundle cố định:

| Bundle | Nội dung | Câu hỏi |
|---|---|---|
| B0 | OHLCV tối thiểu hoặc E0/model baseline hiện tại | Mốc so sánh |
| B1 | Raw normalized LOB state nhỏ | Raw book tự nó có signal không? |
| B2 | OF/OFI + elapsed time | Representation theo IDEA có hơn raw LOB không? |
| B3 | B2 + spread/microprice/queue imbalance | State bổ sung có giúp flow không? |
| B4 | B3 + quality/activity flags | Gain có phải do stale/gap/activity regime không? |

Thứ tự thực hiện:

1. LightGBM chạy B0–B4.
2. Freeze bundle tốt nhất theo outer validation.
3. XGBoost và LSTM chạy B0, B2 và bundle đã freeze.
4. CatBoost/XGB-RF chỉ chạy B0 và bundle đã freeze.

Cách này đo trực tiếp claim “representation quan trọng hơn architecture” và giữ tổng số cấu hình nhỏ. Feature importance chỉ dùng để giải thích hoặc tạo một vòng prune theo nhóm sau khi bundle đã thắng; không quay lại add-one theo từng level/window.

## 7. Target và multi-horizon

### 7.1 Contract chính

Giữ target close-return 1/2/3 phút để kết quả nối với repo hiện tại. Tree model dùng Direct:

```text
f_1(X_t) → y_1
f_2(X_t) → y_2
f_3(X_t) → y_3
```

LSTM dùng MIMO:

```text
F(sequence_t) → [y_1, y_2, y_3]
```

Hai cách đều tránh recursive error accumulation. LSTM MIMO còn giữ dependency giữa horizon trong shared representation.

### 7.2 Diagnostic theo microstructure

Vì IDEA cho thấy horizon hữu dụng có thể gần vài lần mid-price đổi, báo thêm diagnostic không thay objective chính:

- Kết quả sau 1 và 2 mid-price changes.
- Sign accuracy của future mid-price move.
- Gain theo event activity và spread regime.

Diagnostic này giúp phân biệt “book có signal nhưng target phút làm loãng” với “book representation không có signal”. Nó không được trộn vào rule chọn model chính nếu chưa sửa methodology.

## 8. Validation và chống leakage

- Split walk-forward theo thời gian, không random KFold.
- Dùng `available_ts` để cắt input.
- Replay book riêng từng partition hoặc có checkpoint state hợp lệ tại biên; không để state được xây từ future partition.
- Purge ít nhất đủ target horizon và mọi overlap do window label.
- Mọi model/bundle so trên cùng origin mask.
- Chuẩn hóa, depth scale và clipping chỉ fit trên FIT.
- Origin có sequence gap/stale vượt ngưỡng bị loại hoặc gắn cờ theo một policy khóa trước outer VAL.
- Báo coverage và số sample bị loại theo fold.

Không dùng smoke test trong quá trình này. Ở giai đoạn code sau, validation phải dựa trên data-contract checks tất định, unit checks cho replay/alignment và full configured evaluation khi dữ liệu đã khóa.

## 9. Đánh giá khả năng giao dịch

RMSE/MAE vẫn là metric model chính của repo, nhưng order-book signal chỉ có ý nghĩa thực tế nếu vượt chi phí giao dịch. Report cuối phải có thêm:

- Direction accuracy có điều kiện trên `|prediction|`.
- Coverage/trade count theo threshold.
- Gross PnL và net PnL sau spread, maker/taker fee và slippage.
- Latency từ cutoff dữ liệu đến prediction/action.
- Adverse selection sau entry.
- Xác suất prediction đúng trong toàn bộ một transaction/holding path, không chỉ một điểm label.
- Kết quả theo spread, volatility, depth và activity regime.

Không dùng backtest hai ngày hoặc giả định fill lý tưởng để kết luận model có lợi nhuận. Threshold phải được chọn trên training-side/ES và outer VAL chỉ dùng đánh giá.

## 10. Phạm vi triển khai về sau

Khi được phép code, phần việc nên tách như sau:

1. `data_orderbook`: schema, replay, sequence-gap và checksum.
2. `features_orderbook`: OF/OFI, state, elapsed-time và aggregation.
3. Mở rộng `Store` bằng order-book matrix cho tree model và event/bin sequence cho LSTM.
4. Config bundle B0–B4 và origin-quality policy.
5. Adapter cho bốn tree model vẫn dùng Direct 3 horizon.
6. Adapter LSTM MIMO với context order-book ngắn.
7. Artifact lưu data/schema hash, level depth, event/bin rule, feature bundle, coverage, gap/stale rate và trading assumptions.

Chưa thực hiện bất kỳ thay đổi code nào trong giai đoạn báo cáo này.
