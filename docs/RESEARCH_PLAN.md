# RESEARCH PLAN — BTC 1-phút point forecasting (bản đơn giản hóa)

Cập nhật: **2026-09-04d (rev 10.4 — pass VẬN HÀNH: hai CSV canonical 2 năm vào REPO qua Git LFS (`git clone` + `git lfs pull` là đủ, không scp/derive-lf); audit `git check-ignore` cho artifact experiments/**; TimesFM khoá cách gọi (mọi cấu hình feature được chấm dưới dạng HỆ THỐNG HOÀN CHỈNH `TimesFM-LoRA + XReg(F)`, confirmation là F_raw-system vs F_pruned-system, TFM-final LƯU rồi CHỜ champion replay); `max_branches` 4 (GPU đồng thời vẫn 2); **sự cố TÀI NGUYÊN GPU = ngoại lệ DUY NHẤT được dừng và HỎI USER** (`gpu_stop`, exit 3); `gpu-probe` kiểm UUID phân biệt + probe backend THẬT trong từng worker; agent chuyển pha vận hành (checker · run-monitor · infra · analyst hậu-run · researcher dormant) — KHÔNG đổi khoa học. Xem §0b.14–16, §6.1, §8)** · **2026-09-04c (rev 10.3 — pass THỰC THI: scheduler 2 GPU ĐỐI XỨNG (`gpu_devices`, `gpu.py`/`scheduler.py`, worker khoá vào GPU vật lý bằng `CUDA_VISIBLE_DEVICES`, task sẵn sàng → GPU rảnh, KHÔNG affinity ML/DL); orchestrate DAG nhánh model độc lập; champion HOÃN → `champion-replay` theo thứ tự cố định; TimesFM đặt tên hai HỆ THỐNG HOÀN CHỈNH A = `tfm_lora_baseline` (feature-free) vs B = `tfm_lora_xreg` (LoRA + XReg(F_win)); `scheduler_log.jsonl`. KHÔNG đổi bất kỳ luật khoa học nào — xem §0b.6, §0b.12, §2.2 #4, §3, §7.6, §8)** · **2026-09-04b (rev 10.2 — data 2 NĂM thật `data/BTC_1m_2y.csv` + LF 5' dẫn xuất tất định `derive-lf`; split `rolling_spread`: 5 VAL 3 ngày rải đều trên 2 năm, FIT 120 ngày rolling + ES 5, TEST 30 ngày cuối; anchor `data/data_checksums_2y.json`; S0/Candidate_m regenerate trên data thật — §1.5, §5, §8)** · **2026-09-04 (rev 10.1 — hiệu chỉnh: toàn bộ S0_m khoá tường minh (locked_b0/locked_ext); Candidate_m = C_short \\ overlap(S0_m) tính riêng từng model, KHÔNG lọc toàn cục, tương quan cao chỉ báo cáo; C_short dày 163 cột kể cả Keltner/PSAR cửa sổ reset/rv_med2d/r5_2-3/c5 ema 2-3, dow ngoại lệ; TimesFM-LoRA: calibrate = LoRA FIT + ES, artifact `tfm_lora_native`/`tfm_lora_xreg`; GPU-only và TEST-một-lần ép trong code không hỏi; checker không tương tác → `checker_log.jsonl`)** · 2026-09-03 (rev 10 — vòng EXPANDED-DATA: S0_m khoá từ artifact 15 ngày, candidate = C_short ≤ 15', TimesFM-LoRA → freeze → XReg search, fold-parallel tổng quát, không vẽ trong training, experiments/** không ignore — xem §0b, §1.5, §2.1, §2.2 #4, §2.3b, §7.5, §8)** · 2026-08-31 (seed/ε: `calib_seed` chỉ cho ES lấy số vòng, 3 `eval_seeds` đo ε theo `noise_cell = 100·std/mean` từng ô → ε = RMS 15 ô (không seed nào làm mốc), một `selection_seed` cho mọi bước selection) · 2026-08-29 (code §8 + checker review: TEST = 2.728 origin, prune PI theo cờ PI+ ≥ 2/3 horizon, latency §7.4 thread mặc định thư viện, checksum §6.1 bắt buộc trong CLI) · 2026-08-28 (rev 9b: gộp 3 seed bằng mean RMSE từng ô; cửa sổ visualize ở 3 ngày VAL vol thấp/trung bình/cao; rev 9: bỏ safety-net, chỉ prune PI; confirmation 3 seed → win_m; figure win vs champion + Final mọi model; rev 8: B0* là điểm xuất phát chung của mọi model; mỗi model calibrate 15fixed_m riêng trên B0* rồi tự feature search → F*_m; cờ + = > 0 ở ≥ 2/3 horizon; rev 7: training chỉ GPU, ExtraTrees → XGB-RF, lọc B0 theo R1–R4 không tier, ensemble theo skill vs E0, latency p95/p99/max, màu cố định; rev 6: latency §7.4; rev 5: lọc B0). Thay thế hoàn toàn roadmap 2026-08-24 (lưu tại `docs/archive/RESEARCH_PLAN_2026-08-24_detailed.md`, không còn hiệu lực).

Luồng nghiên cứu (vòng EXPANDED-DATA, 2026-09-03):

```
Data 2 năm (BTC_1m_2y.csv + LF 5' dẫn xuất) → split rolling_spread: 5 VAL 3 ngày RẢI ĐỀU trên 2 năm, FIT 120 + ES 5 rolling, TEST 30 ngày cuối  (§1.5)
→ S0_m = B0* ∪ F_old_m KHOÁ từ artifact vòng 15 ngày; C_short (≤ 15') → Candidate_m = C_short \ overlap(S0_m)  (§0b, §2.3b, lock-s0)
→ Feature search: mỗi model (nhanh → chậm) từ S0_m của CHÍNH nó: calibrate trên data mới → fixed rounds/epoch + ε_m mới
     → add-one Candidate_m → F*_raw → prune PI CHỈ cột mới → confirmation 3 seed raw vs pruned → win_m
     LightGBM → XGBoost → CatBoost → TimesFM-LoRA (LoRA per fold → freeze → XReg search → tfm-final) → XGB-RF
     → AutoTS (WR/MR probe từ S0 nhánh → autots-search → AutoTS-final) → LSTM
→ Sau mỗi model: so với champion, log đổi/giữ (KHÔNG vẽ)  → Ensemble → Final (TEST một lần) → `visualize` hậu kỳ
→ (để sau) scale data §5.3
```

Trạng thái: **vòng 15 ngày đã XONG 2026-09-01 (kết quả: MEMORY "Experiment Findings", artifact `experiments/15d/`); code migration vòng expanded-data xong 2026-09-03 + pass hiệu chỉnh 2026-09-04 (139 unit test PASS, smoke synthetic PASS); TRAINING: LOCKED — chưa chạy training vòng mới, chưa chạm TEST.** Training chỉ chạy trên Vast khi user nói unlock (§8).

---

## 0. Giữ nguyên từ thiết kế đã có (và các điểm sửa theo user)

- **Target**: `y_h(t) = log(C[t+h] / C[t])`, h = 1, 2, 3 phút. Model dự báo one-step return `r` phải cộng dồn `ŷ_h = Σ_{i≤h} r̂_{t+i}` trước khi chấm.
- **Data**: BTC 1-phút OHLCV + amount (Binance). Vòng expanded-data (2 năm): `data/BTC_1m_2y.csv` (cột `ts`, alias trong bộ nhớ) + LF 5' dẫn xuất `data/BTC_5m_2y.csv` (anchor `data/data_checksums_2y.json`, §1.5); vòng 15 ngày: `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv`. File 5 phút chỉ dùng cho feature ở resolution 5 phút. Không dùng `*_close.csv`, không cross-asset (thêm sau nếu user muốn).
- **Baseline** (các nghĩa, dùng xuyên suốt):
  - **B0-306** = 306 feature của `Baseline_LGBM.py` (file không sửa) chạy bằng LightGBM code gốc. Luôn log làm reference. Lưu ý: `TargetTransform` trong file B0 có bug nhân in-place (`(n,1) *= (1,3)` → ValueError) — harness dùng `src/p0/transform.py` tái hiện đúng công thức; mọi thứ khác (feature 306, `_make_model`, `LGBMConfig`, `build_lgbm_matrix`) import thẳng từ B0.
  - **Feature baseline = B0\*** = bộ tốt nhất chọn từ B0-306 và R1–R4 (§1.4, chọn cột trong harness). Vòng 15 ngày: mọi model bắt đầu vòng lặp feature từ cùng B0\*; **vòng expanded-data: từ S0_m = B0\* ∪ F_old_m khoá của chính model đó (§0b)**; không model nào kế thừa feature set của model khác (§2.1).
  - **Model baseline / champion ban đầu** = LightGBM đúng code gốc (`fit_lgbm_baseline`, `LGBMConfig` không đổi: Huber alpha 0.9, TargetTransform `y / (rv60·√h)` fit train-only; seed theo vai trò §1.3) trên F\*_LGBM. Gain của các model khác đo so với champion hiện tại (§3).
  - **E0** (ŷ = 0 ⇔ P̂ = C_t) luôn log.
- **Prediction và metric**: model dự báo **log return** `ŷ_h`; metric tính trên **giá**: `P̂_{t+h} = C_t · exp(ŷ_h)`, lỗi `e_h = P̂_{t+h} − C_{t+h}` (USD).
  - `RMSE_h`, `MAE_h` trên `e_h`; `Gain = 1 − RMSE_cand / RMSE_base` per horizon × fold; tóm tắt **MedianGain, WinRate, P10Gain, WorstGain** trên 15 ô (5 fold × 3 h). WinRate = tỷ lệ ô có Gain > 0; P10Gain = phân vị 10% của 15 ô (đuôi xấu, ~ô tệ thứ 2); WorstGain = ô tệ nhất.
  - **Chỉ MedianGain (so với ε) là tiêu chí quyết định** ở mọi chỗ: KEEP/DROP §2.1, chọn B0\* §1.4, đổi champion §3, thành viên ensemble §3. WinRate/P10Gain/WorstGain chỉ báo cáo để nhìn ổn định. PI/MI/standalone chỉ dùng để lập các bộ R1–R4 ở §1.4 (và permutation importance ở bước prune tùy chọn cuối vòng lặp), không tham gia KEEP/DROP của candidate.
  - Pearson r và directional accuracy tính trên **thay đổi giá** `P̂_{t+h} − C_t` so với `C_{t+h} − C_t` (r trên giá tuyệt đối vô nghĩa vì bị mức giá chi phối); directional accuracy bỏ bar có `C_{t+h} = C_t`.
  - Importance/permutation importance/MI chỉ là diagnostic hoặc bộ lọc §1.4, không phải metric quyết định. Loss/transform bên trong model không đổi (z-space Huber); chỉ evaluation trên giá.
- **Inference latency** (§7.4): thời gian predict **một origin** (batch 1) per model × horizon, tóm tắt p95/p99/max. **Chỉ để theo dõi** — không ảnh hưởng training, loss, KEEP/DROP, champion hay bất kỳ quyết định nào.
- **Split**: walk-forward FIT → ES → purge 60' → VAL; TEST cuối không chạm cho tới Final. 15 ngày: §1.2; data đầy đủ (vòng hiện tại): §1.5 / §5.
- **Quy tắc biên**: origin t thuộc `[T_start, T_end)` chỉ khi `t ≥ T_start` và `t + 3' < T_end`. Feature chỉ dùng dữ liệu τ ≤ t. TargetTransform/scaler fit train-only mỗi fold.
- **Runtime**: giai đoạn 15 ngày chạy trên **Vast** (GPU detect tại runtime — có thể là RTX 3090, không hard-code; `LGBMConfig(require_p100=False)`). **Training chỉ trên GPU — cấm training bằng CPU, không CPU fallback**: LightGBM (build GPU), XGBoost (`device=cuda`), CatBoost (`task_type=GPU`), XGB-RF (XGBoost GPU), TimesFM/LSTM (torch GPU), model 1-feature standalone (LightGBM GPU), AutoTS với `regression_model` LightGBM/XGBoost cấu hình GPU (chốt khi audit). ExtraTrees sklearn không có GPU → thay bằng XGB-RF. CPU chỉ cho việc không phải training: tính feature, metric, MI/PI, unit/smoke test local, và predict của thư viện mặc định chạy CPU (LightGBM/CatBoost predict).

---

## 0b. Vòng EXPANDED-DATA (quyết định user 2026-09-03) — thay đổi có hiệu lực

Mọi luật target/metric/leakage/seed/ε/confirmation/champion/ensemble/TEST-một-lần giữ nguyên. Thay đổi:

1. **S0_m khoá** (`src/p0/s0.py`, `python run.py lock-s0`): với mỗi model m đã có winner ở vòng 15 ngày, `S0_m = B0* ∪ F_old_m` dựng từ `experiments/15d/wins/<m>.json` + `b0_star.json` (không gõ tay). `locked = F_old_m`: không phải candidate, không bị prune PI, không thể bỏ. AutoTS: `autots_wr`/`autots_mr` kế thừa đúng bộ thắng của nhánh đó (không dùng AutoTS-final). TimesFM: `S0 = ∅` (TimesFM-final cũ = native, không bịa covariate kế thừa). Kích thước thật: B0* 72 cột chung; ext khoá lgbm 14 · xgb 11 · cat 5 · xgbrf 12 · lstm 23 · autots_wr 21 · autots_mr 8.
2. **Toàn bộ S0_m là khoá (rev 10.1)**: artifact `s0/<m>.json` ghi tường minh `locked_b0 == b0` (mọi cột B0\*) và `locked_ext == ext` (mọi cột ext thắng cũ); `ColSet` không cho bỏ cột B0 (`without_b0` luôn từ chối) lẫn ext khoá (`without_ext`); prune PI chỉ xét cột ext mới; candidate không bao giờ chứa cột đã có trong S0_m; đọc được artifact cũ ({"b0","ext"} / {"locked"}). Tên gọi: **S0_m** = tập xuất phát khoá; **F_raw_m / F_pruned_m / F_best_m** = tập do vòng tìm kiếm mới sinh ra.
3. **Candidate mới = CHỈ C_short** (§2.3b): định nghĩa ngắn hạn ≤ 15' sinh MỘT lần trong `features_short.py` (tiện thực thi — không phải stage nghiên cứu, không gọi "master feature pool"). **`Candidate_m = C_short \\ overlap(C_short, S0_m)` tính RIÊNG cho từng model** và lưu riêng `s0/candidates_<m>.json` (kích thước có thể khác nhau). Overlap = cột C_short đã có trong S0_m của CHÍNH model đó: (i) trùng tên/định nghĩa cùng timestamp, hoặc (ii) khác tên nhưng giá trị giống hệt cùng timestamp (kiểm bằng số, `max|a−b| ≤ 1e-4·std`). **KHÔNG** bỏ vì tương quan cao, xấp xỉ, cửa sổ gần nhau, cùng họ, có trong B0-306 nhưng không trong B0\* của model, hay từng là candidate cũ mà không trong S0. Cùng indicator khác lag (`fine:t-4m:rsi15_centered` vs `rsi15_centered` tại t) **không** phải trùng. Không có bước lọc toàn cục theo B0-306 / 39 candidate cũ / "vũ trụ feature". Tương quan cao (|ρ| ≥ 0.995) chỉ **báo cáo** (`near_vs_s0`, `intra_short_near`) — không bao giờ tự xoá. Candidate cũ §2.3 (KEEP lẫn DROP) không nằm trong C_short theo cách xây lưới (§6).
4. **Pipeline mỗi model giữ nguyên §2.1** với `S := S0_m`, prune PI chỉ xét cột mới. **Không kế thừa** số vòng/epoch, ε, RMSE VAL, điểm champion cũ — tất cả đo lại trên split mới.
5. **TimesFM** (§2.2 #4, đặt tên rõ ở rev 10.3): pretrained → LoRA per fold (FIT học, ES chọn epoch, VAL không thấy) → freeze → CÙNG adapter cho toàn bộ XReg add-one → F_raw → prune PI → F_pruned → **confirmation F_raw vs F_pruned → F_win** → rồi mới so **HAI HỆ THỐNG HOÀN CHỈNH**: A = TimesFM-LoRA baseline (0 feature, 0 B0\*, 0 covariate) vs B = CÙNG adapter + XReg(F_win) → TimesFM-final. Không gọi là "XReg vs LoRA" (XReg không phải model độc lập). Chi phí LoRA ở mức fold × seed, không fold × candidate.
6. **Scheduler GPU đối xứng** (rev 10.3; `gpu.py` + `scheduler.py`, `fold_parallel.py` là adapter): số worker = `len(gpu_devices) × gpu_slots_per_device` (máy thí nghiệm: **2 × RTX 5000 Ada 32 GB → 2 worker, 1 task nặng/GPU**, KHÔNG gộp 64 GB, KHÔNG oversubscribe). Mỗi worker là một process khoá vào ĐÚNG một GPU vật lý bằng `CUDA_VISIBLE_DEVICES` đặt trước mọi import CUDA → LightGBM/XGBoost/CatBoost/PyTorch/TimesFM-jax/AutoTS đều chạy đúng GPU được giao. **Không GPU nào mang vai trò ML hay DL; không pin model family vào GPU**: task sẵn sàng → GPU nào rảnh nhận (round-robin giữa các nhánh đang hoạt động, FIFO trong nhánh). 5 fold của một cấu hình rải động lên hai GPU; **candidate vẫn TUẦN TỰ** (S đổi sau KEEP); prune PI chạy trọn trong MỘT worker (giữ nguyên một dòng RNG như bản tuần tự). Kết quả ghép theo thứ tự fold cố định → y hệt tuần tự. Không CPU fallback: OOM/thiếu GPU → task fail rõ ràng. Log mỗi task: `scheduler_log.jsonl` (§7.6).
7. **Không vẽ trong training**: mọi bước chỉ lưu artifact; `python run.py visualize` dựng lại figure §7.3 sau khi xong (§7.5).
8. **experiments/** không bao giờ bị gitignore**; nhị phân đi Git LFS; không commit checkpoint TimesFM gốc (chỉ id + revision).
9. **Data 2 năm + split `rolling_spread`** (2026-09-04b, §1.5): `data/BTC_1m_2y.csv` (alias `ts`) + LF 5' dẫn xuất tất định `data/BTC_5m_2y.csv` (`derive-lf`, sidecar sha nguồn); anchor `data/data_checksums_2y.json`; 5 VAL 3 ngày rải đều trên 2 năm, FIT 120 ngày rolling + ES 5, TEST 30 ngày cuối; LF phải phủ HF.
10. **Bất biến cứng ép trong code, không hỏi user (rev 10.1)**: GPU không có / backend không thực sự CUDA (preflight XGBoost kiểm `build_info USE_CUDA` + booster device) → dừng ngay; TEST đúng một lần → `final/TEST_SENTINEL.json` ghi trước khi chạm TEST, lần hai dừng ngay (chỉ `--force-test-rerun` recovery tường minh); checksum/LF/biên/artifact S0 malformed → dừng ngay. Mọi vi phạm ghi ERROR vào `checker_log.jsonl`.
14. **Data canonical NẰM TRONG REPO (rev 10.4, quyết định user 2026-09-04d)**: `data/BTC_1m_2y.csv` (sha `559ce040…f097`,
    1.051.201 dòng) và `data/BTC_5m_2y.csv` (sha `0e5fb9ad…f2fef`, 210.239 dòng) được TRACK qua **Git LFS** (chỉ đúng hai file này;
    `.gitignore` giữ `data/*.csv` + hai dòng `!` ngoại lệ). Quy trình chuẩn: `git clone` → `git lfs pull` → `check-data` — **không
    còn bước scp, không cần `derive-lf`** (lệnh này ở lại như công cụ tái lập/kiểm chứng: LF phải dẫn xuất được ĐÚNG sha từ HF).
    Anchor `data/data_checksums_2y.json` + sidecar `data/BTC_5m_2y.derivation.json` phải khớp hai file đã commit.
    **Không đường dẫn nào dưới `experiments/**` bị ignore** — kiểm bằng `git check-ignore`, có test tự động.
15. **Sự cố TÀI NGUYÊN GPU = ngoại lệ tương tác DUY NHẤT (rev 10.4, §10 yêu cầu user)**: không có GPU, GPU được giao biến mất,
    CUDA/backend không train được trên GPU, phát hiện CPU fallback, định tuyến GPU sai (UUID trùng), worker CUDA chết, OOM chặn
    đường GPU → `checker_log.gpu_stop`: dừng an toàn (giữ artifact đã xong), KHÔNG CPU fallback, KHÔNG tự đổi
    batch/hyperparameter/seed/methodology, ghi ERROR `ref=USER_DECISION_REQUIRED` rồi **HỎI USER** (exit 3). Mọi vi phạm bất biến
    KHOA HỌC khác vẫn dừng tự động, không hỏi, không có tuỳ chọn "chạy tiếp". `gpu-probe` bắt buộc: UUID của các worker phải
    PHÂN BIỆT, và mỗi backend (torch, XGBoost + kiểm booster device, LightGBM theo `device_type` đã resolve, CatBoost, jax,
    import timesfm) phải chạy được một phép tính GPU nhỏ THẬT **bên trong worker đã mask**; backend chưa cài = WARN môi trường.
16. **Agent theo pha VẬN HÀNH (rev 10.4)**: `checker` (đúng hai điểm: trước `orchestrate`, trước `final`) · **`run-monitor`**
    (agent chính trong lúc run: đọc `scheduler_log`/`orchestrate_log`/`checker_log` + `nvidia-smi`, báo branch/model/stage/candidate/
    fold/seed/GPU, util/VRAM, worker chết, đói việc, ETA từ task thật — CHỈ ĐỌC) · `infra` (GPU/env hỏng) · `analyst` (HẬU run) ·
    `researcher` (DORMANT — chỉ khi user yêu cầu đổi methodology). `max_branches: 4` (nhiều nhánh SẢN XUẤT task) trong khi
    `gpu_slots_per_device: 1` giữ **tối đa 2 task nặng** chạy cùng lúc.
12. **Champion HOÃN + replay theo thứ tự cố định (rev 10.3, §3)**: khi `defer_champion: true`, mỗi nhánh chỉ SINH artifact đại diện (`wins/<m>.json` + `champion_extra`); `python run.py champion-replay` đọc các artifact đã đóng băng và so champion theo THỨ TỰ METHODOLOGY cố định lgbm → xgb → cat → tfm(TFM-final) → xgbrf → autots(AutoTS-final) → lstm, dùng ĐÚNG luật `> +ε_champion`. Thứ tự nhánh chạy xong (do lịch GPU) KHÔNG bao giờ ảnh hưởng champion. Replay chỉ đọc file: không train, không inference, không cần data/GPU. `orchestrate` chạy DAG nhánh (loop độc lập ‖ ; tfm-final ← loop tfm; autots-search ← autots_wr + autots_mr) rồi replay + ensemble; **`final` (TEST) luôn là lệnh riêng, orchestrate không bao giờ chạm TEST**.
13. **Probe/cấu hình nội bộ không bao giờ đụng champion (rev 10.3)**: `champion_step` từ chối cứng `tfm_lora_baseline`, `tfm_lora_xreg`, `autots_wr`, `autots_mr` (CHAMPION_INELIGIBLE) — chỉ đại diện mới đủ tư cách champion/ensemble/Final.
11. **Checker không tương tác (rev 10.1)**: agent `checker` và code ghi finding vào `experiments/<run>/checker_log.jsonl` (timestamp, stage, model, severity PASS/INFO/WARN/ERROR, check_id, message, file, ref); ERROR = chặn run tới khi sửa + PASS cùng check_id; WARN/INFO (tương quan cao, nghi dư thừa, gain bất thường, runtime, ghi chú methodology) = ghi rồi tiếp tục; không bao giờ hỏi "tiếp hay dừng".

---

## 1. Bước 1 — Fix dataset và lọc B0 (§1.1–§1.4 = vòng 15 ngày, ĐÃ XONG; §1.5 = split vòng expanded-data)

### 1.1 Snapshot và kiểm tra (một lần)

`data/BTC_hf_1min.csv` hiện có 21.916 dòng, `2026-01-18 16:15 → 2026-02-02 21:30 UTC` (file bị cắt đúng 2 MiB, dòng cuối cụt → bỏ). Đã kiểm tra read-only: lưới 60 s, không duplicate, không gap, `H ≥ max(O,C)`, `L ≤ min(O,C)`, `amount/volume` nằm trong `[L, H]`. Trước khi chạy:

1. Adapter header lowercase → `Open/High/Low/Close/Volume` (không sửa B0); giữ `amount`.
2. Ghi checksum + số dòng + range vào `data/data_checksums.json`, nhãn `btc_1min_15d_2026-01-18_02-02`. Mọi kết quả giai đoạn này gắn nhãn dataset đó.
3. `BTC_lf_5min.csv` (đến 2026-03-26, đủ phủ 15 ngày): nhãn `T` = gộp 5 bar 1-phút `(T−4 … T]` → chỉ join bar có `T ≤ t`.
4. Lưu ý regime: 15 ngày này BTC 95.156 → 78.299 (−18%), vol cao; kết quả chọn feature/model là trên một regime, sẽ kiểm tra lại khi có data đầy đủ (§5).

B0-eligible origins: 21.258, `01-19 02:46 → 02-02 21:27` (warmup 631 bar).

### 1.2 Fold cho 15 ngày [đã chốt]

Expanding train, VAL = 1 ngày UTC, ES = ngày liền trước VAL (trừ 60' purge), TEST = 2 ngày cuối:

| Fold | FIT (expanding, từ 01-19 02:46) | ES | purge | VAL |
|---|---|---|---|---|
| 1 | → 01-25 23:56 (~9.9k origin) | 01-26 00:00 → 22:56 | 60' | 01-27 00:00 → 23:56 |
| 2 | → 01-26 23:56 (~11.3k) | 01-27 00:00 → 22:56 | 60' | 01-28 |
| 3 | → 01-27 23:56 (~12.8k) | 01-28 00:00 → 22:56 | 60' | 01-29 |
| 4 | → 01-28 23:56 (~14.2k) | 01-29 00:00 → 22:56 | 60' | 01-30 |
| 5 | → 01-29 23:56 (~15.7k) | 01-30 00:00 → 22:56 | 60' | 01-31 |
| TEST (refit) | → 01-30 23:56 (~17.1k) | 01-31 00:00 → 22:56 | 60' | 02-01 00:00 → 02-02 21:27 |

- Mỗi VAL = 1.437 origin; 5 fold = 7.185 origin, 15 ô. ES = 1.377 origin. TEST = 2.728 origin (02-01 Chủ nhật, 02-02 Thứ hai; origin cuối 02-02 21:27). FIT thực tế theo eligible của B0 (đã chạy `check-data` trên snapshot): 9.887 / 11.327 / 12.767 / 14.207 / 15.647 origin (fold 1–5), Final 17.087 — B0 loại 24 origin ngày 01-24/01-25 (3 bar bất thường lan theo lag), chỉ nằm trong FIT.
- Expanding thay vì rolling vì data ít; so sánh candidate vs baseline luôn trong cùng fold nên train size không làm lệch Gain.
- Mọi partition half-open, `t + 3' < T_end`.
- Cố định trong toàn bộ §1.4 và Bước 2, cho mọi model: cùng fold, cùng tập origin (= eligible của B0), **cùng `selection_seed`** (§1.3), cùng config từng model.
- Lookback candidate ≤ 1440 phút → ext có thể NaN ở ngày đầu FIT; tree nhận NaN native; model không nhận NaN (LSTM, AutoTS tùy API) điền 0 sau chuẩn hóa train-only.

### 1.3 Nhiễu seed và số vòng cố định [đã chốt] — calibrate riêng cho từng phase và từng model

Nguyên tắc: **"số vòng cố định" = chính `best_iteration` mà early stopping dừng ở một run calibrate** (`calib_seed`, ES trên ES set, per fold × horizon → 15 giá trị) — không phải ước lượng thống kê. ES trên 1.377 dòng nhiễu, nên ES chỉ chạy **một lần cho mỗi (phase, model), trên đúng feature set của phase đó**; mọi run còn lại của phase dùng đúng 15 số vòng ấy (`fixed_rounds`, B0 hỗ trợ sẵn) để chênh lệch Gain chỉ do feature. **Không dùng chéo**: `15fixed_306` chỉ cho lọc B0; số vòng của LightGBM không dùng cho XGBoost/CatBoost và ngược lại.

Lịch calibrate:

| Phase | Feature set calibrate | Model | Kết quả | Dùng cho |
|---|---|---|---|---|
| A. Lọc B0 (§1.4) | B0-306 | LightGBM | `15fixed_306` (ES, `calib_seed`) + ε_LGBM(B0-306) từ 3 evaluation seed + **run baseline `15fixed_306` tại `selection_seed`** (15 model dùng cho PI, và là mốc RMSE của R1–R4) | 4 run kiểm chứng R1–R4 → B0\* |
| B. Feature search (§2.1) | **S0_m** (vòng expanded-data, §0b; vòng 15 ngày: B0\* chung cho mọi model) | từng model có early stopping: LightGBM, XGBoost, CatBoost (số vòng), LSTM (số epoch) — mỗi model một run | `15fixed_LGBM`, `15fixed_XGB`, `15fixed_Cat`, `fixed_epoch_LSTM` + ε_m | toàn bộ Candidate_m (39 candidate ở vòng 15 ngày) và prune PI của chính model đó (tính một lần, dùng chung cả phase) |
| C. Prune PI + confirmation (§2.1) | F\*_m và F\*_m^prune của chính model | model m, ES bật, **3 evaluation seed** mỗi configuration | bảng `RMSE̅` 15 ô (mean 3 seed từng ô) mỗi configuration → Gain prune vs unprune từng ô → MedianGain → **win_m** (+ `best_iteration`/best epoch ghi lại cho Final refit) | so với champion (§3), figure §7.3 |

```
B0-306 + ES (LGBM) → 15fixed_306 → R1–R4 → B0*
   → LGBM(B0*) + ES → 15fixed_LGBM → LightGBM add-one 39 candidate → F*_LGBM
   → XGB(B0*)  + ES → 15fixed_XGB  → XGBoost  add-one 39 candidate → F*_XGB
   → Cat(B0*)  + ES → 15fixed_Cat  → CatBoost add-one 39 candidate → F*_Cat
   → LSTM(B0*) + ES theo epoch → fixed_epoch_LSTM → LSTM add-one 39 candidate → F*_LSTM
   (XGB-RF: 1 vòng boosting cố định; TimesFM-LoRA (rev 10): ES theo epoch → fixed_epoch_TFM như LSTM, ε trên native — vòng 15 ngày zero-shot; AutoTS: cơ chế riêng — cũng từ S0/B0*, chỉ đo ε_m, không ép fixed_rounds)
```

- **LSTM** có epoch nên cũng calibrate: một run ES theo epoch trên B0\* (patience 5, ≤ 50 epoch) → `fixed_epoch_LSTM` per fold (head 3 output nên một số epoch cho cả 3 h); mọi candidate của LSTM train đúng số epoch ấy; confirmation bật lại ES.
- **XGB-RF** (1 vòng boosting cố định, `num_parallel_tree` cố định) không có gì để calibrate; **TimesFM** (rev 10) là LoRA có epoch → calibrate như LSTM (`fixed_epoch_TFM`, ε đo trên native; vòng 15 ngày: zero-shot không train); **AutoTS** cố định số vòng của regression_model bên trong theo cơ chế của AutoTS trong config. Ba model này xử lý theo cơ chế riêng, không ép khái niệm fixed_rounds; chỉ đo ε_m.
**Ba vai trò seed, tách bạch hoàn toàn** (áp dụng cho CẢ phase A và phase B):

| seed | dùng ở đâu | không được dùng làm gì |
|---|---|---|
| `calib_seed` (seed0) = 8586 | **chỉ** run ES tìm số vòng/epoch cố định của phase đó | không tham gia đo ε, không dùng cho bất kỳ bước selection nào |
| `eval_seeds` (seed1/2/3) = 8587, 8588, 8589 | đo ε (số vòng cố định) và confirmation 3 seed §2.1b | không dùng làm mốc/mẫu số của nhau |
| `selection_seed` = 8587 (mặc định = `eval_seeds[0]`) | **một giá trị duy nhất** cho MỌI bước selection: PI/SA/MI và 4 run R1–R4 (phase A); baseline S0_m (vòng 15 ngày: B0\*) + toàn bộ candidate add-one + prune PI (phase B); refit Final | không được đổi seed giữa các Rk hoặc giữa các candidate |

Lý do dùng một `selection_seed` cố định: chênh lệch RMSE giữa hai feature set khi đó **chỉ đến từ feature set**, không lẫn nhiễu seed. Ba evaluation seed chỉ để **đo** mức nhiễu đó (ε) và đo độ ổn định, không tham gia chọn feature.

- **ε_m** đo ngay sau calibrate của phase đó: chạy `m` trên feature set của phase (B0-306 ở phase A, B0\* ở phase B) với **3 evaluation seed**, số vòng/epoch cố định vừa có (XGB-RF, AutoTS, TimesFM dùng config cố định của nó), 5 fold → 3 bảng RMSE 15 ô. Với **mỗi ô** `(f, h)` có `R_1, R_2, R_3`:

  `mu = mean(R_1, R_2, R_3)`, `sigma = std(R_1, R_2, R_3, ddof = 0)`, `noise_{f,h} = 100 · sigma / mu` (pp — cùng đơn vị với Gain)

  rồi gộp 15 ô: **`ε_m = max(0.005 pp, sqrt(mean(noise_{f,h}²)))`** (RMS). **Không seed nào được dùng làm mốc/mẫu số** — ε là độ phân tán của chính 3 giá trị trong từng ô, không phải Gain của seed này so với seed kia. LightGBM đo ε hai lần: phase A trên B0-306 (dùng để chọn B0\*) và phase B trên B0\*.

### 1.4 Lọc 306 feature của B0 → B0\* (một lần, trước Bước 2) [mới theo quyết định user]

Lý do: 306 cột của B0 (22 fine × 8 lag + 16 coarse × 8 lag + rv60/log_rv60) có thể chứa nhiều cột nhiễu; lọc một lần bằng LightGBM gốc trên 5 fold §1.2 rồi mới chạy Bước 2. `Baseline_LGBM.py` không đổi — lọc bằng chọn cột trong harness. B0-306 nguyên bản vẫn log làm reference ở mọi bảng.

Flow lọc:

```
B0-306 + ES (LightGBM, calib_seed) → 15fixed_306 + baseline tại selection_seed (15 model cho PI)
   ├─ (a) PI : xáo từng cột trong VAL × 3 lần → ΔRMSE giá per horizon (median 5 fold)
   ├─ (b) SA : LightGBM chỉ 1 cột, 5 fold × 3 h → Gain vs E0 và vs B0-306
   └─ (c) MI : mutual_info_regression(X_j, z_h) trên FIT − MI với target xáo trộn
→ cờ per cột: PI+ / SA+ / MI+  khi điểm số > 0 ở ≥ 2/3 horizon
→ R1 = PI+ ∨ SA+ ∨ MI+   R2 = PI+ ∨ (SA+ ∧ MI+)   R3 = PI+   R4 = SA+
→ 4 run kiểm chứng (LightGBM, 15fixed_306) so với B0-306 → MedianGain 15 ô
→ B0* = bộ không tệ hơn (≥ −ε_LGBM) có MedianGain cao nhất (hòa → nhỏ hơn); không bộ nào đạt → B0-306
→ experiments/b0_filter.csv: 306 dòng (điểm số, cờ, giữ/bỏ theo R1–R4) + 4 kết quả kiểm chứng
```

Ba điểm số cho từng cột `j` (per horizon; gộp 5 fold bằng median), tất cả trên giá:

**(a) Permutation importance (PI)** — dùng đúng 15 model B0-306 của run baseline §1.3 (`15fixed_306`, `selection_seed`) — chính là run làm mốc RMSE cho R1–R4, nên PI và kiểm chứng dùng cùng một baseline. Trên VAL mỗi fold: xáo trộn cột `j` giữa các origin VAL, predict lại, `PI_{j,h,f} = RMSE_perm − RMSE_gốc` (USD); lặp 3 lần xáo (seed khác nhau) lấy trung bình; median qua 5 fold. `PI ≤ 0` = xáo không làm model xấu đi → cột không được dùng hữu ích. Chi phí: chỉ predict, vài phút. Ghi thêm PI theo nhóm (xáo cùng lúc 8 lag của một base feature, 38 nhóm) để đọc, không dùng để quyết định.

**(b) Standalone 1-feature** — với từng cột `j`: LightGBM code gốc (cùng config, cùng TargetTransform, ES trên ES set) chỉ trên `[j]`, 5 fold × 3 h; Gain trên giá so với **E0** và so với **B0-306**. `Gain_E0 ≤ 0` = không có tín hiệu độc lập. Nếu có cột thắng B0-306 (`MedianGain_B0 > +ε_LGBM`) → ghi cờ đỏ (B0 bị nhiễu chi phối); không có luật riêng — R3/R4 sẽ tự thắng ở bước kiểm chứng. Model 1-feature dùng ES riêng từng fit (1 cột, rẻ). Chi phí: 306 × 15 fit tí hon trên GPU ≈ 2–4 h.

**(c) Mutual information (MI)** — `mutual_info_regression(X_j, z_h)` trên FIT của từng fold (train-only), `z_h` = target sau TargetTransform (đúng đại lượng model học), `n_neighbors = 3`, seed cố định. Null: `MI_null_j = MI(X_j, z_h xáo trộn)`. `MI − MI_null ≤ 0` (median 5 fold) = không đo được phụ thuộc. Chi phí ≈ 30 phút.

Cờ per cột (không có tier) [đã chốt]: với mỗi tiêu chí, cột được cờ **+** khi điểm số > 0 ở **ít nhất 2 trong 3 horizon** (`PI+`, `SA+`, `MI+`). Ví dụ PI > 0 ở h1, h2 nhưng < 0 ở h3 → `PI+`; PI > 0 chỉ ở h1 → không `+`. Không dùng bộ riêng theo horizon.

Bốn bộ candidate, định nghĩa thẳng bằng cờ; mỗi cột có giữ/bỏ riêng cho từng bộ:

| Bộ | Giữ cột khi | Ý nghĩa |
|---|---|---|
| R1 | `PI+` hoặc `SA+` hoặc `MI+` | chỉ bỏ cột âm cả ba tiêu chí (nhẹ) |
| R2 | `PI+` hoặc (`SA+` và `MI+`) | bỏ cột model không dùng, trừ khi có tín hiệu độc lập và phụ thuộc đo được |
| R3 | `PI+` | chỉ giữ cột model đang dùng hữu ích (mạnh) |
| R4 | `SA+` | chỉ giữ cột có tín hiệu độc lập (mạnh nhất; đây là cách xử lý trường hợp một cột đơn thắng B0-306) |

Kiểm chứng (4 run, LightGBM gốc, `15fixed_306` của phase A, **cùng `selection_seed` cho cả 4 run và cho baseline B0-306**, 5 fold): mỗi bộ train → Gain trên giá so với B0-306 trên 15 ô. Chọn **B0\***: trong các bộ có `MedianGain ≥ −ε_LGBM` (không tệ hơn B0-306), lấy bộ có MedianGain cao nhất; chênh nhau < ε → lấy bộ nhỏ hơn. Không bộ nào đạt → B0\* = B0-306, ghi rõ "lọc không giúp". Chỉ MedianGain quyết định (ε_LGBM đo ở phase A trên B0-306); WinRate/P10/Worst báo cáo. B0\* là điểm xuất phát chung; sau đó mỗi model calibrate riêng trên B0\* (§1.3 phase B) rồi mới vào vòng lặp của nó.

Output: `experiments/b0_filter.csv` (306 dòng: PI per h, SA Gain vs E0 / vs B0-306 per h, MI − null per h, cờ PI+/SA+/MI+, giữ/bỏ theo R1, R2, R3, R4), kết quả 4 run kiểm chứng + bộ được chọn, bảng nhóm 38 base feature để đọc, danh sách cột B0\* đóng băng trong config.

Áp dụng cho model khác: mọi model bắt đầu Bước 2 từ B0\* (cột); LSTM dùng per phút các fine feature còn ≥ 1 cột trong B0\* (+ rv60); AutoTS base regressor = B0\*. Bộ lọc là theo LightGBM, model khác có thể đánh giá cột khác đi — chấp nhận để giữ một base chung. Tổng chi phí §1.4 ≈ 2–4 h. **Kết quả (2026-08-31): B0\* = R4, 72 cột** — giữ nguyên làm phần B0 của mọi S0_m ở vòng expanded-data (không lọc lại).

### 1.5 Data 2 năm và split `rolling_spread` — vòng expanded-data (2026-09-04) [`split.make_rolling_spread`, `RollingSpec`]

**Data** (quyết định user 2026-09-04b): nguồn canonical `data/BTC_1m_2y.csv` — 1.051.201 bar 60 s, `2024-09-03 16:29 → 2026-09-03 16:29 UTC` (đúng 730 ngày), 0 gap, 0 dup, OHLC/amount hợp lệ, sha256 `559ce040efd737d38f6d541b26e1533f4afc4682b5af2a94ef5cf842e31f8097`; cột `ts` (không phải `timestamp`) được alias trong bộ nhớ ở `read_ohlcv_csv` (có cả hai thì phải trùng; `datetime` kiểm khớp epoch UTC rồi dựng lại). **LF 5' dẫn xuất tất định** từ HF bằng `python run.py derive-lf` (`data.derive_lf_5min`): nhóm `(T−4' … T]`, nhãn `T` = bar 1' cuối (bội 300 s — đúng quy ước LF cũ và `asof_index` `T ≤ t`), open = first, high = max, low = min, close = last, volume/amount = sum; nhóm thiếu bar ở đầu/cuối bị bỏ (2 nhóm) → `data/BTC_5m_2y.csv` 210.239 bar (`2024-09-03 16:35 → 2026-09-03 16:25`), sha256 `0e5fb9ad20478dd4cc8b26c3669453ff35b3ceff1c72b81d20a6337540f52fef`; sidecar `data/BTC_5m_2y.derivation.json` ghi sha nguồn — `check-data` hard-fail nếu LF không dẫn xuất từ HF hiện tại. Anchor §6.1: `data/data_checksums_2y.json` (không ghi đè anchor 15 ngày). CSV raw/dẫn xuất không vào git. Không dùng `data/BTC_hf_1min_full.csv`.

**Split** (không hard-code ngày; tham số `configs/p0_full.json` → `split`): `mode = rolling_spread`, `n_folds = 5`, `val_days = 3`, `fit_days = 120`, `es_days = 5`, `test_days = 30`, `purge_minutes = 60`.

- `t_end` = sau bar 1' cuối; **TEST** = `[t_end − 30d, t_end)` — chỉ chạm ở `final`, một lần.
- `latest_val_start = test_start − 3d`; `earliest_val_start = first_origin + 120d + 5d` (origin eligible đầu đã gồm warmup B0 631 bar); 5 `val_start` = `linspace(earliest, latest, 5)` làm tròn lưới phút → **5 VAL rải đều trên toàn bộ lịch sử trước TEST**, tách rời (khoảng cách ≈ 142,7 ngày), không liền kề.
- Mỗi fold ROLLING (không expanding): `FIT = [val_start − 125d, val_start − 5d)` (đúng 120 ngày ≈ 172,8k bar), `ES = [val_start − 5d, val_start − 60')`, `VAL = [val_start, val_start + 3d)`; fold độc lập, chạy song song được.
- **Final refit**: `FIT = [T_test − 125d, T_test − 5d)`, `ES = [T_test − 5d, T_test − 60')`. Quy tắc biên giữ nguyên: origin thuộc partition khi `t + 3' < T_end`.
- Vì sao rải đều: EDA data 2 năm — năm 1 +94 %, năm 2 −27,9 %, max drawdown ≈ −54 %, vol 1' 3,7 → 9,7 bp theo tháng, hướng target cân bằng ≈ 50 % ⇒ giá trị là ĐA DẠNG REGIME; 5 ô VAL phải lấy mẫu regime khác nhau (giữ 15 ô = 5 fold × 3 horizon). Vì sao FIT 120: ×3 data/fold so với 40 ngày mà vẫn chạy được với 163 candidate; không FIT 365/730 hay expanding cho từng candidate (user quyết sau nếu muốn).

Kết quả resolve trên data thật (`check-data`, B0-eligible 1.049.358 origin, first 2024-09-04 03:00 UTC):

| Fold | FIT (120 ngày) | ES | VAL | n FIT / ES / VAL |
|---|---|---|---|---|
| 1 | 2024-09-04 03:00 → 2025-01-02 03:00 | → 2025-01-07 02:00 | 2025-01-07 03:00 → 01-10 03:00 | 172.753 / 7.121 / 4.317 |
| 2 | 2025-01-25 00:22 → 05-25 00:22 | → 05-29 23:22 | 2025-05-30 00:22 → 06-02 00:22 | 172.727 / 7.137 / 4.309 |
| 3 | 2025-06-16 21:45 → 10-14 21:45 | → 10-19 20:45 | 2025-10-19 21:45 → 10-22 21:45 | 172.409 / 7.137 / 4.317 |
| 4 | 2025-11-06 19:08 → 2026-03-06 19:08 | → 03-11 18:08 | 2026-03-11 19:08 → 03-14 19:08 | 172.554 / 7.137 / 4.317 |
| 5 | 2026-03-29 16:30 → 07-27 16:30 | → 08-01 15:30 | 2026-08-01 16:30 → 08-04 16:30 | 172.703 / 7.109 / 4.300 |
| final | 2026-04-01 16:30 → 07-30 16:30 | → 08-04 15:30 | TEST 2026-08-04 16:30 → 09-03 16:30 | 172.703 / 7.085 / 42.918 |

Split 15 ngày (§1.2, `make_folds`) và mode `rolling_from_end` (rev 10) giữ nguyên cho lịch sử/test. Scale data §5.3 để sau.

---

## 2. Bước 2 — Feature selection theo từng model

Nguyên tắc: **chạy từng model một, theo thứ tự thời gian chạy tăng dần** (§2.2); **mọi model xuất phát từ S0_m của chính nó** (vòng expanded-data, §0b; vòng 15 ngày: cùng B0\*); mỗi model calibrate riêng `15fixed_m` và ε_m trên S0_m (§1.3) rồi tự chạy cùng một vòng lặp add-one qua Candidate_m (§2.3b; vòng 15 ngày: danh sách §2.3) theo cùng thứ tự, bằng chính model đó. Không để model nào tìm trước rồi model khác kế thừa. Kết quả: các feature set riêng F\*_LGBM, F\*_XGB, F\*_Cat, … có thể khác nhau.

### 2.1 Vòng lặp feature (áp dụng y hệt cho mỗi model `m`, xuất phát từ S0_m — vòng 15 ngày: B0\*)

Trước vòng lặp: calibrate phase B của `m` trên S0_m (§1.3; vòng 15 ngày: B0\*) → `15fixed_m` (LightGBM/XGBoost/CatBoost) hoặc `fixed_epoch_LSTM` / `fixed_epoch_TFM` (LSTM, TimesFM-LoRA) bằng `calib_seed`, và ε_m từ 3 evaluation seed; XGB-RF/AutoTS không có gì để calibrate ngoài ε_m. **Run baseline S0_m và toàn bộ candidate đều chạy ở `selection_seed`** (một giá trị duy nhất) với số vòng cố định.

`S_m := S0_m` (vòng expanded-data, §0b; vòng 15 ngày: `S_m := B0*`). Với từng feature `f` trong `Candidate_m` (§2.3b; vòng 15 ngày: §2.3), theo đúng thứ tự:

1. Input = S0_m (B0\* + ext khoá) + các cột ext đang KEEP của `m` + `f` (giá trị tại origin t, lag 0; với LSTM là chuỗi theo phút của cùng cột). **TimesFM**: S0 = ∅ — backbone LoRA đã freeze trên r1, covariate = các candidate ext đang KEEP + `f`, KHÔNG có cột B0\* (§2.2 #4).
2. Train `m` × 5 fold với số vòng/epoch cố định của `m` (config §2.2).
3. Metric trên giá tại VAL; Gain per ô với **base = `m` trên S_m hiện tại** (ghi thêm Gain vs `m` trên B0\*, vs E0, và Gain standalone §2.4).
4. `MedianGain ≥ −ε_m` → **KEEP** (tốt hơn hoặc gần như không đổi), `S_m := S_m + f`; `MedianGain < −ε_m` → **DROP**.
5. Feature tiếp theo.

Hết danh sách → **F\*_m** (bộ sau vòng lặp). Không còn safety-net. **Tên gọi rev 10.1**: F\*_m ≡ **F_raw_m**, F\*_m^prune ≡ **F_pruned_m**, win_m ≡ **F_best_m** — đều là tập do vòng tìm kiếm mới sinh, phân biệt với **S0_m** (tập xuất phát khoá). Sau đó:

(a) **Prune PI** (vẫn số vòng/epoch cố định): tính permutation importance trên VAL cho các cột ext **MỚI** của F\*_m (cột khoá S0_m không xét, không bao giờ bị bỏ); bỏ đồng thời mọi cột ext mới không có cờ PI+ (PI > 0 ở ≥ 2/3 horizon — cùng quy ước cờ §1.4; PI = median 5 fold của 3 lần xáo) → **F\*_m^prune**.

(b) **Confirmation 3 seed → win_m** (phase C): mỗi configuration (F\*_m và F\*_m^prune) chạy **3 evaluation seed** (8587, 8588, 8589; ES bật; `best_iteration`/best epoch ghi lại cho Final refit) → 3 bảng RMSE 5 fold × 3 horizon = 15 ô. Với mỗi ô (f, h) lấy **mean RMSE của 3 seed** → một bảng `RMSE̅` 15 ô duy nhất cho mỗi configuration. Sau đó từng ô:

```
Gain_{f,h} = 1 − RMSE̅^prune_{f,h} / RMSE̅^unprune_{f,h}
```

rồi **MedianGain = median của 15 Gain**, so với ngưỡng nhiễu ε_m của model đang xét (WinRate/P10/Worst tính trên cùng 15 ô, chỉ báo cáo). `MedianGain ≥ −ε_m` → **win_m = F\*_m^prune**; thấp hơn → **win_m = F\*_m** (unpruned). Bảng `RMSE̅` của win_m là bảng dùng cho §3 và figure §7.3.

Kết quả của model `m`: **win_m** (feature set + bảng `RMSE̅` mean 3 seed) + bảng `keepdrop_<m>.csv` + bảng prune. Sang model kế tiếp (cũng từ S0 của nó). TimesFM không có feature dạng cột: xuất phát không covariate (native LoRA), thử thêm lần lượt Candidate_TFM làm covariate qua XReg trên adapter đã freeze (§2.2 #4).

### 2.2 Thứ tự model (thời gian chạy tăng dần), config và cách chọn feature

| # | Model | Config (một config, không sweep) | Chọn feature | Ước lượng tổng cho 39 candidate (15 ngày, Vast) |
|---|---|---|---|---|
| 1 | LightGBM | `LGBMConfig` gốc (B0) | §2.1 từ S0_LGBM với `15fixed_LGBM` → F\*_LGBM | 15 fit × ~5 s ≈ 1–2 phút/candidate → **≈ 1–1.5 h** |
| 2 | XGBoost | `hist`, `device=cuda`, `reg:pseudohubererror` (huber_slope 0.9), lr 0.03, max_depth 6, seed theo vai trò §1.3; cùng TargetTransform | §2.1 từ S0_XGB với `15fixed_XGB` → F\*_XGB | **≈ 1–2 h** |
| 3 | CatBoost | GPU, `Huber:delta=0.9`, lr 0.03, depth 6, seed theo vai trò §1.3; cùng TargetTransform | §2.1 từ S0_Cat với `15fixed_Cat` → F\*_Cat | **≈ 1–2 h** |
| 4 | **TimesFM-LoRA** (2026-09-03; `models_tfm.TimesFMLoRAModel`, `lora.py`; audit `docs/reference/audit_timesfm_lora.md`) | Checkpoint `google/timesfm-2.5-200m-pytorch` rev `1d952420…`, `timesfm[torch]==2.0.2`, `torch_compile=False`. **LoRA per fold** trên chuỗi r1: cửa sổ FIT (context 512 → target `r1[t+1..t+3]`, t ∈ FIT) để học, ES để chọn epoch (patience 5, ≤ 20 epoch), VAL/TEST không bao giờ vào training. LoRA tự chứa (không peft): r=8, α=16, dropout 0, target `stacked_xf.{i}.attn.qkv_proj / attn.out / ff0 / ff1` × 20 = 80 nn.Linear (2.048.000 tham số, 0,885 %), base đóng băng, fp32, AdamW lr 1e-4 wd 0.01, batch 64; **loss = MSE trên ŷ_h = cumsum(r̂) vs y_h** (mean head kênh 0, patch cuối, 3 horizon, đơn vị log-return chia hằng số std). `train_forward` tái hiện đúng `compiled_decode` (normalize, running stats, revin, flip) có grad — canary bit-exact. Vai trò seed §1.3 áp cho adapter: `calib_seed` → ES → `fixed_epoch_TFM`; `eval_seeds` → 3 adapter → ε_TFM (đo trên native); `selection_seed` → **một adapter FROZEN** cho baseline + toàn bộ add-one + prune PI; confirmation 3 seed ES bật (adapter dùng chung cho native / raw / pruned). Adapter cache theo (fold, seed, epoch-mode) → `experiments/<run>/lora/<key>.pt` + `.json` (sha256, curve, cấu hình); hash LoRA được assert không đổi sau mỗi predict. Inference giữ nguyên: `infer_is_positive=False`, head mean `quantile[...,0]`, đường covariate `per_core_batch_size=1`, **1 origin/lời gọi**, covariate **dịch 1 bar**, 3 bước tương lai giữ f(t), xreg trên jax GPU (`PREALLOCATE=false`), một module dùng chung cho wrapper point và covariate | **Một đường**: `loop --model tfm` xuất phát S0 = ∅ → calibrate = LoRA FIT + ES chọn epoch → ε → add-one Candidate_TFM **qua XReg trên adapter đã freeze** (thêm candidate = fit lại xreg, KHÔNG động trọng số). **Mỗi cấu hình feature được chấm dưới dạng HỆ THỐNG HOÀN CHỈNH `TimesFM-LoRA + XReg(S ∪ candidate)`** — XReg không bao giờ là model đứng riêng → F_raw → prune PI → F_pruned → **confirmation = {TimesFM-LoRA + XReg(F_raw)} vs {TimesFM-LoRA + XReg(F_pruned)} (CÙNG adapter đã freeze, code assert danh tính adapter) → F_win** (`wins/tfm_lora_xreg.json`, metadata `feature_set_source` ghi rõ which/n_new) → rồi mới dựng **hệ thống A = TimesFM-LoRA baseline** (0 feature, 0 B0\*, 0 covariate; confirmation 3 seed trên CÙNG adapter → `wins/tfm_lora_baseline.json`, tên cũ `tfm_lora_native.json` vẫn đọc được). `tfm-final`: so **HAI HỆ THỐNG HOÀN CHỈNH** — **B = {TimesFM-LoRA + XReg(F_win)} vs A = {TimesFM-LoRA baseline, feature-free}** bằng luật §3 (MedianGain > +ε_TFM → B, ngược lại A) → **TimesFM-final** (`wins/tfm.json`; `lora_adapters` của A và B phải TRÙNG — cùng adapter đã freeze). TFM-final **được LƯU rồi CHỜ** — không so champion ngay; champion chỉ diễn ra ở `champion-replay` theo thứ tự cố định → champion → ensemble → Final (refit LoRA trên fold final, ES). Mọi artifact TimesFM ghi metadata tường minh (backbone, finetuned=true, finetune_method=LoRA, native, covariates, input_series, context 512, horizon 3, target y1..y3 cộng dồn). **Calibrate CHÍNH LÀ bước LoRA FIT + ES chọn epoch** (không có "LoRA trước, calibrate sau"): pretrained → FIT: train LoRA → ES: chọn epoch → fixed_epoch_TFM → adapter cho các seed § 1.3 → freeze → baseline TimesFM-LoRA native + XReg candidate search. XReg không phải model độc lập; không có nhánh `tfm_b0`/`tfm_ext`, B0\* không bao giờ vào TimesFM. Với k ≥ 1 covariate, model nhận phần dư OLS của xreg — lệch train/serve vốn có, ghi nhận | LoRA: 7 adapter/fold (1 calib + 3 eval + 3 confirmation) × 5 fold; XReg: |Candidate_TFM| ≈ 163 pass × 5 fold × 4.320 origin — **chưa đo**, canary 1 fold × 1 epoch + 1 candidate trước khi cam kết ETA; fold-parallel 5 worker LoRA chỉ đủ VRAM cho batch ≤ ~35 (audit §7) → có thể giảm worker cho bước này |
| 5 | XGB-RF (thay ExtraTrees) | XGBoost random-forest mode trên GPU: `num_parallel_tree=500`, `subsample=0.63`, `colsample_bynode=0.3`, `learning_rate=1`, 1 vòng boosting, `max_depth=8`, `min_child_weight=500`, squared error trên z-target, `device=cuda`, seed theo vai trò §1.3 | §2.1 từ S0_XGBRF (1 vòng boosting cố định — không có gì để calibrate ngoài ε) → F\*_XGBRF | 15 fit × ~10 s ≈ 2–3 phút/candidate → **≈ 1.5–2 h** |
| 6 | AutoTS — 2 model cố định | `WindowRegression` (regression_model LightGBM, GPU) và `MultivariateRegression` (regression_model XGBoost, GPU). **Audit chốt** (`docs/reference/audit_autots.md`): `autots==1.0.4`, import `autots.models.sklearn`, gọi thẳng class cho probe, `regression_model={"model": "LightGBM"|"xgboost", "model_params": {...GPU...}}`, rolling `fit` 1 lần/fold + `fit_data`/`predict` mỗi origin (không refit); regressor **dịch theo model** (MR: `R.loc[s]=f(s−1)`; WR: `R.loc[s]=f(s+window−1)`, predict truyền `f(t)`), vá bug `sklearn.py:3337` ở phía ta; chấm trên **toàn bộ origin** như mọi model | **Hai giai đoạn.** (i) **Probe**: WR và MR, mỗi cái xuất phát từ S0 của nhánh đó (F_WR_old / F_MR_old khoá; vòng 15 ngày: B0\*) và chạy đủ §2.1 (add-one Candidate_m → prune PI → confirmation) → **F_WR_best**, **F_MR_best** (= win sau confirmation, không phải output vừa kết thúc add-one). Probe chỉ dò feature: không so champion, không vào ensemble, không refit ở Final. (ii) **Framework** (`autots-search`): FREEZE hai bộ đó rồi chạy framework AutoTS **riêng cho từng bộ** (dedup nếu hai bộ trùng) — `initial_template` do ta khai báo (mọi dòng ép GPU), `max_generations=0`, `transformer_max_depth=0`, chạy **chỉ trên training-side của fold** (FIT+ES, dừng trước purge 60') → freeze template → refit + rolling predict outer VAL bằng `ModelMonster`. So `result_WR` vs `result_MR` bằng **metric project** (không dùng điểm nội bộ AutoTS) → **AutoTS-final** → champion → ensemble → Final. Template chia theo **nhóm cùng shift regressor** vì AutoTS chỉ nhận một `future_regressor` | probe ≈ 1–4 h/model; framework ≈ 5–9 h (4 template / 2 nhóm × `num_validations`=10 × 5 fold × ≤ 2 bộ) |
| 7 | LSTM-DMH | context 512; input mỗi phút = các fine feature B0 còn trong B0\* (+ rv60) + ext đang KEEP; 1 lớp LSTM hidden 64; head linear 3 output; Huber trên z-target (TargetTransform B0); Adam lr 1e-3, batch 256, ≤ 50 epoch, ES patience 5 trên ES set chỉ ở run calibrate và confirmation; seed theo vai trò §1.3; NaN → 0 sau chuẩn hóa | §2.1 từ B0\* với `fixed_epoch_LSTM` (per fold, một số epoch cho cả 3 h), 1 seed duy nhất (`selection_seed`) trong vòng lặp; confirmation F\*_LSTM 3 evaluation seed, ES bật. Dự phòng nếu hết thời gian: chạy LSTM trên từng F\*_m của các model khác (4–6 run) và chọn bộ tốt nhất theo metric — không có cách biết trước bộ nào hợp | 1 fit ≈ 1–3 phút GPU cho cả 3 h; 5 fold ≈ 5–15 phút/candidate → **≈ 3–10 h** (chậm nhất → chạy cuối) |

Ràng buộc chung cho regressor/covariate (AutoTS, TimesFM): giá trị dùng để dự báo bar `s` chỉ được tính từ dữ liệu `≤ s−1`; dự báo 3 bước từ t giữ nguyên giá trị tại t (cách truyền cụ thể chốt khi audit API; kiểm tra bằng §6.4). AutoTS/TimesFM không thấy VAL/TEST. Audit version trước khi code; cài package chỉ khi user cho phép.

Tổng Bước 2 trên 15 ngày ≈ 12–25 giờ máy nếu chạy tuần tự đủ 7 model (cộng §1.4 ≈ 3–5 h vì standalone chạy GPU).

Không thêm: KNeighbors (306 chiều, low SNR → ≈ E0 hoặc noise), Bagging/ExtraTrees sklearn (CPU-only — họ bagging đại diện bởi XGB-RF trên GPU), LinearRegression/Lars (OLS trên cột collinear không ổn định), AutoTS **genetic self-search** (`max_generations ≥ 1`) — audit 2026-08-31 (`docs/reference/audit_autots.md` §12.4): `generate_regressor_params` không bao giờ sinh khoá `device`/`device_type`, bảng regressor là sklearn CPU-only, và mọi generation ≥ 1 ghi đè params GPU ta nạp ⇒ ~100% số fit sẽ chạy CPU, vi phạm invariant §0. Thay bằng **bake-off template GPU** (`max_generations=0`, template do ta khai báo) ở §2.2 #6 giai đoạn (iii). yfinance.

### 2.3 Danh sách candidate vòng 15 ngày (LỊCH SỬ — không còn là candidate; cột KEEP nằm trong S0_m, xem §2.3b)

Thứ tự trong bảng = thứ tự thử. Mọi cột chỉ dùng dữ liệu ≤ t, cửa sổ kết thúc tại t, lookback ≤ 1440 phút.

Ký hiệu: `C, O, H, L, V` = close/open/high/low/volume của bar; `A` = amount (quote volume); `TP = (H + L + C)/3`; `r1 = log(C_t / C_{t−1})`; `rv_k = sqrt(mean_k(r1²))`; `EMA_k` = ewm(span k, min_periods k) trên log C; `ret_k = log(C_t / C_{t−k})`.

**A. VWAP thật từ amount** — `A/V = Σ p·q / Σ q` của các trade trong bar (Binance quote volume / base volume) → là VWAP thật, không phải proxy.

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 1 | `vwap_amt_gap_1` | `log(C / (A/V))` | vị trí close so với giá trung bình bar; gần `close_position` |
| 2 | `vwap_amt_gap_15` | `log(C / (Σ_15 A / Σ_15 V))` | |
| 3 | `vwap_amt_gap_60` | `log(C / (Σ_60 A / Σ_60 V))` | |
| 4 | `vwap_amt_gap_240` | `log(C / (Σ_240 A / Σ_240 V))` | |

**B. Return / rolling statistics ngoài B0** (B0 có ret 1, 5, 8, 32 và các lag; rv5/rv60, rv8/rv64)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 5 | `ret_60` | `log(C_t / C_{t−60})` | |
| 6 | `ret_240` | `log(C_t / C_{t−240})` | |
| 7 | `ret_1440` | `log(C_t / C_{t−1440})` | |
| 8 | `log_rv15_rv240` | `log(rv_15 / rv_240)` | |
| 9 | `log_rv60_rv1440` | `log(rv_60 / rv_1440)` | |
| 10 | `ret_skew_60` | skew của r1 trên 60 bar | |
| 11 | `dd_240` | `log(C / max_240(C))` | drawdown |
| 12 | `ru_240` | `log(C / min_240(C))` | run-up |

**C. MA / EMA / HMA** (B0 có EMA 5/20, 8/32, 16/64, 32/128, HMA 16)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 13 | `log_c_ema60` | `log C − EMA_60` | |
| 14 | `log_c_ema240` | `log C − EMA_240` | |
| 15 | `log_c_ema1440` | `log C − EMA_1440` | |
| 16 | `log_ema60_ema240` | `EMA_60 − EMA_240` | |
| 17 | `hma_slope64_volnorm` | `diff(HMA_64(log C)) / rv60` | |

**D. RSI / MACD multi-scale** (B0 có RSI 15/64; MACD 5/20/7 và 16/64/16)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 18 | `rsi240_centered` | `RSI_240(r1)/100 − 0.5` | |
| 19 | `macd_hist_60_240_60_volnorm` | `((EMA_60 − EMA_240) − EMA_60(EMA_60 − EMA_240)) / rv60` | |

**E. Bollinger** (trên log C; `SMA_n`, `σ_n` với `min_periods = n`, `ddof = 0`)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 20 | `bb_pctb_20` | `(log C − SMA_20) / (2·σ_20)` | mean-reversion z-score |
| 21 | `bb_pctb_60` | `(log C − SMA_60) / (2·σ_60)` | |
| 22 | `bb_logbw_20` | `log(σ_20)` | bandwidth |

**F. ATR / Keltner** (`TR = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|)`, `ATR_n` = Wilder EMA alpha 1/n, reset sau gap)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 23 | `log_atr14_c` | `log(ATR_14 / C)` | |
| 24 | `log_atr14_rv14` | `log((ATR_14 / C) / rv_14)` | range-vol ÷ close-vol; giả thuyết liên quan bounce/reversal ở h = 1 |
| 25 | `kcw_20` | `log(2·ATR_20 / EMA_20(C))` | Keltner channel width |

**G. MFI** (money flow = A; `A⁺` khi `TP_t > TP_{t−1}`, `A⁻` khi `TP_t < TP_{t−1}`)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 26 | `mfi14_centered` | `ΣA⁺ / (ΣA⁺ + ΣA⁻)` trên 14 bar `− 0.5` | NaN nếu mẫu = 0 |
| 27 | `mfi60_centered` | như trên, 60 bar | |

**H. A/D dạng rolling** (không dùng tích lũy vì level phi dừng; `CLV = ((C−L) − (H−C)) / (H−L)`, = 0 khi `H = L`; `CLV = 2·close_position` của B0)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 28 | `ad_vwclv_5` | `Σ_5 CLV·V / Σ_5 V` | |
| 29 | `ad_vwclv_15` | `Σ_15 CLV·V / Σ_15 V` | |
| 30 | `ad_vwclv_60` | `Σ_60 CLV·V / Σ_60 V` | |

**I. Parabolic SAR** (AF 0.02, bước 0.02, max 0.2; `SAR_t` tính từ bar ≤ t; reset sau gap NaN)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 31 | `psar_dir` | +1 uptrend / −1 downtrend | |
| 32 | `psar_logdist` | `log(C / SAR_t)` | |
| 33 | `psar_age_log` | `log1p(số bar từ lần flip gần nhất)` | |

**J. Regime / calendar**

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 34 | `dow_sin`, `dow_cos` | `sin/cos(2π·weekday/7)` | thử như một cặp; trên 15 ngày chỉ có 2 tuần → ý nghĩa hạn chế |
| 35 | `log_rv60_med2d` | `log(rv60 / median_2880(rv60))` | vol regime |
| 36 | `log_range_240` | `log((max_240 H − min_240 L) / C)` | compression/breakout |

**K. Resolution 5 phút** (`BTC_lf_5min.csv`, chỉ bar đã đóng, as-of join `T ≤ t`; về thông tin trùng 1-phút, chỉ khác representation → để cuối)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 37 | `r5_1` | log return của bar 5' đã đóng gần nhất | |
| 38 | `r5_12` | `log(C5_T / C5_{T−12})` | |
| 39 | `log_c5_ema5_12` | `log C5 − EMA_12(log C5)` | |

Ghi chú: RMSE/MAE/R²/MAPE là metric, không phải feature. Feature ngoài danh sách chỉ thêm khi có giả thuyết rõ, và thêm vào cuối danh sách.

### 2.3b C_short — candidate vòng EXPANDED-DATA (2026-09-03, hiệu chỉnh 2026-09-04; `src/p0/features_short.py`)

Chỉ feature NGẮN HẠN ≤ 15 phút, lưới `W = {1, 2, 3, 4, 5, 8, 10, 15}` làm DÀY mọi họ; một cửa sổ chỉ bị bỏ khi (i) **suy biến / không xác định về toán**, hoặc (ii) chính là một candidate cũ §2.3 (KEEP hay DROP đều không quay lại, §6). **KHÔNG** bỏ vì tương quan cao, xấp xỉ một cột khác (ATR15 vs ATR14, MFI15 vs MFI14, HMA15 vs HMA16, Keltner vs ATR), cửa sổ nhỏ "nhiễu", dạng chỉnh lưu (dd_2/ru_2), hay vì B0-306 có cột cùng tên ngoài B0\* (ret_1/ret_5/ret_8, log_rv5_rv60, rsi15 tại t) — vòng lặp chọn model, không phải bộ sinh, quyết định feature có ích hay không. `dow` là **ngoại lệ có tài liệu**: weekday không có horizon cuộn tự nhiên → không sinh biến thể. Ký hiệu như §2.3. Thứ tự thử = họ A→S, cửa sổ tăng dần. **Tổng 163 cột** (lý do bỏ máy đọc được: `s0/short_pool.json`).

| Họ | Cột | Cửa sổ dùng | Cửa sổ bỏ (lý do chính xác) |
|---|---|---|---|
| A | `vwap_amt_gap_k` = log(C / (Σ_k A / Σ_k V)) | 2,3,4,5,8,10 | 1, 15: chính là candidate cũ |
| B | `ret_k` = log(C_t/C_{t−k}) | 1,2,3,4,5,8,10,15 | — |
| C | `log_rv{k}_rv60` | 2,3,4,5,8,10,15 | 1: rv_1 = |r1| → log 0 khi r1 = 0 (không xác định) |
| D | `log_c_ema{k}` = log C − EMA_k | 2,3,4,5,8,10,15 | 1: ≡ 0 |
| D | `log_ema{a}_ema{b}` | (2,5), (2,8), (3,10), (4,10), (5,15) | — |
| E | `rsi{k}_centered` | 1,2,3,4,5,8,10,15 | — |
| F | `bb_pctb_k` | 2,3,4,5,8,10,15 | 1: σ_1 = 0 → chia 0 |
| F | `bb_logbw_k` = log σ_k(log C) | 3,4,5,8,10,15 | 1: σ = 0 → log 0; 2: σ_2 = |r1|/2 → log 0 khi r1 = 0 |
| G | `log_atr{k}_c` | 1,2,3,4,5,8,10,15 | — |
| G | `log_atr{k}_rv{k}` | 2,3,4,5,8,10,15 | 1: rv_1 = |r1| → log 0 |
| H | `kcw_k` = log(2·ATR_k / EMA_k(C)) (Keltner) | 2,3,4,5,8,10,15 | 1: = log 2 + log_atr1_c (đồng nhất thức) |
| I | `mfi{k}_centered` | 1,2,3,4,5,8,10,15 | — |
| J | `ad_vwclv_k` | 1,2,3,4,8,10 | 5, 15: chính là candidate cũ |
| K | `dd_k`, `ru_k` | 2,3,4,5,8,10,15 | 1: ≡ 0 |
| L | `log_range_k` = log((max_k H − min_k L)/C) | 1,2,3,4,5,8,10,15 | — |
| M | `ret_skew_k` | 3,4,5,8,10,15 | 1, 2: skew cần ≥ 3 quan sát (không xác định) |
| N | `hma_slope{k}_volnorm` | 1,2,3,4,5,8,10,15 | — (HMA của B0 xác định với mọi k ≥ 1) |
| O | `macd_hist_{f}_{s}_{sig}_volnorm` | (2,5,2), (2,8,3), (3,10,3), (4,10,4), (5,15,5) | — |
| P | `psar_dir_W`, `psar_logdist_W`, `psar_age_log_W` — PSAR Wilder **cửa sổ reset**: khởi tạo lại trạng thái tại đầu W bar cuối `[t−W+1 … t]`, chạy causal, lấy trạng thái tại t (AF 0.02/0.02/0.2, cùng quy tắc `_psar_segment`) | 2,3,4,5,8,10,15 | 1: PSAR cần ≥ 2 bar để khởi tạo hướng/EP (trạng thái không xác định) |
| Q | `log_rv{k}_med2d` = log(rv_k / median_2880(rv_k)) | 2,3,4,5,8,10,15 | 1: rv_1 = |r1| → log 0 |
| R | `r5_k` = log(C5_T / C5_{T−k}) trên bar 5' đã đóng (as-of T ≤ t) | 2, 3 (= 10', 15') | 1 (= r5_1), 12 (= r5_12): candidate cũ |
| S | `log_c5_ema5_k` = log C5 − EMA_k(log C5) | 2, 3 | 1: ≡ 0; 12: candidate cũ |
| T | `dow` | **không sinh** | ngoại lệ: weekday không có cửa sổ cuộn tự nhiên (dow_sin/dow_cos là candidate cũ §2.3) |

Candidate_m = C_short \\ overlap(C_short, S0_m) tính riêng từng model (§0b.3). Kết quả xem trước trên data 15 ngày thật: Candidate_m = 163 cho mọi model (không cột C_short nào trùng tên/giá trị với cột S0_m); near vs S0 (chỉ báo): lgbm/xgb/cat/xgbrf/lstm 1, autots_wr 7, autots_mr 3; 27 cặp near nội bộ C_short (chỉ báo). Có thể giới hạn pool bằng `short_candidates` trong config (danh sách con của C_short, ghi rõ khi dùng — không phải mặc định).

### 2.4 Khi candidate thua hoặc không đổi vì base đã nhiều feature

Gain so với `S_m` đo **thông tin tăng thêm** so với B0\* (đã có return/vol/candle/volume/RSI/MACD/EMA/HMA ở nhiều lag). Vì vậy phần lớn candidate sẽ ra gần 0: đó là kết luận hợp lệ ("không thêm thông tin mới"), không phải lỗi. Thua rõ (< −ε_m) nghĩa là feature thêm noise/overfit → DROP vẫn đúng. Xử lý để hiểu vì sao thua:

1. **Gain standalone (diagnostic, không dùng để KEEP/DROP)**: với mỗi candidate `f`, cùng cách tính với §1.4(b): LightGBM code gốc chỉ trên `[f]`, Gain trên giá so với E0 (và so với B0\*). Ghi vào `keepdrop_<m>.csv` cột `gain_standalone`. Đọc kết hợp: standalone > 0 nhưng vs S_m ≈ 0 → có tín hiệu nhưng trùng base; standalone ≈ 0 → không có tín hiệu. Chi phí: 15 fit tí hon ≈ vài giây/candidate.
2. **KEEP khi không đổi** (luật user) giữ lại feature bị che khuất; bước **prune PI** cuối vòng lặp (§2.1a) dọn feature thuần noise. Không có safety-net.
3. Không đổi hyperparameter của model để "giúp" feature (config cố định suốt vòng lặp).

---

## 3. Bước 3 — So sánh với champion (ngay sau mỗi model) + ensemble

- **Champion ban đầu** = LightGBM code gốc trên win_LGBM (sau vòng lặp #1 từ S0_LGBM — vòng 15 ngày: B0\* — prune PI, 3 seed). B0-306 và B0\* nguyên bản đều được log làm reference.
- Sau khi mỗi model `m` có **win_m** (§2.1: prune PI + 3 seed): tính từng ô `Gain_{f,h} = 1 − RMSE̅^win_{f,h} / RMSE̅^champion_{f,h}` với `RMSE̅` = bảng mean 3 seed của mỗi bên (cùng cách gộp như §2.1b) → MedianGain = median 15 Gain. `MedianGain > +ε_champion` (ε của champion đo ở §1.3 bằng 3 evaluation seed) → **đổi champion** = win_m; ngược lại → **giữ champion**. Cả hai trường hợp ghi đầy đủ vào `champion_log.csv` (§7.2) và vẽ figure win vs champion (§7.3).
- TimesFM: **TimesFM-final** = hệ thống B {TimesFM-LoRA + XReg(F_win)} nếu MedianGain vs hệ thống A {TimesFM-LoRA baseline, feature-free} > +ε_TFM, ngược lại A (§2.2 #4; artifact `wins/tfm_lora_xreg.json` / `wins/tfm_lora_baseline.json` → `wins/tfm.json`; F_win phải đã thắng confirmation F_raw vs F_pruned TRƯỚC khi so với A; vòng 15 ngày: bộ tốt hơn giữa hai nhánh zero-shot). AutoTS: **AutoTS-final** = bộ tốt hơn giữa {framework(F_WR_best), framework(F_MR_best)} (§2.2 #6).
- **Thứ tự so champion là methodology, không phải thứ tự chạy xong** (rev 10.3): lgbm → xgb → cat → tfm → xgbrf → autots → lstm. Khi các nhánh chạy song song trên nhiều GPU (`defer_champion: true`), champion được HOÃN: `python run.py champion-replay` đọc `wins/<m>.json` (RMSE̅ 15 ô, ε, `champion_extra`) và áp đúng luật trên theo thứ tự cố định — không train, không inference. `tfm_lora_baseline`/`tfm_lora_xreg`/`autots_wr`/`autots_mr` bị chặn cứng khỏi champion.
- Latency (§7.4) ghi kèm trong `champion_log.csv` như thông tin; không phải tiêu chí đổi/giữ.
- **Ensemble** (sau model cuối): thành viên = champion + mọi model có `MedianGain vs E0 > 0` trên 15 ô (có skill thật; B0-306/B0\* là reference, không phải thành viên; TimesFM/AutoTS-final/LSTM là thành viên nếu đạt). **Probe AutoTS-WR/MR không phải thành viên và không so champion** — chúng chỉ dò feature; đại diện AutoTS ở §3/§4 là **AutoTS-final** (§2.2 #6 giai đoạn iii). (a) trung bình đều; (b) trọng số `1/MSE_VAL` (trên giá) per horizon; lấy cấu hình tốt hơn trên VAL rồi so với champion bằng đúng luật trên (`> +ε_champion` → champion = ensemble). Nếu < 2 thành viên thì không ensemble. Chọn cấu hình cuối **trước** khi chạm TEST; ghi thành viên + trọng số vào `champion_log.csv`.

---

## 4. Final evaluation (một lần; vòng expanded-data: TEST 30 ngày = fold `final_TEST` §1.5 — vòng 15 ngày: TEST 2 ngày)

- (vòng expanded-data) Refit **mọi** model (không chỉ champion) trên fold `final_TEST` §1.5: FIT 120 ngày + ES 5 ngày (trừ purge 60') ngay trước TEST 30 ngày cuối (2026-08-04 16:30 → 09-03 16:30, 42.918 origin; FIT 172.703 / ES 7.085 origin). TimesFM-LoRA refit adapter (ES) trên fold final; AutoTS-final bake-off lại trên FIT+ES của fold final; win_m dùng đúng colset (kể cả cột khoá).
- (vòng 15 ngày, đã xong) Refit trên FIT `→ 01-30 23:56`, ES `01-31 00:00 → 22:56`, purge 60', TEST `02-01 00:00 → 02-02 21:27` (2.728 origin).
- Report per horizon trên giá: RMSE, MAE, Pearson r (thay đổi giá), directional accuracy; Gain vs B0-306, vs B0\* và vs E0; cho E0, B0-306, B0\*, từng model §2.2 tại F\*_m, ensemble. Xuất `all_models_test.csv`; **lưu prediction TEST** `final/<key>.npz` + `final/index.json` (origin, ŷ, colset, best_iters, champion). Figure Final (heatmap, Fig P, Fig T mọi model) sinh SAU bằng `python run.py visualize` (§7.5), không trong `final`.
- Đo latency §7.4 trên toàn bộ origin TEST cho mọi model (pass riêng, batch 1) → `latency_summary.csv`.
- Không sửa gì sau khi xem TEST. TEST chỉ được đọc bởi script final. TEST 2 ngày là kiểm tra one-shot của giai đoạn 15 ngày (đã xong); TEST 30 ngày = vòng expanded-data (§1.5).

---

## 5. Data đầy đủ — §5.1–5.2 là VÒNG HIỆN TẠI (2026-09-03), §5.3–5.4 để sau

1. Data 2 năm đã có (2026-09-04b): `data/BTC_1m_2y.csv` (1.051.201 bar, 2024-09-03 16:29 → 2026-09-03 16:29 UTC, 0 gap/dup) + LF dẫn xuất `data/BTC_5m_2y.csv` (210.239 bar); `derive-lf` → `check-data --write-checksums` → `data/data_checksums_2y.json` — đã chạy (§1.5).
2. Split feature-selection + TEST theo §1.5 (`rolling_spread`: 5 VAL 3 ngày rải đều, FIT 120 + ES 5 rolling, TEST 30 ngày cuối). Kiểm tra lại B0\* (giữ) và F\*_m đã chọn trên 15 ngày (nay là S0_m khoá) trên regime khác + tìm thêm feature ngắn hạn (§2.3b); `lock-s0` đã chạy trên data thật: Candidate_m = 163 cho mọi model, 0 overlap.
3. (để sau) Scale data từng model với F_best_m: train region 120 → 240 → 365 → full (từ origin eligible đầu), neo vào cùng 5 VAL rải đều; chọn D\*_m = mức mà MedianGain so với mức trước `< +ε_m` hoặc full nếu vẫn cải thiện. TimesFM-LoRA: scale data áp dụng cho cửa sổ FIT của LoRA (adapter train lại ở mỗi mức); XReg không đổi.
4. So sánh champion + ensemble tại (F\*_m, D\*_m) → Final trên TEST 30 ngày.

---

## 6. Checklist mỗi experiment (bắt buộc, ghi vào log)

1. **Input**: checksum khớp anchor của config (§1.1 / §1.5: `data/data_checksums_2y.json` cho data 2 năm; LF 5' dẫn xuất từ đúng HF theo sidecar `.derivation.json`, phủ HF); số dòng, khoảng thời gian, UTC, lưới 60 s, không dup, gap = 0; danh sách cột S0_m khớp `s0/<m>.json` (vòng 15 ngày: B0\* khớp config đóng băng).
2. **Target**: `y_h = log(C[t+h]/C[t])` kiểm tra tay vài origin; E0 trên VAL: RMSE giá của `P̂ = C_t` khớp `sqrt(mean((C_{t+h} − C_t)²))`.
3. **Time alignment**: partition half-open, origin cuối = `T_end − 4'`; bar 5' chỉ join khi `T ≤ t`.
4. **Leakage**: feature tính trên chuỗi cắt tại t và trên chuỗi đầy đủ phải cho cùng giá trị tại t; không `rolling(center=True)`, không shift âm; TargetTransform/scaler fit trên FIT của fold; MI §1.4 chỉ tính trên FIT; PI xáo trộn chỉ trong VAL; ES ≠ VAL; TEST chưa đọc; regressor/covariate (AutoTS, TimesFM) chỉ từ dữ liệu `≤ s−1`.
5. **Biên**: FIT/ES/VAL rời nhau; purge 60' giữa ES và VAL, và giữa train cuối và TEST.
6. **Metric**: tính trên **giá** sau decode + `exp` (`P̂ = C_t·exp(ŷ)`), không tính trên log return hay z-space; base của Gain ghi rõ (S_m / B0-306 / B0\* / E0 / champion); MedianGain trên 15 ô; AutoTS chấm trên đúng tập origin đã khai báo (thưa hay đầy đủ).
7. **Decode**: prediction qua `TargetTransform.decode` với rv60 của đúng origin (tree/LSTM) rồi `exp`; encode → decode round-trip khớp; AutoTS/TimesFM cộng dồn one-step đúng thứ tự trước khi `exp`.
8. **Hợp lý**: số vòng cố định đúng theo §1.3; `std(ŷ) ≪ std(y)` là bình thường (tín hiệu 1-phút chỉ cỡ 0.1–0.2 pp RMSE); Gain > ~1 pp vs B0/E0 → nghi leakage/bug, kiểm tra lại trước khi tin; xem figure §7.3 (sinh bằng `visualize`) của vài origin để chắc prediction không lệch pha.
9. **S0 / candidate (rev 10.1)**: `s0/<m>.json` có `locked_b0 == b0` (== B0\* của `experiments/15d/b0_star.json`) và `locked_ext == ext` (== F_old_m từ `experiments/15d/wins/<m>.json`); `candidates_<m>.json` = C_short \\ overlap(C_short, S0_m) của CHÍNH model (không lọc toàn cục; `near_vs_s0` chỉ báo); `audit_dataset_label` == dataset của config; TimesFM S0 = ∅, không B0\*.
10. **TimesFM-LoRA (2026-09-03)**: cửa sổ train chỉ từ FIT (`t + 3 < FIT_end`), ES chỉ để chọn epoch, VAL/TEST không vào `train_lora`; adapter inject sau `load_checkpoint`, base `requires_grad=False`, 80 module / 2.048.000 tham số; `train_forward` == `compiled_decode` (canary); hash adapter không đổi qua toàn bộ add-one + prune (assert trong `_predict_with`); mỗi fold nạp đúng adapter của fold đó; `tfm-final` so +XReg với native bằng ε_TFM đo trên native.
11. **Bất biến cứng & checker_log (rev 10.1)**: GPU preflight thật (XGBoost: build CUDA + booster device cuda); `final/TEST_SENTINEL.json` chặn TEST lần hai; `checker_log.jsonl` không có ERROR chưa đóng trước run (`scripts/checker_record.py --blocking`).

---

## 7. Log và visualize

Layout mẫu của mọi bảng/figure dưới đây, với **số giả**: `reports/smoke_visualize.md` (sinh bởi `reports/smoke_visualize.py`, seed 8586, không đọc data thật) — chỉ để thống nhất hình dạng output trước khi code; không phải kết quả, không trích dẫn.

### 7.1 Log mỗi run

`experiments/log.csv`: `exp_id, step, model, feature_set (danh sách cột ext), dataset_label, config_hash, seed, RMSE/MAE giá 15 ô, Gain 15 ô, MedianGain, WinRate, P10Gain, WorstGain, base, decision, ghi chú`. Thư mục `experiments/runs/<exp_id>/` chứa config, số vòng, importance, prediction VAL (ŷ và P̂ theo origin). Run phải tái tạo được từ config + seed (GPU có thể lệch bit nhỏ; ghi nhận).

### 7.2 Log quyết định

- `experiments/b0_filter.csv` (§1.4): 306 dòng — cột, base feature, lag, PI per h (median fold), SA Gain vs E0 / vs B0-306 per h, MI − null per h, cờ PI+/SA+/MI+ (> 0 ở ≥ 1 horizon), giữ/bỏ theo từng bộ R1, R2, R3, R4; kèm bảng nhóm 38 base feature và kết quả 4 run kiểm chứng + bộ được chọn.
- `experiments/keepdrop_<model>.csv`: mỗi candidate một dòng — thứ tự, cột, MedianGain/WinRate/P10/Worst vs S_m, Gain vs B0\*, Gain vs E0, `gain_standalone`, decision KEEP/DROP, size S_m sau quyết định, exp_id.
- `experiments/champion_log.csv`: mỗi model một dòng khi so với champion — model, win_m (cột ext sau prune), champion trước, metric per horizon (giá) của cả hai, bảng `RMSE̅` (mean 3 seed) của hai bên và Gain 15 ô, MedianGain/WinRate/P10/Worst, ε_champion, decision **đổi / giữ**, champion sau, exp_id. Kèm `prune_<model>.csv`: F\*_m vs F\*_m^prune (3 seed → `RMSE̅` → Gain 15 ô → MedianGain) → win_m. Ensemble và lựa chọn cuối cũng ghi vào đây.
- `experiments/summary/all_models_test.csv` (TEST; bảng VAL tương đương đọc từ `wins/*.json` + `champion_log.csv`, không có file `all_models.csv` riêng): mọi model (E0, B0-306, B0\*, từng model tại F\*_m, TimesFM các biến thể, AutoTS ×2, LSTM, ensemble) × fold × horizon × {RMSE, MAE, r, dir-acc, Gain vs B0-306, Gain vs B0\*, Gain vs E0, Gain vs champion} + latency p95/p99/max (ms) per model × horizon (§7.4); kèm TEST riêng.

### 7.3 Visualize (theo origin, không vẽ chuỗi dự báo liên tục)

- **Màu**: actual **luôn đen**; E0 xám nét đứt. Ảnh so sánh win vs champion dùng màu theo vai trò — win = blue `#2a78d6` (▲), champion = red `#e34948` (●): cặp xa nhau nhất trong palette. Ảnh nhiều model dùng màu + marker **cố định cho từng model** (palette categorical đã validate bằng validator của skill dataviz: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, yellow `#eda100`, magenta `#e87ba4`, green `#008300`, violet `#4a3aa7`, red `#e34948` — thứ tự cố định, không xoay vòng); tối đa 8 màu mỗi panel, vượt thì tách nhóm; reference B0-306/B0\* xám nét đứt; heatmap diverging xanh↔đỏ, cùng thang màu khi so sánh. Mapping cụ thể: `STYLE` trong `reports/smoke_visualize.py`.

**Sau mỗi model — win_m vs champion hiện tại** (2 file):
- **Fig P — forecast path** (1 ảnh = 3 panel, mỗi panel MỘT origin `t`): trục x = `t, t+1, t+2, t+3`; trục y = **thay đổi giá so với `C_t`** (USD). Trong panel: **actual** `[0, C_{t+1}−C_t, C_{t+2}−C_t, C_{t+3}−C_t]` (đen), **prediction của win** và **của champion** `[0, P̂_{t+h}−C_t]` với `P̂_{t+h} = C_t·exp(ŷ_h)`, **E0 = đường ngang 0**. Ba origin = ba ngày VAL/fold khác nhau đại diện mức biến động **thấp / trung bình / cao** (xếp 5 ngày VAL theo std r1 trong ngày, lấy min / trung vị / max), mỗi ngày lấy **origin cố định đầu tiên ≥ 12:00 UTC** — chọn theo quy tắc cố định, **không** chọn theo error/prediction.
- **Fig HM**: 2 heatmap 15 ô (fold × horizon) — của win và của champion, giá trị = Gain vs E0 tính từ bảng `RMSE̅` mean 3 seed, cùng thang màu; tiêu đề ghi MedianGain/WinRate/P10/Worst của mỗi bên và của win vs champion.
- **Fig T_h — trajectory** (3 ảnh độc lập h = 1, 2, 3): chạy dọc **toàn bộ VAL** (5 fold) theo thời gian, vẽ **giá BTC thô** — `actual_h(t) = C[t+h]` (đen) và `pred_h(t) = C[t]·exp(ŷ_h(t))` của win và của champion; trục x = **timestamp t+h**. Mỗi fold là một đoạn riêng, **không nối đường xuyên qua khoảng trống giữa các fold** (vạch đứt xám ở ranh giới). Dùng để nhìn model có bám mức giá và có lệch pha/regime nào không — bổ sung cho Fig P, không thay thế.
- Lưu `experiments/summary/fig_path_<model>_vs_champion.png`, `fig_HM_<model>_vs_champion.png`, `fig_traj_h{1,2,3}_<model>_vs_champion.png`.

**Final (TEST)**:
- **Heatmap của mọi model** (B0-306, B0\*, mọi win_m, ensemble; một panel mỗi model, cùng thang màu): ô = khối 6 giờ × horizon (TEST 2 ngày ≈ 8 khối; TEST 30 ngày = 120 khối), giá trị Gain vs E0.
- **Fig P của mọi model**: cùng định nghĩa forecast path, vẽ prediction của **tất cả model** trên **cùng 3 origin** TEST — chọn theo std r1 của khối 60 origin không chồng nhau: thấp nhất / trung vị / cao nhất, origin đại diện = origin đầu của khối; tách 2 hàng (nhóm A: tree + ensemble; nhóm B: TimesFM/AutoTS/LSTM + reference) để mỗi panel ≤ 8 màu; actual đen ở mọi panel. Lưu `summary/fig_final_paths_all_models.png`.
- **Fig T_h của mọi model** (3 ảnh, h = 1, 2, 3): cùng định nghĩa trajectory nhưng chạy dọc **toàn bộ TEST**, vẽ prediction của mọi model đang được visualize; cùng cách tách 2 nhóm. Lưu `summary/fig_final_traj_h{1,2,3}_all_models.png`.
- Trajectory chỉ được vẽ ở **đúng hai chỗ** này (sau khi win_m so với champion, và Final) — không vẽ ở calibrate, lọc B0, add-one, prune PI hay bước trung gian nào.
- Fig D latency (§7.4) chỉ để theo dõi.
- Figure chỉ để nhìn; quyết định vẫn theo metric §0. **Từ 2026-09-03 mọi figure trên đều sinh HẬU KỲ** bằng `python run.py visualize` (§7.5), không trong đường chạy training.

### 7.5 Artifact và visualize hậu kỳ (2026-09-03)

- Các lệnh training/search (`calibrate`, `lock-s0`, `loop`, `tfm-final`, `autots-search`, `ensemble`, `final`) **không vẽ**; chúng lưu đủ để dựng lại: `wins/<m>.json` (colset kể cả `locked_b0`/`locked_ext`, RMSE̅, E0, ε, best_iters, seed thật; TimesFM: metadata LoRA/native/covariates) + `wins/<m>_seed<k>.npz` (origin, ŷ log-return theo fold), `calib/<m>_base.json`, `keepdrop_<m>.csv`, `prune_pi_<m>.csv`, `prune_<m>.csv`, `champion_log.csv` (bảng RMSE̅ hai bên + Gain 15 ô), `tfm_final.csv`, `autots_search.csv`, `latency_*.csv`, `lora/<key>.pt + .json` (adapter LoRA), `final/<key>.npz` + `final/index.json` (prediction TEST), `runs/<exp_id>/` (config + prediction từng run). Test khoá bằng AST: hàm `cmd_*` (trừ `cmd_visualize`) không tham chiếu `plots`/`matplotlib`.
- `python run.py visualize --config <cfg> [--out DIR]` (`src/p0/visualize.py`): đọc data (chỉ để lấy actual/giá) + artifact → Fig P / Fig T_h / Fig HM cho từng dòng so champion trong `champion_log.csv`, cặp `tfm_lora_xreg_vs_baseline` (hệ thống B vs hệ thống A), `autots_wr_vs_autots_mr`, và Final (heatmap khối 6h × h, Fig P, Fig T_h mọi model). Không train, không inference, không cần GPU.
- **Git**: không đường dẫn nào dưới `experiments/**` bị ignore; `.gitattributes` đưa `experiments/**/*.{npz,safetensors,pt,pth,png}` vào Git LFS (file thêm mới; 27 `.npz` vòng 15 ngày đã commit dạng blob thường, giữ nguyên). Không commit checkpoint TimesFM gốc — ghi `repo_id`, `revision`, phiên bản môi trường (`experiments/<run>/env.txt`, meta adapter).
- **Data canonical trong repo (rev 10.4)**: hai CSV đi Git LFS và được commit; `check-data` verify sha256 của CHÍNH file trong repo với `data/data_checksums_2y.json`. Sau `git clone` phải chạy `git lfs pull` (nếu quên: file là pointer ~130 byte → `check-data` báo checksum lệch). `derive-lf` vẫn tái lập được LF từ HF ra ĐÚNG sha (công cụ kiểm chứng, không bắt buộc).
- **Artifact bắt buộc khác (rev 10.1)**: `s0/<m>.json` (locked_b0/locked_ext), `s0/candidates_<m>.json` (Candidate_m + overlap + near diagnostic), `s0/collisions.json`, `s0/short_pool.json`, `checker_log.jsonl` (finding PASS/INFO/WARN/ERROR), `final/TEST_SENTINEL.json` (một lần TEST: status, config_hash, checksum data, champion, sha wins), `lora/<key>.pt + .json` (adapter LoRA + meta), **`wins/tfm_lora_baseline.json`** (hệ thống A; tên cũ `tfm_lora_native.json` chỉ ĐỌC được), `wins/tfm_lora_xreg.json` (hệ thống B, có `feature_set_source`), `wins/tfm.json` (TFM-final; metadata LoRA/native/covariates/`lora_adapters`), `scheduler_log.jsonl` + `orchestrate_log.jsonl` + `champion_replay.csv/json` (§7.6, chỉ thực thi). Không đổi tên artifact `experiments/15d/`.

### 7.6 Log thực thi — scheduler và orchestrate (rev 10.3, chỉ quan sát, không ảnh hưởng quyết định)

- `experiments/<run>/scheduler_log.jsonl` — MỘT dòng cho mỗi task GPU: `timestamp_start`, `timestamp_end`, `t_start`/`t_end` (epoch),
  `task_id`, `kind` (`run_fold` | `prune_pi` | `autots_bakeoff` | `autots_score` | `backend_probe` | `probe`), `stage` (calibrate/add_one/prune_pi/confirmation/gpu_probe…),
  `branch`, `model`, `fold`, `seed`, `candidate`, `gpu_physical_id`, `worker_id`, `status`, `duration_sec`, `queue_wait_sec`,
  `peak_vram_mb` (khi đo được), `error`. Dùng để thấy GPU nào bận/rảnh và chỉnh scheduling về sau — **không bao giờ dùng để đổi
  quyết định khoa học**.
- `experiments/<run>/orchestrate_log.jsonl` — trạng thái từng nhánh DAG (`branch`, `deps`, `status`, `duration_sec`, `error`).
- `experiments/<run>/champion_replay.csv` + `champion_replay.json` — thứ tự replay cố định, artifact nguồn, champion trước/sau.
- `python run.py gpu-probe` (rev 10.4) — preflight thiết bị: mỗi worker báo GPU vật lý + `CUDA_VISIBLE_DEVICES` + tên + **UUID**;
  **UUID của các worker phải PHÂN BIỆT** và mọi worker/GPU cấu hình phải được probe (thiếu → dừng + hỏi user); trong TỪNG worker đã
  mask chạy phép tính GPU nhỏ THẬT cho từng backend (torch, XGBoost + kiểm booster device, LightGBM theo `device_type` đã resolve,
  CatBoost, jax, import timesfm — AutoTS thừa hưởng LightGBM/XGBoost); backend đã cài mà không dùng được GPU → `gpu_stop` (§0b.15),
  backend chưa cài → WARN môi trường. Kết quả lưu `experiments/<run>/gpu_probe.json`. Không training, không đọc data, không tải checkpoint.

### 7.4 Inference latency (chỉ theo dõi — không ảnh hưởng training/loss/quyết định)

- **Đo gì**: thời gian gọi `predict` cho **một origin** (batch size 1) → ra `ŷ_h` (và `P̂`). Tree (3 model độc lập theo h): đo riêng từng horizon. Model một lần gọi ra cả 3 bước (LSTM head 3 output, TimesFM, AutoTS `fit_data + predict`): đo một lần gọi, gán chung cho h = 1, 2, 3 và đánh dấu `shared = true`. Chưa gồm thời gian tính feature (pipeline hiện tính feature theo cả frame; latency end-to-end để sau khi có pipeline incremental).
- **Đo khi nào**: pass riêng **sau khi train xong**, chỉ ở run confirmation F\*_m (§2.1c) và Final (§4), trên toàn bộ origin VAL/TEST; không đo trong vòng lặp candidate (ở đó predict theo batch cho nhanh). Pass đo không được thay đổi kết quả: assert prediction theo batch == prediction batch-1 (sai số ≤ 1e-6).
- **Cách đo**: `time.perf_counter_ns` quanh đúng lời gọi predict; model chạy GPU (LSTM/TimesFM, XGBoost nếu predict trên GPU) gọi `torch.cuda.synchronize()`/tương đương trước và sau; bỏ 50 lần gọi đầu (warm-up); số thread = mặc định thư viện (batch 1 không phụ thuộc thread; ghi cột `threads`); ghi train/predict device, phiên bản thư viện, GPU/CPU của instance.
- **Output**: `experiments/runs/<exp_id>/latency.csv` (origin, horizon, ms, shared); tóm tắt `experiments/summary/latency_summary.csv` (model × horizon × {p95, p99, max} ms, VAL và TEST riêng; p50 không cần) và cột p95/p99/max trong `all_models.csv`; ghi `train device` (luôn GPU) và `predict device` thực tế (LightGBM/CatBoost predict trên CPU là đặc tính thư viện; XGBoost/XGB-RF/LSTM/TimesFM predict GPU; AutoTS pipeline CPU quanh regression_model GPU).
- **Không dùng** latency cho KEEP/DROP, champion, ensemble hay bất kỳ quyết định nào trong plan này; chỉ ghi nhận và báo cáo.

---

## 8. Triển khai — trạng thái và lệnh chạy

**Vòng 15 ngày (2026-08-31 → 09-01): đã chạy hết Phase A → B → C → final trên Vast RTX 3090; artifact `experiments/15d/`; kết quả trong MEMORY.** **Vòng expanded-data (2026-09-03): code/config/doc migration xong + pass hiệu chỉnh 2026-09-04 + wiring data 2 năm 2026-09-04b — 150 unit test PASS (CPU, data tổng hợp), `smoke-e2e` synthetic PASS qua lock-s0 → loop → final (sentinel) → visualize; `derive-lf` → `check-data` → `lock-s0` đã chạy trên data 2 năm thật; TRAINING: LOCKED.** `src/p0/`: `data`, `split` (§1.2 + `RollingSpec`/`make_rolling_spread` §1.5, `make_rolling_from_end` rev 10), `features_ext` (39 cột §2.3, lịch sử), **`features_short` (C_short §2.3b)**, **`s0` (S0_m khoá, collision audit, Candidate_m)**, `metrics`, `transform`, `models` (LightGBM/XGBoost/XGB-RF/CatBoost), `models_lstm`, **`models_tfm` (`TimesFMLoRAModel` + `lora.py`)**, `models_autots`, `autots_search`, `harness` (`ColSet` có `locked_b0`/`locked_ext`, `run_config` với fold-parallel, `calibrate`, `seed_noise`), **`checker_log`** (finding không tương tác), **`gpu` + `scheduler` + `fold_parallel`** (scheduler 2 GPU đối xứng §0b.6), **`orchestrate`** (DAG nhánh + champion replay §0b.12), `filter_b0`, `loop` (prune chỉ cột mới, confirmation song song + latency), `latency`, `plots` (định nghĩa figure), **`visualize` (hậu kỳ)**, `logs`, `cli`. Config: `configs/p0_full.json` (vòng mới; `experiments/full`, `prev_run_dir: experiments/15d`, `split` rolling, `gpu_devices: [0, 1]` + `gpu_slots_per_device: 1` + `max_branches: 4` + `defer_champion: true` — chỉ THỰC THI, không vào `config_hash`; `models.tfm.lora`), `configs/p0_15d.json` (lịch sử). Test: `python -m pytest -q` (gồm `tests/test_expanded_round.py`, `tests/test_tfm_lora.py`); smoke: `python run.py smoke-e2e --out tmp_smoke --days 6`. Prompt Vast: `docs/VAST_SESSION_PROMPT.md`; bootstrap: `scripts/vast_bootstrap.sh`; môi trường: `requirements.txt`.

Ràng buộc trong CLI: (i) `calibrate / filter-b0 / loop / tfm-final / autots-search / final` từ chối khi `.claude/MEMORY.md` còn `TRAINING: LOCKED`, và preflight GPU trước khi train; (ii) `--smoke` / `--allow-cpu` chỉ được chấp nhận khi `dataset_label` bắt đầu bằng `synthetic`; (iii) mọi bước sau `check-data` verify sha256 của CSV với file checksum của config (`checksums`); (iv) `loop` đầu tiên phải là `lgbm`; (v) `loop` cần `experiments/<run>/s0/<m>.json` + `candidates_<m>.json` (từ `lock-s0`); (vi) `tfm-final` cần cả `wins/tfm_lora_baseline.json` (tên cũ `tfm_lora_native.json` vẫn đọc được) và `wins/tfm_lora_xreg.json` (metadata LoRA, không B0\*, `feature_set_source` chứng minh F_win đến từ confirmation, `lora_adapters` của hai hệ thống trùng nhau); (vii) LF 5' phải phủ HF; (viii) `final` từ chối khi `final/TEST_SENTINEL.json` đã tồn tại (chỉ `--force-test-rerun` recovery vượt, ghi WARN); (ix) mọi từ chối ở (i)–(viii) là `checker_log.hard_fail` — ghi ERROR rồi dừng, không hỏi user; `loop` từ chối khi `candidates_<m>.json` audit trên dataset khác config.

**Quy trình vận hành (rev 10.4)**: `git clone` → `git lfs install && git lfs pull` (đã có CẢ `data/BTC_1m_2y.csv` lẫn
`data/BTC_5m_2y.csv`) → `bash scripts/vast_bootstrap.sh` → `python run.py gpu-probe` → `check-data` → `lock-s0` →
agent `checker` (preflight) → **USER UNLOCK** → `python run.py orchestrate` (agent `run-monitor` theo dõi) →
`champion-replay` (thứ tự cố định) → `ensemble` → agent `analyst` (kết quả VAL) → agent `checker` (trước Final) →
`final` (TEST một lần) → `visualize` → `analyst` tổng kết. Agent `researcher` KHÔNG nằm trong đường này.

Lệnh theo bước trên Vast (vòng expanded-data; máy 2 × RTX 5000 Ada: `export P0_GPU_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false` — KHÔNG đặt `P0_FOLD_WORKERS` trừ khi cố ý oversubscribe; một GPU: `export P0_GPU_DEVICES=0`):

1. `git lfs install && git lfs pull` (data 2 năm đã nằm trong repo — KHÔNG scp) → `bash scripts/vast_bootstrap.sh` → `scripts/vast_canary.py` + `scripts/canary_xreg_gpu.py` → `python run.py gpu-probe --config configs/p0_full.json` (worker 0 → GPU vật lý 0, worker 1 → GPU vật lý 1, **UUID phải khác nhau**, mọi backend đã cài phải chạy được GPU) → (tuỳ chọn kiểm chứng) `python run.py derive-lf --config configs/p0_full.json --force` phải tái lập ĐÚNG sha `0e5fb9ad…` → `python run.py check-data --config configs/p0_full.json` (verify anchor `data/data_checksums_2y.json` đã commit; in 5 fold `rolling_spread` + final như bảng §1.5) → `python run.py lock-s0 --config configs/p0_full.json` (S0_m, Candidate_m = 163, `s0/collisions.json`, audit label = dataset 2 năm) → `pytest -q -x` → `scripts/checker_record.py --exp experiments/full --blocking` sạch ERROR.
2. **Một lệnh (khuyến nghị trên máy 2 GPU)**: `python run.py orchestrate --config configs/p0_full.json` — chạy DAG nhánh (loop lgbm/xgb/cat/tfm/xgbrf/autots_wr/autots_mr/lstm chạy song song theo GPU rảnh; `tfm-final` ngay khi nhánh tfm xong; `autots-search` ngay khi CẢ hai probe AutoTS xong) → `champion-replay` (thứ tự cố định, chỉ đọc artifact) → `ensemble`. **Không chạm TEST.** `--dry-run` in DAG; `--models a,b` giới hạn nhánh.
   **Hoặc từng bước (tương đương, chạy tuần tự)**: `loop --model lgbm` → `xgb` → `cat` → `tfm` → `tfm-final` → `loop --model xgbrf` → `autots_wr` → `autots_mr` → `autots-search` → `loop --model lstm` → `champion-replay` (khi `defer_champion: true`). Mỗi `loop` = calibrate riêng trên S0_m (số vòng/epoch/ε mới) → add-one Candidate_m → prune PI cột mới → confirmation 3 seed → win_m → latency → artifact đại diện. Không vẽ.
3. `python run.py ensemble` (§3, nếu chưa chạy trong orchestrate) → `python run.py final` (§4: TEST một lần → `summary/all_models_test.csv`, `final/*.npz`, `latency_summary.csv`; **chạy tuần tự trên một GPU, không qua scheduler — TEST một lần, tối ưu thực thi không đáng đánh đổi rủi ro**) → `python run.py visualize` (§7.5: mọi figure).
4. Sau mỗi model: cập nhật MEMORY, commit + push (experiments/** tracked, LFS). §5.3 scale data để sau.

## 9. Đã bỏ / đổi so với plan 2026-08-24 (và so với rev 9b — mục 2026-09-03)

- **2026-09-04d (rev 10.4, VẬN HÀNH — không đổi một luật khoa học nào)**: data 2 năm ngoài git (scp/derive) → **hai CSV canonical trong repo qua Git LFS** (`git clone` + `git lfs pull` là đủ; `derive-lf` thành công cụ kiểm chứng); thêm audit `git check-ignore` cho `experiments/**` + test; TimesFM khoá cách gọi (mọi cấu hình = HỆ THỐNG HOÀN CHỈNH `TimesFM-LoRA + XReg(F)`, confirmation là hai hệ thống hoàn chỉnh trên cùng adapter, cấm "XReg vs XReg"/"XReg vs LoRA"); TFM-final và AutoTS-final **LƯU rồi CHỜ** champion replay; `max_branches` 2 → **4** (GPU đồng thời vẫn 2 vì `gpu_slots_per_device: 1`); **sự cố tài nguyên GPU = ngoại lệ DUY NHẤT được hỏi user** (`checker_log.gpu_stop`, ERROR `ref=USER_DECISION_REQUIRED`, exit 3, không CPU fallback, không đổi tham số) trong khi vi phạm bất biến khoa học vẫn dừng im lặng; `gpu-probe` kiểm UUID phân biệt + probe backend THẬT trong từng worker đã mask (lưu `gpu_probe.json`); agent chuyển pha vận hành (thêm `run-monitor`, `analyst` hậu-run, `researcher` dormant).
- **2026-09-04c (rev 10.3, CHỈ THỰC THI — không đổi một luật khoa học nào)**: fold-parallel một pool chung → **scheduler 2 GPU đối xứng** (`gpu.py`/`scheduler.py`; worker khoá GPU vật lý bằng `CUDA_VISIBLE_DEVICES`, hàng đợi task sẵn sàng, round-robin giữa nhánh, 1 task nặng/GPU, không affinity ML/DL, không CPU fallback); chạy tuần tự từng model → **orchestrate DAG** nhánh độc lập song song; champion so ngay trong `loop`/`tfm-final`/`autots-search` → **`champion-replay` hoãn, thứ tự cố định, chỉ đọc artifact**; TimesFM `tfm_lora_native` → **`tfm_lora_baseline`** (tên cũ vẫn đọc) và mọi mô tả đổi thành **hai hệ thống hoàn chỉnh A vs B** (A = LoRA baseline feature-free, B = LoRA + XReg(F_win) sau confirmation raw-vs-pruned); thêm `scheduler_log.jsonl` / `orchestrate_log.jsonl` / `champion_replay.*`, lệnh `gpu-probe`. Bất biến khoa học (target, feature, S0, Candidate_m, thứ tự candidate, KEEP/DROP, PI, confirmation, seed, ε, hyperparameter, split, TEST, metric, luật champion/ensemble) giữ NGUYÊN 100 %.
- **2026-09-04 (rev 10.1)**: S0 khoá TOÀN BỘ tường minh (`locked_b0`/`locked_ext`); bỏ lọc toàn cục C_short theo B0-306/candidate cũ và bỏ mọi việc xoá theo tương quan (bản 2026-09-03 đã ghi sai rằng "user bỏ Keltner ngắn vì corr ≥ 0.999999" — không đúng; tương quan cao chỉ báo cáo) → `Candidate_m = C_short \\ overlap(C_short, S0_m)` per model; C_short 97 → **163** (dày đủ, thêm Keltner, PSAR cửa sổ reset, log_rv_k_med2d, r5_2/3, log_c5_ema5_2/3; dow ngoại lệ); TimesFM artifact `tfm_native`/`tfm_xreg` → `tfm_lora_native`/`tfm_lora_xreg` + metadata, calibrate = LoRA FIT + ES; GPU-only và TEST-một-lần ép trong code (sentinel, không hỏi); checker không tương tác → `checker_log.jsonl`.
- **2026-09-03 (rev 10)**: điểm xuất phát mỗi model B0\* → **S0_m khoá** (B0\* ∪ F_old_m từ artifact 15 ngày); candidate 39 cột §2.3 → **C_short 97 cột ≤ 15'** (§2.3b), candidate cũ không quay lại; TimesFM hai nhánh zero-shot (`tfm_b0`/`tfm_ext`) → **một đường TimesFM-LoRA → freeze → XReg search → tfm-final (+XReg vs native)**; fold-parallel chỉ TimesFM → **mọi model** (`fold_parallel.py`); figure sau mỗi model/Final trong đường chạy → **`visualize` hậu kỳ**; `experiments/runs/`, `cache/`, `bootstrap.log` ignore → **experiments/** tracked + LFS**; split 15 ngày calendar → **rolling neo cuối data** (§1.5) với checksum riêng; artifact vòng 15 ngày chuyển vào `experiments/15d/`.
- HOLDOUT-NEAR/FAR → TEST 2 ngày (15 ngày) và TEST 30 ngày (data đầy đủ — vòng hiện tại §1.5).
- P0 gate / canonical-pilot framework → kiểm tra §1.1; giai đoạn hiện tại chạy trên snapshot 15 ngày theo quyết định user.
- "Không ablate lại 306 feature B0" → đổi: lọc nhiễu B0 một lần bằng PI + standalone + MI + kiểm chứng (§1.4) thành B0\*; file `Baseline_LGBM.py` vẫn không sửa; B0-306 vẫn log làm reference.
- Feature dossier, Wave-1, D-family discovery, second wave → danh sách §2.3 thử lần lượt, từng model một.
- Một feature set chung → mỗi model một feature set riêng F\*_m.
- KEEP/DROP 3 vùng, daily-block paired test, confirmation framework → luật §2.1 với ε_m seed; safety-net đã bỏ, chỉ prune PI; confirmation 3 seed = mean RMSE từng ô → Gain 15 ô prune vs unprune → MedianGain ≥ −ε_m chọn prune → win_m.
- Metric trên log return → metric trên giá (USD); prediction vẫn log return.
- Scale data → để sau (§5).
- TimesFM ladder QMEAN/RECENTER/BTC-CAL → TFM-POINT (+ covariate loop nếu API có) → LoRA khi thắng E0.
- AutoTS: WR/MR cố định = **probe dò feature** (mỗi cái add-one → prune → confirmation → F_WR_best / F_MR_best) → framework AutoTS chạy **riêng cho từng bộ** (`max_generations=0`, template GPU do ta khai báo) → AutoTS-final. **Stage union (`F_WR ∪ F_MR`) đã bỏ.** Genetic self-search bỏ vì không thể ép GPU-only.
- (rev 9b — đã thay bằng rev 10, xem mục 2026-09-03 ở trên) TimesFM: **hai nhánh** (từ B0\* và từ ∅) chạy song song cùng một protocol §2.1, so nhau bằng metric project → TimesFM-final. Bỏ cách chọn "một chiến lược covariate duy nhất" (`b0star_full` / `b0star_subset` / `ext_only`) — không còn freeze `ext_only` như quyết định cuối.
- ExtraTrees (sklearn, CPU-only) → XGB-RF trên GPU [đã chốt]; training chỉ GPU, cấm CPU training.
- Số vòng cố định dùng chung từ B0-306 → mỗi model calibrate riêng `15fixed_m` trên B0\* trước vòng lặp của nó (`15fixed_306` chỉ cho lọc B0); B0\* là điểm xuất phát chung, không model nào kế thừa F\* của model khác.
- Q1–Q15, yfinance, paper 2407.18334 → bỏ.
- "True VWAP không có" → sửa: `amount/volume` là VWAP thật theo trade trong bar; chỉ biến thể từ TP·V mới cần tên `proxy`.
