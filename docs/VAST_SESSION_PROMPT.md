# Vast — tiếp tục sau HF BLOCKED, tìm data thay thế và FULL TRAINING

Bản này thay prompt khởi động HF cũ. Không yêu cầu tải lại archive đã thất bại.
Trong Claude, mở `/goal` và dán nội dung `docs/VAST_GOAL.txt` vào goal condition (giới hạn 4.000 ký tự).
Tài liệu này là chỉ dẫn đầy đủ, không phải nội dung để dán cả file vào goal condition.

## Điểm xuất phát và đích đến

User đi ngủ. Session chính tiếp tục tự chủ trên instance Vast hiện có, không chờ user duyệt từng bước:
**giữ công việc → sửa W1 → tìm/chọn data thay thế → download/prepare thật → FULL TRAINING → summary/report/backup**.

Theo báo cáo Vast user cung cấp: e169df1 trên OB chưa push; HF đã tải đủ 4 file nhưng chỉ reconstruct được
300,6 giây, FIT/VAL rỗng, chưa train cell nào. Đọc reports/commit thật để xác nhận trạng thái đang tiếp nối.
Không coi package đã cài hoặc checker không có ERROR là bằng chứng toàn bộ code/fit đã đúng.

Đọc `.claude/CLAUDE.md`, `.claude/MEMORY.md`, `.claude/AGENT.md`, `.claude/agents/checker.md`,
`src_OB/README.md`, config, DATA_REPORT/RUN_REPORT/CHECKER_FINDINGS và code liên quan.
Quyết định mới ở đây thay quy định HF-only hoặc yêu cầu hỏi lại khi đổi nguồn trong context cũ.
Giữ `src/p0`, `Baseline_LGBM.py`, `experiments/15d` và kết quả cũ. Không dùng workflow OHLCV/run.py/docs/archive.

## Quyền tự làm và cách vận hành

Được tìm/chọn nguồn public miễn phí hoặc nguồn project đã được cấp quyền dùng; pin revision mới;
viết downloader/adapter/config riêng; sửa lỗi correctness/env/API; prepare và training thật; recovery;
ghi artifact, commit/push OB. Không hỏi unlock hoặc duyệt giữa những bước này.
Không tự mua data, đăng ký trả phí/trial cần thanh toán, liên hệ nhà cung cấp, thuê/đổi/stop/destroy instance.
Dùng auth đã có đúng scope, không in/commit secret. Thiếu key thì xét nguồn khác, không chờ user giữa đêm.

Session chính hoàn thiện data/code/env và chạy CLI. CLI tự lặp model/fold/horizon, không dựng agent điều phối.
Checker chỉ đọc evidence sau prepare, khi có lỗi correctness cụ thể và cuối run; không train/infer/prepare,
sửa code hay gọi subagent. Session chính xử lý findings, không coi WARN là miễn xử lý correctness.
Nếu không có subagent capability thì đọc cùng checklist và ghi rõ thiếu independent review.

Cấm mọi smoke/canary/unit/integration test, synthetic run, trial fit, GPU-probe fit, benchmark pass, warmup riêng.
Không pytest, scripts/vast_bootstrap.sh, run.py gpu-probe hoặc workflow cũ. Được đọc code/data/log/metadata,
xem process/nvidia-smi và artifact lượt chạy thật. Guard runtime sequence/causality/GPU vẫn bắt buộc.
Training chỉ trên Vast, fit GPU-only, cấm CPU fallback. CPU dùng cho IO/reconstruction/features/scaler/metric
và native inference LightGBM/CatBoost theo policy hiện tại. Không fit thử để chọn backend.

## 1. Giữ công việc và môi trường đang có

Giữ e169df1, các commit tiếp theo và thay đổi chưa commit. Không reset/reclone đè hoặc force push.
Push bằng auth sẵn có. Remote OB tiến thêm thì fetch/merge phù hợp, giữ công việc cả hai phía.
Nếu thiếu auth, tạo Git bundle và gói reports/config/log/metadata/artifact cần phục hồi, kèm manifest/hash.
Bundle không chứa untracked files hoặc nội dung LFS: sao lưu riêng artifact và LFS objects chưa push.
Chuyển tới đích ngoài instance của user đã được cấu hình/cấp quyền, xác nhận đích nhận khớp manifest.
Không upload repo lên dịch vụ công cộng tùy ý hoặc phân phối raw trái điều kiện nguồn.
Bundle nằm trên cùng ổ Vast chưa phải backup. Thiếu đích/auth thì ghi BACKUP_PENDING và tiếp tục việc khác.
Không tự tắt instance; cuối run báo máy còn chạy và trạng thái backup. Lưu định kỳ trong quá trình làm.

Reuse venv GPU đã build nếu phù hợp; giữ fix LightGBM shared NCCL để tránh lặp lỗi ABI cũ.
Không cài JAX/XReg. Ghi code/config/data/GPU/driver/package provenance vào run mới.

## 2. Sửa W1 từ evidence thật

Đọc finding W1 về buffer depth trước snapshot và replay/schema liên quan. Sửa nguyên nhân nếu xác nhận lỗi.
Đồng bộ snapshot với buffered messages theo update IDs và semantics nguồn; không phát state quá khứ từ
snapshot tương lai, không replay trùng, không nối gap. Ghi rõ xử lý timestamps/bridge và evidence.
Giữ prepared cũ; fix ảnh hưởng kết quả thì tạo prepared version mới, không sửa manifest giả tương thích.
Checker đọc bản sửa, không chạy test. Không prepare lại HF chỉ để lặp bằng chứng blocker đã có.

## 3. Tự chọn nguồn historical thay thế

HF MaximumLeverage/crypto-lob-stream SHA 873f31e729ae23b1c309cd5dcb33feed27c407de là evidence nguồn lỗi,
không còn là nguồn duy nhất bắt buộc. Không sửa raw, tải mirror của cùng archive như nguồn mới, hoặc giả rằng
nâng collector khôi phục được depth đã bị ghi đè.

Chỉ BTCUSDT Binance Spot L2. Ưu tiên snapshot anchor + incremental depth liên tục đủ reconstruct top 10.
Full snapshots thật chỉ dùng khi tần suất/semantics đáp ứng age/context; ghi OF là flow quan sát giữa snapshots,
không giả là đầy đủ message-level flow. Không dùng futures/exchange/symbol khác, OHLCV/trades-only/L1/synthetic.
Không nối nhiều nguồn để che missing sequence; source boundary là segment boundary.

Tự tìm bằng tài liệu/metadata/schema và dữ liệu thật. Nguồn trả phí chỉ dùng nếu project đã có quyền truy cập.
Ghi SOURCE_REPORT.md: URL, revision/manifest/hash, access/license, schema/time semantics, coverage/gaps,
dung lượng và lý do chọn/loại từng ứng viên. Dùng config/raw/prepared/output riêng, không trộn với HF lỗi.
Đọc metadata/dung lượng trước tải; ingest theo chunk/partition nếu cần, không làm đầy disk hoặc xóa kết quả lấy chỗ.
Không cần đủ 2 năm. 30d/58d chỉ là ngưỡng lịch theo split cho 1/5 fold, không đảm bảo đủ origins/context.

Nếu sau khoảng 60 phút tìm nguồn chủ động chưa có ứng viên phù hợp, ghi các nguồn đã xét và blocker.
Giới hạn này không ngắt download/prepare/training đang tiến triển. Không tìm vô hạn, chờ upstream qua đêm
hoặc thu mới nhiều tuần để gọi là giải quyết historical run này. Có nguồn phù hợp thì chuyển ngay sang chạy thật.
Không có nguồn/access/tài nguyên phù hợp: làm hết W1/report/backup có thể làm, rồi kết thúc với BLOCKED có evidence.

## 4. Giữ phương pháp khi đổi nguồn

- Historical data cố định trong run, pin manifest/revision; actual coverage, không bịa/pad lịch sử.
- 10 bid + 10 ask, một mid `(bestBid + bestAsk)/2`. Depth là diff: quantity tuyệt đối, 0 xóa;
  gom trọn message, kiểm tra U/u, prune đúng semantics nguồn rồi lấy top 10; không giữ ghost levels.
- ID/timestamp gap hoặc book invalid đóng segment, chờ snapshot hợp lệ, không nối/forward-fill qua thiếu data.
  Hard gap 2026-07-05 20:56–21:39 UTC thuộc collector HF lỗi, không áp mù lên nguồn độc lập có continuity
  được chứng minh. Ghi source-specific gap policy rõ ràng trong config/report.
- OF theo price/quantity giữa states; OFI = bidOF - askOF cho 10 levels. OF trước same-mid drop và cộng flow
  đến mid-change; giữ raw timestamp/mid timeline trước drop. Baseline OF/OFI + timing, không distances.
- Direct h60/120/180 giây, một model/adapter mỗi fold/horizon. Label log(MP(t+h)/MP(t)), quote cuối <=t+h,
  age guard theo config; feature/window/label cùng segment. FIT21d/gap6d (>5)/VAL3d/step7d, tối đa5 fold.
  > Cập nhật 2026-09-11 (user quyết định trong session): nguồn Zenodo 21 ngày dùng FIT 9d/gap 1d/VAL 2d/step 2d,
  > tối đa 5 fold; gap chỉ cần > horizon dài nhất (180 s). Config HF giữ FIT21d/gap6d.
- Common origins đủ mọi family/horizon. TimesFM512 tại h180 cần 25h33 context liên tục trước origin,
  cộng label. Không giảm context/gap/batch/epoch/search budget, đổi seed/split/target/h để ép data chạy được.
- Đủ lgbm,xgb,cat,xgbrf,lstm,autots,tfm_zero_shot,tfm_lora. Không thêm DeepLOB hoặc feature search.
  AutoTS tự search trong GPU allowlist. TimesFM zero-shot chỉ mid; LoRA + OF head, không XReg.
- RMSE/MAE/R² raw price. rmse_gain_vs_e0 và r2_os_vs_e0 reuse E0 cùng fold/horizon/origins;
  denominator0 thì null có lý do. Không metric trên log-return hoặc tạo baseline thay thế.
- Latency trong inference thật: batch1 p95/p99/max quan sát, batch stats riêng, CUDA synchronization quanh
  timing; không thêm inference benchmark/warmup pass, không gọi observed max là hard bound.

## 5. Download/prepare thật rồi FULL TRAINING ngay

Dùng config nguồn đã chọn và downloader/entrypoint đúng adapter; ghi lệnh thực tế. Không dùng mặc định HF lỗi.
Chạy download rồi prepare thật. DATA_REPORT ghi actual coverage, raw/kept states, segment distribution,
gap/reset reasons, eligible FIT/VAL origins và số fold bằng đúng hàm chọn origins của train.
Dùng data-report nếu commit Vast đã thêm, không giả CLI có lệnh/flag chưa tồn tại.
Prepared_dir tồn tại thì đọc manifest/status, chỉ reuse khi complete và contract tương thích; không xóa chạy đè.

Checker đọc correctness và masks. Có ít nhất một fold hợp lệ cho đầy đủ family/horizon thì train ngay,
không dừng chờ user xác nhận hoặc viết plan mới. Thay <run_config> bằng đường dẫn config thực:

```bash
export P0_OB_VAST=1
export CUDA_VISIBLE_DEVICES=0
python -m src_OB train --config <run_config>
python -m src_OB summarize --config <run_config>
```

Chỉ chạy summarize sau train thành công. CLI tự lặp mọi cell; giữ tmux/log để SSH/context đổi không mất run.
Một process nặng/GPU, không cộng VRAM nhiều GPU; nếu chia fold trên GPU được cấp thì các tập cell phải rời nhau.
Actual fit lỗi thì lưu traceback/artifact, sửa env/API/implementation rồi tiếp tục, không fallback CPU/bỏ model
hoặc lặp nguyên lệnh lỗi vô hạn. Không dừng chỉ vì data xong, một model xong, thời gian dài hay compact.

Recovery: đọc CLI thật, không giả định --resume/--horizons. Được bổ sung recovery theo cell với provenance:
giữ failed attempts, reuse completed khi code/config/data/population tương thích, chạy cell chưa hoàn tất.
Checkpoint thiếu optimizer không gọi là resume chính xác. Fix correctness làm artifact cũ invalid thì lưu riêng,
ghi cell phải chạy lại. Không train lại cell thành công chỉ để bổ sung latency hoặc trộn các config/revision.

## 6. Kết quả, backup và điều kiện kết thúc

COMPLETE chỉ khi >=1 fold đủ FIT/VAL và đủ actual_folds × 8 family × 3 horizon. Không coi 0 cell là thành công.
Mỗi cell có completed.json, metrics.json/metrics.csv, predictions.parquet, latency.json/inference_latency.csv
và checkpoint/adapter phù hợp; zero-shot ghi pretrained ID/revision. Summary có per_fold_per_horizon.csv và
by_model_horizon.csv với price metrics/E0 gains/latency. Checker đọc nội dung và population/provenance,
không chỉ đếm folder hoặc exit code, không infer/fit lại.

Lưu SOURCE_REPORT, DATA_REPORT, CHECKER_FINDINGS, RUN_REPORT, BACKUP_STATUS trong output run.
Báo coverage/segments/origins/cells, kết quả, runtime, GPU/env/code/config/data provenance, lỗi đã sửa,
giới hạn, cell thiếu, vị trí backup xác nhận hoặc BACKUP_PENDING. Cập nhật MEMORY và exact next step,
run/config/process trước compact, rồi tiếp tục. Commit/push scope OB, LFS cho file lớn có quyền phân phối;
không git add -A, force push, commit secrets/pretrained gốc hay ignore metrics/predictions/log/checkpoint mới.

Nếu bị chặn thật sau khi làm hết phần được phép: báo BLOCKED có evidence, ứng viên đã loại, việc đã xong,
cell thiếu, backup đã ở đâu hoặc chưa chuyển được, và quyết định tối thiểu cần user khi quay lại.
Không hỏi user giữa các việc độc lập, không báo COMPLETE giả. Không tự tắt instance; báo trạng thái máy rõ ràng.
