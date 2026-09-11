PHASE: OB — Vast 2026-09-11: HF pinned BLOCKED; user chọn Zenodo 20046390 (21 ngày) + fold FIT9/gap1/VAL2/step2
TRAINING: COMPLETE — Zenodo 120/120 cell (5 fold × 8 family × 3h); AutoTS = adapter v2 (log-return quanh origin, user
duyệt); checker ZF + ZA không ERROR. Không còn quyết định mở.

## Kết quả (có bằng chứng; chi tiết `experiments/orderbook_zenodo/RUN_REPORT.md`)

- Config `configs/orderbook_zenodo.json`; raw `data/orderbook/zenodo_20046390/` (tar md5 `58507a0f…`, không commit);
  prepared `data/orderbook/prepared_zenodo_20046390/` (replay 2, code c68fc45, LFS); output `experiments/orderbook_zenodo/`.
- Không family nào thắng E0 nhất quán (13/120 cell gain > 0). tfm_zero_shot gần E0 nhất theo pooled (+0,002/+0,014 ở
  h120/h180, mean theo fold vẫn âm). AutoTS v2 thứ hai theo pooled (−0,004/−0,044/−0,014) nhưng nhiều model gần hằng số
  (ZA-I2); fold5 h120 −0,334 (ZA-I3). Cây OF/OFI gần E0 (xgbrf/cat tốt nhất h60). LSTM, LoRA tệ hơn E0 mọi cell (LoRA
  overfit). Latency p95 batch 1: LSTM 0,34 ms, cat/lgbm ~0,8, xgb ~1,9, AutoTS ~2,7, xgbrf ~2,9, zero-shot ~82, LoRA ~95.
- AutoTS v1 (giá thô, code 2a5a1c3, RMSE 1,8–38× E0) giữ ở `experiments/orderbook_zenodo/superseded/autots_raw_price/`.
- Sự cố đã sửa (giữ phương pháp): LightGBM GOSS CUDA crash + `linear_tree` âm thầm fit CPU (attempt1, bị bỏ) →
  688fefd; SIGSEGV max_bin 1000 → 2a5a1c3; pandas 3 từ chối weighted sample của AutoTS NewGeneticTemplate → shim 6108a1d.
  `train --resume` có từ 688fefd; attempts ở `experiments/orderbook_zenodo/attempts/`.
- Train code theo cell: d8797c5 (15), 2a5a1c3 (90 non-AutoTS), 37aae35 (7 AutoTS v2), 6108a1d (8 AutoTS v2).
- Backup: toàn bộ push `origin/OB` (xem `experiments/orderbook_zenodo/BACKUP_STATUS.md`).
- Figures hậu kỳ: `python -m src_OB visualize --config configs/orderbook_zenodo.json` → `experiments/orderbook_zenodo/figures/`
  (30 ảnh path TimesFM ZS/LoRA + AutoTS v2, 2 heatmap). `docs/visualize.txt` không tồn tại trong repo lúc làm (2026-09-11);
  làm theo yêu cầu session + quy ước RESEARCH_PLAN §7.3 (RUN_REPORT §8).

## Nếu có việc tiếp theo

Không có yêu cầu mở. Mọi lượt chạy mới: config/output riêng hoặc `--resume` (không ghi đè completed), launcher
`experiments/orderbook_zenodo/run_meta/run_train.sh [--models …] [--resume]` (giữ `HF_HOME`, `P0_OB_VAST=1`,
`CUDA_VISIBLE_DEVICES=0`), commit code trước khi chạy, checker sau run. Không smoke/test/trial fit.

## Môi trường Vast (C.50503596, RTX 3090 24 GB, driver 595.84, CUDA 12.8, không volume)

- Venv `/home/ubuntu/venv-ob` (`experiments/orderbook_hf/run_meta/install_env.sh`; LightGBM 4.7.0 CUDA cần
  `BUILD_WITH_SHARED_NCCL=ON`). HF_HOME phải là `/home/ubuntu/.cache/huggingface` (global trỏ `/workspace/.hf_home` của
  root). pandas 3.0.5 + AutoTS 1.0.4 cần shim weighted sample trong `src_OB/autots_native.py`.
- Push: SSH key, `remote.origin.pushurl` = SSH; commit với `-c user.name/-c user.email`; stage theo scope.

## HF (đóng, BLOCKED — chỉ tham khảo)

- HF `873f31e7…`: depth ≈300 s/giờ (collector v1 lỗi), segment tốt nhất 300,6 s, 0 origin. Replay v2 đã sửa (W1),
  chưa prepare thật trên HF; R3-I1/R3-I2 còn mở (chỉ diff replay). Reports `experiments/orderbook_hf/*.md`,
  nguồn đã xét `SOURCE_REPORT.md` (27 nhóm).

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook_zenodo.json` (run hiện tại), `configs/orderbook.json` (HF), `.claude/CLAUDE.md`,
`.claude/AGENT.md`, `.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`.
Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
