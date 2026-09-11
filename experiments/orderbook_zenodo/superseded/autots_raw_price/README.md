# AutoTS hồi quy giá thô: 15 cell đã bị thay thế

Thư mục này chứa 15 cell AutoTS gốc của run Zenodo (fold1–fold5 × h60/h120/h180). Các cell được chuyển nguyên trạng
bằng `git mv` (lịch sử giữ nguyên), không xoá. Chúng nằm ngoài glob `fold*/*/h*s` của summary, nên không vào bảng
kết quả.

- **Code:** adapter AutoTS v1, train code `2a5a1c3` (`run.json` → `train_code`).
- **Kết quả gốc:** commit `40b4c45`, tại `experiments/orderbook_zenodo/fold{1..5}/autots/h{60,120,180}s/`.
- **Lý do thay thế:** v1 đưa chuỗi mức giá thô trên grid h giây vào `WindowRegression` (`normalize_window=False`,
  `scale=False`), với target là giá thô tại t+h. Model cây vì vậy bị kẹp quanh các mức giá dày đặc của FIT và không
  ngoại suy được. RMSE bằng 1,8–38× E0; gain trung bình theo fold so với E0 là −15,40 / −11,61 / −9,78 (h60/h120/h180).
  Chi tiết: `RUN_REPORT.md` §3b, checker ZF-W1.
- **Quyết định:** user duyệt ngày 2026-09-11 đổi sang adapter v2. Mỗi cửa sổ được center theo giá origin của chính
  nó: X = log(P_window) − log(P_origin), target log(MP(t+h)/MP(t)). 15 cell mới nằm lại ở vị trí chuẩn
  `fold{1..5}/autots/h*s/`.
- **Attempt cũ:** `experiments/orderbook_zenodo/attempts/fold1/autots/h60s/attempt{1,2}` cũng thuộc adapter v1
  (các lần chạy lỗi trước khi sửa GPU allowlist) và được giữ nguyên tại chỗ.

Không gộp các cell này với kết quả v2 vào cùng một bảng.
