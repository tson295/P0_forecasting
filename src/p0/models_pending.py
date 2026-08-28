"""TimesFM (§2.2 #4) và AutoTS (§2.2 #6): chưa implement — plan yêu cầu `researcher` audit version/API trước khi code.

Giao diện dự kiến khi audit xong:
- tfm: input chuỗi r1 kết thúc tại t (context 512 hoặc tối đa API), zero-shot → r̂_{t+1..t+3} → cộng dồn ŷ_h;
  covariate loop nếu API có (giá trị cho 3 bước = giữ giá trị tại t); LoRA chỉ khi TFM-POINT thắng E0.
- autots_wr / autots_mr: WindowRegression (regression_model LightGBM, GPU) / MultivariateRegression (XGBoost, GPU);
  base regressor = B0*, candidate làm regressor (giá trị dùng để dự báo bar s chỉ từ ≤ s−1); rolling-origin fit_data + predict.
"""
from __future__ import annotations


class PendingModel:
    supports_rounds = False
    train_device = "GPU"

    def __init__(self, name: str):
        self.name = name

    def fit_predict(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name}: chưa implement — cần researcher audit API/version (plan §2.2) và ghi docs/reference/audit_<lib>.md trước."
        )


def pending(name: str) -> PendingModel:
    return PendingModel(name)
