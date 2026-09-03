"""LoRA tối giản, tự chứa (W·x + (α/r)·B·A·x) cho module torch bất kỳ + vòng train theo epoch với ES trên ES set.

Dùng cho TimesFM (§2.2 #4, quyết định user 2026-09-03: pretrained → LoRA fine-tune trên chuỗi r1 → freeze adapter →
XReg feature search trên CHÍNH adapter đó). Không phụ thuộc `peft`: audit `docs/reference/audit_timesfm_lora.md` chốt
tên module đích; hàm `inject_lora` thay đúng các `nn.Linear` đó bằng `LoRALinear` (base frozen, chỉ A/B học).

Vai trò dữ liệu (bất biến §6.4): `train_lora` chỉ nhận cửa sổ FIT để cập nhật trọng số và cửa sổ ES để chọn epoch;
VAL/TEST không bao giờ đi qua đây. Trạng thái adapter = dict tensor A/B (+ meta) — lưu bằng `torch.save` (.pt, LFS).
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np


def _torch():
    import torch

    return torch


class LoRALinear:  # nn.Module thật được tạo ở runtime (tránh import torch khi chỉ đọc module)
    pass


def _make_cls():
    torch = _torch()
    nn = torch.nn

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError(f"LoRA chỉ bọc nn.Linear, nhận {type(base).__name__}")
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r, self.scaling = int(r), float(alpha) / float(r)
            self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=base.weight.dtype, device=base.weight.device))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.reset_lora()

        def reset_lora(self, generator=None):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)  # như PEFT
            nn.init.zeros_(self.lora_B)  # B = 0 → lúc khởi tạo model == pretrained

        def forward(self, x):
            return self.base(x) + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    global LoRALinear
    LoRALinear = _LoRALinear
    return _LoRALinear


def inject_lora(module, target_names, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Thay các `nn.Linear` có tên đầy đủ kết thúc bằng một trong `target_names` (ví dụ "qkv_proj", "attn.o_proj") bằng LoRALinear.
    Trả danh sách tên đã thay; không thay được gì → lỗi (tên đích sai so với audit)."""
    torch = _torch()
    cls = _make_cls()
    targets = tuple(target_names)
    replaced = []
    for name, sub in list(module.named_modules()):
        if not isinstance(sub, torch.nn.Linear) or isinstance(sub, cls):
            continue
        if not any(name == t or name.endswith("." + t) or name.endswith(t) for t in targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, cls(sub, r, alpha, dropout))
        replaced.append(name)
    if not replaced:
        raise KeyError(f"inject_lora: không tìm thấy nn.Linear nào khớp {targets}")
    return replaced


def freeze_except_lora(module) -> tuple[int, int]:
    """requires_grad chỉ cho lora_A/lora_B. Trả (số tham số học, tổng)."""
    n_train = n_all = 0
    for name, p in module.named_parameters():
        train = ("lora_A" in name) or ("lora_B" in name)
        p.requires_grad_(train)
        n_all += p.numel()
        n_train += p.numel() if train else 0
    return n_train, n_all


def lora_parameters(module):
    return [p for n, p in module.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(module) -> dict:
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters() if "lora_A" in n or "lora_B" in n}


def load_lora_state_dict(module, sd: dict) -> None:
    own = {n: p for n, p in module.named_parameters() if "lora_A" in n or "lora_B" in n}
    missing = sorted(set(own) - set(sd))
    extra = sorted(set(sd) - set(own))
    if missing or extra:
        raise KeyError(f"adapter không khớp module: thiếu {missing[:3]}, thừa {extra[:3]}")
    with _torch().no_grad():
        for n, p in own.items():
            p.copy_(sd[n].to(p.device, p.dtype))


def state_sha256(sd: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(sd):
        h.update(n.encode())
        h.update(np.ascontiguousarray(sd[n].detach().cpu().float().numpy()).tobytes())
    return h.hexdigest()


def reset_lora(module, seed: int) -> None:
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for sub in module.modules():
        if hasattr(sub, "reset_lora") and hasattr(sub, "lora_A"):
            with torch.no_grad():
                a = torch.empty_like(sub.lora_A, device="cpu")
                torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=g)
                sub.lora_A.copy_(a.to(sub.lora_A.device))
                sub.lora_B.zero_()


def train_lora(forward_fn: Callable, module, X_fit: np.ndarray, Y_fit: np.ndarray, X_es: np.ndarray | None, Y_es: np.ndarray | None,
               *, epochs: int | None, max_epochs: int = 20, patience: int = 5, lr: float = 1e-4, batch_size: int = 64,
               seed: int = 0, device: str = "cuda", weight_decay: float = 0.0, log: Callable[[str], None] | None = None) -> dict:
    """Vòng train LoRA. forward_fn(x: Tensor(B, L)) → Tensor(B, H) = dự báo one-step (mean head) H bước; loss = MSE.

    epochs=None → early stopping theo MSE trên ES (patience, ≤ max_epochs), trả best_epoch (calibrate/confirmation);
    epochs=k → train đúng k epoch, không nhìn ES (`fixed_epoch`, §1.3). Seed điều khiển khởi tạo A và thứ tự batch.
    Chỉ tham số LoRA được cập nhật (`freeze_except_lora`)."""
    torch = _torch()
    torch.manual_seed(int(seed))
    reset_lora(module, seed)
    n_train, n_all = freeze_except_lora(module)
    params = lora_parameters(module)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    dev = torch.device(device)
    xf = torch.as_tensor(np.asarray(X_fit, np.float32), device=dev)
    yf = torch.as_tensor(np.asarray(Y_fit, np.float32), device=dev)
    xe = torch.as_tensor(np.asarray(X_es, np.float32), device=dev) if X_es is not None and epochs is None else None
    ye = torch.as_tensor(np.asarray(Y_es, np.float32), device=dev) if Y_es is not None and epochs is None else None
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n_epochs = int(epochs) if epochs is not None else int(max_epochs)

    def es_loss() -> float:
        module.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(xe), batch_size):
                pb = forward_fn(xe[s:s + batch_size])
                tot += float(torch.mean((pb - ye[s:s + batch_size]) ** 2).item()) * len(pb)
                cnt += len(pb)
        return tot / max(cnt, 1)

    best_state, best_loss, best_epoch, bad = None, float("inf"), 0, 0
    curve = []
    for epoch in range(1, n_epochs + 1):
        module.train()
        perm = torch.randperm(len(xf), generator=gen).to(dev)
        tr_tot, tr_cnt = 0.0, 0
        for s in range(0, len(perm), batch_size):
            sel = perm[s:s + batch_size]
            pred = forward_fn(xf[sel])
            loss = torch.mean((pred - yf[sel]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_tot += float(loss.item()) * len(sel)
            tr_cnt += len(sel)
        row = {"epoch": epoch, "train_mse": tr_tot / max(tr_cnt, 1)}
        if epochs is None:
            cur = es_loss()
            row["es_mse"] = cur
            if cur < best_loss - 1e-12:
                best_loss, best_epoch, bad = cur, epoch, 0
                best_state = lora_state_dict(module)
            else:
                bad += 1
                if bad >= patience:
                    curve.append(row)
                    break
        else:
            best_epoch = epoch
        curve.append(row)
        if log:
            log(f"epoch {epoch}: " + ", ".join(f"{k}={v:.3e}" for k, v in row.items() if k != "epoch"))
    if best_state is not None:
        load_lora_state_dict(module, best_state)
    module.eval()
    sd = lora_state_dict(module)
    return {"best_epoch": int(best_epoch), "curve": curve, "state": sd, "sha256": state_sha256(sd),
            "n_trainable": int(n_train), "n_params": int(n_all), "mode": "es" if epochs is None else f"fixed{int(epochs)}"}
