"""Chính sách thiết bị GPU (§9 hiệu chỉnh 2026-09-04c) — THỰC THI, không đổi khoa học.

Máy thí nghiệm mới = **2 × RTX 5000 Ada 32 GB độc lập** (không gộp 64 GB). Hai GPU là worker ĐỐI XỨNG:
không GPU nào bị gán vai trò "ML" hay "DL", không model family nào bị pin vào một GPU. Việc phân công là
động (task sẵn sàng → GPU rảnh) và do `p0.scheduler` quyết định.

Cơ chế định tuyến thiết bị (quyết định 2026-09-04c, §17 của yêu cầu):
mỗi worker process được cấp ĐÚNG một GPU vật lý bằng `CUDA_VISIBLE_DEVICES=<physical>` đặt TRƯỚC khi import
bất kỳ thư viện CUDA nào → bên trong worker, `cuda:0` / `device="cuda"` / `task_type="GPU"` / `device_type=cuda|gpu`
đều trỏ đúng GPU vật lý đó. Đây là cơ chế DUY NHẤT mà mọi backend của project cùng tôn trọng
(LightGBM CUDA lẫn OpenCL của NVIDIA, XGBoost, CatBoost, PyTorch, JAX/TimesFM, LightGBM/XGBoost bên trong AutoTS),
nên không đi đường "truyền device id khác nhau cho từng thư viện".

Không có CPU fallback: `require_gpu` mà worker không thấy đúng 1 GPU → lỗi rõ ràng, task fail, không tự đổi thiết bị.
"""
from __future__ import annotations

import os


class GpuResourceError(RuntimeError):
    """Sự cố TÀI NGUYÊN GPU (không có GPU, GPU biến mất, backend không train được trên GPU, routing sai, OOM,
    worker CUDA chết). Đây là TÌNH HUỐNG DUY NHẤT được phép DỪNG AN TOÀN VÀ HỎI USER (quyết định user 2026-09-04d,
    §10) — không bao giờ tự chuyển CPU, không tự đổi batch/hyperparameter/methodology.

    Vi phạm bất biến KHOA HỌC (checksum, leakage, biên target, artifact S0 malformed, TEST lần hai) KHÔNG dùng lớp
    này: chúng vẫn hard-fail tự động, không hỏi (thí nghiệm không hợp lệ, không phải lựa chọn tài nguyên)."""


# dấu hiệu lỗi TÀI NGUYÊN GPU trong thông điệp của thư viện/worker (không dùng để đoán lỗi khoa học)
GPU_FAILURE_PATTERNS = (
    "cuda out of memory", "out of memory", "oom", "cuda error", "cudaerror", "no cuda-capable device",
    "cuda driver", "cuda unavailable", "cuda không", "không có cuda", "cuda_visible_devices", "nvml",
    "gpu preflight", "device=cuda", "cpu fallback", "no kernel image", "insufficient driver",
    "worker process chết", "worker gpu không khởi động", "gpu vật lý", "xla", "cublas", "cudnn", "nvrtc",
)


def is_gpu_failure(message) -> bool:
    """True khi thông điệp lỗi mang dấu hiệu SỰ CỐ TÀI NGUYÊN GPU (→ dừng an toàn + hỏi user, §10)."""
    m = str(message).lower()
    return any(pat in m for pat in GPU_FAILURE_PATTERNS)


ENV_DEVICES = "P0_GPU_DEVICES"           # "0,1" — danh sách GPU vật lý dùng làm worker
ENV_SLOTS = "P0_GPU_SLOTS_PER_DEVICE"    # số task nặng đồng thời trên MỖI GPU (mặc định 1)
ENV_WORKERS = "P0_FOLD_WORKERS"          # override tổng số worker (tương thích ngược §9 bản 2026-09-03)
ENV_PHYSICAL = "P0_GPU_PHYSICAL_ID"      # worker tự ghi: GPU vật lý của chính nó (đọc để log/kiểm tra)
ENV_WORKER_ID = "P0_GPU_WORKER_ID"


def _parse_devices(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = [s for s in str(raw).replace(";", ",").split(",")]
    out = []
    for s in items:
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if len(set(out)) != len(out):
        raise ValueError(f"{ENV_DEVICES}: GPU trùng nhau trong danh sách: {out}")
    return out


def resolve_devices(cfg=None) -> tuple[list[int], int]:
    """(devices, slots_per_device). Ưu tiên env → config → mặc định 1 GPU ([0], 1 slot = hành vi cũ, §16 yêu cầu).

    Mặc định KHÔNG BAO GIỜ oversubscribe: một task nặng / một GPU vật lý.
    """
    devices = _parse_devices(os.environ.get(ENV_DEVICES))
    if not devices:
        devices = _parse_devices(getattr(cfg, "gpu_devices", None))
    if not devices:
        devices = [0]
    slots = os.environ.get(ENV_SLOTS)
    if slots is None:
        slots = getattr(cfg, "gpu_slots_per_device", 1) or 1
    slots = max(1, int(slots))
    return devices, slots


def worker_slots(cfg=None) -> tuple[list[int], int, int]:
    """(devices, slots, n_workers). `P0_FOLD_WORKERS` (nếu đặt) ghi đè TỔNG số worker — oversubscribe là lựa chọn
    TƯỜNG MINH của user, được ghi WARN ở scheduler, không bao giờ tự động."""
    devices, slots = resolve_devices(cfg)
    n = len(devices) * slots
    env = os.environ.get(ENV_WORKERS)
    if env is not None:
        try:
            n = max(1, int(env))
        except ValueError:
            pass
    elif cfg is not None and not _parse_devices(os.environ.get(ENV_DEVICES)) and not _parse_devices(getattr(cfg, "gpu_devices", None)):
        n = max(1, int(getattr(cfg, "fold_workers", 1) or 1))  # config cũ chỉ có fold_workers (1 GPU)
    return devices, slots, n


def device_for_worker(worker_id: int, devices: list[int]) -> int:
    """Worker → GPU vật lý, chia vòng tròn. Không phụ thuộc model/family (yêu cầu §6: GPU đối xứng)."""
    return int(devices[worker_id % len(devices)])


def bind_worker_device(worker_id: int, device: int) -> dict:
    """Gọi Ở ĐẦU worker process, TRƯỚC khi import torch/xgboost/lightgbm/catboost/jax.

    Sau lệnh này process chỉ nhìn thấy MỘT GPU (index 0 = `device` vật lý) → mọi backend chạy đúng GPU được giao.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # index ổn định giữa các process/lần chạy
    os.environ[ENV_PHYSICAL] = str(device)
    os.environ[ENV_WORKER_ID] = str(worker_id)
    return {"worker_id": int(worker_id), "gpu_physical_id": int(device), "cuda_visible_devices": str(device)}


def device_report(require_gpu: bool = True) -> dict:
    """Kiểm tra THẬT rằng process này chỉ thấy đúng GPU được giao. Trả metadata để ghi scheduler_log."""
    rep = {"gpu_physical_id": int(os.environ.get(ENV_PHYSICAL, -1)), "worker_id": int(os.environ.get(ENV_WORKER_ID, -1)),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "torch_device_count": None,
           "device_name": None, "device_uuid": None, "backend": None}
    try:
        import torch

        rep["torch_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            rep["device_name"] = str(props.name)
            rep["device_uuid"] = str(getattr(props, "uuid", "") or "")
            rep["backend"] = "torch"
    except Exception as e:  # torch chưa cài (worker chỉ chạy tree model): KHÔNG xác nhận được uuid/số GPU thấy được
        rep["torch_error"] = f"{type(e).__name__}: {e}"  # → `gpu-probe` sẽ báo WARN vì thiếu bằng chứng uuid
    if require_gpu:
        n = rep["torch_device_count"]
        if n is not None and n != 1:
            raise RuntimeError(f"worker {rep['worker_id']}: thấy {n} GPU (mong đúng 1 = GPU vật lý {rep['gpu_physical_id']}) — "
                               f"CUDA_VISIBLE_DEVICES={rep['cuda_visible_devices']!r}; không có CPU fallback.")
    return rep


BACKENDS = ("torch", "xgboost", "lightgbm", "catboost", "jax", "timesfm")


def backend_probe(backends=BACKENDS, lgbm_device_type: str = "cuda") -> dict:
    """Chạy MỘT phép tính GPU nhỏ THẬT cho từng backend, NGAY TRONG worker đã mask (§11).

    Trả {backend: {"status": ok | fail | missing, "detail": ...}}. `ok` = thư viện đã cài VÀ chạy được trên GPU
    được giao; `fail` = đã cài nhưng KHÔNG dùng được GPU (→ sự cố tài nguyên GPU, §10); `missing` = chưa cài
    (vấn đề môi trường/bootstrap, không phải sự cố GPU). KHÔNG train thật, không tải checkpoint.
    """
    import numpy as np

    out: dict[str, dict] = {}
    x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
    for b in backends:
        try:
            if b == "torch":
                import torch

                if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                    raise RuntimeError(f"torch.cuda.is_available()={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
                t = torch.ones(1024, device="cuda")
                val = float((t * 2).sum().item())
                props = torch.cuda.get_device_properties(0)
                out[b] = {"status": "ok", "detail": f"{props.name} | uuid={getattr(props, 'uuid', '')} | sum={val:.0f}"}
            elif b == "xgboost":
                import xgboost as xgb

                if not bool(xgb.build_info().get("USE_CUDA", False)):
                    raise RuntimeError("wheel XGBoost không build CUDA (USE_CUDA=False)")
                m = xgb.XGBRegressor(n_estimators=3, device="cuda", tree_method="hist")
                m.fit(x, x[:, 0])
                import json as _json

                dev = str(_json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"])
                if not dev.startswith("cuda"):
                    raise RuntimeError(f"booster báo device={dev!r} ≠ cuda")
                out[b] = {"status": "ok", "detail": f"booster device={dev}"}
            elif b == "lightgbm":
                import lightgbm as lgb

                m = lgb.LGBMRegressor(n_estimators=8, num_leaves=7, max_bin=63, device_type=lgbm_device_type,
                                      gpu_use_dp=False, verbose=-1, n_jobs=1)
                m.fit(x, x[:, 0])
                out[b] = {"status": "ok", "detail": f"device_type={lgbm_device_type} fit OK"}
            elif b == "catboost":
                from catboost import CatBoostRegressor

                CatBoostRegressor(iterations=3, task_type="GPU", devices="0", verbose=False, allow_writing_files=False).fit(x, x[:, 0])
                out[b] = {"status": "ok", "detail": "task_type=GPU devices=0 fit OK"}
            elif b == "jax":
                import jax
                import jax.numpy as jnp

                devs = jax.devices()
                plat = {str(d.platform) for d in devs}
                if not plat & {"gpu", "cuda"}:
                    raise RuntimeError(f"jax.devices() = {devs} (không phải GPU)")
                val = float(jnp.ones(1024).sum())
                out[b] = {"status": "ok", "detail": f"{devs[0].device_kind} | sum={val:.0f}"}
            elif b == "timesfm":
                import timesfm  # noqa: F401  — chỉ kiểm đã cài đúng version; KHÔNG tải checkpoint trong probe
                from importlib import metadata as _md

                out[b] = {"status": "ok", "detail": f"timesfm {_md.version('timesfm')} (import OK, không tải checkpoint)"}
        except ImportError as e:
            out[b] = {"status": "missing", "detail": f"{type(e).__name__}: {e}"}
        except Exception as e:  # đã cài nhưng KHÔNG chạy được trên GPU → sự cố tài nguyên GPU
            out[b] = {"status": "fail", "detail": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def peak_vram_mb(reset: bool = True) -> float | None:
    """Peak VRAM (MB) của task vừa chạy nếu đo được: torch (allocator) hoặc pynvml (device). None = không đo được."""
    val = None
    try:
        import torch

        if torch.cuda.is_available():
            val = float(torch.cuda.max_memory_allocated()) / 1e6
            if reset:
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        val = None
    if not val:
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)  # đã mask → index 0 = GPU được giao
            val = float(pynvml.nvmlDeviceGetMemoryInfo(h).used) / 1e6
        except Exception:
            return val
    return val
