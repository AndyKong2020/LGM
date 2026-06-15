from contextlib import nullcontext

import torch


def get_torch_device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, dtype=torch.float16):
    if device.type in {"cuda", "npu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()
