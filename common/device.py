"""Device and mixed-precision helpers that behave the same on a laptop, Colab and Kaggle."""

import contextlib
from collections.abc import Mapping

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """cuda if available, then Apple mps, then cpu. Pass prefer="cpu" to force one."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: str | torch.device) -> torch.dtype | None:
    """bf16 on Ampere or newer (A100, RTX 30xx+), fp16 on older GPUs (T4, P100, GTX 16xx), None off-GPU.

    Checks the compute capability directly because torch.cuda.is_bf16_supported() also
    says yes to emulated bf16, which is slower than plain fp32 on a T4.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device: str | torch.device, dtype: torch.dtype | None = None, enabled: bool = True):
    """torch.autocast with the right dtype, or a no-op context where AMP does not apply."""
    device = torch.device(device)
    dtype = dtype or amp_dtype(device)
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def grad_scaler(device: str | torch.device, dtype: torch.dtype | None = None) -> torch.amp.GradScaler:
    """A GradScaler that is only switched on for fp16.

    bf16 and fp32 get a disabled scaler, so the same scaler.scale(loss).backward() /
    scaler.step(optimizer) / scaler.update() loop works everywhere.
    """
    device = torch.device(device)
    dtype = dtype or amp_dtype(device)
    return torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)


def to_device(batch, device: str | torch.device, non_blocking: bool = True):
    """Move a tensor, or any dict / list / tuple of tensors, to device. Other values pass through."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {key: to_device(value, device, non_blocking) for key, value in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return type(batch)(*(to_device(value, device, non_blocking) for value in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(value, device, non_blocking) for value in batch)
    if hasattr(batch, "to"):  # e.g. a Hugging Face BatchEncoding
        return batch.to(device)
    return batch
