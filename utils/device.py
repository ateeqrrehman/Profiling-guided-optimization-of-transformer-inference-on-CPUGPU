from __future__ import annotations

import platform

import psutil
import torch


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_devices(requested: str) -> list[str]:
    available = get_available_devices()
    if requested == "auto":
        return available
    if requested not in available:
        raise ValueError(
            f"Requested device '{requested}' is not available. Available devices: {available}"
        )
    return [requested]


def get_environment_summary() -> dict[str, object]:
    summary: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers_device_count": len(get_available_devices()),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary["gpu_name"] = props.name
        summary["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
        summary["cuda_version"] = torch.version.cuda
        summary["bf16_supported"] = torch.cuda.is_bf16_supported()
    return summary


def get_device_name(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
