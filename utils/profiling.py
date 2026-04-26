from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from utils.metrics import run_forward, synchronize


def _sanitize_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)


def capture_profile(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: str,
    output_dir: Path,
    label: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)
    trace_path = output_dir / f"{safe_label}_trace.json"
    summary_path = output_dir / f"{safe_label}_ops.csv"

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            run_forward(model, inputs)
        synchronize(device)

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception:
        trace_path = Path("")

    rows = []
    for event in prof.key_averages():
        rows.append(
            {
                "op_name": event.key,
                "count": event.count,
                "self_cpu_time_ms": event.self_cpu_time_total / 1000.0,
                "cpu_time_ms": event.cpu_time_total / 1000.0,
                "self_cuda_time_ms": getattr(event, "self_cuda_time_total", 0.0) / 1000.0,
                "cuda_time_ms": getattr(event, "cuda_time_total", 0.0) / 1000.0,
                "cpu_memory_mb": event.cpu_memory_usage / 1e6,
                "cuda_memory_mb": getattr(event, "cuda_memory_usage", 0.0) / 1e6,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_column = "cuda_time_ms" if "cuda_time_ms" in frame.columns and device == "cuda" else "cpu_time_ms"
        frame = frame.sort_values(sort_column, ascending=False)
    frame.to_csv(summary_path, index=False)

    return {
        "profile_trace_path": str(trace_path) if str(trace_path) else "",
        "profile_summary_path": str(summary_path),
    }
