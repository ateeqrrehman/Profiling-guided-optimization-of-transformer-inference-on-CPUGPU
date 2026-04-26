from __future__ import annotations

import math
import statistics
import subprocess
import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class BenchmarkMeasurement:
    latencies_sec: list[float]
    mean_latency_ms: float
    std_latency_ms: float
    samples_per_sec: float
    tokens_per_sec: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float


def synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
    outputs = model(**inputs)
    _ = outputs.logits[:, -1, :1]
    return outputs


def _read_gpu_utilization_percent() -> float:
    """Best-effort GPU utilization query using nvidia-smi.

    This is intentionally optional because not every environment exposes
    nvidia-smi, especially CPU-only machines, some notebooks, or restricted
    cloud runtimes. Returning NaN is better than failing the benchmark suite.
    """
    if not torch.cuda.is_available():
        return math.nan

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        first_value = completed.stdout.strip().splitlines()[0]
        return float(first_value)
    except Exception:
        return math.nan


def measure_inference(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
    seq_length: int,
    warmup_runs: int,
    measure_runs: int,
) -> BenchmarkMeasurement:
    process = psutil.Process()

    # Warm-up runs are deliberately separated from measured runs. This avoids
    # first-run artifacts from kernel initialization, cache effects, and compile
    # setup from contaminating steady-state timing.
    for _ in range(warmup_runs):
        with torch.inference_mode():
            run_forward(model, inputs)
        synchronize(device)

    latencies_sec: list[float] = []
    memory_samples: list[float] = []
    cpu_util_samples: list[float] = []
    gpu_util_samples: list[float] = []

    # Prime psutil's cpu_percent measurement so later samples are meaningful.
    psutil.cpu_percent(interval=None)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(measure_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            run_forward(model, inputs)
        synchronize(device)
        latencies_sec.append(time.perf_counter() - start)

        cpu_util_samples.append(psutil.cpu_percent(interval=None))
        gpu_util_samples.append(_read_gpu_utilization_percent())

        if device == "cuda" and torch.cuda.is_available():
            memory_samples.append(torch.cuda.memory_allocated() / 1e6)
        else:
            memory_samples.append(process.memory_info().rss / 1e6)

    if device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_memory_mb = max(memory_samples) if memory_samples else 0.0

    mean_latency = statistics.mean(latencies_sec)
    std_latency = statistics.stdev(latencies_sec) if len(latencies_sec) > 1 else 0.0

    valid_gpu_samples = [value for value in gpu_util_samples if not math.isnan(value)]

    return BenchmarkMeasurement(
        latencies_sec=latencies_sec,
        mean_latency_ms=mean_latency * 1000.0,
        std_latency_ms=std_latency * 1000.0,
        samples_per_sec=batch_size / mean_latency if mean_latency else math.nan,
        tokens_per_sec=(batch_size * seq_length) / mean_latency if mean_latency else math.nan,
        peak_memory_mb=peak_memory_mb,
        avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0.0,
        cpu_utilization_percent=statistics.mean(cpu_util_samples) if cpu_util_samples else math.nan,
        gpu_utilization_percent=statistics.mean(valid_gpu_samples) if valid_gpu_samples else math.nan,
    )
