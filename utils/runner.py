from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

import config
from utils.device import get_device_name
from utils.metrics import measure_inference
from utils.modeling import load_artifacts, precision_supported
from utils.profiling import capture_profile
from utils.prompts import build_inputs
from utils.quality import evaluate_quality


@dataclass(frozen=True)
class BenchmarkCondition:
    experiment: str
    variant: str
    device: str
    batch_size: int
    seq_label: str
    seq_length: int
    precision: str
    compiled: bool
    measure_quality: bool
    capture_profile: bool


def _stringify_notes(notes: list[str]) -> str:
    return " | ".join(note for note in notes if note)


def _cleanup_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_conditions(
    conditions: list[BenchmarkCondition],
    settings,
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    artifact_cache: dict[tuple[str, str, bool], object] = {}

    for condition in conditions:
        if not precision_supported(condition.device, condition.precision):
            continue

        cache_key = (condition.device, condition.precision, condition.compiled)
        if cache_key not in artifact_cache:
            artifact_cache[cache_key] = load_artifacts(
                settings=settings,
                device=condition.device,
                precision=condition.precision,
                compiled=condition.compiled,
            )
        artifacts = artifact_cache[cache_key]
        inputs = build_inputs(
            tokenizer=artifacts.tokenizer,
            batch_size=condition.batch_size,
            seq_length=condition.seq_length,
            device=condition.device,
        )

        print(
            "  "
            f"{condition.experiment}: device={condition.device}, "
            f"batch={condition.batch_size}, seq={condition.seq_length}, "
            f"precision={condition.precision}, compiled={artifacts.compiled}"
        )

        notes = list(artifacts.notes)
        try:
            measurement = measure_inference(
                model=artifacts.model,
                inputs=inputs,
                device=condition.device,
                batch_size=condition.batch_size,
                seq_length=condition.seq_length,
                warmup_runs=settings.warmup_runs,
                measure_runs=settings.measure_runs,
            )
        except Exception as exc:
            if condition.compiled:
                notes.append(f"Compiled execution failed at runtime: {exc}")
                fallback_artifacts = load_artifacts(
                    settings=settings,
                    device=condition.device,
                    precision=condition.precision,
                    compiled=False,
                )
                artifacts = fallback_artifacts
                inputs = build_inputs(
                    tokenizer=artifacts.tokenizer,
                    batch_size=condition.batch_size,
                    seq_length=condition.seq_length,
                    device=condition.device,
                )
                measurement = measure_inference(
                    model=artifacts.model,
                    inputs=inputs,
                    device=condition.device,
                    batch_size=condition.batch_size,
                    seq_length=condition.seq_length,
                    warmup_runs=settings.warmup_runs,
                    measure_runs=settings.measure_runs,
                )
            else:
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment": condition.experiment,
                        "variant": condition.variant,
                        "device": condition.device,
                        "device_name": get_device_name(condition.device),
                        "model_name": artifacts.model_name,
                        "model_source": artifacts.model_source,
                        "batch_size": condition.batch_size,
                        "seq_label": condition.seq_label,
                        "seq_length": condition.seq_length,
                        "precision": condition.precision,
                        "compiled": artifacts.compiled,
                        "requested_compiled": condition.compiled,
                        "warmup_runs": settings.warmup_runs,
                        "measure_runs": settings.measure_runs,
                        "mean_latency_ms": float("nan"),
                        "std_latency_ms": float("nan"),
                        "samples_per_sec": float("nan"),
                        "tokens_per_sec": float("nan"),
                        "peak_memory_mb": float("nan"),
                        "avg_memory_mb": float("nan"),
                        "quality_loss": float("nan"),
                        "quality_perplexity": float("nan"),
                        "profile_trace_path": "",
                        "profile_summary_path": "",
                        "notes": _stringify_notes(
                            list(artifacts.notes) + [f"Benchmark failed: {exc}"]
                        ),
                    }
                )
                continue

        row: dict[str, object] = {
            "run_id": run_id,
            "experiment": condition.experiment,
            "variant": condition.variant,
            "device": condition.device,
            "device_name": get_device_name(condition.device),
            "model_name": artifacts.model_name,
            "model_source": artifacts.model_source,
            "batch_size": condition.batch_size,
            "seq_label": condition.seq_label,
            "seq_length": condition.seq_length,
            "precision": condition.precision,
            "compiled": artifacts.compiled,
            "requested_compiled": condition.compiled,
            "warmup_runs": settings.warmup_runs,
            "measure_runs": settings.measure_runs,
            "mean_latency_ms": measurement.mean_latency_ms,
            "std_latency_ms": measurement.std_latency_ms,
            "samples_per_sec": measurement.samples_per_sec,
            "tokens_per_sec": measurement.tokens_per_sec,
            "peak_memory_mb": measurement.peak_memory_mb,
            "avg_memory_mb": measurement.avg_memory_mb,
            "quality_loss": float("nan"),
            "quality_perplexity": float("nan"),
            "profile_trace_path": "",
            "profile_summary_path": "",
            "notes": _stringify_notes(notes),
        }

        if condition.measure_quality:
            try:
                quality = evaluate_quality(
                    model=artifacts.model,
                    tokenizer=artifacts.tokenizer,
                    device=condition.device,
                    prompts=config.QUALITY_PROMPTS,
                    seq_length=settings.quality_seq_length,
                )
                row.update(quality)
            except Exception as exc:
                row["notes"] = _stringify_notes(
                    [row["notes"], f"Quality evaluation failed: {exc}"]
                )

        if condition.capture_profile:
            try:
                profile_output_dir = (
                    settings.profiles_dir
                    / run_id
                    / condition.experiment
                )
                profile_paths = capture_profile(
                    model=artifacts.model,
                    inputs=inputs,
                    device=condition.device,
                    output_dir=profile_output_dir,
                    label=condition.variant,
                )
                row.update(profile_paths)
            except Exception as exc:
                row["notes"] = _stringify_notes(
                    [row["notes"], f"Profiler capture failed: {exc}"]
                )

        rows.append(row)

    for device, _, _ in artifact_cache:
        _cleanup_device(device)

    return pd.DataFrame(rows)
