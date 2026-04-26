from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Recommendation:
    objective: str
    workload_device: str
    workload_batch_size: int
    workload_seq_length: int
    used_exact_match: bool
    candidate_count: int
    recommendation: dict[str, object]
    baseline: dict[str, object] | None
    notes: list[str]


def _default_results_path() -> Path:
    return Path("results/raw/latest_all_results.csv")


def _load_results(results_path: str | None) -> pd.DataFrame:
    path = Path(results_path) if results_path else _default_results_path()
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def _select_candidates(
    frame: pd.DataFrame,
    device: str,
    batch_size: int,
    seq_length: int,
) -> tuple[pd.DataFrame, bool]:
    exact = frame[
        (frame["device"] == device)
        & (frame["batch_size"] == batch_size)
        & (frame["seq_length"] == seq_length)
    ]
    if not exact.empty:
        return exact.copy(), True

    device_only = frame[frame["device"] == device].copy()
    if device_only.empty:
        raise ValueError(f"No rows found for device '{device}'.")

    device_only["distance"] = (
        (device_only["batch_size"] - batch_size).abs()
        + (device_only["seq_length"] - seq_length).abs() / max(seq_length, 1)
    )
    min_distance = device_only["distance"].min()
    return device_only[device_only["distance"] == min_distance].copy(), False


def _score_candidates(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    aggregated = (
        frame.groupby(["precision", "compiled"], as_index=False)
        .agg(
            mean_latency_ms=("mean_latency_ms", "mean"),
            tokens_per_sec=("tokens_per_sec", "mean"),
            peak_memory_mb=("peak_memory_mb", "mean"),
            avg_memory_mb=("avg_memory_mb", "mean"),
            quality_perplexity=("quality_perplexity", "mean"),
            profile_summary_path=("profile_summary_path", "first"),
        )
    )

    if objective == "latency":
        aggregated["score"] = -aggregated["mean_latency_ms"]
    elif objective == "throughput":
        aggregated["score"] = aggregated["tokens_per_sec"]
    elif objective == "memory":
        aggregated["score"] = -aggregated["peak_memory_mb"]
    else:
        cols = ["mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]
        numeric = aggregated[cols].astype(float)
        normalized = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        aggregated["score"] = (
            -normalized["mean_latency_ms"]
            + normalized["tokens_per_sec"]
            - normalized["peak_memory_mb"]
        )

    return aggregated.sort_values("score", ascending=False).reset_index(drop=True)


def _read_top_ops(profile_summary_path: str) -> list[str]:
    if not profile_summary_path:
        return []
    path = Path(profile_summary_path)
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty or "op_name" not in frame.columns:
        return []
    return frame["op_name"].head(3).tolist()


def recommend_configuration(
    results_path: str | None,
    device: str,
    batch_size: int,
    seq_length: int,
    objective: str,
) -> Recommendation:
    frame = _load_results(results_path)
    candidates, exact = _select_candidates(frame, device, batch_size, seq_length)
    ranked = _score_candidates(candidates, objective)
    best = ranked.iloc[0].to_dict()

    baseline_rows = ranked[
        (ranked["precision"] == "fp32")
        & (ranked["compiled"] == False)
    ]
    baseline = baseline_rows.iloc[0].to_dict() if not baseline_rows.empty else None

    notes = []
    if not exact:
        notes.append("No exact workload match was found, so the selector used the nearest measured condition.")

    top_ops = _read_top_ops(str(best.get("profile_summary_path", "")))
    if top_ops:
        notes.append(f"Representative top operators: {', '.join(top_ops)}")

    if baseline:
        latency_delta = baseline["mean_latency_ms"] - best["mean_latency_ms"]
        memory_delta = baseline["peak_memory_mb"] - best["peak_memory_mb"]
        if best["precision"] != "fp32" and (latency_delta > 0 or memory_delta > 0):
            notes.append("Reduced precision looks beneficial for this workload, which suggests memory pressure matters.")
        elif best["compiled"] and latency_delta > 0:
            notes.append("Compilation improves latency here, which suggests runtime overhead is significant.")
        else:
            notes.append("The workload appears to benefit from a mixed balance of runtime and memory behavior.")

    return Recommendation(
        objective=objective,
        workload_device=device,
        workload_batch_size=batch_size,
        workload_seq_length=seq_length,
        used_exact_match=exact,
        candidate_count=len(ranked),
        recommendation=best,
        baseline=baseline,
        notes=notes,
    )


def format_recommendation(recommendation: Recommendation) -> str:
    best = recommendation.recommendation
    lines = [
        f"Objective: {recommendation.objective}",
        (
            "Workload: "
            f"device={recommendation.workload_device}, "
            f"batch_size={recommendation.workload_batch_size}, "
            f"seq_length={recommendation.workload_seq_length}"
        ),
        (
            "Recommended configuration: "
            f"precision={best['precision']}, compiled={bool(best['compiled'])}"
        ),
        (
            "Expected metrics: "
            f"latency={best['mean_latency_ms']:.2f} ms, "
            f"throughput={best['tokens_per_sec']:.2f} tokens/s, "
            f"peak_memory={best['peak_memory_mb']:.2f} MB"
        ),
    ]

    if recommendation.baseline:
        baseline = recommendation.baseline
        lines.append(
            "Baseline comparison: "
            f"fp32 eager latency={baseline['mean_latency_ms']:.2f} ms, "
            f"throughput={baseline['tokens_per_sec']:.2f} tokens/s, "
            f"peak_memory={baseline['peak_memory_mb']:.2f} MB"
        )

    if recommendation.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in recommendation.notes)

    return "\n".join(lines)
