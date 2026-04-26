from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def cpu_vs_gpu_analysis(df: pd.DataFrame) -> str:
    baseline = df[df["experiment"] == "baseline"]
    if baseline.empty or not {"cpu", "cuda"}.issubset(set(baseline["device"])):
        return "CPU vs GPU comparison not available."

    cpu = baseline[baseline["device"] == "cpu"][["seq_label", "mean_latency_ms"]]
    gpu = baseline[baseline["device"] == "cuda"][["seq_label", "mean_latency_ms"]]
    merged = cpu.merge(gpu, on="seq_label", suffixes=("_cpu", "_gpu"))

    lines = ["CPU vs GPU Analysis:"]
    for _, row in merged.iterrows():
        speedup = row["mean_latency_ms_cpu"] / row["mean_latency_ms_gpu"]
        lines.append(
            f"- {row['seq_label']}: GPU is {speedup:.2f}x faster than CPU"
        )
    return "\n".join(lines)


def batch_analysis(df: pd.DataFrame) -> str:
    batch = df[df["experiment"] == "batch_size"]
    if batch.empty:
        return "Batch analysis not available."

    best = batch.sort_values("tokens_per_sec", ascending=False).iloc[0]
    return (
        "Batch Size Analysis:\n"
        f"- Best throughput at batch={best['batch_size']} on {best['device']}"
    )


def sequence_analysis(df: pd.DataFrame) -> str:
    seq = df[df["experiment"] == "sequence_length"]
    if seq.empty:
        return "Sequence length analysis not available."

    long_seq = seq.sort_values("seq_length", ascending=False).iloc[0]
    return (
        "Sequence Length Analysis:\n"
        f"- Longer sequences increase latency to {long_seq['mean_latency_ms']:.2f} ms"
    )


def precision_analysis(df: pd.DataFrame) -> str:
    precision = df[df["experiment"] == "precision"]
    if precision.empty:
        return "Precision analysis not available."

    best = precision.sort_values("tokens_per_sec", ascending=False).iloc[0]
    return (
        "Precision Analysis:\n"
        f"- Best performance using {best['precision']} on {best['device']}"
    )


def bottleneck_analysis(df: pd.DataFrame) -> str:
    lines = ["Bottleneck Analysis:"]

    if "gpu_utilization_percent" in df.columns:
        avg_gpu_util = df["gpu_utilization_percent"].mean()
        if avg_gpu_util > 70:
            lines.append("- Likely compute-bound on GPU (high utilization)")
        else:
            lines.append("- Likely memory-bound or IO-bound (low GPU utilization)")

    if "peak_memory_mb" in df.columns:
        max_mem = df["peak_memory_mb"].max()
        lines.append(f"- Peak memory usage observed: {max_mem:.2f} MB")

    return "\n".join(lines)


def generate_report(results_path: Path, output_path: Path) -> None:
    df = load_results(results_path)

    sections = [
        "# Final Analysis Report\n",
        cpu_vs_gpu_analysis(df),
        batch_analysis(df),
        sequence_analysis(df),
        precision_analysis(df),
        bottleneck_analysis(df),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections))


if __name__ == "__main__":
    generate_report(
        Path("results/raw/latest_all_results.csv"),
        Path("results/reports/final_analysis.md"),
    )
