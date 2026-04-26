from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_experiment(frame: pd.DataFrame, experiment_name: str, output_dir: Path, run_id: str) -> None:
    if frame.empty:
        return

    if experiment_name == "baseline":
        _plot_baseline(frame, output_dir / f"{run_id}_{experiment_name}.png")
    elif experiment_name == "batch_size":
        _plot_batch_size(frame, output_dir / f"{run_id}_{experiment_name}.png")
    elif experiment_name == "sequence_length":
        _plot_sequence_length(frame, output_dir / f"{run_id}_{experiment_name}.png")
    elif experiment_name == "precision":
        _plot_precision(frame, output_dir / f"{run_id}_{experiment_name}.png")
    elif experiment_name == "compilation":
        _plot_compilation(frame, output_dir / f"{run_id}_{experiment_name}.png")


def plot_summary_dashboard(frame: pd.DataFrame, output_dir: Path, run_id: str) -> Path | None:
    if frame.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    _plot_baseline_overview(frame, axes[0])
    _plot_gpu_speedup(frame, axes[1])
    _plot_batch_scaling(frame, axes[2])
    _plot_precision_overview(frame, axes[3])
    _plot_compilation_overview(frame, axes[4])
    _plot_memory_overview(frame, axes[5])

    fig.suptitle("Transformer Inference Performance Dashboard", fontsize=16, y=1.02)
    output_path = output_dir / f"{run_id}_summary_dashboard.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_baseline(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]
    titles = ["Latency (ms)", "Throughput (tokens/s)", "Peak Memory (MB)"]

    labels = frame["device"] + "-" + frame["seq_label"]
    for axis, metric, title in zip(axes, metrics, titles):
        axis.bar(labels, frame[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)

    _save_figure(fig, path)


def _plot_batch_size(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]
    titles = ["Latency (ms)", "Throughput (tokens/s)", "Peak Memory (MB)"]

    for axis, metric, title in zip(axes, metrics, titles):
        for device, group in frame.groupby("device"):
            ordered = group.sort_values("batch_size")
            axis.plot(ordered["batch_size"], ordered[metric], marker="o", label=device)
        axis.set_title(title)
        axis.set_xlabel("Batch Size")
        axis.legend()

    _save_figure(fig, path)


def _plot_sequence_length(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]
    titles = ["Latency (ms)", "Throughput (tokens/s)", "Peak Memory (MB)"]

    for axis, metric, title in zip(axes, metrics, titles):
        for device, group in frame.groupby("device"):
            ordered = group.sort_values("seq_length")
            axis.plot(ordered["seq_length"], ordered[metric], marker="o", label=device)
        axis.set_title(title)
        axis.set_xlabel("Sequence Length")
        axis.legend()

    _save_figure(fig, path)


def _plot_precision(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["mean_latency_ms", "peak_memory_mb", "quality_perplexity"]
    titles = ["Latency (ms)", "Peak Memory (MB)", "Perplexity"]

    for axis, metric, title in zip(axes, metrics, titles):
        pivot = frame.pivot_table(index="precision", columns="device", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Precision")
        axis.tick_params(axis="x", rotation=0)

    _save_figure(fig, path)


def _plot_compilation(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["mean_latency_ms", "tokens_per_sec", "quality_perplexity"]
    titles = ["Latency (ms)", "Throughput (tokens/s)", "Perplexity"]

    frame = frame.copy()
    frame["mode"] = frame["compiled"].map({True: "compiled", False: "eager"})
    for axis, metric, title in zip(axes, metrics, titles):
        pivot = frame.pivot_table(index="seq_label", columns="mode", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Sequence Length")
        axis.tick_params(axis="x", rotation=0)

    _save_figure(fig, path)


def _plot_baseline_overview(frame: pd.DataFrame, axis) -> None:
    baseline = frame[frame["experiment"] == "baseline"].copy()
    if baseline.empty:
        axis.set_visible(False)
        return

    pivot = baseline.pivot_table(
        index="seq_label",
        columns="device",
        values="mean_latency_ms",
        aggfunc="mean",
    )
    pivot = pivot.reindex([label for label in ["short", "medium", "long"] if label in pivot.index])
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Baseline Latency")
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Latency (ms)")
    axis.tick_params(axis="x", rotation=0)


def _plot_gpu_speedup(frame: pd.DataFrame, axis) -> None:
    baseline = frame[frame["experiment"] == "baseline"].copy()
    if baseline.empty or not {"cpu", "cuda"}.issubset(set(baseline["device"].unique())):
        axis.text(0.5, 0.5, "GPU speedup unavailable", ha="center", va="center")
        axis.set_axis_off()
        return

    cpu = baseline[baseline["device"] == "cpu"][["seq_label", "mean_latency_ms"]].rename(
        columns={"mean_latency_ms": "cpu_latency_ms"}
    )
    gpu = baseline[baseline["device"] == "cuda"][["seq_label", "mean_latency_ms"]].rename(
        columns={"mean_latency_ms": "gpu_latency_ms"}
    )
    merged = cpu.merge(gpu, on="seq_label", how="inner")
    merged["speedup"] = merged["cpu_latency_ms"] / merged["gpu_latency_ms"]
    merged["seq_order"] = merged["seq_label"].map({"short": 0, "medium": 1, "long": 2})
    merged = merged.sort_values("seq_order")

    axis.plot(merged["seq_label"], merged["speedup"], marker="o", linewidth=2)
    for _, row in merged.iterrows():
        axis.text(row["seq_label"], row["speedup"] + 0.05, f"{row['speedup']:.1f}x", ha="center")
    axis.set_title("GPU Latency Speedup Over CPU")
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Speedup (x)")


def _plot_batch_scaling(frame: pd.DataFrame, axis) -> None:
    batch = frame[frame["experiment"] == "batch_size"].copy()
    if batch.empty:
        axis.set_visible(False)
        return

    for device, group in batch.groupby("device"):
        ordered = group.sort_values("batch_size")
        axis.plot(
            ordered["batch_size"],
            ordered["tokens_per_sec"],
            marker="o",
            linewidth=2,
            label=device.upper(),
        )
    axis.set_title("Batch Scaling Throughput")
    axis.set_xlabel("Batch Size")
    axis.set_ylabel("Tokens / second")
    axis.legend()


def _plot_precision_overview(frame: pd.DataFrame, axis) -> None:
    precision = frame[frame["experiment"] == "precision"].copy()
    if precision.empty:
        axis.set_visible(False)
        return

    precision["label"] = precision["device"].str.upper() + ":" + precision["precision"]
    colors = ["#4C78A8" if device == "cpu" else "#F58518" for device in precision["device"]]
    axis.bar(precision["label"], precision["tokens_per_sec"], color=colors)
    axis.set_title("Precision Throughput Comparison")
    axis.set_xlabel("Device / Precision")
    axis.set_ylabel("Tokens / second")
    axis.tick_params(axis="x", rotation=30)


def _plot_compilation_overview(frame: pd.DataFrame, axis) -> None:
    compilation = frame[frame["experiment"] == "compilation"].copy()
    if compilation.empty:
        axis.set_visible(False)
        return

    compilation["mode"] = compilation["requested_compiled"].map(
        {False: "eager", True: "compile requested"}
    )
    grouped = (
        compilation.groupby(["device", "mode"], as_index=False)["mean_latency_ms"]
        .mean()
        .pivot(index="device", columns="mode", values="mean_latency_ms")
    )
    grouped.plot(kind="bar", ax=axis)
    axis.set_title("Compilation Experiment")
    axis.set_xlabel("Device")
    axis.set_ylabel("Latency (ms)")
    axis.tick_params(axis="x", rotation=0)


def _plot_memory_overview(frame: pd.DataFrame, axis) -> None:
    baseline = frame[frame["experiment"] == "baseline"].copy()
    if baseline.empty:
        axis.set_visible(False)
        return

    pivot = baseline.pivot_table(
        index="seq_label",
        columns="device",
        values="peak_memory_mb",
        aggfunc="mean",
    )
    pivot = pivot.reindex([label for label in ["short", "medium", "long"] if label in pivot.index])
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Baseline Peak Memory")
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Peak memory (MB)")
    axis.tick_params(axis="x", rotation=0)


def build_visual_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No benchmark data found."

    sections: list[str] = []

    baseline = frame[frame["experiment"] == "baseline"].copy()
    if not baseline.empty and {"cpu", "cuda"}.issubset(set(baseline["device"].unique())):
        cpu = baseline[baseline["device"] == "cpu"][["seq_label", "mean_latency_ms"]].rename(
            columns={"mean_latency_ms": "cpu_latency_ms"}
        )
        gpu = baseline[baseline["device"] == "cuda"][["seq_label", "mean_latency_ms"]].rename(
            columns={"mean_latency_ms": "gpu_latency_ms"}
        )
        merged = cpu.merge(gpu, on="seq_label", how="inner")
        merged["speedup"] = merged["cpu_latency_ms"] / merged["gpu_latency_ms"]
        merged["seq_order"] = merged["seq_label"].map({"short": 0, "medium": 1, "long": 2})
        merged = merged.sort_values("seq_order")
        lines = ["Baseline CPU vs GPU latency speedup:"]
        for _, row in merged.iterrows():
            lines.append(
                f"- {row['seq_label']}: CPU {row['cpu_latency_ms']:.2f} ms vs GPU {row['gpu_latency_ms']:.2f} ms ({row['speedup']:.2f}x)"
            )
        sections.append("\n".join(lines))

    batch = frame[frame["experiment"] == "batch_size"].copy()
    if not batch.empty:
        best_batch = batch.sort_values("tokens_per_sec", ascending=False).iloc[0]
        sections.append(
            "Best throughput in the batch-size sweep:\n"
            f"- {best_batch['device'].upper()} batch={int(best_batch['batch_size'])}, "
            f"seq={int(best_batch['seq_length'])}, throughput={best_batch['tokens_per_sec']:.2f} tokens/s"
        )

    precision = frame[frame["experiment"] == "precision"].copy()
    if not precision.empty:
        best_precision = precision.sort_values("tokens_per_sec", ascending=False).iloc[0]
        sections.append(
            "Best precision configuration:\n"
            f"- {best_precision['device'].upper()} {best_precision['precision']} "
            f"at batch={int(best_precision['batch_size'])}, seq={int(best_precision['seq_length'])}, "
            f"latency={best_precision['mean_latency_ms']:.2f} ms, throughput={best_precision['tokens_per_sec']:.2f} tokens/s"
        )

    compilation = frame[frame["experiment"] == "compilation"].copy()
    if not compilation.empty:
        failed = compilation[
            compilation["requested_compiled"].fillna(False)
            & compilation["notes"].fillna("").str.contains("failed", case=False)
        ]
        if not failed.empty:
            issues = failed[["device", "notes"]].drop_duplicates().values.tolist()
            lines = ["Compilation caveats:"]
            for device, note in issues:
                short_note = str(note).splitlines()[0]
                lines.append(f"- {str(device).upper()}: {short_note}")
            sections.append("\n".join(lines))

    return "\n\n".join(sections)
