from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("results/raw/latest_all_results.csv")
OUTPUT_PATH = Path("results/reports/final_analysis.md")


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def _fmt(value: float, unit: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}{unit}"


def _bool_label(value) -> str:
    return "compiled" if bool(value) else "eager"


def coverage_summary(df: pd.DataFrame) -> str:
    lines = ["## Proposal Coverage Check", ""]
    experiments = set(df["experiment"].dropna()) if "experiment" in df else set()
    devices = set(df["device"].dropna()) if "device" in df else set()
    batches = sorted(df["batch_size"].dropna().unique().tolist()) if "batch_size" in df else []
    seqs = sorted(df["seq_length"].dropna().unique().tolist()) if "seq_length" in df else []
    precisions = sorted(df["precision"].dropna().unique().tolist()) if "precision" in df else []

    checks = [
        ("Baseline CPU/GPU experiment", "baseline" in experiments and {"cpu", "cuda"}.issubset(devices)),
        ("Batch-size experiment", "batch_size" in experiments and {1, 2, 4, 8, 16}.issubset(set(map(int, batches)))),
        ("Sequence-length experiment", "sequence_length" in experiments and {32, 128, 256}.issubset(set(map(int, seqs)))),
        ("Precision / quantization experiment", "precision" in experiments and {"fp32", "int8_dynamic"}.issubset(set(precisions))),
        ("Compilation experiment", "compilation" in experiments),
        ("Profiler-enabled output fields", {"profile_trace_path", "profile_summary_path"}.issubset(df.columns)),
        ("Quality metric fields", {"quality_loss", "quality_perplexity"}.issubset(df.columns)),
        ("Utilization fields", {"cpu_utilization_percent", "gpu_utilization_percent"}.issubset(df.columns)),
    ]
    for label, passed in checks:
        lines.append(f"- {'✅' if passed else '⚠️'} {label}")
    lines.append("")
    lines.append(f"Measured devices: {', '.join(sorted(devices)) if devices else 'N/A'}")
    lines.append(f"Measured batch sizes: {batches}")
    lines.append(f"Measured sequence lengths: {seqs}")
    lines.append(f"Measured precision modes: {', '.join(precisions) if precisions else 'N/A'}")
    return "\n".join(lines)


def cpu_vs_gpu_analysis(df: pd.DataFrame) -> str:
    baseline = df[df["experiment"] == "baseline"].copy()
    if baseline.empty or not {"cpu", "cuda"}.issubset(set(baseline["device"])):
        return "## CPU vs GPU Analysis\n\nCPU vs GPU comparison is not available in this run."

    cpu = baseline[baseline["device"] == "cpu"][["seq_label", "seq_length", "mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]]
    gpu = baseline[baseline["device"] == "cuda"][["seq_label", "seq_length", "mean_latency_ms", "tokens_per_sec", "peak_memory_mb"]]
    merged = cpu.merge(gpu, on=["seq_label", "seq_length"], suffixes=("_cpu", "_gpu"))
    merged = merged.sort_values("seq_length")

    lines = ["## CPU vs GPU Analysis", ""]
    for _, row in merged.iterrows():
        speedup = row["mean_latency_ms_cpu"] / row["mean_latency_ms_gpu"]
        lines.append(
            f"- **{row['seq_label']} ({int(row['seq_length'])} tokens):** "
            f"CPU latency={_fmt(row['mean_latency_ms_cpu'], ' ms')}, "
            f"GPU latency={_fmt(row['mean_latency_ms_gpu'], ' ms')}, "
            f"GPU speedup={speedup:.2f}x."
        )
    lines.append("")
    lines.append(
        "Interpretation: the GPU is expected to win because transformer inference contains many matrix operations that can be parallelized. "
        "The size of the speedup depends on sequence length, batch size, and overhead."
    )
    return "\n".join(lines)


def batch_analysis(df: pd.DataFrame) -> str:
    batch = df[df["experiment"] == "batch_size"].copy()
    if batch.empty:
        return "## Batch Size Analysis\n\nBatch-size analysis is not available."

    lines = ["## Batch Size Analysis", ""]
    for device, group in batch.groupby("device"):
        ordered = group.sort_values("batch_size")
        best = ordered.sort_values("tokens_per_sec", ascending=False).iloc[0]
        lines.append(f"### {device.upper()}")
        for _, row in ordered.iterrows():
            lines.append(
                f"- batch={int(row['batch_size'])}: latency={_fmt(row['mean_latency_ms'], ' ms')}, "
                f"throughput={_fmt(row['tokens_per_sec'], ' tokens/s')}, "
                f"peak memory={_fmt(row['peak_memory_mb'], ' MB')}"
            )
        lines.append(
            f"- Best throughput for {device.upper()} occurs at batch={int(best['batch_size'])} "
            f"with {_fmt(best['tokens_per_sec'], ' tokens/s')}."
        )
        lines.append("")

    lines.append(
        "Interpretation: batching usually improves throughput by increasing hardware utilization, but very large batches can increase latency and memory pressure. "
        "The best batch size is therefore workload- and hardware-dependent."
    )
    return "\n".join(lines)


def sequence_analysis(df: pd.DataFrame) -> str:
    seq = df[df["experiment"] == "sequence_length"].copy()
    if seq.empty:
        return "## Sequence Length Analysis\n\nSequence-length analysis is not available."

    lines = ["## Sequence Length Analysis", ""]
    for device, group in seq.groupby("device"):
        ordered = group.sort_values("seq_length")
        lines.append(f"### {device.upper()}")
        for _, row in ordered.iterrows():
            lines.append(
                f"- seq={int(row['seq_length'])}: latency={_fmt(row['mean_latency_ms'], ' ms')}, "
                f"throughput={_fmt(row['tokens_per_sec'], ' tokens/s')}, "
                f"peak memory={_fmt(row['peak_memory_mb'], ' MB')}"
            )
        lines.append("")

    lines.append(
        "Interpretation: sequence length is central for transformer workloads because longer inputs increase the amount of token interaction and memory movement in attention-related computation."
    )
    return "\n".join(lines)


def precision_analysis(df: pd.DataFrame) -> str:
    precision = df[df["experiment"] == "precision"].copy()
    if precision.empty:
        return "## Precision Analysis\n\nPrecision analysis is not available."

    lines = ["## Precision Analysis", ""]
    for device, group in precision.groupby("device"):
        best = group.sort_values("tokens_per_sec", ascending=False).iloc[0]
        lines.append(f"### {device.upper()}")
        for _, row in group.sort_values("precision").iterrows():
            quality = _fmt(row.get("quality_perplexity", float("nan")))
            lines.append(
                f"- {row['precision']}: latency={_fmt(row['mean_latency_ms'], ' ms')}, "
                f"throughput={_fmt(row['tokens_per_sec'], ' tokens/s')}, "
                f"peak memory={_fmt(row['peak_memory_mb'], ' MB')}, "
                f"perplexity proxy={quality}"
            )
        lines.append(
            f"- Best throughput for {device.upper()} is {best['precision']} with {_fmt(best['tokens_per_sec'], ' tokens/s')}."
        )
        lines.append("")

    lines.append(
        "Interpretation: reduced precision can improve speed or memory efficiency, but the result depends on hardware support and implementation overhead. "
        "The quality column is a fixed-prompt perplexity proxy, so it should be interpreted as a lightweight sanity check rather than a full language-model benchmark."
    )
    return "\n".join(lines)


def compilation_analysis(df: pd.DataFrame) -> str:
    comp = df[df["experiment"] == "compilation"].copy()
    if comp.empty:
        return "## Compilation Analysis\n\nCompilation analysis is not available."

    lines = ["## Compilation Analysis", ""]
    if "requested_compiled" not in comp.columns:
        lines.append("The results do not include requested_compiled, so compile status cannot be separated from eager mode.")
        return "\n".join(lines)

    requested = comp[comp["requested_compiled"] == True]
    actually_compiled = requested[requested["compiled"] == True]
    fallback = requested[requested["compiled"] == False]

    lines.append(f"- Compile-requested rows: {len(requested)}")
    lines.append(f"- Actually compiled rows: {len(actually_compiled)}")
    lines.append(f"- Fallback/non-compiled rows after compile request: {len(fallback)}")
    lines.append("")

    for device, group in comp.groupby("device"):
        lines.append(f"### {device.upper()}")
        for seq_length, sg in group.groupby("seq_length"):
            eager = sg[(sg["requested_compiled"] == False)].sort_values("mean_latency_ms")
            compiled = sg[(sg["requested_compiled"] == True)].sort_values("mean_latency_ms")
            if eager.empty or compiled.empty:
                continue
            e = eager.iloc[0]
            c = compiled.iloc[0]
            delta = e["mean_latency_ms"] - c["mean_latency_ms"]
            lines.append(
                f"- seq={int(seq_length)}: eager={_fmt(e['mean_latency_ms'], ' ms')}, "
                f"compile-requested={_fmt(c['mean_latency_ms'], ' ms')}, "
                f"delta={_fmt(delta, ' ms')}, actual compiled={bool(c['compiled'])}."
            )
        lines.append("")

    if not fallback.empty:
        lines.append(
            "Important caveat: some compile-requested runs did not remain marked as compiled in the final results. "
            "This usually means PyTorch compilation fell back or the local environment did not fully support the requested backend. "
            "The benchmark still records those rows so the limitation is visible rather than hidden."
        )
    else:
        lines.append(
            "All compile-requested rows remained compiled, so this run provides a direct eager-vs-compiled comparison."
        )
    return "\n".join(lines)


def bottleneck_analysis(df: pd.DataFrame) -> str:
    lines = ["## Compute-bound vs Memory-bound Interpretation", ""]

    gpu_rows = df[df["device"] == "cuda"].copy() if "device" in df.columns else pd.DataFrame()
    if not gpu_rows.empty and "gpu_utilization_percent" in gpu_rows.columns:
        avg_gpu_util = gpu_rows["gpu_utilization_percent"].dropna().mean()
        max_gpu_util = gpu_rows["gpu_utilization_percent"].dropna().max()
        lines.append(f"- Average measured GPU utilization: {_fmt(avg_gpu_util, '%')}")
        lines.append(f"- Maximum measured GPU utilization: {_fmt(max_gpu_util, '%')}")
        if avg_gpu_util >= 70:
            lines.append("- Interpretation: this run shows signs of compute pressure because utilization is high on average.")
        elif avg_gpu_util >= 35:
            lines.append("- Interpretation: this run shows mixed behavior; some workloads use the GPU well, while others are limited by overhead, memory, or small batch effects.")
        else:
            lines.append("- Interpretation: this run is not consistently compute-saturated; overhead and memory movement likely matter.")

    batch = df[df["experiment"] == "batch_size"].copy()
    if not batch.empty:
        for device, group in batch.groupby("device"):
            ordered = group.sort_values("batch_size")
            mem_growth = ordered["peak_memory_mb"].iloc[-1] - ordered["peak_memory_mb"].iloc[0]
            thr_growth = ordered["tokens_per_sec"].iloc[-1] / ordered["tokens_per_sec"].iloc[0]
            lines.append(
                f"- {device.upper()} batch sweep: memory increases by {_fmt(mem_growth, ' MB')} from smallest to largest batch, "
                f"while throughput changes by {thr_growth:.2f}x."
            )

    lines.append(
        "Overall interpretation: the project should not claim a single universal bottleneck. "
        "Instead, the evidence should be framed as workload-dependent: small batches may be overhead-limited, larger batches improve throughput until memory pressure or saturation appears, and longer sequences increase transformer-specific memory/computation cost."
    )
    return "\n".join(lines)


def selector_guidance() -> str:
    return """## Workload-aware Selector Validation

Run these commands after generating `latest_all_results.csv`:

```bash
python selector.py --device cuda --batch-size 8 --seq-length 128 --objective throughput
python selector.py --device cpu --batch-size 4 --seq-length 128 --objective latency
```

The selector output should be included in the final report as the original contribution. It demonstrates that the project goes beyond benchmarking by using empirical results to recommend a configuration for a given workload.
"""


def next_steps() -> str:
    return """## Recommended Final Steps

1. Save this report with the final submission materials.
2. Include the summary dashboard figure from `results/plots/`.
3. Include 1-2 profiler operator summaries from `results/profiles/`.
4. Report selector examples for one CPU workload and one GPU workload.
5. If compile-requested rows fall back to eager mode, explain this as an environment-dependent systems limitation rather than hiding it.
"""


def generate_report(results_path: Path = RESULTS_PATH, output_path: Path = OUTPUT_PATH) -> None:
    df = load_results(results_path)

    sections = [
        "# Final Analysis Report\n",
        coverage_summary(df),
        cpu_vs_gpu_analysis(df),
        batch_analysis(df),
        sequence_analysis(df),
        precision_analysis(df),
        compilation_analysis(df),
        bottleneck_analysis(df),
        selector_guidance(),
        next_steps(),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Final analysis report written to: {output_path}")


if __name__ == "__main__":
    generate_report()
