from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from experiments import EXPERIMENT_REGISTRY
from utils.device import get_environment_summary, resolve_devices
from utils.io import make_run_id, save_dataframe, write_json, write_text
from utils.plotting import build_visual_summary, plot_experiment, plot_summary_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MSML 605 transformer inference benchmark suite."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Which device set to run. 'auto' uses every available device.",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated list such as baseline,batch_size,precision or 'all'.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a reduced setting for quick verification.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture profiler traces for representative conditions.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the default Hugging Face model name.",
    )
    return parser.parse_args()


def resolve_experiments(raw_value: str) -> list[str]:
    if raw_value == "all":
        return list(config.DEFAULT_EXPERIMENTS)

    requested = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [name for name in requested if name not in EXPERIMENT_REGISTRY]
    if invalid:
        valid = ", ".join(sorted(EXPERIMENT_REGISTRY))
        raise ValueError(f"Unknown experiments: {invalid}. Valid names: {valid}")
    return requested


def build_console_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No benchmark rows were produced."

    columns = [
        "experiment",
        "device",
        "batch_size",
        "seq_length",
        "precision",
        "compiled",
        "mean_latency_ms",
        "tokens_per_sec",
        "peak_memory_mb",
    ]
    display = frame.loc[:, columns].copy()
    display["compiled"] = display["compiled"].map({True: "yes", False: "no"})
    return display.to_string(index=False)


def main() -> None:
    args = parse_args()
    settings = config.make_settings(smoke_test=args.smoke_test, model_name=args.model_name)
    devices = resolve_devices(args.device)
    experiments_to_run = resolve_experiments(args.experiments)
    run_id = make_run_id(prefix="suite")

    print(f"Run ID: {run_id}")
    print(f"Devices: {', '.join(devices)}")
    print(f"Experiments: {', '.join(experiments_to_run)}")
    print(f"Model: {settings.model_name}")
    print(f"Smoke test: {args.smoke_test}")
    print(f"Profiler enabled: {args.profile}")

    environment_path = settings.reports_dir / f"{run_id}_environment.json"
    write_json(get_environment_summary(), environment_path)

    all_frames: list[pd.DataFrame] = []
    for experiment_name in experiments_to_run:
        print(f"\nRunning experiment: {experiment_name}")
        runner = EXPERIMENT_REGISTRY[experiment_name]
        frame = runner(
            settings=settings,
            devices=devices,
            capture_profile=args.profile,
            run_id=run_id,
        )
        all_frames.append(frame)

        experiment_csv = settings.raw_results_dir / f"{run_id}_{experiment_name}.csv"
        latest_csv = settings.raw_results_dir / f"latest_{experiment_name}.csv"
        save_dataframe(frame, experiment_csv)
        save_dataframe(frame, latest_csv)

        plot_experiment(frame, experiment_name, settings.plots_dir, run_id)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined_csv = settings.raw_results_dir / f"{run_id}_all_results.csv"
    latest_combined_csv = settings.raw_results_dir / "latest_all_results.csv"
    save_dataframe(combined, combined_csv)
    save_dataframe(combined, latest_combined_csv)

    summary_text = build_console_summary(combined)
    summary_path = settings.reports_dir / f"{run_id}_summary.txt"
    latest_summary_path = settings.reports_dir / "latest_summary.txt"
    write_text(summary_text, summary_path)
    write_text(summary_text, latest_summary_path)

    dashboard_path = plot_summary_dashboard(combined, settings.plots_dir, run_id)
    visual_summary = build_visual_summary(combined)
    visual_summary_path = settings.reports_dir / f"{run_id}_visual_summary.md"
    latest_visual_summary_path = settings.reports_dir / "latest_visual_summary.md"
    write_text(visual_summary, visual_summary_path)
    write_text(visual_summary, latest_visual_summary_path)

    print("\nBenchmark summary")
    print(summary_text)
    if dashboard_path:
        print(f"\nSummary dashboard saved to: {dashboard_path}")
    print(f"\nCombined results saved to: {combined_csv}")
    print(f"Latest combined results saved to: {latest_combined_csv}")


if __name__ == "__main__":
    main()
