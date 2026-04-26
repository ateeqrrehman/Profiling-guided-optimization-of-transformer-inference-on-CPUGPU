from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.io import write_text
from utils.plotting import build_visual_summary, plot_summary_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact visualization dashboard from benchmark results."
    )
    parser.add_argument(
        "--results",
        default="results/raw/latest_all_results.csv",
        help="Path to the benchmark CSV to visualize.",
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Prefix used for generated output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    frame = pd.read_csv(results_path)
    plots_dir = Path("results/plots")
    reports_dir = Path("results/reports")

    dashboard_path = plot_summary_dashboard(frame, plots_dir, args.run_id)
    summary_text = build_visual_summary(frame)
    summary_path = reports_dir / f"{args.run_id}_visual_summary.md"
    write_text(summary_text, summary_path)

    if dashboard_path:
        print(f"Dashboard saved to: {dashboard_path.resolve()}")
    print(f"Visual summary saved to: {summary_path.resolve()}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
