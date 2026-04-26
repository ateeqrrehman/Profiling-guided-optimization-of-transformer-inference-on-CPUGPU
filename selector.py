from __future__ import annotations

import argparse

from utils.recommendation import format_recommendation, recommend_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend an inference configuration from collected benchmark data."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional path to a benchmark CSV. Defaults to results/raw/latest_all_results.csv.",
    )
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--seq-length", required=True, type=int)
    parser.add_argument(
        "--objective",
        default="balanced",
        choices=["latency", "throughput", "memory", "balanced"],
    )
    args = parser.parse_args()
    recommendation = recommend_configuration(
        results_path=args.results,
        device=args.device,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        objective=args.objective,
    )
    print(format_recommendation(recommendation))


if __name__ == "__main__":
    parse_args()
