from __future__ import annotations

from utils.runner import BenchmarkCondition, run_conditions


def run(settings, devices: list[str], capture_profile: bool, run_id: str):
    conditions = []
    for device in devices:
        for seq_label, seq_length in settings.sequence_lengths.items():
            conditions.append(
                BenchmarkCondition(
                    experiment="baseline",
                    variant=f"{device}_{seq_label}",
                    device=device,
                    batch_size=1,
                    seq_label=seq_label,
                    seq_length=seq_length,
                    precision="fp32",
                    compiled=False,
                    measure_quality=False,
                    capture_profile=capture_profile and seq_label == "medium",
                )
            )
    return run_conditions(conditions, settings=settings, run_id=run_id)
