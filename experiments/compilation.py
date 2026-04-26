from __future__ import annotations

from utils.runner import BenchmarkCondition, run_conditions


def run(settings, devices: list[str], capture_profile: bool, run_id: str):
    conditions = []
    batch_size = min(4, max(settings.batch_sizes))

    for device in devices:
        for seq_label, seq_length in settings.sequence_lengths.items():
            for compiled in (False, True):
                conditions.append(
                    BenchmarkCondition(
                        experiment="compilation",
                        variant=f"{device}_{seq_label}_{'compiled' if compiled else 'eager'}",
                        device=device,
                        batch_size=batch_size,
                        seq_label=seq_label,
                        seq_length=seq_length,
                        precision="fp32",
                        compiled=compiled,
                        measure_quality=True,
                        capture_profile=capture_profile and compiled and seq_label == "medium",
                    )
                )
    return run_conditions(conditions, settings=settings, run_id=run_id)
