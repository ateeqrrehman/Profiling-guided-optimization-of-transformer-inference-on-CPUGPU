from __future__ import annotations

from utils.runner import BenchmarkCondition, run_conditions


def run(settings, devices: list[str], capture_profile: bool, run_id: str):
    seq_label = "medium" if "medium" in settings.sequence_lengths else next(iter(settings.sequence_lengths))
    seq_length = settings.sequence_lengths[seq_label]
    conditions = []

    for device in devices:
        for batch_size in settings.batch_sizes:
            conditions.append(
                BenchmarkCondition(
                    experiment="batch_size",
                    variant=f"{device}_batch_{batch_size}",
                    device=device,
                    batch_size=batch_size,
                    seq_label=seq_label,
                    seq_length=seq_length,
                    precision="fp32",
                    compiled=False,
                    measure_quality=False,
                    capture_profile=capture_profile and batch_size == max(settings.batch_sizes),
                )
            )
    return run_conditions(conditions, settings=settings, run_id=run_id)
