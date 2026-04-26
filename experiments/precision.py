from __future__ import annotations

import config
from utils.runner import BenchmarkCondition, run_conditions


def run(settings, devices: list[str], capture_profile: bool, run_id: str):
    seq_label = "medium" if "medium" in settings.sequence_lengths else next(iter(settings.sequence_lengths))
    seq_length = settings.sequence_lengths[seq_label]
    batch_size = min(4, max(settings.batch_sizes))

    conditions = []
    for device in devices:
        for precision in config.precision_modes_for_device(device):
            conditions.append(
                BenchmarkCondition(
                    experiment="precision",
                    variant=f"{device}_{precision}",
                    device=device,
                    batch_size=batch_size,
                    seq_label=seq_label,
                    seq_length=seq_length,
                    precision=precision,
                    compiled=False,
                    measure_quality=True,
                    capture_profile=capture_profile and precision != "fp32",
                )
            )
    return run_conditions(conditions, settings=settings, run_id=run_id)
