from __future__ import annotations

from experiments.baseline import run as run_baseline
from experiments.batch_size import run as run_batch_size
from experiments.compilation import run as run_compilation
from experiments.precision import run as run_precision
from experiments.sequence_length import run as run_sequence_length

EXPERIMENT_REGISTRY = {
    "baseline": run_baseline,
    "batch_size": run_batch_size,
    "sequence_length": run_sequence_length,
    "precision": run_precision,
    "compilation": run_compilation,
}
