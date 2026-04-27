from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_ROOT / "raw"
PLOTS_DIR = RESULTS_ROOT / "plots"
PROFILES_DIR = RESULTS_ROOT / "profiles"
REPORTS_DIR = RESULTS_ROOT / "reports"

for path in (RESULTS_ROOT, RAW_RESULTS_DIR, PLOTS_DIR, PROFILES_DIR, REPORTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_NAME = "distilgpt2"
ALLOW_RANDOM_MODEL_FALLBACK = True

BATCH_SIZES = [1, 2, 4, 8, 16]
SEQUENCE_LENGTHS = {
    "short": 32,
    "medium": 128,
    "long": 256,
}

CPU_PRECISION_MODES = ["fp32", "int8_dynamic"]
CUDA_PRECISION_MODES = ["fp32", "fp16", "bf16"]

QUALITY_PROMPTS = [
    "Transformers are widely used because",
    "In machine learning systems, performance matters when",
    "A workload becomes memory bound when",
    "Batching helps throughput because",
]

DEFAULT_EXPERIMENTS = [
    "baseline",
    "batch_size",
    "sequence_length",
    "precision",
    "compilation",
]


@dataclass
class RunSettings:
    model_name: str = DEFAULT_MODEL_NAME
    allow_random_fallback: bool = ALLOW_RANDOM_MODEL_FALLBACK
    batch_sizes: list[int] = field(default_factory=lambda: list(BATCH_SIZES))
    sequence_lengths: dict[str, int] = field(
        default_factory=lambda: dict(SEQUENCE_LENGTHS)
    )
    warmup_runs: int = 3
    measure_runs: int = 10
    quality_seq_length: int = 64
    compile_backend: str | None = None
    output_root: Path = RESULTS_ROOT
    raw_results_dir: Path = RAW_RESULTS_DIR
    plots_dir: Path = PLOTS_DIR
    profiles_dir: Path = PROFILES_DIR
    reports_dir: Path = REPORTS_DIR
    default_experiments: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXPERIMENTS)
    )


def make_settings(smoke_test: bool = False, model_name: str | None = None) -> RunSettings:
    settings = RunSettings()
    if model_name:
        settings.model_name = model_name
    if smoke_test:
        settings.batch_sizes = [1, 2]
        settings.sequence_lengths = {"short": 16, "medium": 64}
        settings.warmup_runs = 1
        settings.measure_runs = 2
        settings.quality_seq_length = 32
    return settings


def precision_modes_for_device(device: str) -> list[str]:
    if device == "cuda":
        return list(CUDA_PRECISION_MODES)
    return list(CPU_PRECISION_MODES)
