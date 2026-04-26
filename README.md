# MSML 605 Transformer Inference Benchmark Suite

This project implements the full proposal in `MSML 605 Course Project Proposal.pdf` as a reproducible benchmarking suite for comparing transformer inference across devices and execution settings.

It covers:

- CPU vs GPU baseline comparison
- Batch size tradeoffs
- Sequence length tradeoffs
- Precision tradeoffs
- `torch.compile` tradeoffs
- A lightweight workload-aware selector derived from collected results

The suite measures latency, throughput, peak memory, average memory, and a basic quality proxy (loss and perplexity on a fixed prompt set). It can also capture `torch.profiler` traces for explanation-oriented analysis.

## Project Layout

```text
CPUvsGPU-Transformers/
|-- config.py
|-- run_suite.py
|-- selector.py
|-- experiments/
|   |-- __init__.py
|   |-- baseline.py
|   |-- batch_size.py
|   |-- sequence_length.py
|   |-- precision.py
|   `-- compilation.py
`-- utils/
    |-- __init__.py
    |-- device.py
    |-- io.py
    |-- metrics.py
    |-- modeling.py
    |-- plotting.py
    |-- profiling.py
    |-- prompts.py
    |-- quality.py
    |-- recommendation.py
    `-- runner.py
```

## Setup

Create or activate a virtual environment, then install the dependencies:

```powershell
.venv\Scripts\pip install -r requirements.txt
```

The default workload model is `distilgpt2`. On first use, Hugging Face may download it if it is not already cached. If model download is unavailable, the code can fall back to a small local random GPT-style model so the benchmark pipeline still runs offline.

## Running The Full Suite

Run every proposal experiment on all available devices:

```powershell
.venv\Scripts\python run_suite.py --device auto --experiments all
```

Run a faster smoke test on CPU only:

```powershell
.venv\Scripts\python run_suite.py --device cpu --experiments all --smoke-test --model-name random-fallback
```

Capture profiler traces while running:

```powershell
.venv\Scripts\python run_suite.py --device auto --experiments all --profile
```

Run only selected experiments:

```powershell
.venv\Scripts\python run_suite.py --device auto --experiments baseline,precision,compilation
```

## Results

The suite writes results under `results/`:

- `results/raw/` for per-experiment CSV files and combined CSV outputs
- `results/plots/` for generated figures
- `results/profiles/` for profiler traces and operator summaries
- `results/reports/` for markdown summaries

## Using The Selector

After collecting results, ask the selector for the best configuration for a workload:

```powershell
.venv\Scripts\python selector.py --device cpu --batch-size 4 --seq-length 128 --objective balanced
```

You can also point it at a specific CSV:

```powershell
.venv\Scripts\python selector.py --results results\raw\latest_all_results.csv --device cuda --batch-size 8 --seq-length 256 --objective throughput
```

## Notes

- The current local environment can verify CPU paths. The code automatically enables CUDA benchmarks when run on a GPU-enabled machine.
- Precision support is device-aware. CPU benchmarks use `fp32` and dynamic `int8` quantization. CUDA benchmarks use `fp32`, `fp16`, and `bf16` when supported.
- Compilation is optional and guarded. If `torch.compile` is unavailable or fails on a given setup, the suite records a note and continues.
- `--model-name random-fallback` skips external downloads and uses a local randomly initialized GPT-style model for offline smoke tests.
