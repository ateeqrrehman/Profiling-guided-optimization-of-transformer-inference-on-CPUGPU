# MSML 605 Transformer Inference Benchmark Suite

## Project Title

**Profiling-Guided Optimization of Transformer Inference on CPU/GPU: Batch Size, Sequence Length, Precision, and Compilation Tradeoffs**

This repository implements the MSML 605 course project as a reproducible machine learning systems benchmark suite. The project studies how transformer inference performance changes across hardware, workload size, numeric precision, and runtime execution mode.

The workload model is **DistilGPT2**, loaded through Hugging Face Transformers. The focus is inference performance, not model training.

## Project Goal

Modern transformer inference is shaped by more than model architecture. Runtime behavior depends on device type, batch size, input sequence length, numeric precision, memory pressure, utilization, and compiler/runtime support.

This project asks:

> Under what conditions does transformer inference become compute-bound versus memory-bound, and how do batch size, sequence length, precision, and compilation influence this behavior on CPU and GPU?

The suite is designed to move beyond simple timing numbers. It collects measurements, generates plots, captures profiler traces, and supports a lightweight workload-aware configuration selector.

## What This Project Covers

The repository currently covers the major experiments from the project proposal:

1. CPU vs GPU baseline inference
2. Batch-size tradeoffs
3. Sequence-length tradeoffs
4. Precision and quantization tradeoffs
5. Eager execution vs `torch.compile`
6. Profiler-guided analysis using `torch.profiler`
7. A workload-aware configuration selector derived from benchmark results

## Current Status

The project has been executed successfully on both CPU and CUDA GPU environments. The latest full benchmark run was completed in Google Colab using a Linux CUDA GPU environment, which also allowed the `torch.compile` experiment to complete successfully.

A local Windows run was also tested on an NVIDIA RTX 4050 laptop GPU. CPU/GPU inference, precision experiments, profiling, and selector functionality worked locally, but `torch.compile` had Windows backend/runtime compatibility issues. The Colab/Linux run was used to complete the compilation portion reliably.

This is an important systems observation: compiler-based optimization depends on the local software stack, operating system, backend, and hardware environment.

## Key Preliminary Findings

From the latest full Colab benchmark run:

- GPU inference was much faster than CPU inference for DistilGPT2.
- GPU speedup was about **15.96x** for short sequences, **28.85x** for medium sequences, and **26.01x** for long sequences.
- CUDA batch-size throughput improved up to batch size 8, then decreased at batch size 16, suggesting saturation or memory/overhead effects.
- CUDA FP16 produced the strongest throughput in the precision experiment.
- `torch.compile` completed successfully in Colab and improved several CPU and GPU workloads, but not every case.
- GPU utilization showed mixed behavior, which supports the conclusion that bottlenecks are workload-dependent rather than universal.

Detailed results are available in:

```text
results/raw/latest_all_results.csv
results/reports/final_analysis.md
results/plots/
results/profiles/
```

## Project Layout

```text
CPUvsGPU-Transformers/
|-- README.md
|-- requirements.txt
|-- config.py
|-- run_suite.py
|-- selector.py
|-- analysis/
|   |-- __init__.py
|   `-- final_analysis.py
|-- experiments/
|   |-- __init__.py
|   |-- baseline.py
|   |-- batch_size.py
|   |-- sequence_length.py
|   |-- precision.py
|   `-- compilation.py
|-- utils/
|   |-- __init__.py
|   |-- device.py
|   |-- io.py
|   |-- metrics.py
|   |-- modeling.py
|   |-- plotting.py
|   |-- profiling.py
|   |-- prompts.py
|   |-- quality.py
|   |-- recommendation.py
|   `-- runner.py
`-- results/
    |-- raw/
    |-- plots/
    |-- profiles/
    `-- reports/
```

## Repository Components

### `run_suite.py`

Main entry point for running the benchmark suite. It can run all experiments or selected experiments on CPU, CUDA, or both.

### `selector.py`

Workload-aware recommendation tool. It reads collected benchmark results and recommends an inference configuration based on device, batch size, sequence length, and objective.

### `config.py`

Central configuration file for model name, batch sizes, sequence lengths, precision modes, warm-up runs, measured runs, and output directories.

### `experiments/`

Contains one module per experiment:

- `baseline.py` for CPU vs GPU baseline comparison
- `batch_size.py` for batch-size scaling
- `sequence_length.py` for input length scaling
- `precision.py` for FP32, FP16, BF16, and INT8 comparisons
- `compilation.py` for eager vs `torch.compile` comparison

### `utils/`

Contains reusable support code for device detection, model loading, metrics, profiling, plotting, quality evaluation, result handling, and selector logic.

### `analysis/final_analysis.py`

Generates the markdown analysis report from the latest benchmark CSV.

### `results/`

Stores generated outputs:

- `results/raw/` contains raw CSV files
- `results/plots/` contains generated plots and dashboard figures
- `results/profiles/` contains profiler traces and operator summaries
- `results/reports/` contains text and markdown reports

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ateeqrrehman/Profiling-guided-optimization-of-transformer-inference-on-CPUGPU.git
cd Profiling-guided-optimization-of-transformer-inference-on-CPUGPU
```

### 2. Create or activate an environment

A clean Python environment is recommended. Python 3.10 or 3.11 is preferred for best compatibility with PyTorch and compiler tooling.

Example using `venv` on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Example using Linux or Colab:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using an NVIDIA GPU locally, make sure the installed PyTorch build supports CUDA. A quick check is:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected CUDA output should include:

```text
CUDA available: True
```

## Running the Benchmark Suite

### Run all experiments on all available devices

```bash
python run_suite.py --device auto --experiments all --profile
```

This is the main command used for the final full benchmark run. It automatically uses CPU and CUDA if both are available.

### Run CPU only

```bash
python run_suite.py --device cpu --experiments all --profile
```

### Run CUDA only

```bash
python run_suite.py --device cuda --experiments all --profile
```

### Run selected experiments only

```bash
python run_suite.py --device auto --experiments baseline,precision,compilation --profile
```

Valid experiment names are:

```text
baseline
batch_size
sequence_length
precision
compilation
```

### Run a fast smoke test

```bash
python run_suite.py --device cpu --experiments all --smoke-test --model-name random-fallback
```

The smoke test is useful for checking code paths quickly. It uses fewer settings and can run with a local random GPT-style fallback model.

## Generating the Final Analysis Report

After running the benchmark suite, generate the analysis report:

```bash
python analysis/final_analysis.py
```

The report is written to:

```text
results/reports/final_analysis.md
```

This file summarizes proposal coverage, CPU/GPU speedups, batch-size behavior, sequence-length behavior, precision tradeoffs, compilation results, and compute/memory interpretation.

## Using the Workload-Aware Selector

After results have been collected, run the selector to recommend a configuration for a given workload.

### Example 1: CUDA throughput-focused workload

```bash
python selector.py --device cuda --batch-size 8 --seq-length 128 --objective throughput
```

### Example 2: CPU latency-focused workload

```bash
python selector.py --device cpu --batch-size 4 --seq-length 128 --objective latency
```

### Example 3: Use a specific results file

```bash
python selector.py --results results/raw/latest_all_results.csv --device cuda --batch-size 8 --seq-length 256 --objective throughput
```

Supported objectives include latency, throughput, memory, and balanced configurations.

## Output Files

After a full run, the most important outputs are:

```text
results/raw/latest_all_results.csv
results/reports/final_analysis.md
results/reports/latest_summary.txt
results/reports/latest_visual_summary.md
results/plots/
results/profiles/
```

For a progress report or final report, the most useful files are:

- `results/reports/final_analysis.md`
- the summary dashboard in `results/plots/`
- CPU/GPU comparison plots
- batch-size throughput plots
- selected profiler operator summaries from `results/profiles/`

## Benchmarking Methodology

The suite separates warm-up runs from measured runs. This helps avoid first-run artifacts from kernel initialization, caching effects, and compilation overhead.

Each benchmark condition uses repeated measurements. The current configuration uses:

```text
warmup_runs = 3
measure_runs = 10
```

The benchmark records:

- mean latency
- latency standard deviation
- samples per second
- tokens per second
- peak memory usage
- average memory usage
- CPU utilization
- GPU utilization
- quality loss
- perplexity proxy
- profiler trace paths
- profiler operator summary paths

## Precision Support

Precision support is device-aware:

- CPU uses `fp32` and `int8_dynamic`
- CUDA uses `fp32`, `fp16`, and `bf16` when supported

The quality metric is a lightweight fixed-prompt loss/perplexity proxy. It is intended as a sanity check for quality changes, not as a full language-model evaluation benchmark.

## Compilation Notes

The project includes an eager vs `torch.compile` experiment.

During local Windows testing, `torch.compile` had backend/runtime compatibility issues. This was recorded and handled by the benchmark suite without stopping the full experiment pipeline.

The final Colab/Linux CUDA run completed the `torch.compile` experiment successfully. The latest results include rows where:

```text
requested_compiled = True
compiled = True
```

This provides a direct eager-vs-compiled comparison. It also shows that compiler-based optimization is environment-dependent.

## Profiling Notes

Profiler traces are captured for representative conditions when the `--profile` flag is used. The profiler outputs include Chrome trace JSON files and operator-level CSV summaries.

These files are stored under:

```text
results/profiles/
```

Profiler outputs support the compute-bound vs memory-bound analysis by showing which operators dominate runtime and how memory behavior changes across workload conditions.

## Recommended Reproduction Workflow

For a clean reproduction, use the following sequence:

```bash
pip install -r requirements.txt
python run_suite.py --device auto --experiments all --profile
python analysis/final_analysis.py
python selector.py --device cuda --batch-size 8 --seq-length 128 --objective throughput
python selector.py --device cpu --batch-size 4 --seq-length 128 --objective latency
```

Then review:

```text
results/raw/latest_all_results.csv
results/reports/final_analysis.md
results/plots/
results/profiles/
```

## Packaging for Submission

For course submission, include the following files and folders in the code zip:

```text
README.md
requirements.txt
config.py
run_suite.py
selector.py
analysis/
experiments/
utils/
results/raw/
results/plots/
results/profiles/
results/reports/
```

Do not include local environment folders or cache directories such as:

```text
.git/
.venv/
__pycache__/
.cache/
Hugging Face cache folders
```

## Summary

This repository now contains a complete Phase 1 implementation of the transformer inference benchmarking project. It includes CPU/GPU experiments, batch-size and sequence-length studies, precision and compilation comparisons, profiler outputs, generated plots, final analysis summaries, and a workload-aware selector.

The current results show that inference performance is workload-dependent. GPU execution provides large speedups, batching improves throughput until saturation, reduced precision can improve performance depending on hardware support, and compilation benefits vary with workload and environment.
