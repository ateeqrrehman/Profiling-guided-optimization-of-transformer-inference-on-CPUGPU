# Final Analysis Report


## Proposal Coverage Check

- ✅ Baseline CPU/GPU experiment
- ✅ Batch-size experiment
- ✅ Sequence-length experiment
- ✅ Precision / quantization experiment
- ✅ Compilation experiment
- ✅ Profiler-enabled output fields
- ✅ Quality metric fields
- ✅ Utilization fields

Measured devices: cpu, cuda
Measured batch sizes: [1, 2, 4, 8, 16]
Measured sequence lengths: [32, 128, 256]
Measured precision modes: bf16, fp16, fp32, int8_dynamic

## CPU vs GPU Analysis

- **short (32 tokens):** CPU latency=326.37 ms, GPU latency=30.67 ms, GPU speedup=10.64x.
- **medium (128 tokens):** CPU latency=548.22 ms, GPU latency=22.42 ms, GPU speedup=24.46x.
- **long (256 tokens):** CPU latency=729.32 ms, GPU latency=81.45 ms, GPU speedup=8.95x.

Interpretation: the GPU is expected to win because transformer inference contains many matrix operations that can be parallelized. The size of the speedup depends on sequence length, batch size, and overhead.

## Batch Size Analysis

### CPU
- batch=1: latency=555.36 ms, throughput=230.48 tokens/s, peak memory=4901.28 MB
- batch=2: latency=686.14 ms, throughput=373.10 tokens/s, peak memory=4931.99 MB
- batch=4: latency=1216.78 ms, throughput=420.78 tokens/s, peak memory=4879.99 MB
- batch=8: latency=2043.23 ms, throughput=501.17 tokens/s, peak memory=5087.67 MB
- batch=16: latency=4183.19 ms, throughput=489.58 tokens/s, peak memory=5292.66 MB
- Best throughput for CPU occurs at batch=8 with 501.17 tokens/s.

### CUDA
- batch=1: latency=66.51 ms, throughput=1924.64 tokens/s, peak memory=363.55 MB
- batch=2: latency=74.63 ms, throughput=3430.03 tokens/s, peak memory=389.47 MB
- batch=4: latency=131.16 ms, throughput=3903.63 tokens/s, peak memory=442.49 MB
- batch=8: latency=231.92 ms, throughput=4415.30 tokens/s, peak memory=545.01 MB
- batch=16: latency=337.25 ms, throughput=6072.65 tokens/s, peak memory=752.84 MB
- Best throughput for CUDA occurs at batch=16 with 6072.65 tokens/s.

Interpretation: batching usually improves throughput by increasing hardware utilization, but very large batches can increase latency and memory pressure. The best batch size is therefore workload- and hardware-dependent.

## Sequence Length Analysis

### CPU
- seq=32: latency=295.16 ms, throughput=108.42 tokens/s, peak memory=4926.85 MB
- seq=128: latency=493.91 ms, throughput=259.16 tokens/s, peak memory=4946.69 MB
- seq=256: latency=621.69 ms, throughput=411.78 tokens/s, peak memory=4979.54 MB

### CUDA
- seq=32: latency=25.06 ms, throughput=1276.79 tokens/s, peak memory=343.95 MB
- seq=128: latency=25.72 ms, throughput=4976.60 tokens/s, peak memory=363.55 MB
- seq=256: latency=35.01 ms, throughput=7311.34 tokens/s, peak memory=389.68 MB

Interpretation: sequence length is central for transformer workloads because longer inputs increase the amount of token interaction and memory movement in attention-related computation.

## Precision Analysis

### CPU
- fp32: latency=1187.05 ms, throughput=431.32 tokens/s, peak memory=4916.58 MB, perplexity proxy=2593.87
- int8_dynamic: latency=1079.04 ms, throughput=474.50 tokens/s, peak memory=5462.94 MB, perplexity proxy=4082.06
- Best throughput for CPU is int8_dynamic with 474.50 tokens/s.

### CUDA
- bf16: latency=78.00 ms, throughput=6563.70 tokens/s, peak memory=728.54 MB, perplexity proxy=2726.65
- fp16: latency=48.12 ms, throughput=10640.25 tokens/s, peak memory=560.52 MB, perplexity proxy=2595.38
- fp32: latency=124.74 ms, throughput=4104.49 tokens/s, peak memory=442.49 MB, perplexity proxy=2593.88
- Best throughput for CUDA is fp16 with 10640.25 tokens/s.

Interpretation: reduced precision can improve speed or memory efficiency, but the result depends on hardware support and implementation overhead. The quality column is a fixed-prompt perplexity proxy, so it should be interpreted as a lightweight sanity check rather than a full language-model benchmark.

## Compilation Analysis

- Compile-requested rows: 6
- Actually compiled rows: 0
- Fallback/non-compiled rows after compile request: 6

### CPU
- seq=32: eager=301.79 ms, compile-requested=541.03 ms, delta=-239.24 ms, actual compiled=False.
- seq=128: eager=1154.66 ms, compile-requested=1126.02 ms, delta=28.64 ms, actual compiled=False.
- seq=256: eager=1993.53 ms, compile-requested=2026.27 ms, delta=-32.74 ms, actual compiled=False.

### CUDA
- seq=32: eager=110.90 ms, compile-requested=61.95 ms, delta=48.95 ms, actual compiled=False.
- seq=128: eager=132.94 ms, compile-requested=120.36 ms, delta=12.58 ms, actual compiled=False.
- seq=256: eager=188.04 ms, compile-requested=158.40 ms, delta=29.64 ms, actual compiled=False.

Important caveat: some compile-requested runs did not remain marked as compiled in the final results. This usually means PyTorch compilation fell back or the local environment did not fully support the requested backend. The benchmark still records those rows so the limitation is visible rather than hidden.

## Compute-bound vs Memory-bound Interpretation

- Average measured GPU utilization: 58.21%
- Maximum measured GPU utilization: 90.20%
- Interpretation: this run shows mixed behavior; some workloads use the GPU well, while others are limited by overhead, memory, or small batch effects.
- CPU batch sweep: memory increases by 391.39 MB from smallest to largest batch, while throughput changes by 2.12x.
- CUDA batch sweep: memory increases by 389.30 MB from smallest to largest batch, while throughput changes by 3.16x.
Overall interpretation: the project should not claim a single universal bottleneck. Instead, the evidence should be framed as workload-dependent: small batches may be overhead-limited, larger batches improve throughput until memory pressure or saturation appears, and longer sequences increase transformer-specific memory/computation cost.

## Workload-aware Selector Validation

Run these commands after generating `latest_all_results.csv`:

```bash
python selector.py --device cuda --batch-size 8 --seq-length 128 --objective throughput
python selector.py --device cpu --batch-size 4 --seq-length 128 --objective latency
```

The selector output should be included in the final report as the original contribution. It demonstrates that the project goes beyond benchmarking by using empirical results to recommend a configuration for a given workload.


## Recommended Final Steps

1. Save this report with the final submission materials.
2. Include the summary dashboard figure from `results/plots/`.
3. Include 1-2 profiler operator summaries from `results/profiles/`.
4. Report selector examples for one CPU workload and one GPU workload.
5. If compile-requested rows fall back to eager mode, explain this as an environment-dependent systems limitation rather than hiding it.
