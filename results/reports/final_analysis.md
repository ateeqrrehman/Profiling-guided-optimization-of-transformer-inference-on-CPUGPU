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

- **short (32 tokens):** CPU latency=174.05 ms, GPU latency=24.99 ms, GPU speedup=6.96x.
- **medium (128 tokens):** CPU latency=325.22 ms, GPU latency=31.23 ms, GPU speedup=10.41x.
- **long (256 tokens):** CPU latency=429.93 ms, GPU latency=59.12 ms, GPU speedup=7.27x.

Interpretation: the GPU is expected to win because transformer inference contains many matrix operations that can be parallelized. The size of the speedup depends on sequence length, batch size, and overhead.

## Batch Size Analysis

### CPU
- batch=1: latency=787.58 ms, throughput=162.52 tokens/s, peak memory=4915.71 MB
- batch=2: latency=917.64 ms, throughput=278.98 tokens/s, peak memory=4946.38 MB
- batch=4: latency=1501.97 ms, throughput=340.89 tokens/s, peak memory=4890.23 MB
- batch=8: latency=2404.44 ms, throughput=425.88 tokens/s, peak memory=5098.19 MB
- batch=16: latency=4488.41 ms, throughput=456.29 tokens/s, peak memory=5303.31 MB
- Best throughput for CPU occurs at batch=16 with 456.29 tokens/s.

### CUDA
- batch=1: latency=19.71 ms, throughput=6493.65 tokens/s, peak memory=363.55 MB
- batch=2: latency=18.42 ms, throughput=13895.93 tokens/s, peak memory=389.47 MB
- batch=4: latency=41.49 ms, throughput=12340.22 tokens/s, peak memory=442.49 MB
- batch=8: latency=79.34 ms, throughput=12906.81 tokens/s, peak memory=545.01 MB
- batch=16: latency=119.37 ms, throughput=17156.24 tokens/s, peak memory=752.84 MB
- Best throughput for CUDA occurs at batch=16 with 17156.24 tokens/s.

Interpretation: batching usually improves throughput by increasing hardware utilization, but very large batches can increase latency and memory pressure. The best batch size is therefore workload- and hardware-dependent.

## Sequence Length Analysis

### CPU
- seq=32: latency=141.16 ms, throughput=226.69 tokens/s, peak memory=4935.10 MB
- seq=128: latency=794.55 ms, throughput=161.10 tokens/s, peak memory=4958.08 MB
- seq=256: latency=802.75 ms, throughput=318.90 tokens/s, peak memory=4982.87 MB

### CUDA
- seq=32: latency=25.59 ms, throughput=1250.72 tokens/s, peak memory=343.95 MB
- seq=128: latency=35.51 ms, throughput=3605.02 tokens/s, peak memory=363.55 MB
- seq=256: latency=22.22 ms, throughput=11520.91 tokens/s, peak memory=389.68 MB

Interpretation: sequence length is central for transformer workloads because longer inputs increase the amount of token interaction and memory movement in attention-related computation.

## Precision Analysis

### CPU
- fp32: latency=1442.94 ms, throughput=354.83 tokens/s, peak memory=4927.31 MB, perplexity proxy=2593.87
- int8_dynamic: latency=1425.37 ms, throughput=359.20 tokens/s, peak memory=5432.96 MB, perplexity proxy=4082.06
- Best throughput for CPU is int8_dynamic with 359.20 tokens/s.

### CUDA
- bf16: latency=24.67 ms, throughput=20754.66 tokens/s, peak memory=728.54 MB, perplexity proxy=2726.65
- fp16: latency=38.67 ms, throughput=13240.92 tokens/s, peak memory=560.52 MB, perplexity proxy=2595.38
- fp32: latency=36.99 ms, throughput=13843.30 tokens/s, peak memory=442.49 MB, perplexity proxy=2593.88
- Best throughput for CUDA is bf16 with 20754.66 tokens/s.

Interpretation: reduced precision can improve speed or memory efficiency, but the result depends on hardware support and implementation overhead. The quality column is a fixed-prompt perplexity proxy, so it should be interpreted as a lightweight sanity check rather than a full language-model benchmark.

## Compilation Analysis

- Compile-requested rows: 6
- Actually compiled rows: 0
- Fallback/non-compiled rows after compile request: 6

### CPU
- seq=32: eager=592.64 ms, compile-requested=729.05 ms, delta=-136.41 ms, actual compiled=False.
- seq=128: eager=1386.91 ms, compile-requested=1006.20 ms, delta=380.70 ms, actual compiled=False.
- seq=256: eager=2044.45 ms, compile-requested=2001.40 ms, delta=43.05 ms, actual compiled=False.

### CUDA
- seq=32: eager=53.85 ms, compile-requested=96.25 ms, delta=-42.41 ms, actual compiled=False.
- seq=128: eager=123.21 ms, compile-requested=107.51 ms, delta=15.70 ms, actual compiled=False.
- seq=256: eager=219.06 ms, compile-requested=215.43 ms, delta=3.64 ms, actual compiled=False.

Important caveat: some compile-requested runs did not remain marked as compiled in the final results. This usually means PyTorch compilation fell back or the local environment did not fully support the requested backend. The benchmark still records those rows so the limitation is visible rather than hidden.

## Compute-bound vs Memory-bound Interpretation

- Average measured GPU utilization: 42.45%
- Maximum measured GPU utilization: 83.30%
- Interpretation: this run shows mixed behavior; some workloads use the GPU well, while others are limited by overhead, memory, or small batch effects.
- CPU batch sweep: memory increases by 387.60 MB from smallest to largest batch, while throughput changes by 2.81x.
- CUDA batch sweep: memory increases by 389.30 MB from smallest to largest batch, while throughput changes by 2.64x.
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
