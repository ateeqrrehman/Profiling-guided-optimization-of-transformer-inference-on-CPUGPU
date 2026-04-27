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

- **short (32 tokens):** CPU latency=102.28 ms, GPU latency=6.41 ms, GPU speedup=15.96x.
- **medium (128 tokens):** CPU latency=311.66 ms, GPU latency=10.80 ms, GPU speedup=28.85x.
- **long (256 tokens):** CPU latency=478.49 ms, GPU latency=18.39 ms, GPU speedup=26.01x.

Interpretation: the GPU is expected to win because transformer inference contains many matrix operations that can be parallelized. The size of the speedup depends on sequence length, batch size, and overhead.

## Batch Size Analysis

### CPU
- batch=1: latency=287.11 ms, throughput=445.83 tokens/s, peak memory=1701.31 MB
- batch=2: latency=438.38 ms, throughput=583.97 tokens/s, peak memory=1701.54 MB
- batch=4: latency=911.85 ms, throughput=561.50 tokens/s, peak memory=1744.89 MB
- batch=8: latency=1831.17 ms, throughput=559.20 tokens/s, peak memory=1815.06 MB
- batch=16: latency=3403.00 ms, throughput=601.82 tokens/s, peak memory=2002.48 MB
- Best throughput for CPU occurs at batch=16 with 601.82 tokens/s.

### CUDA
- batch=1: latency=10.90 ms, throughput=11742.97 tokens/s, peak memory=370.89 MB
- batch=2: latency=15.99 ms, throughput=16009.50 tokens/s, peak memory=396.81 MB
- batch=4: latency=20.13 ms, throughput=25432.55 tokens/s, peak memory=449.83 MB
- batch=8: latency=37.39 ms, throughput=27384.97 tokens/s, peak memory=552.35 MB
- batch=16: latency=81.96 ms, throughput=24986.97 tokens/s, peak memory=760.18 MB
- Best throughput for CUDA occurs at batch=8 with 27384.97 tokens/s.

Interpretation: batching usually improves throughput by increasing hardware utilization, but very large batches can increase latency and memory pressure. The best batch size is therefore workload- and hardware-dependent.

## Sequence Length Analysis

### CPU
- seq=32: latency=102.49 ms, throughput=312.23 tokens/s, peak memory=2050.51 MB
- seq=128: latency=317.86 ms, throughput=402.69 tokens/s, peak memory=2050.83 MB
- seq=256: latency=448.44 ms, throughput=570.87 tokens/s, peak memory=2051.23 MB

### CUDA
- seq=32: latency=6.81 ms, throughput=4700.14 tokens/s, peak memory=351.29 MB
- seq=128: latency=10.85 ms, throughput=11793.00 tokens/s, peak memory=370.89 MB
- seq=256: latency=15.68 ms, throughput=16329.25 tokens/s, peak memory=397.02 MB

Interpretation: sequence length is central for transformer workloads because longer inputs increase the amount of token interaction and memory movement in attention-related computation.

## Precision Analysis

### CPU
- fp32: latency=815.89 ms, throughput=627.54 tokens/s, peak memory=2056.74 MB, perplexity proxy=2593.88
- int8_dynamic: latency=803.23 ms, throughput=637.42 tokens/s, peak memory=2226.42 MB, perplexity proxy=4082.07
- Best throughput for CPU is int8_dynamic with 637.42 tokens/s.

### CUDA
- bf16: latency=31.37 ms, throughput=16321.81 tokens/s, peak memory=748.46 MB, perplexity proxy=2635.04
- fp16: latency=8.36 ms, throughput=61224.26 tokens/s, peak memory=574.15 MB, perplexity proxy=2594.58
- fp32: latency=21.27 ms, throughput=24072.78 tokens/s, peak memory=449.83 MB, perplexity proxy=2593.89
- Best throughput for CUDA is fp16 with 61224.26 tokens/s.

Interpretation: reduced precision can improve speed or memory efficiency, but the result depends on hardware support and implementation overhead. The quality column is a fixed-prompt perplexity proxy, so it should be interpreted as a lightweight sanity check rather than a full language-model benchmark.

## Compilation Analysis

- Compile-requested rows: 6
- Actually compiled rows: 6
- Fallback/non-compiled rows after compile request: 0

### CPU
- seq=32: eager=252.57 ms, compile-requested=224.31 ms, delta=28.26 ms, actual compiled=True.
- seq=128: eager=990.77 ms, compile-requested=824.87 ms, delta=165.89 ms, actual compiled=True.
- seq=256: eager=1748.19 ms, compile-requested=1696.65 ms, delta=51.54 ms, actual compiled=True.

### CUDA
- seq=32: eager=9.43 ms, compile-requested=13.36 ms, delta=-3.93 ms, actual compiled=True.
- seq=128: eager=20.99 ms, compile-requested=16.11 ms, delta=4.88 ms, actual compiled=True.
- seq=256: eager=38.73 ms, compile-requested=34.40 ms, delta=4.33 ms, actual compiled=True.

All compile-requested rows remained compiled, so this run provides a direct eager-vs-compiled comparison.

## Compute-bound vs Memory-bound Interpretation

- Average measured GPU utilization: 45.19%
- Maximum measured GPU utilization: 80.80%
- Interpretation: this run shows mixed behavior; some workloads use the GPU well, while others are limited by overhead, memory, or small batch effects.
- CPU batch sweep: memory increases by 301.17 MB from smallest to largest batch, while throughput changes by 1.35x.
- CUDA batch sweep: memory increases by 389.30 MB from smallest to largest batch, while throughput changes by 2.13x.
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
