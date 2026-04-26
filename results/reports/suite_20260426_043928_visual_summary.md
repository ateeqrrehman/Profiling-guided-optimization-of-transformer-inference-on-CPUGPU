Baseline CPU vs GPU latency speedup:
- short: CPU 174.05 ms vs GPU 24.99 ms (6.96x)
- medium: CPU 325.22 ms vs GPU 31.23 ms (10.41x)
- long: CPU 429.93 ms vs GPU 59.12 ms (7.27x)

Best throughput in the batch-size sweep:
- CUDA batch=16, seq=128, throughput=17156.24 tokens/s

Best precision configuration:
- CUDA bf16 at batch=4, seq=128, latency=24.67 ms, throughput=20754.66 tokens/s

Compilation caveats:
- CPU: Compiled execution failed at runtime: name 'torch' is not defined
- CUDA: Compiled execution failed at runtime: name 'torch' is not defined