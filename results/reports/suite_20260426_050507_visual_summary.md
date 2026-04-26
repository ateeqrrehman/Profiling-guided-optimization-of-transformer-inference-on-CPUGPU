Baseline CPU vs GPU latency speedup:
- short: CPU 326.37 ms vs GPU 30.67 ms (10.64x)
- medium: CPU 548.22 ms vs GPU 22.42 ms (24.46x)
- long: CPU 729.32 ms vs GPU 81.45 ms (8.95x)

Best throughput in the batch-size sweep:
- CUDA batch=16, seq=128, throughput=6072.65 tokens/s

Best precision configuration:
- CUDA fp16 at batch=4, seq=128, latency=48.12 ms, throughput=10640.25 tokens/s

Compilation caveats:
- CPU: Compiled execution failed at runtime: name 'torch' is not defined
- CUDA: Compiled execution failed at runtime: name 'torch' is not defined