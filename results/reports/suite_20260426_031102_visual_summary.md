Baseline CPU vs GPU latency speedup:
- short: CPU 172.31 ms vs GPU 56.91 ms (3.03x)
- medium: CPU 767.69 ms vs GPU 22.15 ms (34.66x)
- long: CPU 1017.73 ms vs GPU 23.89 ms (42.61x)

Best throughput in the batch-size sweep:
- CUDA batch=8, seq=128, throughput=25107.03 tokens/s

Best precision configuration:
- CUDA bf16 at batch=4, seq=128, latency=25.61 ms, throughput=19990.11 tokens/s

Compilation caveats:
- CPU: Compiled execution failed at runtime: backend='inductor' raised:
- CUDA: Compiled execution failed at runtime: backend='inductor' raised: