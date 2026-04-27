Baseline CPU vs GPU latency speedup:
- short: CPU 102.28 ms vs GPU 6.41 ms (15.96x)
- medium: CPU 311.66 ms vs GPU 10.80 ms (28.85x)
- long: CPU 478.49 ms vs GPU 18.39 ms (26.01x)

Best throughput in the batch-size sweep:
- CUDA batch=8, seq=128, throughput=27384.97 tokens/s

Best precision configuration:
- CUDA fp16 at batch=4, seq=128, latency=8.36 ms, throughput=61224.26 tokens/s