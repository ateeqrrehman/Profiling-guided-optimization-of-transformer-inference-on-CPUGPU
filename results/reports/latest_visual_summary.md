Baseline CPU vs GPU latency speedup:
- short: CPU 19.31 ms vs GPU 5.78 ms (3.34x)
- medium: CPU 61.29 ms vs GPU 8.08 ms (7.58x)
- long: CPU 102.23 ms vs GPU 10.20 ms (10.03x)

Best throughput in the batch-size sweep:
- CUDA batch=8, seq=128, throughput=36212.57 tokens/s

Best precision configuration:
- CUDA bf16 at batch=4, seq=128, latency=12.03 ms, throughput=42554.57 tokens/s

Compilation caveats:
- CPU: Compiled execution failed at runtime: RuntimeError: Compiler: cl is not found.
- CUDA: Compiled execution failed at runtime: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton