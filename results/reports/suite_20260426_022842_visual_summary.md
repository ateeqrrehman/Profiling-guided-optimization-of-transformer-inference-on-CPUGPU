Best throughput in the batch-size sweep:
- CPU batch=16, seq=128, throughput=306.70 tokens/s

Best precision configuration:
- CPU int8_dynamic at batch=4, seq=128, latency=1893.43 ms, throughput=270.41 tokens/s

Compilation caveats:
- CPU: Compiled execution failed at runtime: RuntimeError: Compiler: cl is not found.