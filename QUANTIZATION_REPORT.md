# QUANTIZATION_REPORT.md

## Empirical Precision Tradeoff Report

This report compares the physical footprint and inference latency of varying numerical precisions.

| Precision | Size (MB) | TTFT (ms) | ITL (ms) | Throughput (Tokens/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Base)** | 4.40 | 14.47 | 0.89 | 638.12 |
| **FP16 (Half)** | 2.21 | 3.18 | 1.30 | 719.14 |
| **INT8 (Quantized)** | 1.43 | 2.35 | 1.76 | 559.48 |

### Architectural Findings:
- **Memory:** INT8 significantly reduces the model's physical footprint compared to FP32.
- **Throughput:** While INT8 lowers memory bandwidth requirements, execution speed is highly dependent on specific CPU/GPU instruction sets for 8-bit math.