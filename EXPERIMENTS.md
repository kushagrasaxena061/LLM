# EXPERIMENTS.md
## Architectural Configuration Tradeoffs

This document tracks empirical experiments comparing different hyperparameter setups.

| Experiment | Layers | d_model | Heads | Params (M) | Tokens/sec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wide & Shallow** | 2 | 256 | 8 | 2.35M | 70432.6 |
| **Narrow & Deep** | 8 | 128 | 4 | 2.23M | 36756.7 |

### Decision Log
- **Result:** Deeper networks generally compound latency linearly due to sequential matrix multiplications, whereas wider networks can better utilize parallel compute up to the hardware's limit.
- **Decision:** For the final 151M model, a balanced approach (12 layers, 768 d_model) was selected to optimize the tradeoff between representational depth and memory bandwidth.