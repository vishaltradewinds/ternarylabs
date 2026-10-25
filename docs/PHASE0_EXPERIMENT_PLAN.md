# Phase 0 — Ternary Weight Feasibility Gate

## Question
For a fixed, reproducible linear workload, what representation-size reduction is possible with ternary weights, and what output error does a simple post-training ternary quantizer introduce?

## Scope
- Synthetic, seeded FP32 inputs and weights.
- Per-output-channel ternary codes in {-1, 0, +1}.
- Per-channel FP32 scales.
- Output MAE, RMSE, relative L2 error, nonzero fraction, and theoretical packed storage.
- Correctness check for the reconstructed ternary matrix.

## Explicit exclusions
This phase does not establish:
- model/task accuracy on a real neural network;
- faster execution, because ordinary NumPy matmul is used;
- measured energy, power, chip area, or silicon performance;
- novelty or freedom to operate.

## Reproduction
Install Python 3.10+ and NumPy, then run:

```bash
python experiments/phase0_ternary_weights.py --output results/phase0.json
```

Change the seed, shape, and threshold only as separate recorded runs. Keep the FP32 reference workload identical.

## Gate
Proceed to a real-model benchmark only after:
1. script runs reproducibly;
2. representation correctness passes;
3. the error/storage trade-off is recorded;
4. baseline and limitations are documented.

## Shakti Standards evidence
- Reality/context: this is a synthetic representation experiment.
- Feasibility: low-cost software-only test.
- Validation: deterministic seed and numerical assertion.
- Measurement: metrics saved as JSON.
- Assurance: no unsupported hardware or energy claims.
