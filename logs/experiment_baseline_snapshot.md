# Fixed Baseline Snapshot - 2026-09-12

Copied from [the current METRICS report](../docs/METRICS.md), without recomputation or changing the evaluation method. Underlying real [test script output](../abyss/logs/phase4_corrected_map.json) used the baseline-640 test split.

Checkpoint SHA-256: `baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d`.

| Class | AP@0.5 | AP@0.5:0.95 |
| --- | ---: | ---: |
| Plane | 0.779424 | 0.378747 |
| Ship | 0.957086 | 0.674665 |
| Shipwreck | 0.144992 | 0.058058 |
| **Overall mAP** | **0.627168** | **0.370490** |

This snapshot is immutable for comparisons in the bounded experiment. Candidate weights and experimental results do not replace the current baseline without user approval.
