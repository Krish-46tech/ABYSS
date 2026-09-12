# ABYSS Improvement Attempts

## 2026-09-12: Validation-only raw-confidence threshold sweep

**Reason.** The corrected test run showed 131 false positives at raw confidence 0.05 versus 44 at the deployed 0.25 threshold. A threshold change is cheap to validate without retraining or test-set selection.

**Method.** On 136 baseline-640 validation images, sweep raw detector confidence from 0.05 through 0.95 in 0.05 steps, using class-aware one-to-one IoU >= 0.5 matching and micro-F1 as the selection criterion. [Full sweep](../abyss/logs/phase4_validation_threshold_sweep/validation_threshold_sweep.csv), [selection JSON](../abyss/logs/phase4_validation_threshold_sweep/summary.json). Test was not used to select the threshold.

| Split / raw threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation / 0.25 current | 77 | 56 | 47 | 0.578947 | 0.620968 | 0.599222 |
| Validation / 0.55 selected by micro-F1 | 73 | 23 | 51 | 0.760417 | 0.588710 | 0.663636 |
| Test / 0.25 current | 77 | 44 | 34 | 0.636364 | 0.693694 | 0.663793 |
| Test / 0.55 preselected | 69 | 10 | 42 | 0.873418 | 0.621622 | 0.726316 |

Test rows come from [the fixed-threshold test report](../abyss/logs/phase4_corrected_details/summary.json), after validation selected 0.55. The threshold experiment itself was motivated by inspection of this test report, so the 0.55 test comparison is **exploratory, not a fresh confirmatory estimate**. The 0.55 test confusion matrix has **zero Plane predictions** and 13/17 Plane ground truths missed entirely. **Verdict: discard as a deployed threshold change** despite higher aggregate F1; aircraft detection is a core use case, and 0.55 is an unacceptable class tradeoff. Keep 0.25 until a class-aware operating-point policy is designed on validation and reviewed.

**Other ideas assessed, not claimed as attempts.** Plane has only 25 training labels but AP@0.5 is 0.779424; Shipwreck has 115 training labels yet AP@0.5 is 0.144992. These numbers alone do not establish that imbalance is the cause, so no class-weighted or oversampling retrain was run or claimed. V4 calibration is fresh for the unchanged checkpoint; no output-distribution change requires a refit. The test ECE gain and validation ECE loss are both documented in `docs/METRICS.md`, not hidden or tuned against test.

## 2026-09-12: Validation-ECE bin-count investigation

**Reason.** Ten-bin validation ECE rose from 0.083756 raw to 0.097176 final, while ten-bin test ECE fell from 0.072560 to 0.059635. The validation result needed a binning and sample-size audit before finalization.

**Real rerun.** [ECE bin audit](../abyss/logs/phase3_calibration_improved_v4/ece_bin_audit.json) verified the v4 checkpoint hash, the exact fitting validation split, 136 images, 124 labels (24 Plane, 74 Ship, 26 Shipwreck), and 227 in-sample predictions. Five-bin ECE raw/fused/final: **0.063979/0.044664/0.045771**. Ten-bin ECE: **0.083756/0.096462/0.097176**. The smallest occupied final bin contains 19 versus 2 predictions. **Verdict:** the apparent raw-to-final validation regression is not robust to binning and is more consistent with small-sample measurement sensitivity than a proven global calibration failure. Temperature still slightly worsens validation ECE relative to fused at both bin counts; no independent validation holdout exists. No calibration parameters were changed or selected using test data.

## 2026-09-12: Remove filename-based Plane relabeling

**Reason.** Backend logic changed `Ship` to `Plane` when an upload filename contained an aircraft token, masking the actual model error and making displayed class depend on a user-controlled filename.

**Real rerun and fix.** A validation-assigned raw Plane image was run through baseline-640/v4 CLI inference; the native output was **Ship, class 1, raw confidence 0.8526390790939331** ([saved prediction](../abyss/logs/final_plane_native_inference/plane-001_png.rf.dd5de3545eefa071ea7034204a08c0c1_detections.json)). Removed filename-dependent relabeling and filename forwarding. The API returned the same `Ship` class and raw confidence for both `plane-001.jpg` and `neutral.jpg`; the replacement test compares both responses with a direct YOLO prediction. Backend suite: **7 passed, 1 existing Starlette deprecation warning**. **Verdict: keep the integrity fix.** Classification metrics do not improve; the real failure is now visible. No commit was created.

## 2026-09-12: Bounded model experiment - baseline locked

The unchanged reference is saved in [experiment_baseline_snapshot.md](experiment_baseline_snapshot.md): Plane AP50/AP50:95 0.779424/0.378747, Ship 0.957086/0.674665, Shipwreck 0.144992/0.058058, overall mAP50/mAP50:95 0.627168/0.370490, checkpoint SHA-256 `baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d`. Candidate training and evaluation must be compared to this fixed published test run. No candidate is deployed or marked current without explicit user approval.

The [predeclared protocol](bounded_experiment_protocol.md) fixes four independent 25-epoch candidates, all initialized from `baseline_v1.pt` and selected on validation. `bounded_shipwreck_weight` began real training with custom Shipwreck-only frequency-derived classification weighting (other class weights unchanged relative to one another), while `bounded_shipwreck_aug` received a separately built train-only dataset with 115 transformed Shipwreck images added to 546 originals. Its generated YAML points to the untouched baseline validation/test directories; the builder confirmed zero train/held-out filename overlap. No candidate result is asserted until its checkpoint and evaluation JSON exist.

### Attempt 1: Shipwreck-focused classification loss weight

**Executed:** 25-epoch independent retrain, checkpoint [best.pt](../abyss/ml/detection/runs/bounded_shipwreck_weight/weights/best.pt), SHA-256 `75fea5d1e0f38b4ba28b9b6585efadb95514517278fe247d1a2bfd0144e6d644`. The saved checkpoint contains actual normalized class loss weights `[0.8060, 0.8060, 1.3880]` for Plane/Ship/Shipwreck. [Validation output](../abyss/logs/bounded_shipwreck_weight_val.json), [test output](../abyss/logs/bounded_shipwreck_weight_test.json), [rerunnable comparison](../abyss/logs/bounded_shipwreck_weight_comparison.json).

| Test class | Baseline AP50 / AP50:95 | Weighted AP50 / AP50:95 |
| --- | ---: | ---: |
| Plane | 0.779424 / 0.378747 | 0.735472 / 0.457477 |
| Ship | 0.957086 / 0.674665 | 0.941927 / 0.668797 |
| Shipwreck | 0.144992 / 0.058058 | 0.149287 / 0.074910 |
| **Overall mAP** | **0.627168 / 0.370490** | **0.608895 / 0.400395** |

**Validation selection check:** Shipwreck AP50 rose 0.218147 -> 0.248880, but Plane AP50 fell 0.703030 -> 0.639907; overall mAP50 fell 0.606812 -> 0.591787 and mAP50:95 fell 0.380731 -> 0.364403. On held-out test, Shipwreck AP50 gained only 0.004295 while Plane AP50 fell 0.043952 and overall mAP50 fell 0.018272; test mAP50:95 rose 0.029905. **Verdict: discard** as a Shipwreck-focused change because the Plane/Ship and overall AP50 tradeoffs violate the stated condition; the apparent test mAP50:95 gain does not override the validation selection failure. No deployment or calibration change.

### Attempt 2: Shipwreck-only augmented oversampling

**Executed:** independent 25-epoch retrain with 115 train-only Shipwreck affine transforms added to 546 original training images; no validation or test augmentation. Checkpoint [best.pt](../abyss/ml/detection/runs/bounded_shipwreck_aug/weights/best.pt), SHA-256 `cb3d3a3ca9f614988dda6479ef4fe5ae911389b2ee3095abaaecea14259d5a0b`. [Validation output](../abyss/logs/bounded_shipwreck_aug_val.json), [test output](../abyss/logs/bounded_shipwreck_aug_test.json), [comparison](../abyss/logs/bounded_shipwreck_aug_comparison.json).

| Test class | Baseline AP50 / AP50:95 | Augmented AP50 / AP50:95 |
| --- | ---: | ---: |
| Plane | 0.779424 / 0.378747 | 0.653440 / 0.353499 |
| Ship | 0.957086 / 0.674665 | 0.951313 / 0.627589 |
| Shipwreck | 0.144992 / 0.058058 | 0.200944 / 0.085831 |
| **Overall mAP** | **0.627168 / 0.370490** | **0.601899 / 0.355640** |

**Validation selection check:** Shipwreck AP50 0.218147 -> 0.273395, Plane 0.703030 -> 0.687521, Ship 0.899260 -> 0.886047; overall mAP50 0.606812 -> 0.615654 but mAP50:95 0.380731 -> 0.375111. On held-out test, Shipwreck AP50 rose 0.055952, but Plane AP50 fell 0.125984, overall mAP50 fell 0.025269, and mAP50:95 fell 0.014850. **Verdict: discard** for deployment; the targeted gain does not justify the overall and Plane regression. No deployment or calibration change.

### Attempt 3: Box regression loss weight 7.5 -> 9.0

**Executed:** independent 25-epoch retrain with only YOLO `box` weight changed; checkpoint [best.pt](../abyss/ml/detection/runs/bounded_box_weight/weights/best.pt), SHA-256 `e0d477460dc220c752dc3787a8747c3bf845e7ff0ce3178339f448f8ca6df117`. [Validation output](../abyss/logs/bounded_box_weight_val.json), [test output](../abyss/logs/bounded_box_weight_test.json), [comparison](../abyss/logs/bounded_box_weight_comparison.json).

| Test class | Baseline AP50 / AP50:95 | Box-weight AP50 / AP50:95 |
| --- | ---: | ---: |
| Plane | 0.779424 / 0.378747 | 0.759924 / 0.389192 |
| Ship | 0.957086 / 0.674665 | 0.924013 / 0.646498 |
| Shipwreck | 0.144992 / 0.058058 | 0.212485 / 0.067935 |
| **Overall mAP** | **0.627168 / 0.370490** | **0.632141 / 0.367875** |

**Validation selection check:** overall mAP50 0.606812 -> 0.594335 and mAP50:95 0.380731 -> 0.380616; Shipwreck AP50 0.218147 -> 0.152828. On test, the intended mAP50:95 target fell 0.002615 even though mAP50 rose 0.004973; Ship AP50 fell 0.033074. **Verdict: discard** for localization, since it did not improve the target metric on either split and validation selection failed. No deployment or calibration change.

### Attempt 4: Close mosaic for final 15 rather than final 10 epochs

**Executed:** independent 25-epoch retrain with only `close_mosaic=15`; checkpoint [best.pt](../abyss/ml/detection/runs/bounded_close_mosaic/weights/best.pt), SHA-256 `f30e51b6c76889b204ef572889ad06d9977f4c882c3ed6baf15f40b16022254c`. [Validation output](../abyss/logs/bounded_close_mosaic_val.json), [test output](../abyss/logs/bounded_close_mosaic_test.json), [comparison](../abyss/logs/bounded_close_mosaic_comparison.json).

| Test class | Baseline AP50 / AP50:95 | Close-mosaic AP50 / AP50:95 |
| --- | ---: | ---: |
| Plane | 0.779424 / 0.378747 | 0.738115 / 0.390198 |
| Ship | 0.957086 / 0.674665 | 0.943036 / 0.649396 |
| Shipwreck | 0.144992 / 0.058058 | 0.219042 / 0.088640 |
| **Overall mAP** | **0.627168 / 0.370490** | **0.633398 / 0.376078** |

**Validation selection check:** overall mAP50 0.606812 -> 0.615485 and mAP50:95 0.380731 -> 0.414709; Plane AP50 0.703030 -> 0.742573, Ship 0.899260 -> 0.908479, Shipwreck 0.218147 -> 0.195402. On held-out test, overall gains are small (+0.006230 mAP50, +0.005588 mAP50:95) while Plane AP50 falls 0.041309 and Ship AP50 falls 0.014050; Shipwreck AP50 rises 0.074050. **Verdict: promising aggregate validation result, but do not adopt automatically.** The test Plane drop matters for aircraft use and the aggregate test gain is modest. Keep baseline-640/v4 current pending the user's explicit decision; candidate-specific calibration and a full corrected report would be needed if adopted. Test metrics were not used to tune or retrain this candidate.

**Bounded-experiment decision:** No candidate delivers a sufficiently clear gain without an important tradeoff. Class weighting, Shipwreck augmentation, and increased box weight are discarded. The close-mosaic candidate is retained only as an experimental checkpoint for review, not as deployed/current. Optional AI4Shipwrecks and 960px retrains were not attempted; no compatibility or high-resolution calibration claims are made. Full comparison is in [bounded_experiment_report.md](bounded_experiment_report.md).

## 2026-09-12: Baseline decision finalized

The user explicitly chose to **keep baseline-640/v4 as the final deployed checkpoint** and adopt no bounded-experiment candidate. No candidate calibration or deployment references were changed. The decision is closed for this phase; the close-mosaic checkpoint remains experimental only. Phase 5 fast-mode frontend work proceeded under this unchanged baseline.
