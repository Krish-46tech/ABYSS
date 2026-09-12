# ABYSS bounded detector experiment (2026-09-12)

This is an experiment report, not a change to the deployed detector. **Final user decision: keep baseline-640/v4; no candidate adopted.** The fixed published baseline is [experiment_baseline_snapshot.md](experiment_baseline_snapshot.md), copied from `docs/METRICS.md` before the attempts. All candidates were independently initialized from `abyss/ml/detection/weights/baseline_v1.pt`, trained for 25 epochs at 640 px using train labels and validation selection, then evaluated with `abyss/ml/detection/evaluate.py` on the original validation and held-out test splits. The test split was not used in fitting or hyperparameter selection. Each candidate has its own checkpoint hash, run configuration, split-specific metric JSON, and checked comparison JSON under `abyss/logs/`.

## Fixed test comparison

Each class cell is AP@0.5 / AP@0.5:0.95; overall is mAP@0.5 / mAP@0.5:0.95. Test contains 129 images and 111 labels (Plane 17, Ship 74, Shipwreck 20). Numbers below are from the saved evaluator JSONs, not estimates.

| Candidate | Plane | Ship | Shipwreck | Overall | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline-640/v4 | 0.779424 / 0.378747 | 0.957086 / 0.674665 | 0.144992 / 0.058058 | 0.627168 / 0.370490 | Current, unchanged |
| Shipwreck loss weight | 0.735472 / 0.457477 | 0.941927 / 0.668797 | 0.149287 / 0.074910 | 0.608895 / 0.400395 | Discard: validation overall declined; Plane/Ship AP50 declined |
| Shipwreck-only augmentation | 0.653440 / 0.353499 | 0.951313 / 0.627589 | 0.200944 / 0.085831 | 0.601899 / 0.355640 | Discard: overall and Plane declined |
| Box weight 9.0 | 0.759924 / 0.389192 | 0.924013 / 0.646498 | 0.212485 / 0.067935 | 0.632141 / 0.367875 | Discard: target mAP@0.5:0.95 declined |
| Close mosaic 15 | 0.738115 / 0.390198 | 0.943036 / 0.649396 | 0.219042 / 0.088640 | 0.633398 / 0.376078 | Experimental only: small overall gain, Plane AP50 decline |

## Validation selection check

The original validation split has 136 images. These results are the pre-test basis for comparing changes; all candidates used the same evaluation script and 640 px input.

| Candidate | Plane AP50 / AP50:95 | Ship AP50 / AP50:95 | Shipwreck AP50 / AP50:95 | Overall mAP50 / mAP50:95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline-640 | 0.703030 / 0.448000 | 0.899260 / 0.623524 | 0.218147 / 0.070668 | 0.606812 / 0.380731 |
| Shipwreck loss weight | 0.639907 / 0.375494 | 0.886575 / 0.617705 | 0.248880 / 0.100010 | 0.591787 / 0.364403 |
| Shipwreck-only augmentation | 0.687521 / 0.469052 | 0.886047 / 0.580937 | 0.273395 / 0.075344 | 0.615654 / 0.375111 |
| Box weight 9.0 | 0.725885 / 0.481569 | 0.904291 / 0.609236 | 0.152828 / 0.051043 | 0.594335 / 0.380616 |
| Close mosaic 15 | 0.742573 / 0.545134 | 0.908479 / 0.616135 | 0.195402 / 0.082856 | 0.615485 / 0.414709 |

## Provenance and decision

| Candidate | Checkpoint SHA-256 | Saved evidence |
| --- | --- | --- |
| Baseline-640/v4 | `baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d` | `abyss/logs/phase4_corrected_map.json`, `abyss/logs/bounded_baseline_val.json` |
| Shipwreck loss weight | `75fea5d1e0f38b4ba28b9b6585efadb95514517278fe247d1a2bfd0144e6d644` | `abyss/logs/bounded_shipwreck_weight_{val,test,comparison}.json` |
| Shipwreck-only augmentation | `cb3d3a3ca9f614988dda6479ef4fe5ae911389b2ee3095abaaecea14259d5a0b` | `abyss/logs/bounded_shipwreck_aug_{val,test,comparison}.json` |
| Box weight 9.0 | `e0d477460dc220c752dc3787a8747c3bf845e7ff0ce3178339f448f8ca6df117` | `abyss/logs/bounded_box_weight_{val,test,comparison}.json` |
| Close mosaic 15 | `f30e51b6c76889b204ef572889ad06d9977f4c882c3ed6baf15f40b16022254c` | `abyss/logs/bounded_close_mosaic_{val,test,comparison}.json` |

For every candidate, `abyss/ml/detection/runs/bounded_*/args.yaml`, `results.csv`, `weights/best.pt`, and `candidate_summary.json` record the real run and its selected checkpoint. `compare_bounded_candidate.py` rejects mismatched splits, checkpoint hashes, data YAML, class IDs, image sizes, or image counts.

**Final decision:** keep baseline-640/v4 as the deployed model. The close-mosaic candidate is the only one with gains in both aggregate validation metrics and both aggregate test metrics, but its test mAP@0.5:0.95 gain is just 0.005588 while Plane AP@0.5 falls 0.041309 and Ship AP@0.5 falls 0.014050. With only 17 Plane test labels, this is not a clean no-tradeoff deployment gain. The candidate remains an experimental artifact and is not wired into inference. The decision is closed for this phase; v4 calibration remains paired only with the unchanged baseline checkpoint.

The optional AI4Shipwrecks extra-data and 960px experiments were not run due compute/time. No claim about their effectiveness or dataset compatibility is made. No Phase 5 work was started.
