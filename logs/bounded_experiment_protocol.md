# Bounded Detector Experiment Protocol

Fixed comparison: [published baseline snapshot](experiment_baseline_snapshot.md), checkpoint SHA-256 `baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d`.

All candidates start from the same `baseline_v1.pt` initialization and use the recorded `improved_v2_baseline_finetune_adamw/args.yaml` 25-epoch, baseline-640, AdamW recipe, except for **one** declared change each. Training and best-checkpoint selection use train/validation only. The current deployed improved_v2 checkpoint and v4 calibration remain unchanged until the user decides.

| Candidate | Sole intentional change | Motivation |
| --- | --- | --- |
| `bounded_shipwreck_weight` | Shipwreck classification-loss weight relative to Plane/Ship is `sqrt(max(other-class training count) / Shipwreck training count)`, normalized to mean 1 by the trainer. Counts come from train labels, not literals. | Raise Shipwreck signal moderately without making the rarer Plane class dominate the focal experiment. |
| `bounded_shipwreck_aug` | Add one deterministic rotation/scale/flip augmented training copy for every Shipwreck-bearing train image; do not augment validation/test or any other class. | Increase effective diversity of Shipwreck training examples. |
| `bounded_box_weight` | Increase YOLO `box` from the recorded 7.5 to 9.0; all other settings unchanged. | Test tighter box localization. |
| `bounded_close_mosaic` | Increase recorded `close_mosaic` from 10 to 15; all other settings unchanged. | Test a longer final period without mosaic. The baseline already disabled mosaic for 10 final epochs. |

Optional higher resolution and AI4Shipwrecks merge are deferred unless the required four runs finish and compute/time permits. A higher-resolution candidate must never reuse v4 calibration. Any candidate selected on validation and later confirmed on test remains experimental until user approval; test results will not drive additional parameter changes.
