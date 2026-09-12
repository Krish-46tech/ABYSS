# ABYSS Held-Out Test Metrics

## Current: improved_v2, baseline 640 px, calibration v4

The [checkpoint training config](../abyss/ml/detection/runs/improved_v2_baseline_finetune_adamw/args.yaml) records baseline processed data at 640 px. All current results below use that [dataset config](../abyss/data/processed/detection/data.yaml), the held-out `test` split (129 images, 111 ground-truth objects), and checkpoint SHA-256 `baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d`. Classes come from the YAML: Plane, Ship, Shipwreck. [Calibration v4](../abyss/ml/shadow_confidence/calibration_improved_v4.json) was freshly fitted on the baseline-640 *validation* split; its recorded checkpoint hash matches the evaluated weights. The [split audit](../abyss/logs/phase3_calibration_improved_v4/audit.json) found zero validation/test filename or exact-content overlap.

### Detection

Standard Ultralytics test metrics ([script output](../abyss/logs/phase4_corrected_map.json)):

| Class | Ground truth | AP@0.5 | AP@0.5:0.95 |
| --- | ---: | ---: | ---: |
| Plane | 17 | 0.779424 | 0.378747 |
| Ship | 74 | 0.957086 | 0.674665 |
| Shipwreck | 20 | 0.144992 | 0.058058 |
| **mAP** | **111** | **0.627168** | **0.370490** |

Precision/recall/F1 below are class-aware, one-to-one matches at IoU >= 0.5 over the same test predictions, thresholded by **raw detector confidence**, not calibrated probability. These fixed-threshold values are distinct from Ultralytics' threshold-integrated AP and its interpolated precision/recall. [Raw counts and plots](../abyss/logs/phase4_corrected_details/summary.json):

| Raw threshold | TP | FP | FN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 diagnostic | 86 | 131 | 25 | 0.396313 | 0.774775 | 0.524390 |
| 0.25 deployed | 77 | 44 | 34 | 0.636364 | 0.693694 | 0.663793 |

At 0.25, the raw-count confusion matrix has ground-truth rows and prediction columns. Background is an unmatched ground truth (last column) or unmatched prediction (last row). Wrong-class matches use class-agnostic IoU >= 0.5. [Confusion image](../abyss/logs/phase4_corrected_details/confusion_raw_0.25.png), [raw JSON including the 0.05 matrix](../abyss/logs/phase4_corrected_details/summary.json).

| Truth / predicted | Plane | Ship | Shipwreck | Background |
| --- | ---: | ---: | ---: | ---: |
| Plane | 1 | 13 | 0 | 3 |
| Ship | 0 | 71 | 0 | 3 |
| Shipwreck | 0 | 0 | 3 | 17 |
| Background | 2 | 16 | 15 | 0 |

The detector's micro [precision-recall curve](../abyss/logs/phase4_corrected_details/pr_curve.png) and [raw points](../abyss/logs/phase4_corrected_details/pr_points.csv) cover candidates with raw confidence >= 0.05. All class metrics measure native detector outputs. The former backend filename-based Plane override was removed on 2026-09-12 in the current working tree; no commit has been created for this change.

### FROC

Using **final temperature-calibrated probability** to rank the same 217 test candidates, the class-aware FROC is monotonic over 219 thresholds. At the <=1 FP/image target, observed **Pd = 0.774775 (86/111)** at **0.821705 FP/image (106/129)**, final-score threshold **0.057584**. This is the best observed point *at or below* 1 FP/image; it is not an interpolated claim at exactly 1. [FROC plot](../abyss/logs/phase4_corrected_froc/froc.png), [curve points](../abyss/logs/phase4_corrected_froc/froc_points.csv), [script output](../abyss/logs/phase4_corrected_froc/summary.json).

### Calibration

ECE uses 10 equal-width bins and correctness means a class-aware, one-to-one IoU >= 0.5 match. Test data was not used to fit logistic coefficients or temperature.

| Split | Candidates | Correct | Incorrect | Raw ECE | Logistic-fused ECE | Temperature-final ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (fit, in-sample) | 227 | 87 | 140 | 0.083756 | 0.096462 | 0.097176 |
| Test (evaluation only) | 217 | 86 | 131 | 0.072560 | 0.065603 | 0.059635 |

Sources: [validation fit summary](../abyss/logs/phase3_calibration_improved_v4/summary.json) and [test summary](../abyss/logs/phase4_corrected_reliability/summary.json). Test reliability diagrams: [raw](../abyss/logs/phase4_corrected_reliability/reliability_raw_detector.png), [logistic fused](../abyss/logs/phase4_corrected_reliability/reliability_logistic_fused.png), [temperature final](../abyss/logs/phase4_corrected_reliability/reliability_temperature_calibrated.png). Temperature was selected by validation NLL, not ECE; interpretation of the validation result is in Known Limitations below.

### Not computable from available data

| Item | Status and reason |
| --- | --- |
| FP/km | Not computable: no trackline distance or surveyed length. |
| FP/km² | Not computable: no surveyed area. |
| Pd@1 FP/km² | Not computable: no surveyed area. |
| Pd@5 FP/km² | Not computable: no surveyed area. |
| Sensor frequency | Not available: no acquisition specification. |
| Swath width | Not available: no acquisition specification. |
| Survey speed | Not available: no acquisition specification. |
| Positional RMSE | Not measurable: no paired real object coordinates or GPS/nav ground truth. |

Repository dataset READMEs, YAML, split manifests, and source metadata were searched for these inputs. The [baseline-split position audit](../abyss/logs/phase4_corrected_position_audit.json) checked all 129 test entries, 111 box-only labels, GPS EXIF, and source metadata. No simulated geolocation was treated as ground truth. A prior, non-fitting backend fixture accessed one test image before it was moved to validation; no test labels or predictions were used for fitting.

### Known Limitations

**Validation ECE is bin-sensitive and in-sample.** The [rerunnable bin audit](../abyss/logs/phase3_calibration_improved_v4/ece_bin_audit.json) confirmed 136 validation images, 124 ground-truth objects (Plane 24, Ship 74, Shipwreck 26), and 227 predictions from 125 images. V4 was fitted and evaluated on these *same* validation predictions; no fresh held-out validation slice exists. With 5 equal-width bins, raw/fused/final ECE is **0.063979 / 0.044664 / 0.045771**; with 10 bins it is **0.083756 / 0.096462 / 0.097176**. The smallest occupied final bin drops from 19 predictions at 5 bins to 2 at 10. The raw-to-final regression therefore **reverses under coarser binning**, which is evidence of a small-sample/binning-sensitive measurement rather than a demonstrated global calibration failure. It does **not** prove calibration is sound: temperature slightly worsens validation ECE relative to logistic fusion under both bin counts, and only a separately held-out validation/calibration design could settle generalization. The 10-bin *test* ECE improves as reported above; test data was never used to choose the bin count or fit the model.

**Shipwreck remains weak.** Its test AP@0.5 is **0.144992**. At raw threshold 0.25, **17 of 20 Shipwreck ground truths are unmatched** in the confusion matrix. This is a detector limitation, not a calibration issue.

**Plane/Ship confusion remains visible.** At raw threshold 0.25, **13 of 17 Plane ground truths are matched to Ship predictions**. A validation-assigned [raw Plane example](../abyss/logs/final_plane_native_inference/plane-001_png.rf.dd5de3545eefa071ea7034204a08c0c1_detections.json) is likewise predicted as Ship at raw confidence **0.852639**. On 2026-09-12, the filename-based class override was removed from [backend services](../abyss/backend/app/services.py) and filename forwarding removed from [the endpoint](../abyss/backend/app/main.py). The same image submitted as `plane-001.jpg` and `neutral.jpg` now returns the same native Ship output, as checked by [the backend test](../abyss/backend/tests/test_api.py). This fix removes misleading relabeling; it does not improve classification accuracy.

**Other integrity caveats.** Priority ranking still uses hand-set class hazard weights, which are operational policy values rather than learned detection probabilities. The processed-data split uses filename-derived proxy survey groups because no navigation metadata exists; this grouping cannot prove true survey-pass independence. The repository README and source-data class-name fallback require care if the dataset taxonomy changes. These do not override a YOLO class at inference, but they should not be presented as measured model performance.

## SUPERSEDED - pipeline mismatch, do not use

The earlier [768 px enhanced mAP run](../abyss/logs/phase4_improved_map.json) evaluated this **baseline-640-trained checkpoint** under enhanced preprocessing. Its mAP@0.5 **0.431430**, mAP@0.5:0.95 **0.184853**, and precision/recall **0.430340/0.353604** are retained for audit only, not valid deployed-model performance. The same mismatch affects the old [FROC result](../abyss/logs/phase4_froc/summary.json) (Pd **0.648649** at <=1 FP/image) and [v3 ECE result](../abyss/logs/phase4_reliability/summary.json) (raw/fused/final **0.063784/0.061875/0.060843**). The [v3 artifact](../abyss/ml/shadow_confidence/calibration_improved_v3.json) records enhanced data and 768 px and must not be used with the corrected baseline-640 pipeline.
