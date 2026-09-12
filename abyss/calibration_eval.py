"""
calibration_eval.py

Measures calibration (ECE) of ABYSS's detector, comparing:
  Run A: raw YOLO detector confidence
  Run B: composite confidence (detector + image quality + shadow consistency)

Run this AFTER Phase 2 (trained model) and Phase 3 (confidence fusion) are working.

Usage:
    python calibration_eval.py --weights path/to/baseline_v1.pt \
                                --test_images data/processed/test/images \
                                --test_labels data/processed/test/labels

Output:
    - Prints ECE for raw confidence and composite confidence
    - Saves reliability_diagram_raw.png and reliability_diagram_composite.png
    - Saves calibration_results.json with all numbers for your presentation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR / "ml" / "shadow_confidence"))

# import your existing Phase 3 modules
from shadow_features import compute_shadow_features  # type: ignore  # noqa: E402
from image_quality import compute_image_quality_score  # type: ignore  # noqa: E402
from confidence_fusion import fuse_detection_confidence  # type: ignore  # noqa: E402


def load_ground_truth_boxes(label_path, img_w, img_h):
    """Read YOLO-format label file, return list of [x1, y1, x2, y2, class_id] in pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, int(cls)])
    return boxes


def iou(box_a, box_b):
    """Standard IoU between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a[:4]
    xb1, yb1, xb2, yb2 = box_b[:4]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5, require_class_match=True):
    """
    For each predicted box, determine if it's a True Positive (matches a GT box
    at IoU >= threshold, and that GT box hasn't already been claimed by a
    higher-confidence prediction) or a False Positive.

    Returns: list of (confidence, is_correct) tuples.
    """
    # sort predictions by confidence, descending — standard detection matching order
    pred_boxes = sorted(pred_boxes, key=lambda b: b["raw_confidence"], reverse=True)
    gt_claimed = [False] * len(gt_boxes)
    results = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt_claimed[i]:
                continue
            if require_class_match and pred["class_id"] != gt[4]:
                continue
            current_iou = iou(pred["box"], gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i

        is_correct = best_gt_idx >= 0 and best_iou >= iou_threshold
        if is_correct:
            gt_claimed[best_gt_idx] = True

        results.append({
            "raw_confidence": pred["raw_confidence"],
            "composite_confidence": pred["composite_confidence"],
            "is_correct": is_correct,
        })

    return results


def compute_ece(confidences, correctness, n_bins=10):
    """
    Standard Expected Calibration Error.
    ECE = sum over bins of (bin_size / N) * |accuracy_in_bin - avg_confidence_in_bin|
    Also returns per-bin data for plotting the reliability diagram.
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    n = len(confidences)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin inclusive on the right edge
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()
        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue

        acc_in_bin = correctness[mask].mean()
        conf_in_bin = confidences[mask].mean()
        bin_accs.append(acc_in_bin)
        bin_confs.append(conf_in_bin)
        bin_counts.append(int(count))

        ece += (count / n) * abs(acc_in_bin - conf_in_bin)

    return ece, bin_edges, bin_accs, bin_confs, bin_counts


def plot_reliability_diagram(bin_edges, bin_accs, bin_counts, title, save_path):
    n_bins = len(bin_accs)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, edgecolor="black",
           alpha=0.7, label="Actual accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Actual accuracy (precision)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    for i, count in enumerate(bin_counts):
        if count > 0:
            ax.text(bin_centers[i], bin_accs[i] + 0.02, str(count),
                    ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt file")
    parser.add_argument("--test_images", required=True, help="Folder of test images")
    parser.add_argument("--test_labels", required=True, help="Folder of YOLO-format label .txt files")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--output_dir", default="calibration_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.weights} ...")
    model = YOLO(args.weights)
    class_names = getattr(model, "names", {})
    num_classes = len(class_names) if class_names else 0
    require_class_match = num_classes != 1
    if require_class_match:
        print(f"Multi-class detector ({num_classes} classes); requiring predicted class == ground-truth class.")
    else:
        print("Single-class detector; class matching is unnecessary.")

    image_paths = sorted(Path(args.test_images).glob("*.jpg")) + \
                  sorted(Path(args.test_images).glob("*.png"))

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.test_images} — check the path.")

    print(f"Found {len(image_paths)} test images. Running inference...")

    all_results = []
    num_gt_objects = 0

    for img_path in image_paths:
        results = model(str(img_path), verbose=False)[0]
        img_h, img_w = results.orig_shape
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"OpenCV could not read test image: {img_path}")

        label_path = Path(args.test_labels) / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth_boxes(label_path, img_w, img_h)
        num_gt_objects += len(gt_boxes)

        pred_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_conf = float(box.conf[0])
            class_id = int(box.cls[0])

            # call your real Phase 3 functions — NOT placeholders
            shadow = compute_shadow_features(image, [x1, y1, x2, y2])
            quality = compute_image_quality_score(image)
            fusion = fuse_detection_confidence(
                raw_conf,
                quality["image_quality_score"],
                shadow["shadow_consistency_score"],
            )
            composite = fusion["composite_confidence"]

            pred_boxes.append({
                "box": [x1, y1, x2, y2],
                "class_id": class_id,
                "raw_confidence": raw_conf,
                "composite_confidence": composite,
            })

        matched = match_detections_to_gt(
            pred_boxes,
            gt_boxes,
            args.iou_threshold,
            require_class_match=require_class_match,
        )
        all_results.extend(matched)

    print(f"Total matched predictions across all images: {len(all_results)}")

    if len(all_results) == 0:
        raise RuntimeError("No predictions were matched — check your model and label paths.")

    raw_confidences = [r["raw_confidence"] for r in all_results]
    composite_confidences = [r["composite_confidence"] for r in all_results]
    correctness = [r["is_correct"] for r in all_results]
    true_positives = int(sum(correctness))
    false_positives = len(correctness) - true_positives

    # Run A: raw confidence
    ece_raw, bins_raw, accs_raw, confs_raw, counts_raw = compute_ece(
        raw_confidences, correctness, args.n_bins
    )
    plot_reliability_diagram(
        bins_raw, accs_raw, counts_raw,
        f"Raw Detector Confidence (ECE = {ece_raw:.4f})",
        os.path.join(args.output_dir, "reliability_diagram_raw.png"),
    )

    # Run B: composite confidence
    ece_composite, bins_comp, accs_comp, confs_comp, counts_comp = compute_ece(
        composite_confidences, correctness, args.n_bins
    )
    plot_reliability_diagram(
        bins_comp, accs_comp, counts_comp,
        f"Composite Confidence (ECE = {ece_composite:.4f})",
        os.path.join(args.output_dir, "reliability_diagram_composite.png"),
    )

    summary = {
        "model_path": str(Path(args.weights).resolve()),
        "test_images_path": str(Path(args.test_images).resolve()),
        "test_labels_path": str(Path(args.test_labels).resolve()),
        "num_test_images": len(image_paths),
        "num_ground_truth_objects": num_gt_objects,
        "num_predictions_evaluated": len(all_results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "iou_threshold": args.iou_threshold,
        "n_bins": args.n_bins,
        "ece_raw_confidence": round(ece_raw, 4),
        "ece_composite_confidence": round(ece_composite, 4),
        "improvement": round(ece_raw - ece_composite, 4),
        "improvement_note": (
            "Positive value means composite confidence is BETTER calibrated "
            "(lower ECE) than raw detector confidence. Negative means it's worse — "
            "report this honestly either way."
        ),
    }

    with open(os.path.join(args.output_dir, "calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CALIBRATION RESULTS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
