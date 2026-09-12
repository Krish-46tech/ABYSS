"""Compute class-aware FROC from frozen calibrated test predictions.

Run: python abyss/ml/shadow_confidence/evaluate_froc.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from confidence_fusion import IMPROVED_ARTIFACT, load_calibration
from fit_calibration import iou, read_labels

ROOT = Path(__file__).resolve().parents[3]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_test_data(artifact: dict, predictions_csv: Path) -> tuple[list[Path], dict[Path, list], dict[Path, list]]:
    if artifact["fit_split"] != "val":
        raise ValueError("FROC artifact must be fitted on validation")
    config = yaml.safe_load(Path(artifact["data_yaml"]).read_text(encoding="utf-8"))
    root = Path(config["path"]).resolve()
    val_dir = (root / config["val"]).resolve()
    test_dir = (root / config["test"]).resolve()
    if Path(artifact["image_dir"]).resolve() != val_dir or val_dir == test_dir:
        raise ValueError("Artifact fit directory does not match validation split")
    images = sorted(path.resolve() for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("No test images for FROC")
    image_set = set(images)
    labels_by_image = {}
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable test image: {path}")
        height, width = image.shape[:2]
        labels_by_image[path] = read_labels(test_dir.parent / "labels" / f"{path.stem}.txt", width, height)
    predictions_by_image: dict[Path, list] = defaultdict(list)
    with predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["image"]).resolve()
            if path not in image_set:
                raise ValueError(f"FROC CSV references a non-test image: {path}")
            score = float(row["temperature_calibrated_probability"])
            box = ast.literal_eval(row["bbox_xyxy"])
            if not np.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or not np.all(np.isfinite(box)):
                raise ValueError(f"Invalid FROC prediction in {path}")
            predictions_by_image[path].append((score, int(row["class_id"]), box))
    if not any(predictions_by_image.values()):
        raise ValueError("No saved test predictions for FROC")
    for path in images:
        predictions_by_image[path].sort(key=lambda item: item[0], reverse=True)
    return images, labels_by_image, predictions_by_image


def evaluate_threshold(threshold: float, images: list[Path], labels_by_image: dict, predictions_by_image: dict) -> tuple[int, int]:
    tp = fp = 0
    for path in images:
        matched = set()
        for score, cls, box in predictions_by_image[path]:
            if score < threshold:
                break
            candidates = [(iou(box, gt_box), index)
                          for index, (gt_cls, gt_box) in enumerate(labels_by_image[path])
                          if cls == gt_cls and index not in matched]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                matched.add(best_index)
                tp += 1
            else:
                fp += 1
    return tp, fp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=IMPROVED_ARTIFACT)
    parser.add_argument("--predictions-csv", type=Path, default=ROOT / "abyss" / "logs" / "phase4_corrected_reliability" / "test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "abyss" / "logs" / "phase4_corrected_froc")
    parser.add_argument("--fp-per-image", type=float, default=1.0)
    args = parser.parse_args()
    if args.fp_per_image < 0:
        raise ValueError("Operating false positives per image must be nonnegative")
    artifact = load_calibration(args.artifact)
    images, labels, predictions = load_test_data(artifact, args.predictions_csv)
    gt_count = sum(len(labels[path]) for path in images)
    pred_count = sum(len(predictions[path]) for path in images)
    if gt_count == 0:
        raise ValueError("No ground-truth test objects for FROC")
    thresholds = sorted({1.0, 0.0, *(score for rows in predictions.values() for score, _, _ in rows)}, reverse=True)
    points = []
    for threshold in thresholds:
        tp, fp = evaluate_threshold(threshold, images, labels, predictions)
        points.append({"threshold": threshold, "true_positives": tp, "false_positives": fp,
                       "sensitivity": tp / gt_count, "fp_per_image": fp / len(images)})
    monotonic = all(next_row["true_positives"] >= row["true_positives"] and
                    next_row["false_positives"] >= row["false_positives"]
                    for row, next_row in zip(points, points[1:]))
    print(f"Test images: {len(images)}; ground-truth objects: {gt_count}; candidate predictions: {pred_count}", flush=True)
    print(f"Detector candidate confidence floor from validation artifact: {artifact['prediction_conf']}", flush=True)
    print(f"Threshold points: {len(points)}; monotonic TP/FP as threshold decreases: {monotonic}", flush=True)
    if not monotonic:
        raise AssertionError("FROC is not monotonic as the final confidence threshold decreases")
    feasible = [row for row in points if row["fp_per_image"] <= args.fp_per_image]
    if not feasible:
        raise ValueError("No FROC operating point at or below requested FP/image")
    operating = max(feasible, key=lambda row: (row["sensitivity"], -row["fp_per_image"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "froc_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step([row["fp_per_image"] for row in points], [row["sensitivity"] for row in points],
            where="post", color="#0d9488", linewidth=2, label="Final calibrated confidence")
    ax.axvline(args.fp_per_image, color="#64748b", linestyle="--", label=f"{args.fp_per_image:g} FP/image target")
    ax.scatter([operating["fp_per_image"]], [operating["sensitivity"]], color="#c2410c", s=80, zorder=3)
    ax.set(xlabel="False positives per image", ylabel="Sensitivity (Pd)", xlim=(0, max(1.05, points[-1]["fp_per_image"] * 1.05)),
           ylim=(0, 1.02), title="ABYSS test FROC, IoU >= 0.5")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plot_path = args.output_dir / "froc.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    summary = {"artifact": str(args.artifact.resolve()), "evaluation_split": "test",
               "test_image_count": len(images), "ground_truth_count": gt_count,
               "candidate_prediction_count": pred_count, "candidate_detector_confidence_floor": artifact["prediction_conf"],
               "threshold_count": len(points), "monotonic": monotonic,
               "target_fp_per_image": args.fp_per_image, "pd_at_target": operating["sensitivity"],
               "operating_point": operating, "max_observed_fp_per_image": points[-1]["fp_per_image"],
               "plot": str(plot_path), "points_csv": str(csv_path)}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
