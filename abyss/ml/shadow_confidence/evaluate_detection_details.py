"""Report threshold-specific detection metrics from saved held-out test predictions."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from confidence_fusion import load_calibration
from fit_calibration import iou, read_labels


def load_data(artifact_path: Path, predictions_path: Path, split: str = "test"):
    artifact = load_calibration(artifact_path)
    if artifact["fit_split"] != "val" or artifact["preprocess_mode"] != "baseline" or artifact["imgsz"] != 640:
        raise ValueError("Expected validation-fitted baseline-640 calibration")
    if artifact.get("weights_sha256") != hashlib.sha256(Path(artifact["weights"]).read_bytes()).hexdigest():
        raise ValueError("Checkpoint hash changed since calibration")
    config = yaml.safe_load(Path(artifact["data_yaml"]).read_text(encoding="utf-8"))
    names = {int(key): value for key, value in config["names"].items()}
    root = Path(config["path"]).resolve()
    test_dir = (root / config[split]).resolve()
    val_dir = (root / config["val"]).resolve()
    if Path(artifact["image_dir"]).resolve() != val_dir or (split == "test" and test_dir == val_dir):
        raise ValueError("Calibration fit/test split mismatch")
    paths = sorted(p.resolve() for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError("No held-out test images")
    labels = {}
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable test image: {path}")
        h, w = image.shape[:2]
        labels[path] = read_labels(test_dir.parent / "labels" / f"{path.stem}.txt", w, h)
    predictions = {path: [] for path in paths}
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["image"]).resolve()
            if path not in predictions:
                raise ValueError(f"Prediction outside test split: {path}")
            cls = int(row["class_id"])
            score = float(row["detector_confidence"])
            box = ast.literal_eval(row["bbox_xyxy"])
            if cls not in names or not np.isfinite(score) or not 0 <= score <= 1 or len(box) != 4:
                raise ValueError(f"Invalid prediction in {path}")
            predictions[path].append((score, cls, box))
    for path in paths:
        predictions[path].sort(key=lambda value: value[0], reverse=True)
    return artifact, names, paths, labels, predictions


def evaluate(threshold, names, paths, labels, predictions):
    tp = fp = 0
    matrix = np.zeros((len(names) + 1, len(names) + 1), dtype=int)
    background = len(names)
    for path in paths:
        gt = labels[path]
        selected = [item for item in predictions[path] if item[0] >= threshold]
        # Class-aware one-to-one matching for detection precision and recall.
        matched = set()
        for _, cls, box in selected:
            candidates = [(iou(box, gt_box), index) for index, (gt_cls, gt_box) in enumerate(gt)
                          if gt_cls == cls and index not in matched]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                matched.add(best_index)
                tp += 1
            else:
                fp += 1
        # Class-agnostic pairing exposes wrong-class predictions in the confusion matrix.
        used_gt = set()
        for _, cls, box in selected:
            candidates = [(iou(box, gt_box), index) for index, (_, gt_box) in enumerate(gt) if index not in used_gt]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                used_gt.add(best_index)
                matrix[gt[best_index][0], cls] += 1
            else:
                matrix[background, cls] += 1
        for index, (gt_cls, _) in enumerate(gt):
            if index not in used_gt:
                matrix[gt_cls, background] += 1
    gt_count = sum(len(labels[path]) for path in paths)
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": matrix.tolist()}


def plot_confusion(matrix, names, path, threshold):
    labels = [names[index] for index in sorted(names)] + ["Background"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Ground truth", title=f"Test confusion, raw confidence >= {threshold:g}")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-threshold", type=float)
    args = parser.parse_args()
    artifact, names, paths, labels, predictions = load_data(args.artifact, args.predictions_csv)
    if artifact["prediction_conf"] > 0.05:
        raise ValueError("Candidate floor is above requested diagnostic threshold 0.05")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    thresholds = [0.05, 0.25]
    if args.extra_threshold is not None:
        if not artifact["prediction_conf"] <= args.extra_threshold <= 1:
            raise ValueError("Extra threshold must be within the candidate score range")
        if args.extra_threshold not in thresholds:
            thresholds.append(args.extra_threshold)
    for threshold in thresholds:
        result = evaluate(threshold, names, paths, labels, predictions)
        result["confusion_plot"] = str(args.output_dir / f"confusion_raw_{threshold:g}.png")
        plot_confusion(result["confusion_matrix"], names, result["confusion_plot"], threshold)
        results[str(threshold)] = result
    scores = sorted({item[0] for rows in predictions.values() for item in rows}, reverse=True)
    if not scores:
        raise ValueError("No test predictions")
    points = [evaluate(score, names, paths, labels, predictions) for score in scores]
    points.append(results["0.05"])
    with (args.output_dir / "pr_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows({key: point[key] for key in writer.fieldnames} for point in points)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([point["recall"] for point in points], [point["precision"] for point in points], color="#0d9488")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="Test micro PR, raw confidence >= 0.05, IoU >= 0.5")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pr_curve.png", dpi=170)
    plt.close(fig)
    summary = {"artifact": str(args.artifact.resolve()), "test_image_count": len(paths),
               "ground_truth_count": sum(len(labels[path]) for path in paths),
               "candidate_prediction_count": sum(len(predictions[path]) for path in paths),
               "class_names": names, "confusion_rows": "ground_truth", "confusion_columns": "prediction",
               "background_index": len(names), "matching_iou": 0.5,
               "threshold_score": "raw_detector_confidence", "thresholds": results,
               "pr_plot": str(args.output_dir / "pr_curve.png"),
               "pr_points_csv": str(args.output_dir / "pr_points.csv")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
