"""Audit calibration ECE sensitivity to bin count on saved labeled predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from confidence_fusion import load_calibration
from fit_calibration import expected_calibration_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    artifact = load_calibration(args.artifact)
    if artifact["fit_split"] != "val":
        raise ValueError("Artifact was not fit on validation")
    if artifact.get("weights_sha256") != hashlib.sha256(Path(artifact["weights"]).read_bytes()).hexdigest():
        raise ValueError("Checkpoint hash differs from calibration fit")
    config = yaml.safe_load(Path(artifact["data_yaml"]).read_text(encoding="utf-8"))
    val_dir = (Path(config["path"]) / config["val"]).resolve()
    if Path(artifact["image_dir"]).resolve() != val_dir:
        raise ValueError("Artifact image directory is not the dataset validation split")
    val_images = {path.resolve() for path in val_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    labels = {int(key): name for key, name in config["names"].items()}
    ground_truth_counts = {name: 0 for name in labels.values()}
    for path in val_images:
        label_path = val_dir.parent / "labels" / f"{path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            cls = int(line.split()[0])
            if cls not in labels:
                raise ValueError(f"Unknown class {cls}: {label_path}")
            ground_truth_counts[labels[cls]] += 1

    with args.predictions_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != artifact["validation_count"]:
        raise ValueError("Prediction CSV count differs from artifact fit count")
    if not rows or any(Path(row["image"]).resolve() not in val_images for row in rows):
        raise ValueError("Prediction CSV is empty or contains non-validation images")
    y = np.asarray([int(row["correct"]) for row in rows], dtype=int)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Correctness labels must be binary")
    scores = {"raw_detector": np.asarray([float(row["detector_confidence"]) for row in rows]),
              "logistic_fused": np.asarray([float(row["logistic_fused_probability"]) for row in rows]),
              "temperature_calibrated": np.asarray([float(row["temperature_calibrated_probability"]) for row in rows])}
    if any(not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)) for values in scores.values()):
        raise ValueError("Invalid confidence values")
    results = {}
    for count in (5, 10):
        variants = {}
        edges = np.linspace(0, 1, count + 1)
        for name, values in scores.items():
            bins = []
            for index in range(count):
                mask = (values >= edges[index]) & ((values <= edges[index + 1]) if index == count - 1 else (values < edges[index + 1]))
                bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": int(mask.sum()),
                             "mean_confidence": float(values[mask].mean()) if mask.any() else None,
                             "accuracy": float(y[mask].mean()) if mask.any() else None})
            variants[name] = {"ece": expected_calibration_error(y, values, bins=count), "bins": bins,
                              "nonempty_bins": sum(item["count"] > 0 for item in bins),
                              "min_nonempty_bin_count": min(item["count"] for item in bins if item["count"] > 0)}
        results[str(count)] = variants
    report = {"artifact": str(args.artifact.resolve()), "weights_sha256": artifact["weights_sha256"],
              "fit_split": artifact["fit_split"], "evaluation_split": "same_validation_predictions_used_for_fit",
              "validation_image_count": len(val_images), "validation_ground_truth_counts": ground_truth_counts,
              "prediction_count": len(rows), "prediction_image_count": len({row["image"] for row in rows}),
              "correct_count": int(y.sum()), "incorrect_count": int(len(y) - y.sum()), "ece_by_bin_count": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation: {len(val_images)} images; ground-truth labels: {ground_truth_counts}")
    print(f"In-sample fitting/evaluation: {len(rows)} predictions from {report['prediction_image_count']} images; correct={report['correct_count']}, incorrect={report['incorrect_count']}")
    for count, variants in results.items():
        print(f"ECE with {count} equal-width bins:")
        for name, value in variants.items():
            print(f"  {name}: {value['ece']:.9f}; nonempty={value['nonempty_bins']}; smallest occupied bin={value['min_nonempty_bin_count']}")
    print(f"Audit JSON: {args.output_json}")


if __name__ == "__main__":
    main()
