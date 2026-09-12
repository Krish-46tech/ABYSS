"""Evaluate a frozen validation-fitted calibration artifact on held-out test images.

Run: python abyss/ml/shadow_confidence/evaluate_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from confidence_fusion import FEATURE_NAMES, IMPROVED_ARTIFACT, calibrated_probabilities, load_calibration
from fit_calibration import collect_predictions, expected_calibration_error

ROOT = Path(__file__).resolve().parents[3]


def reliability_plot(y: np.ndarray, scores: np.ndarray, title: str, ece: float, path: Path) -> None:
    edges = np.linspace(0, 1, 11)
    confidence, accuracy, counts = [], [], []
    for index in range(10):
        mask = (scores >= edges[index]) & ((scores <= edges[index + 1]) if index == 9 else (scores < edges[index + 1]))
        if np.any(mask):
            confidence.append(float(scores[mask].mean()))
            accuracy.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", label="perfect calibration")
    ax.scatter(confidence, accuracy, s=np.maximum(35, np.asarray(counts) * 5),
               c="#0d9488", edgecolor="#134e4a", alpha=0.8, label="occupied bins")
    ax.plot(confidence, accuracy, color="#0d9488", alpha=0.65)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted correctness probability",
           ylabel="Observed fraction correct", title=f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=IMPROVED_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "abyss" / "logs" / "phase4_corrected_reliability")
    args = parser.parse_args()
    artifact = load_calibration(args.artifact)
    if artifact["fit_split"] != "val":
        raise ValueError("Calibration artifact was not fitted on validation")
    config = yaml.safe_load(Path(artifact["data_yaml"]).read_text(encoding="utf-8"))
    root = Path(config["path"]).resolve()
    val_dir = (root / config["val"]).resolve()
    test_dir = (root / config["test"]).resolve()
    if Path(artifact["image_dir"]).resolve() != val_dir or test_dir == val_dir:
        raise ValueError("Artifact validation provenance does not match dataset")
    if artifact["preprocess_mode"] != "baseline" or artifact["imgsz"] != 640:
        raise ValueError("Corrected improved_v2 test requires baseline preprocessing at 640 px")
    if artifact.get("weights_sha256") != hashlib.sha256(Path(artifact["weights"]).read_bytes()).hexdigest():
        raise ValueError("Calibration artifact checkpoint hash does not match current weights")
    print(f"Frozen artifact: {args.artifact.resolve()}", flush=True)
    print(f"Fit split: {val_dir}", flush=True)
    print(f"Evaluation split: {test_dir}", flush=True)
    print(f"Weights: {artifact['weights']}", flush=True)
    rows = collect_predictions(YOLO(artifact["weights"]), test_dir, artifact["imgsz"], artifact["prediction_conf"])
    if not rows:
        raise ValueError("No test predictions")
    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.asarray([row["correct"] for row in rows], dtype=int)
    fused, final = calibrated_probabilities(x, artifact)
    scores = {"raw_detector": x[:, 0], "logistic_fused": fused, "temperature_calibrated": final}
    ece = {name: expected_calibration_error(y, values) for name, values in scores.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in scores.items():
        path = args.output_dir / f"reliability_{name}.png"
        reliability_plot(y, values, name.replace("_", " ").title(), ece[name], path)
        print(f"Test ECE {name}: {ece[name]:.6f}; reliability plot: {path}", flush=True)
    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "bbox_xyxy", *FEATURE_NAMES,
                                                    "correct", "best_iou", "logistic_fused_probability",
                                                    "temperature_calibrated_probability"])
        writer.writeheader()
        for row, fp, cp in zip(rows, fused, final):
            writer.writerow({**row, "logistic_fused_probability": float(fp),
                             "temperature_calibrated_probability": float(cp)})
    summary = {"artifact": str(args.artifact.resolve()), "fit_split": "val", "evaluation_split": "test",
               "test_dir": str(test_dir), "image_count": len({row["image"] for row in rows}),
               "prediction_count": len(rows), "correct_count": int(y.sum()),
               "incorrect_count": int(len(y) - y.sum()), "ece": ece,
               "plots": {name: str(args.output_dir / f"reliability_{name}.png") for name in scores}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
