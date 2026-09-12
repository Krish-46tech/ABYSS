"""Fit confidence calibration on labeled YOLO validation predictions.

Run: python abyss/ml/shadow_confidence/fit_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "abyss" / "ml" / "detection"))
from utils import DATA_YAML, WEIGHTS_ROOT, write_json  # noqa: E402
from confidence_fusion import DEFAULT_ARTIFACT, FEATURE_NAMES, calibrated_probabilities, feature_vector  # noqa: E402
from image_quality import compute_image_quality_score  # noqa: E402
from shadow_features import compute_shadow_features  # noqa: E402


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ground-truth labels: {path}")
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label: {path}:{line_number}")
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        if not np.all(np.isfinite([cx, cy, w, h])) or not all(0 <= v <= 1 for v in (cx, cy, w, h)):
            raise ValueError(f"Invalid coordinates: {path}:{line_number}")
        labels.append((cls, [(cx - w / 2) * width, (cy - h / 2) * height,
                             (cx + w / 2) * width, (cy + h / 2) * height]))
    return labels


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        raise ValueError("No predictions for ECE")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & ((p <= edges[index + 1]) if index == bins - 1 else (p < edges[index + 1]))
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def collect_predictions(model: YOLO, image_dir: Path, imgsz: int, conf: float) -> list[dict]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No validation images: {image_dir}")
    rows = []
    for number, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        height, width = image.shape[:2]
        labels = read_labels(image_dir.parent / "labels" / f"{path.stem}.txt", width, height)
        result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        quality = compute_image_quality_score(image)["image_quality_score"]
        used_gt = set()
        for box in result.boxes:
            cls = int(box.cls.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf.item())
            matches = [(iou(xyxy, gt_box), idx) for idx, (gt_cls, gt_box) in enumerate(labels)
                       if gt_cls == cls and idx not in used_gt]
            best_iou, best_idx = max(matches, default=(0.0, -1))
            correct = int(best_iou >= 0.5)
            if correct:
                used_gt.add(best_idx)
            shadow = compute_shadow_features(image, xyxy)["shadow_consistency_score"]
            features = feature_vector(score, shadow, quality)
            rows.append({"image": str(path), "class_id": cls, "bbox_xyxy": xyxy,
                         **dict(zip(FEATURE_NAMES, features)), "correct": correct, "best_iou": best_iou})
        if number % 25 == 0 or number == len(paths):
            print(f"Images processed from {image_dir}: {number}/{len(paths)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_ROOT / "baseline_v1.pt")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "abyss" / "logs" / "phase3_calibration")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--collect-only", action="store_true", help="Save real validation features without fitting")
    parser.add_argument("--predictions-csv", type=Path, help="Fit from a previously collected validation feature CSV")
    args = parser.parse_args()
    if not 0 < args.conf < 1 or args.imgsz <= 0:
        raise ValueError("Invalid confidence threshold or image size")
    config = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    image_dir = (Path(config["path"]) / config["val"]).resolve()
    test_dir = (Path(config["path"]) / config["test"]).resolve()
    if image_dir == test_dir or "test" in image_dir.parts or "val" not in image_dir.parts:
        raise ValueError(f"Calibration requires a distinct validation directory: {image_dir}")
    if args.collect_only and args.predictions_csv:
        raise ValueError("--collect-only and --predictions-csv cannot be combined")
    print(f"Validation split: {image_dir}", flush=True)
    if args.predictions_csv:
        with args.predictions_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        val_paths = {p.resolve() for p in image_dir.iterdir() if p.is_file()}
        if any(Path(row["image"]).resolve() not in val_paths for row in rows):
            raise ValueError("Input CSV contains a file outside validation split")
        for row in rows:
            features = feature_vector(*(float(row[name]) for name in FEATURE_NAMES))
            row.update(dict(zip(FEATURE_NAMES, features)))
            row["correct"] = int(row["correct"])
            row["best_iou"] = float(row["best_iou"])
    else:
        rows = collect_predictions(YOLO(str(args.weights)), image_dir, args.imgsz, args.conf)
    if not rows:
        raise ValueError("No validation predictions")
    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows])
    y = np.asarray([row["correct"] for row in rows], dtype=int)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    print(f"Class balance: {positives} correct ({positives / len(y):.1%}), {negatives} incorrect ({negatives / len(y):.1%})", flush=True)
    if min(positives, negatives) < 5 or min(positives, negatives) / len(y) < 0.05:
        raise ValueError("Validation predictions are too imbalanced (<5% in one class); adjust --conf")
    if args.collect_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "validation_features.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "bbox_xyxy", *FEATURE_NAMES, "correct", "best_iou"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Collected validation features without fitting: {output_path}", flush=True)
        return
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x, y)
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    coefficients = logistic.coef_[0]
    print(f"Learned standardized coefficients: {dict(zip(FEATURE_NAMES, coefficients.tolist()))}", flush=True)
    if np.all(np.abs(coefficients) < 0.01):
        raise ValueError("All coefficients near zero; model learned nothing")
    logits = model.decision_function(x)
    def nll(temperature: float) -> float:
        z = logits / temperature
        return float(np.mean(np.logaddexp(0, z) - y * z))
    result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded", options={"xatol": 1e-6})
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(result.x)
    print(f"Validation NLL temperature: {temperature:.6f}", flush=True)
    if temperature < 0.51 or temperature > 4.99:
        print("WARNING: temperature is near the fit boundary", flush=True)
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    artifact = {"feature_names": list(FEATURE_NAMES), "weights": str(args.weights.resolve()),
                "weights_sha256": weights_hash, "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "data_yaml": str(args.data_yaml.resolve()), "fit_split": "val", "image_dir": str(image_dir),
                "imgsz": args.imgsz, "preprocess_mode": "enhanced" if "enhanced" in str(args.data_yaml) else "baseline",
                "prediction_conf": args.conf, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "coefficients": coefficients.tolist(),
                "intercept": float(logistic.intercept_[0]), "temperature": temperature,
                "validation_count": len(rows), "positive_count": positives, "negative_count": negatives}
    fused, calibrated = calibrated_probabilities(x, artifact)
    ece = {"raw_detector": expected_calibration_error(y, x[:, 0]),
           "logistic_fused": expected_calibration_error(y, fused),
           "temperature_calibrated": expected_calibration_error(y, calibrated)}
    for name, value in ece.items():
        print(f"Validation ECE {name}: {value:.6f}", flush=True)
    print("Validation ECE is in-sample; held-out test ECE belongs to Phase 4.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "bbox_xyxy", *FEATURE_NAMES, "correct", "best_iou", "logistic_fused_probability", "temperature_calibrated_probability"])
        writer.writeheader()
        for row, fp, cp in zip(rows, fused, calibrated):
            writer.writerow({**row, "logistic_fused_probability": float(fp), "temperature_calibrated_probability": float(cp)})
    write_json(args.artifact, artifact)
    summary = {"fit_split": "val", "image_count": len(set(row["image"] for row in rows)),
               "prediction_count": len(rows), "positive_count": positives, "negative_count": negatives,
               "coefficients": dict(zip(FEATURE_NAMES, coefficients.tolist())), "temperature": temperature,
               "validation_ece": ece, "artifact": str(args.artifact),
               "predictions_csv": str(args.output_dir / "validation_predictions.csv")}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
