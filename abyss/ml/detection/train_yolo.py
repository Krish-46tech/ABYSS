"""Train the ABYSS YOLOv8 baseline detector.

Usage:
    python abyss/ml/detection/train_yolo.py --epochs 20 --imgsz 640 --batch 8

The script fine-tunes a real Ultralytics YOLO model on the processed ABYSS
train split, validates on the validation split, copies the best weights to
abyss/ml/detection/weights/baseline_v1.pt, and logs per-epoch losses.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

from utils import DATA_YAML, DETECTION_ROOT, WEIGHTS_ROOT, fail_if_missing, setup_logging, write_json


def parse_results_csv(path: Path) -> list[dict[str, float | int]]:
    fail_if_missing(path, "Ultralytics results.csv")
    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            epoch_row: dict[str, float | int] = {}
            for key, value in clean.items():
                if value == "":
                    continue
                epoch_row[key] = int(float(value)) if key == "epoch" else float(value)
            rows.append(epoch_row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on ABYSS processed sonar data.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, mps, 0. Defaults to auto.")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero for real training")
    fail_if_missing(DATA_YAML, "processed YOLO data.yaml")
    logger = setup_logging("phase2_train_yolo")
    logger.info("Starting YOLO training: model=%s epochs=%s imgsz=%s batch=%s", args.model, args.epochs, args.imgsz, args.batch)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(DETECTION_ROOT / "runs"),
        "name": "baseline_v1",
        "exist_ok": True,
        "verbose": True,
        "plots": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    run_dir = DETECTION_ROOT / "runs" / "baseline_v1"
    best_weights = run_dir / "weights" / "best.pt"
    fail_if_missing(best_weights, "trained best.pt weights")
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    final_weights = WEIGHTS_ROOT / "baseline_v1.pt"
    shutil.copy2(best_weights, final_weights)

    epoch_rows = parse_results_csv(run_dir / "results.csv")
    metrics_path = DETECTION_ROOT / "baseline_v1_training_history.json"
    write_json(metrics_path, {"epochs": len(epoch_rows), "rows": epoch_rows, "weights": str(final_weights)})

    logger.info("Training complete. Best weights copied to %s", final_weights)
    logger.info("Training history written to %s", metrics_path)
    logger.info("Per-epoch results:")
    for row in epoch_rows:
        logger.info(row)


if __name__ == "__main__":
    main()

