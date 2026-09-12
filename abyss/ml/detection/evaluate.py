"""Evaluate the trained ABYSS YOLO detector on the held-out test split.

Usage:
    python abyss/ml/detection/evaluate.py --weights abyss/ml/detection/weights/baseline_v1.pt

The script computes real Ultralytics test metrics and measures per-image
inference latency on the processed test images.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

from utils import ABYSS_ROOT, DATA_YAML, LOGS_ROOT, fail_if_missing, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ABYSS YOLO baseline on held-out test data.")
    parser.add_argument("--weights", type=Path, default=ABYSS_ROOT / "ml" / "detection" / "weights" / "baseline_v1.pt")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--output-json", type=Path, default=LOGS_ROOT / "baseline_metrics.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    fail_if_missing(args.weights, "trained model weights")
    fail_if_missing(args.data_yaml, "processed YOLO data.yaml")
    import yaml

    with args.data_yaml.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    dataset_root = Path(data_config["path"])
    split_image_dir = dataset_root / data_config[args.split]
    image_paths = sorted(split_image_dir.glob("*"))
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    if not image_paths:
        raise ValueError(f"No {args.split} images found in {split_image_dir}")

    logger = setup_logging("phase2_evaluate")
    logger.info("Loading model weights from %s", args.weights)
    model = YOLO(str(args.weights))
    val_kwargs = {"data": str(args.data_yaml), "split": args.split, "imgsz": args.imgsz, "verbose": True}
    if args.device:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)

    latencies_ms: list[float] = []
    predict_kwargs = {"imgsz": args.imgsz, "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device
    for image_path in image_paths:
        start = time.perf_counter()
        model.predict(str(image_path), **predict_kwargs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "data_yaml": str(args.data_yaml.resolve()),
        "imgsz": args.imgsz,
        "evaluation_split": args.split,
        "image_count": len(image_paths),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            str(data_config["names"][int(class_id)]): {
                "class_id": int(class_id),
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "ap50": float(metrics.box.ap50[index]),
                "ap50_95": float(metrics.box.ap[index].mean()),
            }
            for index, class_id in enumerate(metrics.box.ap_class_index)
        },
        "latency_ms_per_image": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
    }
    output_path = args.output_json
    write_json(output_path, result)
    logger.info("Evaluation metrics written to %s", output_path)
    logger.info("Metrics: %s", result)
    print(result)


if __name__ == "__main__":
    main()
