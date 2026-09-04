"""Train an MLflow-tracked ABYSS detector experiment.

Usage:
    python abyss/ml/detection/train_experiment.py \
      --run-name improved_v1_yolov8s_clahe \
      --model yolov8s.pt \
      --data-yaml abyss/data/processed/detection_enhanced/data.yaml \
      --epochs 35 --imgsz 768 --batch 4 --device mps
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import time
from pathlib import Path

import mlflow
import yaml

PROJECT_ROOT_FOR_ENV = Path(__file__).resolve().parents[3]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT_FOR_ENV / "abyss" / ".ultralytics"))

from ultralytics import YOLO, settings

from mlflow_utils import configure_mlflow, log_json_artifact
from train_yolo import parse_results_csv
from utils import DETECTION_ROOT, LOGS_ROOT, WEIGHTS_ROOT, fail_if_missing, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLflow-tracked ABYSS YOLO experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--lrf", type=float, default=0.05)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    fail_if_missing(args.data_yaml, "experiment data.yaml")
    configure_mlflow()
    settings.update({"mlflow": False})
    logger = setup_logging(f"{args.run_name}_train")
    is_enhanced_dataset = "detection_enhanced" in str(args.data_yaml)
    preprocessing_name = (
        "percentile_clip_bilateral_clahe_unsharp_letterbox_768"
        if is_enhanced_dataset
        else "median_blur_minmax_letterbox_640"
    )
    augmentation_params = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.08 if not is_enhanced_dataset else 0.12,
        "degrees": 2.0 if not is_enhanced_dataset else 3.0,
        "translate": 0.05 if not is_enhanced_dataset else 0.08,
        "scale": 0.25 if not is_enhanced_dataset else 0.35,
        "mosaic": 0.15 if not is_enhanced_dataset else 0.35,
        "mixup": 0.0,
    }

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model": args.model,
                "data_yaml": str(args.data_yaml),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "lrf": args.lrf,
                "preprocessing": preprocessing_name,
                **augmentation_params,
            }
        )
        model = YOLO(args.model)
        train_kwargs = {
            "data": str(args.data_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project": str(DETECTION_ROOT / "runs"),
            "name": args.run_name,
            "exist_ok": True,
            "verbose": True,
            "plots": True,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "cos_lr": True,
            "close_mosaic": 10,
            **augmentation_params,
            "patience": 15,
        }
        if args.device:
            train_kwargs["device"] = args.device
        logger.info("Starting improved training with args: %s", train_kwargs)
        model.train(**train_kwargs)

        run_dir = DETECTION_ROOT / "runs" / args.run_name
        best_weights = run_dir / "weights" / "best.pt"
        fail_if_missing(best_weights, "trained best.pt weights")
        WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
        final_weights = WEIGHTS_ROOT / f"{args.run_name}.pt"
        shutil.copy2(best_weights, final_weights)

        rows = parse_results_csv(run_dir / "results.csv")
        history_path = DETECTION_ROOT / f"{args.run_name}_training_history.json"
        write_json(history_path, {"epochs": len(rows), "rows": rows, "weights": str(final_weights)})
        for row in rows:
            step = int(row["epoch"])
            mlflow.log_metric("train_box_loss", row["train/box_loss"], step=step)
            mlflow.log_metric("train_cls_loss", row["train/cls_loss"], step=step)
            mlflow.log_metric("train_dfl_loss", row["train/dfl_loss"], step=step)
            mlflow.log_metric("val_map50", row["metrics/mAP50(B)"], step=step)
            mlflow.log_metric("val_precision", row["metrics/precision(B)"], step=step)
            mlflow.log_metric("val_recall", row["metrics/recall(B)"], step=step)
        mlflow.log_artifact(str(final_weights), artifact_path="weights")
        mlflow.log_artifact(str(history_path), artifact_path="training")

        logger.info("Running held-out test evaluation for %s", args.run_name)
        eval_model = YOLO(str(final_weights))
        val_kwargs = {"data": str(args.data_yaml), "split": "test", "imgsz": args.imgsz, "verbose": True}
        predict_kwargs = {"imgsz": args.imgsz, "verbose": False}
        if args.device:
            val_kwargs["device"] = args.device
            predict_kwargs["device"] = args.device
        metrics = eval_model.val(**val_kwargs)
        with args.data_yaml.open("r", encoding="utf-8") as handle:
            data_config = yaml.safe_load(handle)
        test_image_dir = Path(data_config["path"]) / data_config.get("test", "test/images")
        image_paths = sorted(path for path in test_image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        if not image_paths:
            raise ValueError(f"No test images found in {test_image_dir}")
        latencies_ms: list[float] = []
        for image_path in image_paths:
            start = time.perf_counter()
            eval_model.predict(str(image_path), **predict_kwargs)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        test_metrics = {
            "weights": str(final_weights),
            "test_image_count": len(image_paths),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "latency_ms_per_image": {
                "mean": statistics.mean(latencies_ms),
                "median": statistics.median(latencies_ms),
                "min": min(latencies_ms),
                "max": max(latencies_ms),
            },
        }
        metrics_path = LOGS_ROOT / f"{args.run_name}_metrics.json"
        write_json(metrics_path, test_metrics)
        mlflow.log_metric("test_precision", test_metrics["precision"])
        mlflow.log_metric("test_recall", test_metrics["recall"])
        mlflow.log_metric("test_map50", test_metrics["map50"])
        mlflow.log_metric("test_map50_95", test_metrics["map50_95"])
        mlflow.log_metric("latency_mean_ms", test_metrics["latency_ms_per_image"]["mean"])
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        log_json_artifact(
            f"{args.run_name}_summary.json",
            {"history_rows": len(rows), "weights": str(final_weights), "test_metrics": test_metrics},
        )
        print({"run_name": args.run_name, "weights": str(final_weights), "history": str(history_path), "metrics": test_metrics})


if __name__ == "__main__":
    main()
