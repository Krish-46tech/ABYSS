"""Log the existing ABYSS baseline model and metrics to MLflow.

Usage:
    python abyss/ml/detection/log_baseline_to_mlflow.py
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from mlflow_utils import configure_mlflow, log_json_artifact
from utils import ABYSS_ROOT, read_json


def main() -> None:
    configure_mlflow()
    metrics_path = ABYSS_ROOT / "logs" / "baseline_metrics.json"
    history_path = ABYSS_ROOT / "ml" / "detection" / "baseline_v1_training_history.json"
    weights_path = ABYSS_ROOT / "ml" / "detection" / "weights" / "baseline_v1.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Baseline weights missing: {weights_path}")

    metrics = read_json(metrics_path)
    history = read_json(history_path)
    with mlflow.start_run(run_name="baseline_v1_yolov8n_phase2"):
        mlflow.log_params(
            {
                "model_family": "YOLOv8n",
                "preprocessing": "median_blur_minmax_letterbox",
                "epochs": history["epochs"],
                "imgsz": 640,
                "optimizer": "Ultralytics auto AdamW",
                "notes": "Existing Phase 2 baseline logged after training.",
            }
        )
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])
        mlflow.log_metric("test_map50", metrics["map50"])
        mlflow.log_metric("test_map50_95", metrics["map50_95"])
        mlflow.log_metric("latency_mean_ms", metrics["latency_ms_per_image"]["mean"])
        for row in history["rows"]:
            step = int(row["epoch"])
            mlflow.log_metric("train_box_loss", row["train/box_loss"], step=step)
            mlflow.log_metric("train_cls_loss", row["train/cls_loss"], step=step)
            mlflow.log_metric("train_dfl_loss", row["train/dfl_loss"], step=step)
            mlflow.log_metric("val_map50", row["metrics/mAP50(B)"], step=step)
        mlflow.log_artifact(str(weights_path), artifact_path="weights")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(history_path), artifact_path="training")
        log_json_artifact("baseline_v1_summary.json", {"metrics": metrics, "history_rows": len(history["rows"])})
        print({"logged_run": "baseline_v1_yolov8n_phase2", "metrics": metrics, "weights": str(weights_path)})


if __name__ == "__main__":
    main()
