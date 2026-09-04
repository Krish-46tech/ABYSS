"""MLflow helpers for ABYSS detection experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_ROOT = PROJECT_ROOT / "abyss" / "mlflow"
TRACKING_DB = MLFLOW_ROOT / "mlflow.db"
ARTIFACT_ROOT = MLFLOW_ROOT / "artifacts"
EXPERIMENT_NAME = "ABYSS sonar detector"


def configure_mlflow() -> None:
    MLFLOW_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_json_artifact(name: str, payload: dict[str, Any]) -> None:
    artifact_dir = PROJECT_ROOT / "abyss" / "logs" / "mlflow_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(path))
