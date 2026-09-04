"""Shared utilities for ABYSS detection scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ABYSS_ROOT = PROJECT_ROOT / "abyss"
DATA_YAML = ABYSS_ROOT / "data" / "processed" / "detection" / "data.yaml"
DETECTION_ROOT = ABYSS_ROOT / "ml" / "detection"
WEIGHTS_ROOT = DETECTION_ROOT / "weights"
LOGS_ROOT = ABYSS_ROOT / "logs"
INFERENCE_ROOT = DETECTION_ROOT / "inference_outputs"
CLASS_NAMES = ["Plane", "Ship", "Shipwreck"]


def setup_logging(script_name: str) -> logging.Logger:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(LOGS_ROOT / f"{script_name}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def fail_if_missing(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    fail_if_missing(path, "JSON file")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

