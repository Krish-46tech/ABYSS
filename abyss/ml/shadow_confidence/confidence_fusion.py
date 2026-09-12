"""Learned correctness probability from detector and sonar-image features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

FEATURE_NAMES = ("detector_confidence", "shadow_consistency_score", "image_quality_score")
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "calibration_baseline_v1.json"
IMPROVED_ARTIFACT = Path(__file__).resolve().parent / "calibration_improved_v4.json"


def artifact_for_weights(weights_path: Path) -> Path:
    name = weights_path.name
    if name == "baseline_v1.pt":
        return DEFAULT_ARTIFACT
    if name == "improved_v2_baseline_finetune_adamw.pt":
        return IMPROVED_ARTIFACT
    raise ValueError(f"No default calibration artifact for weights {weights_path}; pass --calibration-artifact")


def feature_vector(detector_confidence: float, shadow_consistency_score: float, image_quality_score: float) -> list[float]:
    values = [float(detector_confidence), float(shadow_consistency_score), float(image_quality_score)]
    if not np.all(np.isfinite(values)) or any(value < 0 or value > 1 for value in values):
        raise ValueError(f"Invalid confidence features: {values}")
    return values


def load_calibration(path: Path = DEFAULT_ARTIFACT) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration artifact missing: {path}. Run fit_calibration.py on validation data first.")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Calibration artifact feature order does not match inference")
    if artifact["temperature"] <= 0:
        raise ValueError("Calibration temperature must be positive")
    return artifact


def calibrated_probabilities(features: np.ndarray, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite N x 3 calibration features")
    scaled = (x - np.asarray(artifact["scaler_mean"])) / np.asarray(artifact["scaler_scale"])
    logits = scaled @ np.asarray(artifact["coefficients"]) + artifact["intercept"]
    return expit(logits), expit(logits / artifact["temperature"])


def fuse_detection_confidence(detector_confidence: float, image_quality_score: float, shadow_consistency_score: float) -> dict:
    """Compatibility output for callers that still name the final score composite."""
    artifact = load_calibration()
    features = feature_vector(detector_confidence, shadow_consistency_score, image_quality_score)
    _, final = calibrated_probabilities(np.asarray([features]), artifact)
    return {"composite_confidence": float(final[0])}
