"""Transparent confidence fusion for ABYSS detections.

This is a documented heuristic, not a trained calibration model. It combines
the detector confidence with image quality and sonar-shadow consistency so a
triage UI can surface uncertain detections honestly.
"""

from __future__ import annotations


DEFAULT_WEIGHTS = {
    "detector_confidence": 0.60,
    "image_quality_score": 0.20,
    "shadow_consistency_score": 0.20,
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def fuse_detection_confidence(
    detector_confidence: float,
    image_quality_score: float,
    shadow_consistency_score: float,
    weights: dict[str, float] | None = None,
) -> dict[str, float | dict[str, float]]:
    weights = weights or DEFAULT_WEIGHTS
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Confidence fusion weights must sum to a positive value")

    detector = _clip01(float(detector_confidence))
    quality = _clip01(float(image_quality_score))
    shadow = _clip01(float(shadow_consistency_score))
    composite = (
        weights["detector_confidence"] * detector
        + weights["image_quality_score"] * quality
        + weights["shadow_consistency_score"] * shadow
    ) / total_weight
    return {
        "composite_confidence": _clip01(composite),
        "weights": dict(weights),
        "inputs": {
            "detector_confidence": detector,
            "image_quality_score": quality,
            "shadow_consistency_score": shadow,
        },
    }

