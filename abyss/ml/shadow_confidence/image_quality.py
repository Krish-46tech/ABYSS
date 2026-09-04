"""Real image-quality scoring for sonar tiles.

The score combines measurable signal statistics:

- normalized contrast from intensity standard deviation
- local signal-to-noise ratio
- normalized Laplacian variance as a blur/sharpness proxy

The result is a heuristic 0..1 score, not a learned calibration.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot score an empty image")
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError(f"Unsupported image shape for quality scoring: {image.shape}")


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_image_quality_score(image: np.ndarray) -> dict[str, float]:
    gray = _to_gray(image)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    contrast_score = _clip01(std / 64.0)

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_score = _clip01(blur_variance / 1000.0)

    local_mean = cv2.blur(gray, (15, 15))
    noise = gray - local_mean
    signal_power = float(np.mean(local_mean**2))
    noise_power = float(np.mean(noise**2)) + 1e-6
    snr = signal_power / noise_power
    snr_score = _clip01(np.log1p(snr) / np.log1p(50.0))

    image_quality_score = _clip01(0.40 * contrast_score + 0.35 * snr_score + 0.25 * sharpness_score)
    return {
        "image_quality_score": image_quality_score,
        "contrast_score": contrast_score,
        "snr_score": snr_score,
        "sharpness_score": sharpness_score,
        "mean_intensity": mean,
        "std_intensity": std,
        "laplacian_variance": blur_variance,
        "local_snr": float(snr),
    }

