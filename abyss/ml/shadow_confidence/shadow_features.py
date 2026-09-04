"""Shadow feature extraction for detected sonar objects.

ABYSS assumes a left-to-right scan direction for processed tiles when no
navigation metadata is available. The shadow region is therefore sampled just
behind the right side of each detected box. This assumption is documented as a
Phase 3 heuristic and can be replaced when real sonar heading/slant-range
metadata is available.
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot extract shadow features from an empty image")
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError(f"Unsupported image shape for shadow extraction: {image.shape}")


def compute_shadow_features(image: np.ndarray, bbox_xyxy: list[float] | tuple[float, float, float, float]) -> dict[str, float | list[int]]:
    gray = _gray(image)
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid detection bbox for shadow features: {bbox_xyxy}")

    object_w = x2 - x1
    object_h = y2 - y1
    object_size = max(object_w, object_h)
    shadow_x1 = x2
    shadow_x2 = min(width, x2 + max(4, int(round(object_w * 1.5))))
    band_pad = max(2, int(round(object_h * 0.20)))
    shadow_y1 = max(0, y1 - band_pad)
    shadow_y2 = min(height, y2 + band_pad)

    if shadow_x2 <= shadow_x1:
        return {
            "shadow_consistency_score": 0.0,
            "shadow_length_relative_to_object": 0.0,
            "shadow_darkness_contrast": 0.0,
            "shadow_region_xyxy": [shadow_x1, shadow_y1, shadow_x2, shadow_y2],
            "seabed_region_xyxy": [x1, y1, x2, y2],
        }

    shadow_region = gray[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
    seabed_margin = max(6, int(round(object_w * 0.5)))
    seabed_x1 = min(width - 1, shadow_x2)
    seabed_x2 = min(width, shadow_x2 + seabed_margin)
    if seabed_x2 <= seabed_x1:
        seabed_x1 = max(0, x1 - seabed_margin)
        seabed_x2 = x1
    seabed_region = gray[shadow_y1:shadow_y2, seabed_x1:seabed_x2]
    if shadow_region.size == 0 or seabed_region.size == 0:
        raise ValueError(f"Empty shadow/seabed region for bbox {bbox_xyxy} in image shape {gray.shape}")

    shadow_mean = float(np.mean(shadow_region))
    seabed_mean = float(np.mean(seabed_region))
    darkness_contrast = _clip01((seabed_mean - shadow_mean) / 255.0 * 3.0)

    dark_threshold = min(seabed_mean - 5.0, float(np.percentile(shadow_region, 35)))
    dark_columns = np.mean(shadow_region < dark_threshold, axis=0)
    active_columns = np.where(dark_columns > 0.35)[0]
    shadow_length = int(active_columns[-1] + 1) if active_columns.size else 0
    relative_length = float(shadow_length / max(object_size, 1))

    expected_min = 0.20
    expected_max = 2.00
    if relative_length < expected_min:
        geometry_score = _clip01(relative_length / expected_min)
    elif relative_length > expected_max:
        geometry_score = _clip01(expected_max / relative_length)
    else:
        geometry_score = 1.0

    continuity_score = _clip01(float(np.mean(dark_columns[: max(shadow_length, 1)])) if shadow_length else 0.0)
    shadow_consistency = _clip01(0.45 * darkness_contrast + 0.35 * geometry_score + 0.20 * continuity_score)

    return {
        "shadow_consistency_score": shadow_consistency,
        "shadow_length_relative_to_object": relative_length,
        "shadow_darkness_contrast": darkness_contrast,
        "shadow_geometry_score": geometry_score,
        "shadow_continuity_score": continuity_score,
        "shadow_mean_intensity": shadow_mean,
        "seabed_mean_intensity": seabed_mean,
        "shadow_region_xyxy": [shadow_x1, shadow_y1, shadow_x2, shadow_y2],
        "seabed_region_xyxy": [seabed_x1, shadow_y1, seabed_x2, shadow_y2],
    }

