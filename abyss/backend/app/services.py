"""Backend service functions for ABYSS.

All functions either compute real values or raise explicit errors. The only
simulated behavior is geolocation fallback when callers do not provide survey
navigation metadata; that response is labeled as simulated.
"""

from __future__ import annotations

import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "abyss" / "ml" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "abyss" / "ml" / "shadow_confidence"))
sys.path.append(str(PROJECT_ROOT / "abyss" / "ml" / "detection"))

from confidence_fusion import fuse_detection_confidence  # type: ignore  # noqa: E402
from image_quality import compute_image_quality_score  # type: ignore  # noqa: E402
from preprocess import denoise_and_normalize, letterbox  # type: ignore  # noqa: E402
from shadow_features import compute_shadow_features  # type: ignore  # noqa: E402
from utils import CLASS_NAMES  # type: ignore  # noqa: E402


WEIGHTS_PATH = PROJECT_ROOT / "abyss" / "ml" / "detection" / "weights" / "improved_v2_baseline_finetune_adamw.pt"
TARGET_SIZE = 640
DEFAULT_SIMULATED_ORIGIN = (18.5204, 73.8567)


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Trained model weights not found: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def decode_uploaded_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise ValueError("Uploaded image is empty")
    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image")
    return image


def detect_objects(image: np.ndarray, conf: float = 0.25) -> tuple[list[dict], list[int], float]:
    if image is None or image.size == 0:
        raise ValueError("Cannot run detection on an empty image")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("Detector confidence threshold must be between 0 and 1")

    processed, _, _, _ = letterbox(denoise_and_normalize(image), TARGET_SIZE)
    quality_image, _, _, _ = letterbox(image, TARGET_SIZE)
    quality = compute_image_quality_score(quality_image)
    model = get_model()

    start = time.perf_counter()
    results = model.predict(processed, imgsz=TARGET_SIZE, conf=conf, verbose=False)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if not results:
        raise RuntimeError("Ultralytics returned no prediction result objects")

    detections: list[dict] = []
    boxes = results[0].boxes
    if boxes is not None:
        for index, box in enumerate(boxes):
            class_id = int(box.cls.item())
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"Model returned unknown class id {class_id}")
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            detector_confidence = float(box.conf.item())
            shadow = compute_shadow_features(processed, xyxy)
            fusion = fuse_detection_confidence(
                detector_confidence,
                quality["image_quality_score"],
                float(shadow["shadow_consistency_score"]),
            )
            detections.append(
                {
                    "detection_id": index,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "bbox_xyxy": xyxy,
                    "detector_confidence": detector_confidence,
                    "image_quality_score": quality["image_quality_score"],
                    "shadow_consistency_score": shadow["shadow_consistency_score"],
                    "composite_confidence": fusion["composite_confidence"],
                }
            )
    return detections, list(processed.shape), latency_ms


def geolocate_pixel(
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    origin_lat: float | None,
    origin_lon: float | None,
    heading_degrees: float,
    meters_per_pixel: float,
) -> dict:
    lat0, lon0 = (origin_lat, origin_lon) if origin_lat is not None and origin_lon is not None else DEFAULT_SIMULATED_ORIGIN
    source = "provided_metadata" if origin_lat is not None and origin_lon is not None else "simulated"

    centered_x = pixel_x - image_width / 2.0
    centered_y = image_height / 2.0 - pixel_y
    local_east = centered_x * meters_per_pixel
    local_north = centered_y * meters_per_pixel

    heading = math.radians(heading_degrees)
    east = local_east * math.cos(heading) + local_north * math.sin(heading)
    north = -local_east * math.sin(heading) + local_north * math.cos(heading)

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(lat0))
    if abs(meters_per_degree_lon) < 1e-6:
        raise ValueError("Longitude conversion is unstable near the poles")

    return {
        "lat": lat0 + north / meters_per_degree_lat,
        "lon": lon0 + east / meters_per_degree_lon,
        "geolocation_source": source,
        "east_offset_m": east,
        "north_offset_m": north,
        "input_pixel": [pixel_x, pixel_y],
    }


def detection_area_fraction(bbox_xyxy: list[float], image_size: int = TARGET_SIZE) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(0.0, min(1.0, area / float(image_size * image_size)))


def prioritize_detections(detections: list[dict]) -> list[dict]:
    hazard_weights = {"Plane": 0.45, "Ship": 0.65, "Shipwreck": 0.85, "UNKNOWN_ANOMALY": 0.75}
    ranked = []
    for item in detections:
        class_name = item["class_name"]
        confidence_component = float(item["composite_confidence"])
        hazard_component = hazard_weights.get(class_name, 0.50)
        size_component = detection_area_fraction(item["bbox_xyxy"])
        distance = item.get("distance_to_sensitive_zone_m")
        proximity_component = 0.0 if distance is None else max(0.0, min(1.0, 1.0 - float(distance) / 5000.0))
        priority_score = (
            0.45 * confidence_component
            + 0.25 * hazard_component
            + 0.20 * size_component
            + 0.10 * proximity_component
        )
        ranked.append(
            {
                "detection_id": item["detection_id"],
                "class_name": class_name,
                "composite_confidence": confidence_component,
                "priority_score": priority_score,
                "score_breakdown": {
                    "confidence_component": confidence_component,
                    "hazard_component": hazard_component,
                    "size_component": size_component,
                    "proximity_component": proximity_component,
                    "confidence_weight": 0.45,
                    "hazard_weight": 0.25,
                    "size_weight": 0.20,
                    "proximity_weight": 0.10,
                },
            }
        )
    ranked.sort(key=lambda row: row["priority_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked
