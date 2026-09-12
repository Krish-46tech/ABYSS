"""Real endpoint tests for the ABYSS FastAPI backend."""

from __future__ import annotations

from pathlib import Path
import hashlib

import yaml

from fastapi.testclient import TestClient

from abyss.backend.app.main import app
from abyss.backend.app.services import CALIBRATION_PATH, WEIGHTS_PATH, decode_uploaded_image, get_model
from abyss.ml.preprocessing.preprocess import denoise_and_normalize, letterbox


ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE = ROOT / "abyss" / "data" / "raw" / "side_scan_sonar" / "train" / "images" / "ship-019_png.rf.2cdef02713a62436088a0b14063017b6.jpg"


client = TestClient(app)


def test_deployed_calibration_matches_checkpoint_training_config() -> None:
    training = yaml.safe_load((ROOT / "abyss/ml/detection/runs/improved_v2_baseline_finetune_adamw/args.yaml").read_text())
    artifact = yaml.safe_load(CALIBRATION_PATH.read_text())
    assert Path(artifact["weights"]).resolve() == WEIGHTS_PATH.resolve()
    assert artifact["weights_sha256"] == hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    assert Path(artifact["data_yaml"]).resolve() == (ROOT / training["data"]).resolve()
    assert artifact["imgsz"] == training["imgsz"]
    assert artifact["preprocess_mode"] == "baseline"


def test_detect_returns_real_detections() -> None:
    assert SAMPLE_IMAGE.exists(), f"Missing sample image: {SAMPLE_IMAGE}"
    with SAMPLE_IMAGE.open("rb") as handle:
        response = client.post("/detect?conf=0.25", files={"file": ("sample.jpg", handle, "image/jpeg")})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["filename"] == "sample.jpg"
    assert data["image_shape"] == [640, 640, 3]
    assert data["processing_latency_ms"] > 0
    assert len(data["detections"]) >= 1
    detection = data["detections"][0]
    for key in ("detector_confidence", "image_quality_score", "shadow_consistency_score",
                "logistic_fused_probability", "temperature_calibrated_probability", "composite_confidence"):
        assert 0.0 <= detection[key] <= 1.0
    assert detection["composite_confidence"] == detection["temperature_calibrated_probability"]
    assert len(detection["bbox_xyxy"]) == 4
    assert all(0.0 <= coordinate <= 640.0 for coordinate in detection["bbox_xyxy"])


def test_detect_rejects_corrupt_upload() -> None:
    response = client.post("/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]


def test_plane_image_class_is_native_model_output_for_any_filename() -> None:
    plane_path = ROOT / "abyss/data/raw/side_scan_sonar/train/images/plane-001_png.rf.dd5de3545eefa071ea7034204a08c0c1.jpg"
    assert plane_path.is_file()
    contents = plane_path.read_bytes()
    responses = [client.post("/detect?conf=0.25", files={"file": (name, contents, "image/jpeg")})
                 for name in ("plane-001.jpg", "neutral.jpg")]
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    detections = [response.json()["detections"] for response in responses]
    assert detections[0] and detections[1]
    assert [(item["class_id"], item["class_name"]) for item in detections[0]] == [
        (item["class_id"], item["class_name"]) for item in detections[1]
    ]
    processed, _, _, _ = letterbox(denoise_and_normalize(decode_uploaded_image(contents)), 640)
    native = get_model().predict(processed, imgsz=640, conf=0.25, verbose=False)[0]
    assert native.boxes is not None and len(native.boxes) == len(detections[0])
    assert [item["class_id"] for item in detections[0]] == [int(box.cls.item()) for box in native.boxes]


def test_geolocate_with_provided_metadata() -> None:
    response = client.post(
        "/geolocate",
        json={
            "pixel_x": 320,
            "pixel_y": 300,
            "image_width": 640,
            "image_height": 640,
            "origin_lat": 18.5204,
            "origin_lon": 73.8567,
            "heading_degrees": 45,
            "meters_per_pixel": 0.5,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["geolocation_source"] == "provided_metadata"
    assert -90 <= data["lat"] <= 90
    assert -180 <= data["lon"] <= 180
    assert data["north_offset_m"] != 0


def test_geolocate_simulated_is_labeled() -> None:
    response = client.post(
        "/geolocate",
        json={"pixel_x": 400, "pixel_y": 300, "image_width": 640, "image_height": 640, "heading_degrees": 0, "meters_per_pixel": 1.0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["geolocation_source"] == "simulated"


def test_priority_ranking_has_score_breakdown() -> None:
    response = client.post(
        "/priority",
        json={
            "detections": [
                {
                    "detection_id": "ship-low",
                    "class_name": "Ship",
                    "bbox_xyxy": [28, 15, 552, 545],
                    "composite_confidence": 0.59,
                    "distance_to_sensitive_zone_m": 1200,
                },
                {
                    "detection_id": "wreck-high",
                    "class_name": "Shipwreck",
                    "bbox_xyxy": [176, 342, 276, 368],
                    "composite_confidence": 0.82,
                    "distance_to_sensitive_zone_m": 300,
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    ranked = response.json()["ranked_detections"]
    assert len(ranked) == 2
    assert ranked[0]["priority_score"] >= ranked[1]["priority_score"]
    assert ranked[0]["rank"] == 1
    for item in ranked:
        assert item["score_breakdown"]["confidence_weight"] == 0.45
        assert 0.0 <= item["priority_score"] <= 1.0
