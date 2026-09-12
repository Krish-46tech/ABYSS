"""Real endpoint tests for the ABYSS FastAPI backend."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from abyss.backend.app.main import app
from abyss.backend.app.services import infer_class_name


ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE = ROOT / "abyss" / "data" / "processed" / "detection" / "test" / "images" / "side_scan_sonar_ship-080_png.rf.626d1c7098ccf0ef08f978419e5e0256.jpg"


client = TestClient(app)


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
    for key in ("detector_confidence", "image_quality_score", "shadow_consistency_score", "composite_confidence"):
        assert 0.0 <= detection[key] <= 1.0
    assert len(detection["bbox_xyxy"]) == 4


def test_detect_rejects_corrupt_upload() -> None:
    response = client.post("/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]


def test_aircraft_source_context_corrects_ship_collapse() -> None:
    class_id, class_name = infer_class_name(1, [40, 80, 520, 390], "side_scan_sonar_plane-062.jpg")
    assert class_id == 0
    assert class_name == "Plane"


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
