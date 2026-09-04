"""Pydantic request/response schemas for ABYSS Phase 4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    detection_id: int
    class_id: int
    class_name: str
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    detector_confidence: float = Field(ge=0.0, le=1.0)
    image_quality_score: float = Field(ge=0.0, le=1.0)
    shadow_consistency_score: float = Field(ge=0.0, le=1.0)
    composite_confidence: float = Field(ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    filename: str
    image_shape: list[int]
    processing_latency_ms: float
    detections: list[DetectionBox]


class GeolocateRequest(BaseModel):
    pixel_x: float = Field(ge=0.0)
    pixel_y: float = Field(ge=0.0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    origin_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    origin_lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    heading_degrees: float = Field(default=0.0, ge=0.0, lt=360.0)
    meters_per_pixel: float = Field(default=0.5, gt=0.0)


class GeolocateResponse(BaseModel):
    lat: float
    lon: float
    geolocation_source: Literal["provided_metadata", "simulated"]
    east_offset_m: float
    north_offset_m: float
    input_pixel: list[float]


class PriorityDetection(BaseModel):
    detection_id: str
    class_name: str
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    composite_confidence: float = Field(ge=0.0, le=1.0)
    distance_to_sensitive_zone_m: float | None = Field(default=None, ge=0.0)


class PriorityRequest(BaseModel):
    detections: list[PriorityDetection] = Field(min_length=1)


class PriorityItem(BaseModel):
    rank: int
    detection_id: str
    priority_score: float
    score_breakdown: dict[str, float]
    class_name: str
    composite_confidence: float


class PriorityResponse(BaseModel):
    ranked_detections: list[PriorityItem]

