"""ABYSS FastAPI backend.

Run:
    uvicorn abyss.backend.app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import DetectResponse, GeolocateRequest, GeolocateResponse, PriorityRequest, PriorityResponse
from .services import decode_uploaded_image, detect_objects, geolocate_pixel, prioritize_detections


app = FastAPI(title="ABYSS Sonar Detection API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ABYSS Sonar Detection API"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), conf: float = Query(default=0.25, ge=0.0, le=1.0)) -> dict:
    contents = await file.read()
    try:
        image = decode_uploaded_image(contents)
        detections, image_shape, latency_ms = detect_objects(image, conf=conf, source_name=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "filename": file.filename or "uploaded_image",
        "image_shape": image_shape,
        "processing_latency_ms": latency_ms,
        "detections": detections,
    }


@app.post("/geolocate", response_model=GeolocateResponse)
async def geolocate(payload: GeolocateRequest) -> dict:
    try:
        return geolocate_pixel(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/priority", response_model=PriorityResponse)
async def priority(payload: PriorityRequest) -> dict:
    return {"ranked_detections": prioritize_detections([item.model_dump() for item in payload.detections])}
