# ABYSS

ABYSS is a sonar object detection and triage system for Smart India Hackathon problem statement SIH26057. The project is built phase by phase, with every phase ending in a runnable verification step that reports real outputs from the local data and code.

## Repository Structure

```text
abyss/
  data/
    raw/          # Local raw datasets copied or linked for ingestion. Not committed.
    processed/    # Preprocessed train/val/test data. Not committed.
    splits/       # Split metadata and manifests.
  ml/
    preprocessing/       # Ingestion, splitting, and preprocessing scripts.
    detection/           # YOLO training, evaluation, inference, and committed small weights.
    shadow_confidence/   # Shadow features, image quality, and confidence fusion.
    notebooks/           # Local exploration notebooks.
  backend/
    app/          # FastAPI application.
    tests/        # Pytest backend tests.
  docs/           # Human-readable documentation and generated examples.
  logs/           # Console/file logs and metric JSON files.
frontend/         # React/Vite dashboard for live detection triage.
```

## Setup From Scratch

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Verify the environment:

```bash
python -m pip freeze
```

## Phase Workflow

This repository is intentionally developed in ordered phases:

1. Phase 0: project structure and Python environment.
2. Phase 1: dataset ingestion, survey-aware splits, preprocessing, and examples.
3. Phase 2: YOLOv8 baseline training, evaluation, and image inference.
4. Phase 3: shadow features, image quality scoring, and confidence fusion.
5. Phase 4: FastAPI backend and endpoint tests.
6. Phase 5: React frontend dashboard calling the real backend APIs.

Do not commit local datasets, generated processed data, logs, virtual environments, frontend build output, or large experiment outputs. The small baseline and improved YOLO weights used by the backend are committed so `/detect` works after cloning.

## Dataset Placement

The raw source datasets are expected to be available locally and ingested during Phase 1:

- Side Scan Sonar YOLO dataset: `Side Scan Sonar.v1i.yolov8`
- Shipwreck segmentation dataset: `AI4Shipwrecks` or `AIforShipwrecks`

Phase 1 scripts will copy or index these real files into `abyss/data/raw/` and print measured dataset statistics.

## Phase 1 Commands

Run these commands from the repository root after activating the virtual environment:

```bash
python abyss/ml/preprocessing/ingest_data.py \
  --side-scan-source "/Users/g.o.a.t/Downloads/Side Scan Sonar.v1i.yolov8" \
  --shipwreck-source "/Users/g.o.a.t/Downloads/AI4Shipwrecks"

python abyss/ml/preprocessing/split_data.py
python abyss/ml/preprocessing/preprocess.py
```

Phase 1 writes manifests and summaries to `abyss/data/splits/`, processed YOLO-compatible data to `abyss/data/processed/detection/`, preprocessing comparisons to `abyss/docs/preprocessing_examples/`, and logs to `abyss/logs/`.

## Run The Backend

```bash
PYTHONPATH=. .venv/bin/uvicorn abyss.backend.app.main:app --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Run The Frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Dashboard:

```text
http://127.0.0.1:5173
```
