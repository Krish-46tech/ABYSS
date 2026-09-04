"""Shared helpers for ABYSS Phase 1 preprocessing scripts.

These helpers are intentionally strict: missing files, unreadable images, and
bad labels raise explicit exceptions instead of being skipped silently.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ABYSS_ROOT = PROJECT_ROOT / "abyss"
RAW_ROOT = ABYSS_ROOT / "data" / "raw"
SPLITS_ROOT = ABYSS_ROOT / "data" / "splits"
PROCESSED_ROOT = ABYSS_ROOT / "data" / "processed"
DOCS_ROOT = ABYSS_ROOT / "docs"
LOGS_ROOT = ABYSS_ROOT / "logs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = 640
CLASS_NAMES = ["Plane", "Ship", "Shipwreck"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Sample:
    dataset: str
    source_split: str
    image_path: str
    label_path: str
    label_type: str
    group_id: str
    class_ids: list[int]
    width: int
    height: int
    image_format: str


def setup_logging(script_name: str) -> logging.Logger:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(LOGS_ROOT / f"{script_name}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Dataset source folder does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.name != ".DS_Store":
            shutil.copy2(item, target)


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"OpenCV could not read image file: {path}")
    return image


def parse_yolo_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        raise FileNotFoundError(f"YOLO label file does not exist: {label_path}")
    class_ids: list[int] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO label at {label_path}:{line_number}: expected 5 fields, got {len(parts)}")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"Class id {class_id} out of range in {label_path}:{line_number}")
        class_ids.append(class_id)
    return class_ids


def mask_has_foreground(mask_path: Path) -> bool:
    mask = load_image(mask_path, cv2.IMREAD_GRAYSCALE)
    return bool(np.count_nonzero(mask) > 0)


def yolo_group_from_name(path: Path) -> str:
    """Infer a survey-pass proxy from Roboflow filenames.

    The side-scan dataset does not include navigation or survey metadata. File
    names follow patterns such as ``ship-154_png.rf.<hash>.jpg``. We treat the
    object prefix plus sequential 25-image numeric block as one proxy survey
    pass, keeping nearby captures together during splitting.
    """

    stem = path.stem.split(".rf.")[0]
    match = re.match(r"(?P<prefix>[A-Za-z]+)-(?P<number>\d+)", stem)
    if not match:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
        return f"side_scan_unknown_{safe}"
    prefix = match.group("prefix").lower()
    number = int(match.group("number"))
    block = number // 25
    return f"side_scan_{prefix}_block_{block:03d}"


def shipwreck_group_from_name(path: Path) -> str:
    """Infer the natural wreck/site group from names like ``Barge_No_1_13``."""

    stem = path.stem
    match = re.match(r"(?P<site>.+)_\d+$", stem)
    site = match.group("site") if match else stem
    safe = re.sub(r"[^A-Za-z0-9]+", "_", site).strip("_").lower()
    if not safe:
        raise ValueError(f"Could not infer shipwreck site group from filename: {path.name}")
    return f"shipwreck_{safe}"


def samples_to_dicts(samples: list[Sample]) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in samples]


def load_samples(path: Path | None = None) -> list[Sample]:
    data = read_json(path or SPLITS_ROOT / "raw_manifest.json")
    return [Sample(**item) for item in data["samples"]]


def load_yolo_names(dataset_root: Path) -> list[str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return CLASS_NAMES[:2]
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid names list in {yaml_path}")
    return [str(name) for name in names]

