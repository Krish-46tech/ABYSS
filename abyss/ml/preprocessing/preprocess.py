"""Preprocess ABYSS images into a YOLO-compatible detection dataset.

Usage:
    python abyss/ml/preprocessing/preprocess.py

For each split this script denoises sonar imagery, normalizes intensity,
letterbox-resizes to 640x640, converts/adjusts labels, and writes:

    abyss/data/processed/detection/{train,val,test}/images
    abyss/data/processed/detection/{train,val,test}/labels

It also saves five before/after comparison images under
abyss/docs/preprocessing_examples/.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import (
    CLASS_NAMES,
    DOCS_ROOT,
    PROCESSED_ROOT,
    SPLITS_ROOT,
    TARGET_SIZE,
    Sample,
    load_image,
    read_json,
    setup_logging,
    write_json,
)


def denoise_and_normalize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.medianBlur(gray, 3)
    normalized = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def letterbox(image: np.ndarray, target_size: int = TARGET_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape for letterbox: {image.shape}")
    scale = min(target_size / width, target_size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def convert_yolo_label(label_path: Path, sample: Sample, scale: float, pad_x: int, pad_y: int) -> list[str]:
    lines: list[str] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO label at {label_path}:{line_number}")
        class_id = int(float(parts[0]))
        cx, cy, width, height = [float(value) for value in parts[1:]]
        abs_cx = cx * sample.width * scale + pad_x
        abs_cy = cy * sample.height * scale + pad_y
        abs_w = width * sample.width * scale
        abs_h = height * sample.height * scale
        lines.append(f"{class_id} {abs_cx / TARGET_SIZE:.6f} {abs_cy / TARGET_SIZE:.6f} {abs_w / TARGET_SIZE:.6f} {abs_h / TARGET_SIZE:.6f}")
    return lines


def mask_to_yolo_label(mask_path: Path, class_id: int, scale: float, pad_x: int, pad_y: int) -> list[str]:
    mask = load_image(mask_path, cv2.IMREAD_GRAYSCALE)
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return []
    y_min, x_min = foreground.min(axis=0)
    y_max, x_max = foreground.max(axis=0)
    x1 = float(x_min) * scale + pad_x
    y1 = float(y_min) * scale + pad_y
    x2 = float(x_max + 1) * scale + pad_x
    y2 = float(y_max + 1) * scale + pad_y
    cx = ((x1 + x2) / 2.0) / TARGET_SIZE
    cy = ((y1 + y2) / 2.0) / TARGET_SIZE
    width = (x2 - x1) / TARGET_SIZE
    height = (y2 - y1) / TARGET_SIZE
    return [f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"]


def safe_output_name(sample: Sample) -> str:
    source = Path(sample.image_path)
    return f"{sample.dataset}_{source.stem}.jpg"


def save_comparison(before: np.ndarray, after: np.ndarray, output_path: Path) -> None:
    before_small, _, _, _ = letterbox(before)
    combined = np.hstack([before_small, after])
    cv2.putText(combined, "before", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "after", (TARGET_SIZE + 20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), combined):
        raise IOError(f"Failed to write preprocessing comparison image: {output_path}")


def main() -> None:
    logger = setup_logging("phase1_preprocess")
    split_manifest = read_json(SPLITS_ROOT / "split_manifest.json")
    output_root = PROCESSED_ROOT / "detection"
    if output_root.exists():
        shutil.rmtree(output_root)

    example_dir = DOCS_ROOT / "preprocessing_examples"
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)

    processed_summary: dict[str, object] = {"target_size": TARGET_SIZE, "splits": {}, "examples": []}
    examples_saved = 0

    for split in ("train", "val", "test"):
        samples = [Sample(**item) for item in split_manifest[split]]
        image_out = output_root / split / "images"
        label_out = output_root / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        class_counts: Counter[str] = Counter()

        for sample in samples:
            original = load_image(Path(sample.image_path))
            enhanced = denoise_and_normalize(original)
            processed, scale, pad_x, pad_y = letterbox(enhanced)
            output_name = safe_output_name(sample)
            output_image_path = image_out / output_name
            output_label_path = label_out / f"{Path(output_name).stem}.txt"

            if sample.label_type == "yolo_bbox":
                label_lines = convert_yolo_label(Path(sample.label_path), sample, scale, pad_x, pad_y)
            elif sample.label_type == "binary_mask":
                label_lines = mask_to_yolo_label(Path(sample.label_path), 2, scale, pad_x, pad_y)
            else:
                raise ValueError(f"Unsupported label type {sample.label_type} for {sample.image_path}")

            if not cv2.imwrite(str(output_image_path), processed):
                raise IOError(f"Failed to write processed image: {output_image_path}")
            output_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            for label_line in label_lines:
                class_counts[CLASS_NAMES[int(label_line.split()[0])]] += 1

            if examples_saved < 5:
                example_path = example_dir / f"preprocess_example_{examples_saved + 1}_{Path(output_name).stem}.jpg"
                save_comparison(original, processed, example_path)
                processed_summary["examples"].append(str(example_path))
                examples_saved += 1

        processed_summary["splits"][split] = {
            "images": len(samples),
            "labels": len(list(label_out.glob("*.txt"))),
            "class_counts": dict(sorted(class_counts.items())),
            "image_dir": str(image_out),
            "label_dir": str(label_out),
        }
        logger.info("%s processed: %s", split, processed_summary["splits"][split])

    if examples_saved < 5:
        raise ValueError(f"Expected to save 5 preprocessing examples, saved {examples_saved}")

    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    write_json(SPLITS_ROOT / "processed_summary.json", processed_summary)
    logger.info("Processed dataset summary written to %s", SPLITS_ROOT / "processed_summary.json")
    logger.info("Preprocessing examples: %s", processed_summary["examples"])


if __name__ == "__main__":
    main()
