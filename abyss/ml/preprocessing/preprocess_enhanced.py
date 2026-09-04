"""Build an enhanced YOLO dataset for the improved ABYSS detector.

Usage:
    python abyss/ml/preprocessing/preprocess_enhanced.py

Enhancements over the baseline preprocessing:
- percentile intensity clipping to reduce outlier dominance
- bilateral filtering for sonar speckle while preserving edges
- CLAHE local contrast enhancement
- mild unsharp masking to recover object boundaries
- 768x768 letterbox output for a larger detector input
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import CLASS_NAMES, DOCS_ROOT, PROCESSED_ROOT, SPLITS_ROOT, Sample, load_image, read_json, setup_logging, write_json
from preprocess import convert_yolo_label, mask_to_yolo_label, save_comparison


ENHANCED_SIZE = 768


def enhanced_sonar_preprocess(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.uint8)
    low, high = np.percentile(gray, [1.0, 99.0])
    if high <= low:
        raise ValueError("Image has degenerate intensity range; cannot enhance contrast")
    clipped = np.clip(gray, low, high)
    scaled = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    denoised = cv2.bilateralFilter(scaled, d=7, sigmaColor=45, sigmaSpace=45)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(contrast, 1.25, blurred, -0.25, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def letterbox_enhanced(image: np.ndarray, target_size: int = ENHANCED_SIZE) -> tuple[np.ndarray, float, int, int]:
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


def convert_labels_for_size(sample: Sample, scale: float, pad_x: int, pad_y: int) -> list[str]:
    baseline_size = 640
    label_path = Path(sample.label_path)
    if sample.label_type == "yolo_bbox":
        baseline_lines = convert_yolo_label(label_path, sample, scale, pad_x, pad_y)
    elif sample.label_type == "binary_mask":
        baseline_lines = mask_to_yolo_label(label_path, 2, scale, pad_x, pad_y)
    else:
        raise ValueError(f"Unsupported label type {sample.label_type} for {sample.image_path}")
    # convert_yolo_label imports TARGET_SIZE=640, so rescale the normalized
    # coordinates into the enhanced canvas size. This keeps the inherited
    # conversion logic strict while correcting for the 768 output.
    corrected: list[str] = []
    for line in baseline_lines:
        class_id, cx, cy, width, height = line.split()
        corrected.append(
            f"{class_id} {float(cx) * baseline_size / ENHANCED_SIZE:.6f} "
            f"{float(cy) * baseline_size / ENHANCED_SIZE:.6f} "
            f"{float(width) * baseline_size / ENHANCED_SIZE:.6f} "
            f"{float(height) * baseline_size / ENHANCED_SIZE:.6f}"
        )
    return corrected


def safe_output_name(sample: Sample) -> str:
    source = Path(sample.image_path)
    return f"{sample.dataset}_{source.stem}.jpg"


def main() -> None:
    logger = setup_logging("improved_preprocess_enhanced")
    split_manifest = read_json(SPLITS_ROOT / "split_manifest.json")
    output_root = PROCESSED_ROOT / "detection_enhanced"
    if output_root.exists():
        shutil.rmtree(output_root)

    example_dir = DOCS_ROOT / "preprocessing_examples_enhanced"
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"target_size": ENHANCED_SIZE, "splits": {}, "examples": []}
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
            enhanced = enhanced_sonar_preprocess(original)
            processed, scale, pad_x, pad_y = letterbox_enhanced(enhanced)
            output_name = safe_output_name(sample)
            output_image_path = image_out / output_name
            output_label_path = label_out / f"{Path(output_name).stem}.txt"
            label_lines = convert_labels_for_size(sample, scale, pad_x, pad_y)
            if not cv2.imwrite(str(output_image_path), processed):
                raise IOError(f"Failed to write enhanced processed image: {output_image_path}")
            output_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            for label_line in label_lines:
                class_counts[CLASS_NAMES[int(label_line.split()[0])]] += 1
            if examples_saved < 5:
                example_path = example_dir / f"enhanced_example_{examples_saved + 1}_{Path(output_name).stem}.jpg"
                before_small, _, _, _ = letterbox_enhanced(original)
                combined = np.hstack([before_small, processed])
                cv2.putText(combined, "before", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(combined, "enhanced", (ENHANCED_SIZE + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                if not cv2.imwrite(str(example_path), combined):
                    raise IOError(f"Failed to write enhanced comparison image: {example_path}")
                summary["examples"].append(str(example_path))
                examples_saved += 1
        summary["splits"][split] = {
            "images": len(samples),
            "labels": len(list(label_out.glob("*.txt"))),
            "class_counts": dict(sorted(class_counts.items())),
            "image_dir": str(image_out),
            "label_dir": str(label_out),
        }
        logger.info("%s enhanced processed: %s", split, summary["splits"][split])

    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)
    write_json(SPLITS_ROOT / "processed_enhanced_summary.json", summary)
    logger.info("Enhanced dataset summary written to %s", SPLITS_ROOT / "processed_enhanced_summary.json")


if __name__ == "__main__":
    main()

