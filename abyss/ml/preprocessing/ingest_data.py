"""Ingest ABYSS raw sonar datasets and report measured statistics.

Usage:
    python abyss/ml/preprocessing/ingest_data.py \
      --side-scan-source "/path/to/Side Scan Sonar.v1i.yolov8" \
      --shipwreck-source "/path/to/AI4Shipwrecks"

The script copies the provided datasets into abyss/data/raw/, builds a strict
manifest, and prints real image counts, class counts, dimensions, formats, and
survey/site group counts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    CLASS_TO_ID,
    RAW_ROOT,
    SPLITS_ROOT,
    Sample,
    copy_tree_contents,
    image_paths,
    load_image,
    mask_has_foreground,
    parse_yolo_classes,
    samples_to_dicts,
    setup_logging,
    shipwreck_group_from_name,
    write_json,
    yolo_group_from_name,
)


def build_side_scan_samples(raw_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split_dir in ("train", "valid", "test"):
        image_dir = raw_root / split_dir / "images"
        label_dir = raw_root / split_dir / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Expected YOLO images/labels folders under {raw_root / split_dir}")
        for image_path in image_paths(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            class_ids = parse_yolo_classes(label_path)
            image = load_image(image_path)
            height, width = image.shape[:2]
            samples.append(
                Sample(
                    dataset="side_scan_sonar",
                    source_split=split_dir,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    label_type="yolo_bbox",
                    group_id=yolo_group_from_name(image_path),
                    class_ids=class_ids,
                    width=width,
                    height=height,
                    image_format=image_path.suffix.lower(),
                )
            )
    if not samples:
        raise ValueError(f"No side-scan images found in {raw_root}")
    return samples


def build_shipwreck_samples(raw_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split_dir in ("train", "test"):
        image_dir = raw_root / split_dir / "images"
        label_dir = raw_root / split_dir / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Expected shipwreck images/labels folders under {raw_root / split_dir}")
        for image_path in image_paths(image_dir):
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing mask for shipwreck image {image_path}: expected {label_path}")
            class_ids = [CLASS_TO_ID["Shipwreck"]] if mask_has_foreground(label_path) else []
            image = load_image(image_path)
            height, width = image.shape[:2]
            samples.append(
                Sample(
                    dataset="shipwrecks",
                    source_split=split_dir,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    label_type="binary_mask",
                    group_id=shipwreck_group_from_name(image_path),
                    class_ids=class_ids,
                    width=width,
                    height=height,
                    image_format=image_path.suffix.lower(),
                )
            )
    if not samples:
        raise ValueError(f"No shipwreck images found in {raw_root}")
    return samples


def summarize(samples: list[Sample]) -> dict:
    widths = np.array([sample.width for sample in samples], dtype=np.float32)
    heights = np.array([sample.height for sample in samples], dtype=np.float32)
    class_counts: Counter[str] = Counter()
    for sample in samples:
        if not sample.class_ids:
            class_counts["Unlabeled/Background"] += 1
        for class_id in sample.class_ids:
            class_counts[CLASS_NAMES[class_id]] += 1
    dataset_counts = Counter(sample.dataset for sample in samples)
    group_counts = Counter(sample.group_id for sample in samples)
    format_counts = Counter(sample.image_format for sample in samples)
    return {
        "total_images": len(samples),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "class_distribution": dict(sorted(class_counts.items())),
        "image_dimensions": {
            "width": {"min": int(widths.min()), "max": int(widths.max()), "mean": round(float(widths.mean()), 2)},
            "height": {"min": int(heights.min()), "max": int(heights.max()), "mean": round(float(heights.mean()), 2)},
        },
        "file_formats": dict(sorted(format_counts.items())),
        "survey_or_site_group_count": len(group_counts),
        "largest_groups": dict(group_counts.most_common(10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw ABYSS datasets and print real dataset statistics.")
    parser.add_argument("--side-scan-source", required=True, type=Path)
    parser.add_argument("--shipwreck-source", required=True, type=Path)
    args = parser.parse_args()

    logger = setup_logging("phase1_ingest_data")
    side_dst = RAW_ROOT / "side_scan_sonar"
    wreck_dst = RAW_ROOT / "shipwrecks"
    logger.info("Copying side-scan dataset from %s to %s", args.side_scan_source, side_dst)
    copy_tree_contents(args.side_scan_source, side_dst)
    logger.info("Copying shipwreck dataset from %s to %s", args.shipwreck_source, wreck_dst)
    copy_tree_contents(args.shipwreck_source, wreck_dst)

    samples = build_side_scan_samples(side_dst) + build_shipwreck_samples(wreck_dst)
    summary = summarize(samples)
    manifest = {"class_names": CLASS_NAMES, "summary": summary, "samples": samples_to_dicts(samples)}
    write_json(SPLITS_ROOT / "raw_manifest.json", manifest)
    write_json(SPLITS_ROOT / "ingest_summary.json", summary)

    logger.info("Raw manifest written to %s", SPLITS_ROOT / "raw_manifest.json")
    for key, value in summary.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()

