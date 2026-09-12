"""Audit whether held-out test data supports real positional RMSE.

Run: python abyss/ml/shadow_confidence/audit_position.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
METADATA_EXTENSIONS = {".csv", ".json", ".yaml", ".yml", ".xml", ".gpx", ".nmea", ".txt"}
POSITION_KEYS = {"lat", "lon", "latitude", "longitude", "gps", "navigation", "ground_truth_coordinates"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "abyss" / "data" / "splits" / "split_manifest.json")
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "abyss" / "data" / "processed" / "detection" / "data.yaml")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "abyss" / "data" / "raw")
    parser.add_argument("--output-json", type=Path, default=ROOT / "abyss" / "logs" / "phase4_position_audit.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["test"]
    if not samples:
        raise ValueError("Test manifest is empty")
    config = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    test_root = Path(config["path"]) / config["test"]
    label_root = test_root.parent / "labels"
    sample_fields = sorted({key for sample in samples for key in sample})
    position_fields = sorted(set(sample_fields) & POSITION_KEYS)
    label_count = 0
    exif_gps_count = 0
    for sample in samples:
        source = Path(sample["image_path"])
        processed_label = label_root / f"{sample['dataset']}_{source.stem}.txt"
        if not processed_label.is_file():
            raise FileNotFoundError(f"Missing processed test label: {processed_label}")
        for line in processed_label.read_text(encoding="utf-8").splitlines():
            if len(line.split()) != 5:
                raise ValueError(f"Unexpected test label schema in {processed_label}")
            label_count += 1
        if not source.is_file():
            raise FileNotFoundError(f"Missing raw source image: {source}")
        with Image.open(source) as image:
            if image.getexif().get(34853):
                exif_gps_count += 1
    metadata_files = sorted(path for path in args.raw_root.rglob("*")
                            if path.is_file() and path.suffix.lower() in METADATA_EXTENSIONS
                            and "images" not in path.parts and "labels" not in path.parts)
    keyword_files = [str(path) for path in metadata_files if any(
        term in path.read_text(encoding="utf-8", errors="replace").lower()
        for term in ("latitude", "longitude", "gps", "navigation", "ground_truth_coordinates"))]
    if position_fields or exif_gps_count or keyword_files:
        raise ValueError("Potential positional data exists; inspect it before declaring RMSE not measurable")
    result = {"test_images": len(samples), "test_object_labels": label_count, "manifest_fields": sample_fields,
              "position_fields": position_fields, "raw_images_with_gps_exif": exif_gps_count,
              "metadata_files_inspected": [str(path) for path in metadata_files],
              "metadata_files_with_position_keywords": keyword_files,
              "positional_rmse": "not measurable — no real ground-truth coordinates"}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
