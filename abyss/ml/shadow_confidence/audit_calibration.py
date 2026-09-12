"""Audit calibration split membership and saved prediction balance.

Run: python abyss/ml/shadow_confidence/audit_calibration.py --data-yaml PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing image split: {path}")
    result = sorted(p.resolve() for p in path.iterdir() if p.suffix.lower() in SUFFIXES)
    if not result:
        raise ValueError(f"Empty image split: {path}")
    return result


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--output-json", type=Path, default=ROOT / "abyss" / "logs" / "calibration_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"]).resolve()
    val = images(root / config["val"])
    test = images(root / config["test"])
    val_names = {p.name for p in val}
    test_names = {p.name for p in test}
    name_overlap = sorted(val_names & test_names)
    test_hashes = {digest(p) for p in test}
    content_overlap = sorted(str(p) for p in val if digest(p) in test_hashes)
    print(f"Validation images: {len(val)}, test images: {len(test)}")
    print(f"Validation/test filename overlap: {len(name_overlap)}")
    print(f"Validation/test exact-content overlap: {len(content_overlap)}")
    print("Exact validation file list:")
    for path in val:
        print(path)
    report = {"data_yaml": str(args.data_yaml.resolve()), "validation_files": [str(p) for p in val],
              "validation_count": len(val), "test_count": len(test), "filename_overlap": name_overlap,
              "content_overlap": content_overlap}
    if args.predictions_csv:
        with args.predictions_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Saved prediction CSV is empty")
        used_paths = sorted({Path(row["image"]).resolve() for row in rows})
        if any(p not in val for p in used_paths):
            raise ValueError("Fitting CSV references images outside the selected validation split")
        correct = sum(int(row["correct"]) for row in rows)
        incorrect = len(rows) - correct
        balance = min(correct, incorrect) / len(rows)
        print(f"Fitting predictions: {len(rows)}; correct: {correct}; incorrect: {incorrect}; minority ratio: {balance:.6f}")
        print(f"Exact fitting file count: {len(used_paths)}")
        for path in used_paths:
            print(f"FIT {path}")
        report.update({"prediction_count": len(rows), "correct_count": correct,
                       "incorrect_count": incorrect, "minority_ratio": balance,
                       "fitting_files": [str(p) for p in used_paths]})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audit report: {args.output_json}")
    if name_overlap or content_overlap:
        raise ValueError("Validation/test split overlap found")
    if "minority_ratio" in report and (report["minority_ratio"] < 0.05 or min(correct, incorrect) < 5):
        raise ValueError("Calibration labels are too imbalanced to fit")


if __name__ == "__main__":
    main()
