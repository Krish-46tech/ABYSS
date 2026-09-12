"""Build a train-only Shipwreck augmentation dataset without changing val/test."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_YAML = ROOT / "abyss/data/processed/detection/data.yaml"
DEFAULT_OUTPUT = ROOT / "abyss/data/experiments/bounded_shipwreck_aug"


def transform_box(parts: list[str], matrix: np.ndarray, width: int, height: int) -> str:
    cls = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:])
    if not all(0 <= value <= 1 for value in (cx, cy, bw, bh)):
        raise ValueError(f"Invalid YOLO box: {parts}")
    x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
    y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
    corners = np.asarray([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=float)
    transformed = corners @ matrix.T
    left, top = np.maximum(transformed.min(axis=0), [0, 0])
    right, bottom = np.minimum(transformed.max(axis=0), [width, height])
    if right - left <= 1 or bottom - top <= 1:
        raise ValueError(f"Augmented box became too small: {parts}")
    return f"{cls} {(left + right) / (2 * width):.6f} {(top + bottom) / (2 * height):.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Augmented dataset already exists: {output}")
    config = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    names = {int(key): value for key, value in config["names"].items()}
    wreck_ids = [key for key, value in names.items() if value == "Shipwreck"]
    if len(wreck_ids) != 1:
        raise ValueError(f"Expected one Shipwreck class in source YAML: {names}")
    wreck = wreck_ids[0]
    root = Path(config["path"]).resolve()
    train_images = (root / config["train"]).resolve()
    val_images = (root / config["val"]).resolve()
    test_images = (root / config["test"]).resolve()
    originals = sorted(path for path in train_images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not originals:
        raise ValueError("No source training images")
    val_names = {path.name for path in val_images.iterdir() if path.is_file()}
    test_names = {path.name for path in test_images.iterdir() if path.is_file()}
    if ({path.name for path in originals} & val_names) or ({path.name for path in originals} & test_names):
        raise ValueError("Train filenames overlap validation or test")
    out_images = output / "train/images"
    out_labels = output / "train/labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)
    rng = np.random.default_rng(0)
    augmented = []
    for path in originals:
        label_path = train_images.parent / "labels" / f"{path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        lines = label_path.read_text(encoding="utf-8").splitlines()
        shutil.copy2(path, out_images / path.name)
        shutil.copy2(label_path, out_labels / label_path.name)
        classes = {int(line.split()[0]) for line in lines}
        if wreck not in classes:
            continue
        if classes != {wreck}:
            raise ValueError(f"Mixed-class Shipwreck image requires a separate label policy: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable source image: {path}")
        height, width = image.shape[:2]
        angle = float(rng.uniform(-8, 8))
        scale = float(rng.uniform(0.90, 1.10))
        horizontal_flip = bool(rng.integers(0, 2))
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        transform = np.vstack([rotation, [0, 0, 1]])
        if horizontal_flip:
            flip = np.asarray([[-1, 0, width], [0, 1, 0], [0, 0, 1]], dtype=float)
            transform = transform @ flip
        augmented_image = cv2.warpAffine(image, transform[:2], (width, height),
                                          flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        new_lines = [transform_box(line.split(), transform[:2], width, height) for line in lines]
        new_name = f"aug_shipwreck_{path.name}"
        if new_name in val_names or new_name in test_names:
            raise ValueError(f"Augmented filename overlaps held-out split: {new_name}")
        if not cv2.imwrite(str(out_images / new_name), augmented_image):
            raise OSError(f"Could not write augmented image: {new_name}")
        (out_labels / f"{Path(new_name).stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        augmented.append({"source": str(path), "image": str(out_images / new_name), "label_count": len(new_lines),
                          "rotation_degrees": angle, "scale": scale, "horizontal_flip": horizontal_flip})
    if not augmented:
        raise ValueError("No Shipwreck training examples were augmented")
    candidate_yaml = {"path": str(output), "train": "train/images", "val": str(val_images),
                      "names": config["names"]}
    (output / "data.yaml").write_text(yaml.safe_dump(candidate_yaml, sort_keys=False), encoding="utf-8")
    summary = {"source_yaml": str(BASE_YAML), "candidate_yaml": str(output / "data.yaml"),
               "source_train_image_count": len(originals), "augmented_shipwreck_image_count": len(augmented),
               "candidate_train_image_count": len(originals) + len(augmented),
               "validation_images": str(val_images), "test_images": str(test_images),
               "split_filename_overlap": 0, "augmentations": augmented}
    (output / "augmentation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "augmentations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
