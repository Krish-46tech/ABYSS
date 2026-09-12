"""Train one isolated ABYSS detector candidate using the recorded baseline recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "abyss/ml/detection/runs"
BASE_ARGS = RUN_ROOT / "improved_v2_baseline_finetune_adamw/args.yaml"
BASE_INIT = ROOT / "abyss/ml/detection/weights/baseline_v1.pt"
BASE_DATA = ROOT / "abyss/data/processed/detection/data.yaml"
KINDS = ("shipwreck_weight", "shipwreck_aug", "box_weight", "close_mosaic")


class ShipwreckWeightedTrainer(DetectionTrainer):
    def compute_class_weights(self, class_counts):
        names = {int(key): str(value) for key, value in self.data["names"].items()}
        wreck_ids = [index for index, name in names.items() if name == "Shipwreck"]
        if len(wreck_ids) != 1 or np.any(class_counts <= 0):
            raise ValueError(f"Expected one Shipwreck class and positive train counts: {names}, {class_counts}")
        wreck = wreck_ids[0]
        other_max = max(float(count) for index, count in enumerate(class_counts) if index != wreck)
        weights = np.ones_like(class_counts, dtype=np.float32)
        weights[wreck] = np.sqrt(other_max / float(class_counts[wreck]))
        print(f"Shipwreck-focused class counts: {class_counts.tolist()}; unnormalized loss weights: {weights.tolist()}", flush=True)
        return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--aug-data-yaml", type=Path, help="Train-only augmented YAML for shipwreck_aug")
    args = parser.parse_args()
    baseline = yaml.safe_load(BASE_ARGS.read_text(encoding="utf-8"))
    if not BASE_INIT.is_file() or not BASE_DATA.is_file():
        raise FileNotFoundError("Baseline initialization or data YAML missing")
    if baseline["imgsz"] != 640 or baseline["data"] != "abyss/data/processed/detection/data.yaml":
        raise ValueError("Recorded baseline config changed; inspect before running experiment")
    name = f"bounded_{args.kind}"
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        raise FileExistsError(f"Candidate run already exists: {run_dir}")
    data = BASE_DATA
    if args.kind == "shipwreck_aug":
        if not args.aug_data_yaml or not args.aug_data_yaml.is_file():
            raise ValueError("shipwreck_aug requires a prepared train-only augmented YAML")
        data = args.aug_data_yaml.resolve()
    elif args.aug_data_yaml:
        raise ValueError("Only shipwreck_aug accepts augmented data")

    keys = ("epochs", "patience", "batch", "imgsz", "device", "workers", "optimizer", "lr0", "lrf",
            "weight_decay", "warmup_epochs", "seed", "deterministic", "cos_lr", "close_mosaic",
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "flipud", "fliplr",
            "mosaic", "mixup", "box", "cls", "dfl", "cls_remap", "cache")
    train_args = {key: baseline[key] for key in keys}
    train_args.update({"data": str(data), "project": str(RUN_ROOT), "name": name,
                       "exist_ok": True, "plots": False, "verbose": True})
    if args.kind == "shipwreck_weight":
        train_args["cls_pw"] = 1.0
    elif args.kind == "box_weight":
        train_args["box"] = 9.0
    elif args.kind == "close_mosaic":
        train_args["close_mosaic"] = 15
    print(f"Candidate: {name}; initialization: {BASE_INIT}; train/val YAML: {data}", flush=True)
    print(json.dumps(train_args, indent=2, default=str), flush=True)
    model = YOLO(str(BASE_INIT))
    if args.kind == "shipwreck_weight":
        model.train(trainer=ShipwreckWeightedTrainer, **train_args)
    else:
        model.train(**train_args)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"No trained best checkpoint: {best}")
    result = {"kind": args.kind, "weights": str(best), "weights_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
              "train_args": train_args, "fit_split": "train", "selection_split": "val"}
    output = run_dir / "candidate_summary.json"
    output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
