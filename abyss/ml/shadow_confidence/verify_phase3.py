"""Run five real source-image inferences using a fitted calibration artifact.

Run: python abyss/ml/shadow_confidence/verify_phase3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "abyss" / "ml" / "detection"))
from infer import run_inference  # noqa: E402
from confidence_fusion import DEFAULT_ARTIFACT, load_calibration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--predictions-csv", type=Path, default=ROOT / "abyss" / "logs" / "phase3_calibration" / "validation_predictions.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "abyss" / "data" / "splits" / "split_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "abyss" / "logs" / "phase3_examples")
    args = parser.parse_args()
    artifact = load_calibration(args.artifact)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_by_processed_name = {f"{row['dataset']}_{Path(row['image_path']).stem}": Path(row["image_path"]) for row in manifest["val"]}
    with args.predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    maxima: dict[str, float] = {}
    for row in rows:
        stem = Path(row["image"]).stem
        maxima[stem] = max(maxima.get(stem, 0.0), float(row["detector_confidence"]))
    candidates = sorted(
        (stem for stem in maxima if stem in source_by_processed_name and "test" not in source_by_processed_name[stem].parts),
        key=maxima.get,
    )
    if len(candidates) < 5:
        raise ValueError("Need at least five validation images with detections")
    selected = [candidates[round(i * (len(candidates) - 1) / 4)] for i in range(5)]
    results = []
    for stem in selected:
        source = source_by_processed_name[stem]
        payload = run_inference(source, Path(artifact["weights"]), args.output_dir, artifact["imgsz"],
                                True, artifact["prediction_conf"], artifact.get("preprocess_mode", "baseline"), args.artifact)
        if not payload["detections"]:
            raise ValueError(f"No detections in selected validation example: {source}")
        detection = max(payload["detections"], key=lambda row: row["detector_confidence"])
        result = {"image": str(source), "class_name": detection["class_name"],
                  "detector_confidence": detection["detector_confidence"],
                  "logistic_fused_probability": detection["logistic_fused_probability"],
                  "temperature_calibrated_probability": detection["temperature_calibrated_probability"]}
        results.append(result)
        print(json.dumps(result), flush=True)
    if len({row["image"] for row in results}) != 5:
        raise ValueError("Five examples must use distinct images")
    if len({round(row["detector_confidence"], 4) for row in results}) < 3:
        raise ValueError("Example detector confidences do not vary across images")
    output = args.output_dir / "five_examples.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved five real examples to {output}", flush=True)


if __name__ == "__main__":
    main()
