"""Select a raw-confidence threshold using validation predictions only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_detection_details import evaluate, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact, names, paths, labels, predictions = load_data(args.artifact, args.predictions_csv, split="val")
    if artifact["prediction_conf"] > 0.05:
        raise ValueError("Validation candidates do not cover 0.05")
    thresholds = [round(index / 100, 2) for index in range(5, 96, 5)]
    results = [evaluate(threshold, names, paths, labels, predictions) for threshold in thresholds]
    baseline = next(row for row in results if row["threshold"] == 0.25)
    winner = max(results, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "validation_threshold_sweep.csv"
    keys = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in results)
    summary = {"selection_split": "val", "selection_metric": "micro_f1", "checkpoint_hash": artifact["weights_sha256"],
               "validation_image_count": len(paths), "baseline_0.25": {key: baseline[key] for key in keys},
               "selected": {key: winner[key] for key in keys}, "sweep_csv": str(csv_path)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for row in results:
        print(f"{row['threshold']:.2f}: P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
