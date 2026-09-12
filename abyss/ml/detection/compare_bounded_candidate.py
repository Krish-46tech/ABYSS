"""Compare one fully evaluated candidate to the fixed baseline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_metrics(path: Path, split: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "evaluation_split" not in data and split == "test":
        data["evaluation_split"] = "test"
    if "image_count" not in data and split == "test" and "test_image_count" in data:
        data["image_count"] = data["test_image_count"]
    if data.get("evaluation_split", "test") != split:
        raise ValueError(f"Expected {split} metrics in {path}")
    for key in ("weights_sha256", "map50", "map50_95", "per_class"):
        if key not in data:
            raise ValueError(f"Missing {key} in {path}")
    return data


def compare(reference: dict, candidate: dict) -> dict:
    if reference["weights_sha256"] == candidate["weights_sha256"]:
        raise ValueError("Candidate and reference checkpoint hashes are identical")
    if (reference["imgsz"] != candidate["imgsz"] or reference["image_count"] != candidate["image_count"]
            or Path(reference["data_yaml"]).resolve() != Path(candidate["data_yaml"]).resolve()
            or set(reference["per_class"]) != set(candidate["per_class"])):
        raise ValueError("Candidate and baseline evaluation configurations differ")
    classes = {}
    for name in reference["per_class"]:
        classes[name] = {metric: {"baseline": reference["per_class"][name][metric],
                                  "candidate": candidate["per_class"][name][metric],
                                  "delta": candidate["per_class"][name][metric] - reference["per_class"][name][metric]}
                         for metric in ("ap50", "ap50_95")}
    overall = {metric: {"baseline": reference[metric], "candidate": candidate[metric],
                        "delta": candidate[metric] - reference[metric]} for metric in ("map50", "map50_95")}
    return {"baseline_sha256": reference["weights_sha256"], "candidate_sha256": candidate["weights_sha256"],
            "evaluation_split": candidate["evaluation_split"], "image_count": candidate["image_count"],
            "per_class": classes, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-val", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--candidate-val", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    comparisons = {
        "validation": compare(read_metrics(args.baseline_val, "val"), read_metrics(args.candidate_val, "val")),
        "test": compare(read_metrics(args.baseline_test, "test"), read_metrics(args.candidate_test, "test")),
    }
    if comparisons["validation"]["candidate_sha256"] != comparisons["test"]["candidate_sha256"]:
        raise ValueError("Validation and test results used different candidate checkpoints")
    if comparisons["validation"]["baseline_sha256"] != comparisons["test"]["baseline_sha256"]:
        raise ValueError("Validation and test references used different baseline checkpoints")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    for split, result in comparisons.items():
        print(f"{split.upper()} ({result['image_count']} images):")
        for name, metrics in result["per_class"].items():
            print(f"  {name}: AP50 {metrics['ap50']['baseline']:.6f} -> {metrics['ap50']['candidate']:.6f} ({metrics['ap50']['delta']:+.6f}); "
                  f"AP50:95 {metrics['ap50_95']['baseline']:.6f} -> {metrics['ap50_95']['candidate']:.6f} ({metrics['ap50_95']['delta']:+.6f})")
        print(f"  Overall: mAP50 {result['overall']['map50']['baseline']:.6f} -> {result['overall']['map50']['candidate']:.6f} "
              f"({result['overall']['map50']['delta']:+.6f}); mAP50:95 {result['overall']['map50_95']['baseline']:.6f} "
              f"-> {result['overall']['map50_95']['candidate']:.6f} ({result['overall']['map50_95']['delta']:+.6f})")
    print(f"Saved comparison: {args.output_json}")


if __name__ == "__main__":
    main()
