"""Create survey/site-group train/val/test splits for ABYSS.

Usage:
    python abyss/ml/preprocessing/split_data.py

The split is performed by inferred survey/site group, never by individual image
shuffle. The script fails if any group appears in more than one split.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from common import SPLITS_ROOT, Sample, load_samples, samples_to_dicts, setup_logging, write_json


def assign_groups(samples: list[Sample]) -> dict[str, str]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_id].append(sample)
    group_class_counts = {group_id: class_id_counts(group_samples) for group_id, group_samples in grouped.items()}
    if len(grouped) < 3:
        raise ValueError(f"Need at least 3 groups for train/val/test split; found {len(grouped)}")

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    assignments: dict[str, str] = {}
    order = ["train", "val", "test"]

    families: dict[int, list[tuple[str, list[Sample]]]] = defaultdict(list)
    for group_id, group_samples in grouped.items():
        counts = group_class_counts[group_id]
        dominant_class = counts.most_common(1)[0][0] if counts else -1
        families[dominant_class].append((group_id, group_samples))

    for dominant_class, family_groups in sorted(families.items(), key=lambda item: item[0]):
        family_groups = sorted(family_groups, key=lambda item: (-len(item[1]), item[0]))
        family_total = sum(len(group_samples) for _, group_samples in family_groups)
        family_targets = {split: ratios[split] * family_total for split in ratios}
        family_counts = {"train": 0, "val": 0, "test": 0}

        for index, (group_id, group_samples) in enumerate(family_groups):
            # Seed each class/site family across splits when at least three groups exist.
            # This keeps rare classes inspectable in validation/test while preserving
            # the no-leakage guarantee because whole groups are still assigned together.
            if len(family_groups) >= 3 and index < 3:
                chosen = order[index]
            else:
                chosen = max(order, key=lambda split: family_targets[split] - family_counts[split])
            assignments[group_id] = chosen
            family_counts[chosen] += len(group_samples)
    return assignments


def assert_no_group_leakage(split_samples: dict[str, list[Sample]]) -> dict[str, int]:
    group_to_split: dict[str, str] = {}
    leakage: list[tuple[str, str, str]] = []
    for split, samples in split_samples.items():
        for sample in samples:
            previous = group_to_split.setdefault(sample.group_id, split)
            if previous != split:
                leakage.append((sample.group_id, previous, split))
    if leakage:
        raise AssertionError(f"Survey/site group leakage detected: {leakage[:10]}")
    return {split: len({sample.group_id for sample in samples}) for split, samples in split_samples.items()}


def class_id_counts(samples: list[Sample]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        for class_id in sample.class_ids:
            counts[class_id] += 1
    return counts


def class_counts(samples: list[Sample]) -> dict[str, int]:
    counts = class_id_counts(samples)
    return dict(sorted(counts.items()))


def main() -> None:
    logger = setup_logging("phase1_split_data")
    samples = load_samples()
    assignments = assign_groups(samples)
    split_samples = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_samples[assignments[sample.group_id]].append(sample)

    for split, split_list in split_samples.items():
        if not split_list:
            raise ValueError(f"{split} split is empty after group assignment")

    group_counts = assert_no_group_leakage(split_samples)
    summary = {
        "total_images": len(samples),
        "split_counts": {split: len(items) for split, items in split_samples.items()},
        "split_group_counts": group_counts,
        "split_class_counts_by_id": {split: class_counts(items) for split, items in split_samples.items()},
        "group_leakage_count": 0,
    }
    write_json(SPLITS_ROOT / "split_manifest.json", {split: samples_to_dicts(items) for split, items in split_samples.items()})
    write_json(SPLITS_ROOT / "split_summary.json", summary)
    logger.info("Split manifest written to %s", SPLITS_ROOT / "split_manifest.json")
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
