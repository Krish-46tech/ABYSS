"""Render project metric charts and workflow diagrams from saved real evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "abyss/logs"
DEFAULT_OUTPUT = ROOT / "docs/figures"
CANDIDATES = (
    ("Baseline", "phase4_corrected_map.json"),
    ("Class weight", "bounded_shipwreck_weight_test.json"),
    ("Wreck augment", "bounded_shipwreck_aug_test.json"),
    ("Box weight", "bounded_box_weight_test.json"),
    ("Close mosaic", "bounded_close_mosaic_test.json"),
)
TEAL = "#0b7f80"
CORAL = "#d65c48"
INK = "#24323a"
MUTED = "#677983"
LIGHT = "#e8f0ef"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def metric_charts(output: Path) -> None:
    rows = [(name, read_json(LOGS / file)) for name, file in CANDIDATES]
    baseline_hash = rows[0][1]["weights_sha256"]
    if len({data["weights_sha256"] for _, data in rows}) != len(rows):
        raise ValueError("Candidate checkpoints are not distinct")
    if baseline_hash != "baef8d00dbd8d764e61c59bf411c95da67936834552fb80428771cf69a7d6a5d":
        raise ValueError("Fixed published baseline checkpoint changed")
    if any(data.get("evaluation_split", "test") != "test" or data.get("image_count", data.get("test_image_count")) != 129 for _, data in rows):
        raise ValueError("The comparison must use the original 129-image test split")
    names = [name for name, _ in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10.8, 5.1), layout="constrained")
    width = 0.34
    for offset, key, title, color in ((-width / 2, "map50", "mAP@0.5", TEAL),
                                       (width / 2, "map50_95", "mAP@0.5:0.95", CORAL)):
        values = [data[key] for _, data in rows]
        bars = ax.bar(x + offset, values, width, label=title, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("Mean average precision")
    ax.set_title("Held-out test: baseline versus independent retrains", loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, output / "experiment_overall_map.png")

    classes = ("Plane", "Ship", "Shipwreck")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), sharey=True, layout="constrained")
    for ax, class_name in zip(axes, classes):
        values = [data["per_class"][class_name]["ap50"] for _, data in rows]
        colors = [TEAL if i == 0 else CORAL if i == 4 else MUTED for i in range(len(rows))]
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.65)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(class_name, color=INK, weight="bold")
        ax.set_xlim(0, 1.11)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out test AP@0.5 by class", x=0.01, ha="left", color=INK, weight="bold")
    save(fig, output / "experiment_class_ap50.png")

    reliability = read_json(LOGS / "phase4_corrected_reliability/summary.json")
    if reliability["evaluation_split"] != "test" or reliability["prediction_count"] != 217:
        raise ValueError("Expected the audited held-out reliability run")
    ece = reliability["ece"]
    labels = ("Raw detector", "Logistic fused", "Temperature final")
    keys = ("raw_detector", "logistic_fused", "temperature_calibrated")
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    bars = ax.bar(labels, [ece[key] for key in keys], color=[MUTED, TEAL, CORAL], width=0.62)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_ylim(0, 0.10)
    ax.set_ylabel("10-bin expected calibration error (lower is better)")
    ax.set_title("Deployed baseline: held-out calibration", loc="left", color=INK, weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, output / "baseline_ece_comparison.png")


def node(ax: plt.Axes, x: float, y: float, width: float, height: float,
         title: str, detail: str, fill: str = LIGHT) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.025",
                                linewidth=1, edgecolor="#b4c8c5", facecolor=fill))
    ax.text(x + 0.04, y + height * 0.69, title, ha="left", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(x + 0.04, y + height * 0.31, detail, ha="left", va="center", fontsize=8.5, color=MUTED)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.5, color=TEAL))


def flowcharts(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 3.4), layout="constrained")
    ax.set(xlim=(0, 13.4), ylim=(0, 3.2))
    ax.axis("off")
    stages = (
        ("Data", "Train / val / test split"),
        ("Detector", "YOLOv8, baseline 640 px"),
        ("Features", "Shadow + image quality"),
        ("Calibration", "Logistic + temperature"),
        ("Evaluation", "AP, FROC, ECE, errors"),
    )
    for index, (title, detail) in enumerate(stages):
        x = 0.13 + index * 2.68
        node(ax, x, 1.25, 2.35, 1.05, title, detail, "#e7f2f0" if index != 4 else "#fcebe6")
        if index:
            arrow(ax, (x - 0.27, 1.78), (x - 0.04, 1.78))
    ax.text(0.13, 2.75, "ABYSS Phase 0-4 workflow", color=INK, fontsize=15, weight="bold")
    ax.text(0.13, 0.55, "Detector training uses train; calibration fitting uses validation; final metrics use held-out test.",
            color=MUTED, fontsize=10)
    save(fig, output / "project_pipeline_flow.png")

    fig, ax = plt.subplots(figsize=(11.4, 5.4), layout="constrained")
    ax.set(xlim=(0, 11.4), ylim=(0, 5.4))
    ax.axis("off")
    ax.text(0.25, 5.0, "Bounded retrain decision flow", color=INK, fontsize=15, weight="bold")
    node(ax, 0.25, 3.6, 2.45, 1.0, "Fixed baseline", "640 px + v4, unchanged")
    node(ax, 3.25, 3.6, 2.45, 1.0, "Independent retrain", "One change per candidate")
    node(ax, 6.25, 3.6, 2.45, 1.0, "Validation selection", "Compare all classes + mAP")
    node(ax, 9.15, 3.6, 2.0, 1.0, "Held-out test", "Final tradeoff check")
    for start, end in (((2.72, 4.1), (3.19, 4.1)), ((5.72, 4.1), (6.19, 4.1)),
                       ((8.72, 4.1), (9.09, 4.1))):
        arrow(ax, start, end)
    node(ax, 0.4, 1.0, 4.35, 1.25, "Rejected", "Class weight / augmentation / box loss", "#fcebe6")
    node(ax, 6.55, 1.0, 4.45, 1.25, "Experimental, not deployed", "Close mosaic: small gain, Plane tradeoff", "#e7f2f0")
    arrow(ax, (10.15, 3.55), (8.85, 2.3))
    arrow(ax, (7.05, 3.55), (2.75, 2.3))
    save(fig, output / "experiment_decision_flow.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_charts(args.output_dir)
    flowcharts(args.output_dir)


if __name__ == "__main__":
    main()
