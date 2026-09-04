"""Run ABYSS YOLO inference on one image and save JSON plus annotation.

Usage:
    python abyss/ml/detection/infer.py --image /path/to/image.jpg \
      --weights abyss/ml/detection/weights/baseline_v1.pt

The JSON output contains real detector boxes and confidence values returned by
the trained model. Phase 3 confidence fields are included when
``--include-confidence-fusion`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PROJECT_ROOT / "abyss" / "ml" / "shadow_confidence"))
sys.path.append(str(PROJECT_ROOT / "abyss" / "ml" / "preprocessing"))

from preprocess import denoise_and_normalize, letterbox  # type: ignore  # noqa: E402
from preprocess_enhanced import enhanced_sonar_preprocess, letterbox_enhanced  # type: ignore  # noqa: E402
from utils import ABYSS_ROOT, CLASS_NAMES, INFERENCE_ROOT, fail_if_missing, setup_logging, write_json  # noqa: E402


def draw_detections(image, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        class_name = detection["class_name"]
        confidence = detection["detector_confidence"]
        color = (0, 220, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def run_inference(
    image_path: Path,
    weights_path: Path,
    output_dir: Path,
    imgsz: int,
    include_confidence_fusion: bool,
    conf: float,
    preprocess_mode: str = "baseline",
) -> dict:
    fail_if_missing(image_path, "input image")
    fail_if_missing(weights_path, "trained model weights")
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"OpenCV could not read input image: {image_path}")

    # Use the matching deterministic preprocessing for the selected model.
    # This keeps improved-model inference aligned with its enhanced training
    # distribution instead of mixing pipelines silently.
    if preprocess_mode == "baseline":
        processed, _, _, _ = letterbox(denoise_and_normalize(original), imgsz)
        quality_image, _, _, _ = letterbox(original, imgsz)
    elif preprocess_mode == "enhanced":
        processed, _, _, _ = letterbox_enhanced(enhanced_sonar_preprocess(original), imgsz)
        quality_image, _, _, _ = letterbox_enhanced(original, imgsz)
    else:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
    model = YOLO(str(weights_path))
    start = time.perf_counter()
    results = model.predict(processed, imgsz=imgsz, conf=conf, verbose=False)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if not results:
        raise RuntimeError("Ultralytics returned no prediction result objects")

    detections: list[dict] = []
    result = results[0]
    boxes = result.boxes
    if boxes is not None:
        for index, box in enumerate(boxes):
            class_id = int(box.cls.item())
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"Model returned unknown class id {class_id}")
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            detector_confidence = float(box.conf.item())
            detection = {
                "detection_id": index,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "bbox_xyxy": xyxy,
                "detector_confidence": detector_confidence,
            }
            if include_confidence_fusion:
                from confidence_fusion import fuse_detection_confidence  # type: ignore
                from image_quality import compute_image_quality_score  # type: ignore
                from shadow_features import compute_shadow_features  # type: ignore

                quality = compute_image_quality_score(quality_image)
                shadow = compute_shadow_features(processed, xyxy)
                fusion = fuse_detection_confidence(detector_confidence, quality["image_quality_score"], shadow["shadow_consistency_score"])
                detection.update(
                    {
                        "image_quality_score": quality["image_quality_score"],
                        "image_quality_details": quality,
                        "shadow_consistency_score": shadow["shadow_consistency_score"],
                        "shadow_features": shadow,
                        "composite_confidence": fusion["composite_confidence"],
                        "confidence_fusion_weights": fusion["weights"],
                    }
                )
            detections.append(detection)

    annotated = processed.copy()
    draw_detections(annotated, detections)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / f"{image_path.stem}_annotated.jpg"
    output_json_path = output_dir / f"{image_path.stem}_detections.json"
    if not cv2.imwrite(str(output_image_path), annotated):
        raise IOError(f"Failed to write annotated image: {output_image_path}")

    payload = {
        "image_path": str(image_path),
        "weights_path": str(weights_path),
        "preprocessed_shape": list(processed.shape),
        "processing_latency_ms": latency_ms,
        "preprocess_mode": preprocess_mode,
        "confidence_threshold": conf,
        "detections": detections,
        "annotated_image_path": str(output_image_path),
        "json_output_path": str(output_json_path),
    }
    write_json(output_json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABYSS detector inference on a single sonar image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=ABYSS_ROOT / "ml" / "detection" / "weights" / "baseline_v1.pt")
    parser.add_argument("--output-dir", type=Path, default=INFERENCE_ROOT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold passed to Ultralytics.")
    parser.add_argument("--preprocess-mode", choices=["baseline", "enhanced"], default="baseline")
    parser.add_argument("--include-confidence-fusion", action="store_true")
    args = parser.parse_args()
    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be between 0 and 1")

    logger = setup_logging("phase2_infer")
    payload = run_inference(
        args.image,
        args.weights,
        args.output_dir,
        args.imgsz,
        args.include_confidence_fusion,
        args.conf,
        args.preprocess_mode,
    )
    logger.info("Inference complete for %s", args.image)
    logger.info("Annotated image: %s", payload["annotated_image_path"])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
