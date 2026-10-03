from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


# ------------------------- User configuration -------------------------
# IMPORTANT: replace MODEL_PATH with the checkpoint that you actually want to evaluate.
MODEL_PATH = (r"D:\fjx\ultralyticsPro0401-YOLOv8\rynew\train97\weights\best.pt"
              r"")
DATA_YAML = r"D:\fjx\ultralyticsPro0401-YOLOv8\M4SFWD Dataset New\data.yaml"

EVAL_SPLIT = "test"
EXPECTED_IMAGE_COUNT = 602

DEVICE = 0
BATCH = 1
IMG_SIZE = 640
HALF = True
AP_CONF = 0.001          # retain low-confidence predictions for AP calculation
NMS_IOU = 0.70
MAX_DET = 300

OPERATING_CONF = 0.25
MATCH_IOU = 0.50

OUTPUT_DIR = Path("runs/scale_class_evaluation")

RUN_STANDARD_VAL = False

PREDICT_ONE_IMAGE_AT_A_TIME = True
CUDA_CLEANUP_INTERVAL = 20
PROGRESS_INTERVAL = 25
# ---------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_AREA_MAX = 32.0**2
MEDIUM_AREA_MAX = 96.0**2


def cuda_enabled() -> bool:
    return torch.cuda.is_available() and str(DEVICE).lower() != "cpu"


def cleanup_cuda() -> None:
    """Release Python references and unused CUDA cache blocks."""
    gc.collect()
    if cuda_enabled():
        torch.cuda.empty_cache()


def resolve_dataset(data_yaml: str | Path) -> tuple[dict, Path]:
    yaml_path = Path(data_yaml).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_value = cfg.get("path", yaml_path.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return cfg, root


def resolve_existing_path(
    path_value: str | Path,
    root: Path,
    fallback_base: Path | None = None,
) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()

    candidate = (root / p).resolve()
    if candidate.exists():
        return candidate

    if fallback_base is not None:
        fallback = (fallback_base / p).resolve()
        if fallback.exists():
            return fallback

    return candidate


def collect_images_from_entry(entry: str | Path, root: Path) -> list[Path]:
    source = resolve_existing_path(entry, root)

    if source.is_dir():
        return sorted(
            p.resolve()
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source.resolve()]

    if source.is_file() and source.suffix.lower() == ".txt":
        images: list[Path] = []
        with source.open("r", encoding="utf-8-sig") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                image_path = resolve_existing_path(value, root, source.parent)
                if image_path.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(image_path.resolve())
        return sorted(images)

    raise FileNotFoundError(f"Cannot resolve dataset source: {entry} -> {source}")


def collect_evaluation_images(
    cfg: dict,
    root: Path,
    requested_split: str,
) -> tuple[list[Path], str]:
    split_name = requested_split
    split_entry = cfg.get(split_name)

    if split_entry is None:
        if requested_split == "test" and cfg.get("val") is not None:
            split_name = "val"
            split_entry = cfg["val"]
            print(
                "[WARNING] data.yaml has no 'test' entry; falling back to the 'val' entry. "
            )
        else:
            raise KeyError(
                f"The data.yaml file does not contain a '{requested_split}' entry."
            )

    entries = split_entry if isinstance(split_entry, list) else [split_entry]
    images: list[Path] = []
    for entry in entries:
        images.extend(collect_images_from_entry(entry, root))

    unique = sorted({p.resolve() for p in images})
    if not unique:
        raise RuntimeError(f"No images were found for split '{split_name}'.")
    return unique, split_name


def normalize_names(names_value) -> dict[int, str]:
    if isinstance(names_value, dict):
        return {int(k): str(v) for k, v in names_value.items()}
    if isinstance(names_value, list):
        return {i: str(v) for i, v in enumerate(names_value)}
    raise TypeError("The 'names' field in data.yaml must be a list or dictionary.")


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [i for i, value in enumerate(parts) if value.lower() == "images"]
    if not image_indexes:
        raise ValueError(
            f"Cannot infer label path for {image_path}. "
            "Expected a standard YOLO path containing an 'images' directory."
        )

    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Missing or empty label files are treated as negative images."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    labels = []
    with label_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}")
            cls_id = int(float(fields[0]))
            xc, yc, w, h = map(float, fields[1:5])
            labels.append((cls_id, xc, yc, w, h))
    return labels


def normalized_yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    eval_size: int,
) -> np.ndarray:
    """Convert normalized YOLO xywh to xyxy on a common eval_size canvas."""
    x1 = (xc - w / 2.0) * eval_size
    y1 = (yc - h / 2.0) * eval_size
    x2 = (xc + w / 2.0) * eval_size
    y2 = (yc + h / 2.0) * eval_size
    return np.array(
        [
            np.clip(x1, 0, eval_size),
            np.clip(y1, 0, eval_size),
            np.clip(x2, 0, eval_size),
            np.clip(y2, 0, eval_size),
        ],
        dtype=np.float32,
    )


def xyxy_to_coco_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def build_coco_ground_truth(
    image_paths: list[Path],
    names: dict[int, str],
    eval_size: int,
) -> tuple[dict, dict[int, dict], dict[str, int], set[int]]:
    images = []
    annotations = []
    gt_by_image: dict[int, dict] = {}
    image_id_by_path: dict[str, int] = {}
    negative_image_ids: set[int] = set()
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as im:
            original_width, original_height = im.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": eval_size,
                "height": eval_size,
                "original_width": original_width,
                "original_height": original_height,
            }
        )
        image_id_by_path[str(image_path.resolve()).lower()] = image_id

        label_path = image_to_label_path(image_path)
        labels = read_yolo_labels(label_path)
        gt_boxes = []
        gt_classes = []

        for cls_id, xc, yc, w, h in labels:
            if cls_id not in names:
                raise ValueError(f"Unknown class id {cls_id} in {label_path}")

            xyxy = normalized_yolo_to_xyxy(xc, yc, w, h, eval_size)
            bbox = xyxy_to_coco_xywh(xyxy)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            gt_boxes.append(xyxy)
            gt_classes.append(cls_id)

        if not gt_boxes:
            negative_image_ids.add(image_id)

        gt_by_image[image_id] = {
            "boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
            "classes": np.asarray(gt_classes, dtype=np.int64),
        }

    categories = [
        {"id": cls_id + 1, "name": class_name}
        for cls_id, class_name in sorted(names.items())
    ]
    coco_gt = {
        "info": {"description": "YOLO split converted for COCO-style evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco_gt, gt_by_image, image_id_by_path, negative_image_ids


def count_scale_instances(coco_gt: dict, names: dict[int, str]) -> list[dict]:
    """Count ground-truth instances under COCO-style 32^2 and 96^2 area thresholds."""
    counts: dict[str, dict[str, int]] = {
        "all": {"small": 0, "medium": 0, "large": 0},
        **{
            class_name: {"small": 0, "medium": 0, "large": 0}
            for class_name in names.values()
        },
    }
    category_name = {cls_id + 1: name for cls_id, name in names.items()}

    for annotation in coco_gt["annotations"]:
        area = float(annotation["area"])
        if area < SMALL_AREA_MAX:
            scale = "small"
        elif area < MEDIUM_AREA_MAX:
            scale = "medium"
        else:
            scale = "large"

        counts["all"][scale] += 1
        counts[category_name[int(annotation["category_id"])]][scale] += 1

    rows = []
    for scope, scope_counts in counts.items():
        rows.append(
            {
                "scope": scope,
                "small_instances": scope_counts["small"],
                "medium_instances": scope_counts["medium"],
                "large_instances": scope_counts["large"],
                "total_instances": sum(scope_counts.values()),
            }
        )
    return rows


def convert_prediction_to_eval_canvas(
    xyxy_original: np.ndarray,
    original_width: int,
    original_height: int,
    eval_size: int,
) -> np.ndarray:
    """Map a prediction from original-image coordinates to a common 640x640 canvas."""
    box = xyxy_original.astype(np.float32).copy()
    box[[0, 2]] *= eval_size / float(original_width)
    box[[1, 3]] *= eval_size / float(original_height)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, eval_size)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, eval_size)
    return box


def store_result_on_cpu(
    result,
    image_id: int,
    eval_size: int,
    coco_predictions: list[dict],
    pred_by_image: dict[int, dict],
) -> None:
    """Convert one Ultralytics Result to CPU-only NumPy/Python objects."""
    original_height, original_width = result.orig_shape

    if result.boxes is None or len(result.boxes) == 0:
        return

    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=True)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=True)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64, copy=True)

    eval_boxes = np.stack(
        [
            convert_prediction_to_eval_canvas(
                box,
                original_width,
                original_height,
                eval_size,
            )
            for box in xyxy
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    pred_by_image[image_id] = {
        "boxes": eval_boxes.copy(),
        "classes": classes.copy(),
        "scores": scores.copy(),
    }

    for box, score, cls_id in zip(eval_boxes, scores, classes):
        coco_predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(cls_id) + 1,
                "bbox": [float(v) for v in xyxy_to_coco_xywh(box)],
                "score": float(score),
            }
        )


def run_predictions(
    model: YOLO,
    image_paths: list[Path],
    image_id_by_path: dict[str, int],
    eval_size: int,
) -> tuple[list[dict], dict[int, dict]]:
    """Run low-memory inference and keep only CPU-side prediction data."""
    coco_predictions: list[dict] = []
    pred_by_image: dict[int, dict] = {
        image_id: {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "classes": np.empty((0,), dtype=np.int64),
            "scores": np.empty((0,), dtype=np.float32),
        }
        for image_id in image_id_by_path.values()
    }

    use_half = bool(HALF and cuda_enabled())
    print(
        f"Prediction mode: batch={BATCH}, half={use_half}, "
        f"one_image_at_a_time={PREDICT_ONE_IMAGE_AT_A_TIME}"
    )

    with torch.inference_mode():
        if PREDICT_ONE_IMAGE_AT_A_TIME:
            total = len(image_paths)
            for index, image_path in enumerate(image_paths, start=1):
                results = model.predict(
                    source=str(image_path),
                    imgsz=eval_size,
                    batch=1,
                    conf=AP_CONF,
                    iou=NMS_IOU,
                    max_det=MAX_DET,
                    agnostic_nms=False,
                    device=DEVICE,
                    half=use_half,
                    stream=False,
                    save=False,
                    verbose=False,
                )

                if len(results) != 1:
                    raise RuntimeError(
                        f"Expected one result for {image_path}, received {len(results)}."
                    )

                result = results[0]
                result_path = str(Path(result.path).resolve()).lower()
                if result_path not in image_id_by_path:
                    raise KeyError(
                        f"Prediction path was not found in evaluation list: {result.path}"
                    )

                image_id = image_id_by_path[result_path]
                store_result_on_cpu(
                    result,
                    image_id,
                    eval_size,
                    coco_predictions,
                    pred_by_image,
                )

                # Remove all per-image objects before continuing.
                del result
                del results

                if index % CUDA_CLEANUP_INTERVAL == 0:
                    cleanup_cuda()
                if index % PROGRESS_INTERVAL == 0 or index == total:
                    print(f"Predicted {index}/{total} images")
        else:
            results = model.predict(
                source=[str(p) for p in image_paths],
                imgsz=eval_size,
                batch=BATCH,
                conf=AP_CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                agnostic_nms=False,
                device=DEVICE,
                half=use_half,
                stream=True,
                save=False,
                verbose=False,
            )

            for index, result in enumerate(results, start=1):
                result_path = str(Path(result.path).resolve()).lower()
                if result_path not in image_id_by_path:
                    raise KeyError(
                        f"Prediction path was not found in evaluation list: {result.path}"
                    )

                image_id = image_id_by_path[result_path]
                store_result_on_cpu(
                    result,
                    image_id,
                    eval_size,
                    coco_predictions,
                    pred_by_image,
                )
                del result

                if index % CUDA_CLEANUP_INTERVAL == 0:
                    cleanup_cuda()
                if index % PROGRESS_INTERVAL == 0 or index == len(image_paths):
                    print(f"Predicted {index}/{len(image_paths)} images")

            del results

    cleanup_cuda()
    return coco_predictions, pred_by_image


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)

    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(
        0.0, inter_y2 - inter_y1
    )

    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-9)


def evaluate_class_metrics(
    gt_by_image: dict[int, dict],
    pred_by_image: dict[int, dict],
    names: dict[int, str],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    """Class-aware, confidence-sorted, one-to-one matching at a fixed operating point."""
    counts = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in names}

    for image_id, gt_record in gt_by_image.items():
        pred_record = pred_by_image[image_id]

        for cls_id in names:
            gt_boxes = gt_record["boxes"][gt_record["classes"] == cls_id]
            pred_mask = (
                (pred_record["classes"] == cls_id)
                & (pred_record["scores"] >= confidence_threshold)
            )
            pred_boxes = pred_record["boxes"][pred_mask]
            pred_scores = pred_record["scores"][pred_mask]

            if len(pred_scores):
                pred_boxes = pred_boxes[np.argsort(-pred_scores)]

            matched_gt = np.zeros(len(gt_boxes), dtype=bool)
            for pred_box in pred_boxes:
                available = np.where(~matched_gt)[0]
                if available.size == 0:
                    counts[cls_id]["fp"] += 1
                    continue

                ious = box_iou_one_to_many(pred_box, gt_boxes[available])
                best_local = int(np.argmax(ious)) if ious.size else -1
                if best_local >= 0 and ious[best_local] >= iou_threshold:
                    matched_gt[available[best_local]] = True
                    counts[cls_id]["tp"] += 1
                else:
                    counts[cls_id]["fp"] += 1

            counts[cls_id]["fn"] += int((~matched_gt).sum())

    rows = []
    for cls_id, class_name in sorted(names.items()):
        tp = counts[cls_id]["tp"]
        fp = counts[cls_id]["fp"]
        fn = counts[cls_id]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        fnr = fn / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "FNR": fnr,
            }
        )
    return rows


def mean_coco_precision(
    evaluator: COCOeval,
    area_label: str,
    iou_threshold: float | None = None,
) -> float:
    """Extract mean AP for a specific area range and optional IoU threshold."""
    precision = evaluator.eval.get("precision")
    if precision is None:
        return float("nan")

    area_index = evaluator.params.areaRngLbl.index(area_label)
    max_det_index = evaluator.params.maxDets.index(100)
    values = precision[:, :, :, area_index, max_det_index]

    if iou_threshold is not None:
        threshold_indices = np.where(
            np.isclose(evaluator.params.iouThrs, iou_threshold)
        )[0]
        if threshold_indices.size == 0:
            return float("nan")
        values = values[threshold_indices]

    valid = values[values > -1]
    return float(valid.mean()) if valid.size else float("nan")


def run_coco_evaluation(
    gt_json_path: Path,
    pred_json_path: Path,
    names: dict[int, str],
) -> list[dict]:
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    rows = []

    scopes: list[tuple[str, list[int] | None]] = [("all", None)]
    scopes.extend(
        (class_name, [cls_id + 1])
        for cls_id, class_name in sorted(names.items())
    )

    for scope_name, category_ids in scopes:
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if category_ids is not None:
            evaluator.params.catIds = category_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        rows.append(
            {
                "scope": scope_name,
                "AP50-S": mean_coco_precision(evaluator, "small", 0.50),
                "AP50-M": mean_coco_precision(evaluator, "medium", 0.50),
                "AP50-L": mean_coco_precision(evaluator, "large", 0.50),
                "AP50-95-S": mean_coco_precision(evaluator, "small"),
                "AP50-95-M": mean_coco_precision(evaluator, "medium"),
                "AP50-95-L": mean_coco_precision(evaluator, "large"),
            }
        )
    return rows


def evaluate_negative_images(
    negative_image_ids: set[int],
    pred_by_image: dict[int, dict],
    names: dict[int, str],
    confidence_threshold: float,
) -> list[dict]:
    total_negative = len(negative_image_ids)
    if total_negative == 0:
        return [
            {
                "negative_images": 0,
                "false_alarm_images": 0,
                "false_alarm_image_rate": 0.0,
                "false_positives": 0,
                "FPPI": 0.0,
            }
        ]

    false_alarm_images = 0
    total_false_positives = 0
    class_false_positives = defaultdict(int)

    for image_id in negative_image_ids:
        record = pred_by_image[image_id]
        keep = record["scores"] >= confidence_threshold
        kept_classes = record["classes"][keep]
        count = int(keep.sum())
        total_false_positives += count
        false_alarm_images += int(count > 0)
        for cls_id in kept_classes:
            class_false_positives[int(cls_id)] += 1

    row = {
        "confidence_threshold": confidence_threshold,
        "negative_images": total_negative,
        "false_alarm_images": false_alarm_images,
        "false_alarm_image_rate": false_alarm_images / total_negative,
        "false_positives": total_false_positives,
        "FPPI": total_false_positives / total_negative,
    }
    for cls_id, class_name in sorted(names.items()):
        row[f"{class_name}_false_positives"] = class_false_positives[cls_id]
    return [row]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(
    title: str,
    rows: list[dict],
    percentage_fields: Iterable[str] = (),
) -> None:
    percentage_fields = set(percentage_fields)
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    for row in rows:
        printable = {}
        for key, value in row.items():
            if isinstance(value, float):
                if np.isnan(value):
                    printable[key] = "N/A"
                elif key in percentage_fields:
                    printable[key] = f"{value * 100:.2f}%"
                else:
                    printable[key] = f"{value:.6f}"
            else:
                printable[key] = value
        print(printable)


def run_optional_standard_validation(model: YOLO, split_name: str) -> None:
    """Optional standard validation pass; disabled by default to prevent duplicate GPU usage."""
    if not RUN_STANDARD_VAL:
        return

    use_half = bool(HALF and cuda_enabled())
    print("\nRunning optional standard Ultralytics validation...")
    val_results = model.val(
        data=DATA_YAML,
        split=split_name,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=0,
        device=DEVICE,
        half=use_half,
        conf=AP_CONF,
        iou=NMS_IOU,
        plots=False,
        verbose=True,
    )

    del val_results
    # Different Ultralytics revisions may retain predictor/validator objects.
    for attribute_name in ("predictor", "validator"):
        if hasattr(model, attribute_name):
            try:
                setattr(model, attribute_name, None)
            except Exception:
                pass
    cleanup_cuda()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg, dataset_root = resolve_dataset(DATA_YAML)
    names = normalize_names(cfg["names"])
    image_paths, resolved_split = collect_evaluation_images(
        cfg,
        dataset_root,
        EVAL_SPLIT,
    )

    print(f"Evaluation split: {resolved_split}")
    print(f"Evaluation images: {len(image_paths)}")
    print(f"Classes: {names}")
    print(f"Model path: {MODEL_PATH}")

    if EXPECTED_IMAGE_COUNT is not None and len(image_paths) != EXPECTED_IMAGE_COUNT:
        print(
            f"[WARNING] Expected {EXPECTED_IMAGE_COUNT} images for the manuscript test set, "
            f"but found {len(image_paths)}. Check EVAL_SPLIT and data.yaml before using the "
            "results in the paper."
        )

    if cuda_enabled():
        torch.cuda.set_device(int(DEVICE))
        cleanup_cuda()
        properties = torch.cuda.get_device_properties(int(DEVICE))
        print(
            f"CUDA device: {properties.name}, "
            f"VRAM={properties.total_memory / 1024**3:.2f} GB"
        )

    model = YOLO(MODEL_PATH)
    run_optional_standard_validation(model, resolved_split)

    coco_gt, gt_by_image, image_id_by_path, negative_image_ids = build_coco_ground_truth(
        image_paths,
        names,
        IMG_SIZE,
    )
    print(f"Ground-truth instances: {len(coco_gt['annotations'])}")
    print(f"Negative/background images: {len(negative_image_ids)}")

    scale_count_rows = count_scale_instances(coco_gt, names)

    coco_predictions, pred_by_image = run_predictions(
        model,
        image_paths,
        image_id_by_path,
        IMG_SIZE,
    )
    print(f"Retained predictions for COCO evaluation: {len(coco_predictions)}")

    if not coco_predictions:
        raise RuntimeError(
            "No predictions were retained. Check MODEL_PATH, AP_CONF, and model output."
        )

    gt_json_path = OUTPUT_DIR / "ground_truth_coco.json"
    pred_json_path = OUTPUT_DIR / "predictions_coco.json"
    gt_json_path.write_text(
        json.dumps(coco_gt, ensure_ascii=False),
        encoding="utf-8",
    )
    pred_json_path.write_text(
        json.dumps(coco_predictions, ensure_ascii=False),
        encoding="utf-8",
    )

    scale_rows = run_coco_evaluation(gt_json_path, pred_json_path, names)
    class_rows = evaluate_class_metrics(
        gt_by_image,
        pred_by_image,
        names,
        OPERATING_CONF,
        MATCH_IOU,
    )
    negative_rows = evaluate_negative_images(
        negative_image_ids,
        pred_by_image,
        names,
        OPERATING_CONF,
    )

    write_csv(OUTPUT_DIR / "scale_ap_metrics.csv", scale_rows)

    print_table("Scale instance counts", scale_count_rows)
    print_table(
        "COCO scale AP",
        scale_rows,
        {
            "AP50-S",
            "AP50-M",
            "AP50-L",
            "AP50-95-S",
            "AP50-95-M",
            "AP50-95-L",
        },
    )
  
if __name__ == "__main__":
    main()
