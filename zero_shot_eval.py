#!/usr/bin/env python3
"""
Single-class zero-shot evaluation for a two-class M4SFWD YOLO model.

Source model classes:
    0 = fire
    1 = smoke

Target datasets:
    SMOKE_dataset -> evaluate source class 1 (smoke)
    FLAME_dataset -> evaluate source class 0 (fire)

The target datasets are assumed to use standard YOLO-format labels with a
single target class encoded as class 0.

Example:
    python zero_shot_eval.py \
        --weights weights/cnr_seed0_best.pt \
        --data path/to/smoke_data.yaml \
        --target smoke \
        --seed 0 \
        --device 0

    python zero_shot_eval.py \
        --weights weights/cnr_seed0_best.pt \
        --data path/to/flame_data.yaml \
        --target flame \
        --seed 0 \
        --device 0

Notes:
- No training or fine-tuning is performed.
- The script filters the two-class model output to the corresponding source
  class and evaluates it as a single-class detector.
- It uses the dataset's `test` split from the supplied YAML file.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="M4SFWD-trained checkpoint")
    p.add_argument("--data", required=True, help="Target dataset YAML")
    p.add_argument("--target", required=True, choices=["smoke", "flame"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Low confidence threshold used for AP evaluation")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--output", default=None,
                   help="Optional output CSV path")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(data_yaml: Path, split: str) -> list[Path]:
    data = load_yaml(data_yaml)
    if split not in data or data[split] is None:
        raise KeyError(f"`{split}` is not defined in {data_yaml}")

    root = Path(data.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    entries = data[split]
    if isinstance(entries, str):
        entries = [entries]

    images: list[Path] = []
    for entry in entries:
        p = Path(entry)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()

        if p.is_dir():
            images.extend(
                sorted(x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
            )
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                q = Path(line)
                if not q.is_absolute():
                    q = (p.parent / q).resolve()
                images.append(q)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
        else:
            raise FileNotFoundError(f"Cannot resolve split entry: {p}")

    if not images:
        raise RuntimeError(f"No images found for split `{split}` in {data_yaml}")
    return images


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Replace the last directory named "images" by "labels" if present.
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    # Fallback: sibling labels directory.
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [x1, y1, x2, y2]


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Target datasets are single-class. All class-0 annotations are treated
    as the single evaluation class.
    """
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        vals = line.strip().split()
        if len(vals) < 5:
            continue
        cls = int(float(vals[0]))
        if cls != 0:
            # Single-class target datasets are expected to use class 0.
            continue
        xc, yc, w, h = map(float, vals[1:5])
        boxes.append(xywhn_to_xyxy(xc, yc, w, h, img_w, img_h))

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-16
    return inter / union


def match_predictions(
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresholds: np.ndarray,
) -> np.ndarray:
    """
    Returns [num_predictions, num_iou_thresholds] boolean correctness matrix.
    One-to-one matching is performed independently at each IoU threshold.
    """
    n_pred = len(pred_boxes)
    correct = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
    if n_pred == 0 or len(gt_boxes) == 0:
        return correct

    ious = box_iou_np(pred_boxes, gt_boxes)
    order = np.argsort(-pred_conf)

    for t_idx, thr in enumerate(iou_thresholds):
        matched_gt = set()
        for p_idx in order:
            gt_order = np.argsort(-ious[p_idx])
            for g_idx in gt_order:
                if ious[p_idx, g_idx] < thr:
                    break
                if int(g_idx) not in matched_gt:
                    correct[p_idx, t_idx] = True
                    matched_gt.add(int(g_idx))
                    break
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    101-point interpolated AP, matching the standard YOLO/COCO-style integral.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate(all_conf, all_correct, n_gt):
    if n_gt == 0:
        raise RuntimeError("No ground-truth objects were found.")

    if len(all_conf) == 0:
        return np.zeros(len(IOU_THRESHOLDS), dtype=float)

    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct, axis=0)

    order = np.argsort(-conf)
    conf = conf[order]
    correct = correct[order]

    aps = []
    for t_idx in range(correct.shape[1]):
        tp = correct[:, t_idx].astype(np.float64)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / (n_gt + 1e-16)
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        aps.append(compute_ap(recall, precision))

    return np.asarray(aps, dtype=float)


def main():
    args = parse_args()
    set_seed(args.seed)

    source_class = 1 if args.target == "smoke" else 0
    source_name = "smoke" if source_class == 1 else "fire"

    data_yaml = Path(args.data).resolve()
    image_paths = resolve_split(data_yaml, args.split)

    model = YOLO(args.weights)

    all_conf = []
    all_correct = []
    n_gt = 0

    for image_path in image_paths:
        with Image.open(image_path) as im:
            img_w, img_h = im.size

        gt_boxes = load_ground_truth(
            image_to_label_path(image_path), img_w, img_h
        )
        n_gt += len(gt_boxes)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_conf = np.empty((0,), dtype=np.float32)
        else:
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            keep = cls == source_class
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy()[keep].astype(np.float32)
            pred_conf = result.boxes.conf.detach().cpu().numpy()[keep].astype(np.float32)

        correct = match_predictions(
            pred_boxes, pred_conf, gt_boxes, IOU_THRESHOLDS
        )

        all_conf.append(pred_conf)
        all_correct.append(correct)

    aps = evaluate(all_conf, all_correct, n_gt)
    map50 = aps[0] * 100.0
    map50_95 = aps.mean() * 100.0

    print(f"Target dataset : {args.target}")
    print(f"Source class   : {source_class} ({source_name})")
    print(f"Seed           : {args.seed}")
    print(f"Checkpoint     : {args.weights}")
    print(f"Images         : {len(image_paths)}")
    print(f"Ground truths  : {n_gt}")
    print(f"mAP50          : {map50:.2f}%")
    print(f"mAP50-95       : {map50_95:.2f}%")

    output = args.output
    if output is None:
        output = f"{args.target}_zero_shot_eval.csv"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target", "seed", "checkpoint", "source_class_id",
            "source_class_name", "images", "ground_truths",
            "map50", "map50_95"
        ])
        writer.writerow([
            args.target, args.seed, str(args.weights), source_class,
            source_name, len(image_paths), n_gt,
            f"{map50:.4f}", f"{map50_95:.4f}"
        ])

    print(f"Saved          : {out_path}")


if __name__ == "__main__":
    main()
