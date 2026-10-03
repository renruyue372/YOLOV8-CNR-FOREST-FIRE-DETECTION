# YOLOv8-CNR

Official reproducibility materials for the manuscript on lightweight forest fire/smoke detection based on YOLOv8-CNR.

## 1. Environment

Recommended environment:

- Python 3.8
- PyTorch 2.0.1
- CUDA 11.7
- NVIDIA GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

This repository contains a modified local Ultralytics implementation. Please use the source code included in this repository rather than replacing it with a different Ultralytics version.

## 2. M4SFWD split

The reported M4SFWD experiments use the exact camera-run-level split below:

- Train: 17 runs, 2803 images, Scenes 1–2
- Validation: 5 runs, 569 images, Scene 3
- Test: 6 runs, 602 images, Scene 4

The three subsets are run-level exclusive. Scene 4 is completely held out from training and validation.

Split files:

```text
splits/
├── train.txt
├── val.txt
├── test.txt
├── data.yaml
└── split_manifest.txt
```

## 3. Models

Main configuration:

```text
YOLOv8-CNR.yaml
```

Main custom components:

```text
C3_ConvNeXtv2.py
RepNCSPELAN4.py
SIoU.py
```

The complete model combines:

- A: C3_ConvNeXtv2
- B: RepNCSPELAN4
- C: SIoU

## 4. Training and evaluation

Training:

```bash
python train_rynew.py
```

Standard validation:

```bash
python val_ry.py
```

Scale- and class-wise evaluation:

```bash
python evaluate_scale_class_metrics.py
```

## 5. Released checkpoints

```text
weights/
├── v8_seed0_best.pt
├── v8_seed42_best.pt
├── v8_seed123_best.pt
├── cnr_seed0_best.pt
├── cnr_seed42_best.pt
└── cnr_seed123_best.pt
```

Seeds used in the manuscript: `0`, `42`, and `123`.

## 6. Results corresponding to the manuscript

### Table 2: Ablation study

```text
results/table2_ablation/table2_per_seed.csv
```

This file contains the three-seed results for:

```text
Baseline
A
B
C
A+B
A+C
B+C
A+B+C
```

### Table 4: Class-wise evaluation

The released baseline and YOLOv8-CNR per-seed evaluation outputs contain the class-wise Fire/Smoke results.

### Table 5: Real-world retraining and zero-shot transfer

```text
results/table5_cross_dataset/
├── table5_retrained_per_seed.csv
├── smoke_zero_shot.csv
├── flame_zero_shot.csv
└── label_mapping.md
```

The zero-shot experiments use:

```text
weights/cnr_seed0_best.pt
```

This checkpoint was trained only on M4SFWD with seed 0 and was directly evaluated on the real-world datasets without fine-tuning.

Zero-shot class mapping:

- SMOKE_dataset: M4SFWD class 1 (`smoke`)
- FLAME_dataset: M4SFWD class 0 (`fire`)

Zero-shot evaluation:

```bash
python zero_shot_eval.py --weights weights/cnr_seed0_best.pt --data <target_data.yaml> --target smoke --seed 0
python zero_shot_eval.py --weights weights/cnr_seed0_best.pt --data <target_data.yaml> --target flame --seed 0
```

### Table 6: Scale-stratified evaluation

```text
results/table6_scale/
├── v8_seed0_scale_ap_metrics.csv
├── v8_seed42_scale_ap_metrics.csv
├── v8_seed123_scale_ap_metrics.csv
├── cnr_seed0_scale_ap_metrics.csv
├── cnr_seed42_scale_ap_metrics.csv
└── cnr_seed123_scale_ap_metrics.csv
```

## 7. Statistical test

The paired comparison between the tuned YOLOv8n baseline and YOLOv8-CNR uses the three corresponding mAP50-95 values:

```text
Baseline: 37.62, 36.42, 36.39
CNR:      39.96, 39.14, 39.13
```

Reported paired t-test:

```text
t = 19.98
df = 2
p = 0.0025
```

## 8. Notes

- All reported M4SFWD results use the split released in `splits/`.
- The M4SFWD test set is a Scene-4-held-out test set.
- Zero-shot results are single-run results and are treated as preliminary cross-domain transfer evidence.
- Please use the released model configuration, source code, checkpoints, and split files together when reproducing the results.
