# Zero-shot label mapping

## Source model

The zero-shot model is `cnr_seed0_best.pt`, trained only on the M4SFWD training set with seed 0.

M4SFWD uses two classes:

- class 0: `fire`
- class 1: `smoke`

No fine-tuning on SMOKE_dataset or FLAME_dataset is performed in the zero-shot setting.

## SMOKE_dataset

SMOKE_dataset is a single-class dataset containing only smoke annotations.

For zero-shot evaluation:

- M4SFWD class 1 (`smoke`) is evaluated against the single smoke class in SMOKE_dataset.
- Predictions from M4SFWD class 0 (`fire`) are excluded from metric computation.
- The target dataset's single class is treated as the evaluation class.

Reported result:

- mAP50: 78.3%
- mAP50-95: 46.5%

## FLAME_dataset

FLAME_dataset is a single-class dataset containing only flame/fire annotations.

For zero-shot evaluation:

- M4SFWD class 0 (`fire`) is evaluated against the single flame/fire class in FLAME_dataset.
- Predictions from M4SFWD class 1 (`smoke`) are excluded from metric computation.
- The target dataset's single class is treated as the evaluation class.

Reported result:

- mAP50: 82.7%
- mAP50-95: 49.8%

## Evaluation note

The zero-shot results are single-run results using seed 0 and are reported as preliminary cross-domain transfer evidence rather than statistically established generalization.
