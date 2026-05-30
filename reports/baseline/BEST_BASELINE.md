# Best Baseline

## Model

- Architecture: `resnet18`
- Pretrained weights: `false`
- Input size: `224x224`
- Classes: `edible`, `non_edible`
- Positive / safety-critical class: `non_edible`
- Model artifact: `models/baseline_resnet18.pt`

## Training Setup

- Data directory: `data/processed`
- Train images: `2256`
- Validation images: `282`
- Test images: `282`
- Epochs: `20`
- Best epoch by validation F1: `19`
- Batch size: `32`
- Learning rate: `0.0003`
- Weight decay: `0.0001`
- Device: `cuda`
- Seed: `42`

Run command:

```powershell
.\.venv\Scripts\python.exe src\training\train_baseline.py
```

## Validation Metrics

| Metric | Value |
| --- | ---: |
| Loss | 0.3711 |
| Accuracy | 0.8652 |
| Precision `non_edible` | 0.8615 |
| Recall `non_edible` | 0.8485 |
| F1 `non_edible` | 0.8550 |
| ROC-AUC | 0.9296 |

## Test Metrics

| Metric | Value |
| --- | ---: |
| Loss | 0.3697 |
| Accuracy | 0.8972 |
| Precision `non_edible` | 0.9402 |
| Recall `non_edible` | 0.8333 |
| F1 `non_edible` | 0.8835 |
| ROC-AUC | 0.9355 |

## Classification Report

```text
              precision    recall  f1-score   support

      edible       0.87      0.95      0.91       150
  non_edible       0.94      0.83      0.88       132

    accuracy                           0.90       282
   macro avg       0.90      0.89      0.90       282
weighted avg       0.90      0.90      0.90       282
```

## Saved Artifacts

- `reports/baseline/metrics.json`
- `reports/baseline/history.csv`
- `reports/baseline/classification_report.txt`
- `reports/baseline/confusion_matrix.png`
- `reports/baseline/training_curves.png`
- `models/baseline_resnet18.pt`

## Notes

The current split contains exact duplicate images across `train`, `val`, and `test`.
For this iteration we intentionally keep the provided split, but the limitation should be mentioned in the final report.
