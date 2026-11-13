# Simple ML Pipeline: Breast Cancer Classification

A minimal end-to-end machine learning pipeline that classifies breast tumors as
malignant or benign. Built for the AIPI 510 Week 12 sync activity.

## Results

**Test accuracy = 98.2%** (112 of 114 held-out samples classified correctly)

| Metric | Value |
| --- | --- |
| Test accuracy | 98.2% |
| 5-fold cross-validation accuracy (training set) | 98.0% ± 1.3% |
| Malignant precision / recall | 0.98 / 0.98 |
| Benign precision / recall | 0.99 / 0.99 |

Confusion matrix on the test set:

| | Predicted malignant | Predicted benign |
| --- | --- | --- |
| **Actual malignant** | 41 | 1 |
| **Actual benign** | 1 | 71 |

## Pipeline

1. **Data**: the Breast Cancer Wisconsin (Diagnostic) dataset, loaded with
   `sklearn.datasets.load_breast_cancer`. It has 569 samples and 30 numeric
   features computed from images of cell nuclei, with 212 malignant and 357
   benign cases.
2. **Split**: 80% train / 20% test, stratified by class, with a fixed random seed (42).
3. **Model**: `StandardScaler` followed by `LogisticRegression`, combined in a
   scikit-learn `Pipeline` so the scaler is fit on training data only.
4. **Evaluation**: 5-fold cross-validation on the training set, then a single
   evaluation on the held-out test set (accuracy, confusion matrix,
   precision and recall).

## How to run

```bash
pip install -r requirements.txt
python pipeline.py
```

The script prints all of the results above. The dataset ships with
scikit-learn, so nothing needs to be downloaded.
