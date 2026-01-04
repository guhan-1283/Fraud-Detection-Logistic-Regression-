
# Fraud Detection End-to-End

This repository demonstrates a compact end-to-end fraud detection workflow using classical ML (Logistic Regression). It includes data, an exploratory/training notebook, and a small app for inference or demo.

## Goals

- Provide a reproducible pipeline from raw data to a trained model.
- Show preprocessing choices, feature engineering, and evaluation for an interpretable classifier.
- Offer a lightweight app to run inference or demo results.

## Files (high level)

- `AIML Dataset.csv` — raw dataset (place in project root). Update path in code if stored elsewhere.
- `Model_Training.ipynb` — exploratory data analysis, preprocessing, training, evaluation, and model export (e.g., `joblib` or `pickle`).
- `app.py` — demo/inference script (may be a Streamlit app or a CLI script). Inspect this file for exact runtime behavior and required args.
- `README.md` — this file.

## Dataset

- Location: project root file `AIML Dataset.csv`.
- Typical columns (update to match your CSV): `transaction_id`, `amount`, `timestamp`, `customer_id`, `merchant_id`, `device_type`, `location`, `label` (0 = legit, 1 = fraud).
- Tips: inspect with `pandas.read_csv('AIML Dataset.csv').head()` to confirm schema.
- Link: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download

## Preprocessing (recommended)

- Handle missing values: drop or impute (`median` for numeric, `mode` or `Unknown` for categoricals).
- Feature encoding: one-hot for low-cardinality categoricals, frequency or target encoding for high-cardinality IDs.
- Scaling: standardize numeric features if using distance-based models; for logistic regression scaling helps interpretability.
- Time features: derive hour/day from `timestamp` when present.
- Train/test split: use stratified split on `label` and hold out a validation set or use cross-validation.

Example (notebook shows full code): perform `train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)`.

## Model & Training

- Model: `sklearn.linear_model.LogisticRegression` (good baseline for fraud detection).
- Typical training steps: pipeline with imputer -> encoder -> scaler -> classifier.
- Hyperparameters to consider: `C` (inverse regularization), `class_weight='balanced'` (if classes are imbalanced), solver (`liblinear` for small datasets, `saga` for larger ones).
- Save final model with `joblib.dump(model, 'model.joblib')`.

## Evaluation

- Use metrics appropriate for imbalanced classification: **precision**, **recall**, **F1-score**, and **ROC AUC**.
- Confusion matrix and precision-recall curve are useful for threshold selection.

Example metrics code snippet (in notebook):

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_true, y_pred))
print('ROC AUC:', roc_auc_score(y_true, y_prob[:,1]))
```

## Running the app

- If `app.py` is a Streamlit app:

```bash
streamlit run app.py
```

- If `app.py` is a plain script:

```bash
python app.py
```

- Ensure `AIML Dataset.csv` is in the project root or modify the dataset path in `app.py`.

## Reproducing training locally

1. Create a virtual environment and install dependencies (example below).
2. Open `Model_Training.ipynb` and run cells in order; adjust paths if necessary.
3. After training, ensure the exported model (e.g., `model.joblib`) is available to `app.py`.

Install example (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas scikit-learn numpy joblib streamlit matplotlib seaborn
```

## File structure

- `AIML Dataset.csv` — raw data
- `Model_Training.ipynb` — EDA, preprocessing, training
- `app.py` — demo / inference
- `README.md` — this document

## Troubleshooting

- Missing packages: install from the list above or create a `requirements.txt` (I can generate one from `app.py`).
- Encoding errors reading CSV: try `pd.read_csv('AIML Dataset.csv', encoding='utf-8', errors='replace')`.
- Memory issues: sample the dataset or use chunked processing.

## Next steps / Improvements

- Add automated training script (e.g., `train.py`) and CLI flags for hyperparameters.
- Add unit tests for preprocessing functions.
- Add a `requirements.txt` with exact pinned versions.

---

If you want, I can (pick one):

- generate a `requirements.txt` based on `app.py` imports,
- open `app.py` and extract the exact runtime steps to add to this README,
- or add example screenshots / usage examples for the app.


