# ChurnIQ — Telecom Customer Churn Prediction

An end-to-end machine-learning web application using the Telco Customer Churn CSV dataset.

## Features
- Dataset overview and churn analytics
- Interactive customer churn prediction
- Random Forest classification pipeline
- Held-out test metrics and confusion matrix
- Streamlit frontend suitable for Streamlit Community Cloud

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The training script creates `churn_model.joblib` and `metrics.json` from `data.csv`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create an app, select the repository and `app.py` as the entry point.
4. Deploy. Keep `data.csv`, `train.py`, `requirements.txt`, `churn_model.joblib`, and `metrics.json` in the repository.

## Dataset notes
- Target: `Churn` (Yes/No)
- `customerID` is excluded because it is an identifier.
- `TotalCharges` is converted to numeric; blank values are imputed in the pipeline.
- A stratified 80/20 train-test split is used.

## Important
This is a learning/demo model. Evaluate it on additional data and calibrate thresholds before using predictions for real customer-retention decisions. Do not upload personally identifiable customer data to a public repository.
