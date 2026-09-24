# Customer Churn Prediction and Analytics

**IBM SkillsBuild Data Analytics with AI Internship Project**

---

## Project Overview

This is an end-to-end machine-learning application that analyses telecom customer data to predict whether a customer is likely to churn. It includes full exploratory data analysis, a trained classification model, a REST API, and an interactive dashboard.

---

## Problem Statement

Customer churn — when a customer stops using a company's service — is one of the most costly challenges in the telecommunications industry. Acquiring a new customer costs 5–25× more than retaining one. Early identification of at-risk customers enables targeted retention actions.

---

## Objectives

1. Identify key drivers of customer churn through EDA.
2. Build a reproducible, leakage-free preprocessing pipeline.
3. Train and compare Logistic Regression and Random Forest classifiers.
4. Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. Expose the best pipeline as a Flask REST API and Streamlit dashboard.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | IBM Sample Data Sets — Telco Customer Churn |
| **File** | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Target** | `Churn` (Yes / No) |
| **Class balance** | 73.5% No Churn / 26.5% Churn |
| **Missing values** | 11 rows with blank `TotalCharges` (all have `tenure = 0`) |

---

## Features

| Column | Type | Notes |
|--------|------|-------|
| `tenure` | Numeric | Months as a customer (0–72) |
| `MonthlyCharges` | Numeric | Monthly bill amount |
| `TotalCharges` | Numeric | Total charges to date (11 blank → filled with MonthlyCharges) |
| `Contract` | Ordinal | Month-to-month < One year < Two year |
| `gender` | Binary categorical | Male / Female |
| `SeniorCitizen` | Binary numeric | 0 / 1 |
| `Partner` | Binary categorical | Yes / No |
| `Dependents` | Binary categorical | Yes / No |
| `PhoneService` | Binary categorical | Yes / No |
| `PaperlessBilling` | Binary categorical | Yes / No |
| `MultipleLines` | Multi-level categorical | No / Yes / No phone service |
| `InternetService` | Multi-level categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Multi-level categorical | No / Yes / No internet service |
| `OnlineBackup` | Multi-level categorical | No / Yes / No internet service |
| `DeviceProtection` | Multi-level categorical | No / Yes / No internet service |
| `TechSupport` | Multi-level categorical | No / Yes / No internet service |
| `StreamingTV` | Multi-level categorical | No / Yes / No internet service |
| `StreamingMovies` | Multi-level categorical | No / Yes / No internet service |
| `PaymentMethod` | Multi-level categorical | 4 payment types |

---

## Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.14 | Core language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing pipeline, model training, evaluation |
| matplotlib / seaborn | Visualisations |
| joblib | Model serialisation |
| Flask | REST API backend |
| Streamlit | Interactive dashboard frontend |
| Jupyter Notebook | Analysis and documentation |

---

## Project Structure

```
customer_churn_prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── CustomerChurn_Project.ipynb
├── model/
│   ├── churn_pipeline.pkl        ← trained pipeline (auto-generated)
│   └── metrics.json              ← evaluation metrics (auto-generated)
├── outputs/                      ← EDA charts (auto-generated)
├── backend/
│   └── app.py                    ← Flask REST API
├── train_model.py                ← training script
├── ui.py                         ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone or open the project folder, then install dependencies:
pip install -r requirements.txt
```

> **Note for Windows users on Python 3.14:** A one-time `Unblock-File` call may be needed on newly downloaded packages. The training script handles this automatically via a module stub.

---

## How to Train the Model

```bash
python train_model.py
```

This will:
1. Load and clean the dataset
2. Generate and save 10 EDA charts to `outputs/`
3. Train Logistic Regression and Random Forest
4. Print a full comparison table
5. Save the best pipeline to `model/churn_pipeline.pkl`
6. Save metrics to `model/metrics.json`

**You must run this before starting the Flask API or Streamlit app.**

---

## How to Run Flask

```bash
python backend/app.py
```

The server starts on **http://localhost:5000**

---

## How to Run Streamlit

```bash
streamlit run ui.py
```

The dashboard opens in your browser at **http://localhost:8501**

---

## API Reference

### GET /health

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "model": "Logistic Regression",
  "roc_auc": 0.8418,
  "status": "ok"
}
```

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "Contract": "Month-to-month",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

Response:
```json
{
  "churn_probability": 0.4821,
  "disclaimer": "This prediction is an estimate from a machine-learning model and is not a guarantee of future customer behaviour.",
  "no_churn_probability": 0.5179,
  "prediction": "No Churn"
}
```

---

## Model Evaluation Results

> All metrics are from a **stratified 20% hold-out test set** (1,409 customers).
> Results are generated by `train_model.py` on the actual dataset — not manually entered.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7381 | 0.5043 | **0.7834** | 0.6136 | **0.8418** |
| Random Forest | 0.7764 | 0.5688 | 0.6524 | 0.6077 | 0.8252 |

**Selected model: Logistic Regression**

- **ROC-AUC 0.8418** — correctly ranks a churner above a non-churner ~84% of the time
- **Recall 78.3%** — captures the majority of actual churners (most important for retention)
- Higher ROC-AUC and Recall than Random Forest; similar F1-Score

---

## Results Summary

The trained model identified the following as the **top churn drivers**:

1. **Month-to-month contract** → ~42% churn rate vs ~11% for two-year
2. **Fiber optic internet** → ~42% churn rate
3. **Electronic check payment** → ~45% churn rate
4. **Low tenure (< 12 months)** → highest churn risk window
5. **No tech support / online security** → ~2× churn rate vs customers with those services
6. **Senior citizens** → ~41% churn rate vs ~24% for non-seniors

---

## Limitations

- No time-series features; `tenure` is used as a lifecycle proxy.
- Class imbalance addressed via `class_weight='balanced'` only — SMOTE not explored.
- No external signals (competitor pricing, market events).
- The 0.5 decision threshold is not optimised for a specific business cost/benefit trade-off.

---

## Future Improvements

- XGBoost / LightGBM with `Optuna` hyperparameter tuning
- SMOTE oversampling for improved minority-class performance
- SHAP explainability for per-customer feature attribution
- Decision-threshold optimisation (maximise recall subject to precision ≥ X%)
- Real-time retraining pipeline on new customer batches
- A/B testing framework for retention intervention evaluation

---

## Disclaimer

> Predictions produced by this model are estimates from a statistical algorithm and are
> **not** a guarantee of future customer behaviour. They should be used as one input
> among many in a data-driven retention strategy.
#   C U S T O M E R _ G H U R N _ P R E D I C T I O N  
 