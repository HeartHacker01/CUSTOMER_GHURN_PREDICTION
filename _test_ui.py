"""
Test the prediction logic in ui.py without starting a Streamlit server.
Loads the saved model and runs two sample predictions.
"""
import sys, types as _t

def _patch():
    class _S: pass
    for n in [
        "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
        "sklearn.metrics._pairwise_distances_reduction._base",
        "sklearn.metrics._pairwise_distances_reduction._argkmin",
        "sklearn.metrics._pairwise_distances_reduction._argkmin_classmode",
        "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode",
    ]:
        m = _t.ModuleType(n)
        for a in ["ArgKmin32","ArgKmin64","ArgKminClassMode32","ArgKminClassMode64",
                  "RadiusNeighbors32","RadiusNeighbors64","RadiusNeighborsClassMode32",
                  "RadiusNeighborsClassMode64","_sqeuclidean_row_norms32","_sqeuclidean_row_norms64"]:
            setattr(m, a, _S)
        sys.modules[n] = m
_patch()

import json, joblib, pandas as pd

pipeline = joblib.load("model/churn_model.pkl")
meta     = json.load(open("model/model_meta.json"))
features = meta["feature_columns"]

print(f"Model loaded : {type(pipeline.named_steps['classifier']).__name__}")
print(f"Features     : {len(features)}")
print()

samples = [
    # High-risk: month-to-month, fiber optic, electronic check, short tenure
    {
        "label": "HIGH-RISK customer",
        "tenure": 2, "MonthlyCharges": 99.65, "TotalCharges": 199.30,
        "SeniorCitizen": 0, "Contract": "Month-to-month",
        "gender": "Female", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "PaperlessBilling": "Yes",
        "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "PaymentMethod": "Electronic check",
    },
    # Low-risk: two-year contract, DSL, auto-pay, long tenure
    {
        "label": "LOW-RISK customer",
        "tenure": 60, "MonthlyCharges": 42.30, "TotalCharges": 2538.0,
        "SeniorCitizen": 0, "Contract": "Two year",
        "gender": "Male", "Partner": "Yes", "Dependents": "Yes",
        "PhoneService": "No", "PaperlessBilling": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    },
]

for s in samples:
    label = s.pop("label")
    row   = pd.DataFrame([s])
    pred  = int(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0]
    print(f"{label}")
    print(f"  Prediction      : {'Churn' if pred else 'No Churn'}")
    print(f"  Churn prob      : {proba[1]:.4f}  ({proba[1]*100:.1f}%)")
    print(f"  No-churn prob   : {proba[0]:.4f}  ({proba[0]*100:.1f}%)")
    print()

print("Prediction test: PASSED")
