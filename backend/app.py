"""
backend/app.py
==============
Flask REST API for the Customer Churn Prediction model.

Endpoints:
  GET  /health          → service health check
  POST /predict         → predict churn probability for a customer
"""

# ── sklearn DLL patch (must be FIRST, before any sklearn import) ──────────────
import sys
import types as _types


def _patch_sklearn_blocked_dll():
    class _Stub:
        pass
    _blocked = [
        "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
        "sklearn.metrics._pairwise_distances_reduction._base",
        "sklearn.metrics._pairwise_distances_reduction._argkmin",
        "sklearn.metrics._pairwise_distances_reduction._argkmin_classmode",
        "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode",
    ]
    _attrs = [
        "ArgKmin32", "ArgKmin64",
        "ArgKminClassMode32", "ArgKminClassMode64",
        "RadiusNeighbors32", "RadiusNeighbors64",
        "RadiusNeighborsClassMode32", "RadiusNeighborsClassMode64",
        "_sqeuclidean_row_norms32", "_sqeuclidean_row_norms64",
    ]
    for mod_name in _blocked:
        m = _types.ModuleType(mod_name)
        for attr in _attrs:
            setattr(m, attr, _Stub)
        sys.modules[mod_name] = m


_patch_sklearn_blocked_dll()
# ── end patch ─────────────────────────────────────────────────────────────────

import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "churn_pipeline.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# Load pipeline once at startup
pipeline = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    metrics_data = json.load(f)

EXPECTED_FEATURES = metrics_data["feature_columns"]

# ── Allowed categorical values ─────────────────────────────────────────────────
ALLOWED = {
    "gender":            {"Male", "Female"},
    "SeniorCitizen":     {0, 1},
    "Partner":           {"Yes", "No"},
    "Dependents":        {"Yes", "No"},
    "PhoneService":      {"Yes", "No"},
    "MultipleLines":     {"Yes", "No", "No phone service"},
    "InternetService":   {"DSL", "Fiber optic", "No"},
    "OnlineSecurity":    {"Yes", "No", "No internet service"},
    "OnlineBackup":      {"Yes", "No", "No internet service"},
    "DeviceProtection":  {"Yes", "No", "No internet service"},
    "TechSupport":       {"Yes", "No", "No internet service"},
    "StreamingTV":       {"Yes", "No", "No internet service"},
    "StreamingMovies":   {"Yes", "No", "No internet service"},
    "Contract":          {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling":  {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    },
}


def validate_input(data: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors = []
    for feat in EXPECTED_FEATURES:
        if feat not in data:
            errors.append(f"Missing field: '{feat}'")
            continue
        val = data[feat]
        if feat in ALLOWED and val not in ALLOWED[feat]:
            errors.append(
                f"Invalid value for '{feat}': '{val}'. "
                f"Allowed: {sorted(str(v) for v in ALLOWED[feat])}"
            )
    # Numeric range checks
    if "tenure" in data:
        try:
            t = float(data["tenure"])
            if t < 0:
                errors.append("'tenure' must be >= 0")
        except (ValueError, TypeError):
            errors.append("'tenure' must be numeric")
    if "MonthlyCharges" in data:
        try:
            mc = float(data["MonthlyCharges"])
            if mc < 0:
                errors.append("'MonthlyCharges' must be >= 0")
        except (ValueError, TypeError):
            errors.append("'MonthlyCharges' must be numeric")
    if "TotalCharges" in data:
        try:
            tc = float(data["TotalCharges"])
            if tc < 0:
                errors.append("'TotalCharges' must be >= 0")
        except (ValueError, TypeError):
            errors.append("'TotalCharges' must be numeric")
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({
        "status": "ok",
        "model":  metrics_data.get("best_model", "unknown"),
        "roc_auc": next(
            (r["roc_auc"] for r in metrics_data["all_results"]
             if r["model"] == metrics_data["best_model"]),
            None
        ),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict churn probability.

    Expects JSON with all 19 feature fields (customerID excluded).
    Returns:
      {
        "prediction": "Churn" | "No Churn",
        "churn_probability": float (0-1),
        "no_churn_probability": float (0-1),
        "disclaimer": str
      }
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    errors = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 422

    try:
        # Build a single-row DataFrame in the exact feature order
        row = {feat: [data[feat]] for feat in EXPECTED_FEATURES}
        df  = pd.DataFrame(row)

        pred    = int(pipeline.predict(df)[0])
        probas  = pipeline.predict_proba(df)[0]
        churn_p = round(float(probas[1]), 4)
        no_churn_p = round(float(probas[0]), 4)

        return jsonify({
            "prediction":          "Churn" if pred == 1 else "No Churn",
            "churn_probability":   churn_p,
            "no_churn_probability": no_churn_p,
            "disclaimer": (
                "This prediction is an estimate from a machine-learning model "
                "and is not a guarantee of future customer behaviour."
            ),
        })

    except Exception as exc:
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
