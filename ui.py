"""
ui.py
=====
Streamlit frontend — Customer Churn Prediction
IBM SkillsBuild Data Analytics with AI Internship

Run with:
    streamlit run ui.py
"""

# ── sklearn DLL stub (Python 3.14 / Windows Smart App Control) ───────────────
import sys
import types as _t


def _patch_sklearn():
    class _Stub:
        pass
    for _n in [
        "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
        "sklearn.metrics._pairwise_distances_reduction._base",
        "sklearn.metrics._pairwise_distances_reduction._argkmin",
        "sklearn.metrics._pairwise_distances_reduction._argkmin_classmode",
        "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors",
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode",
    ]:
        _m = _t.ModuleType(_n)
        for _a in [
            "ArgKmin32", "ArgKmin64",
            "ArgKminClassMode32", "ArgKminClassMode64",
            "RadiusNeighbors32", "RadiusNeighbors64",
            "RadiusNeighborsClassMode32", "RadiusNeighborsClassMode64",
            "_sqeuclidean_row_norms32", "_sqeuclidean_row_norms64",
        ]:
            setattr(_m, _a, _Stub)
        sys.modules[_n] = _m


_patch_sklearn()
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import joblib
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "churn_model.pkl")
META_PATH  = os.path.join(BASE, "model", "model_meta.json")
OUT_DIR    = os.path.join(BASE, "outputs")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model and metadata (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

pipeline = load_model()
meta     = load_meta()

best_model   = meta["best_model"]
all_results  = {r["Model"]: r for r in meta["all_results"]}
best_metrics = all_results[best_model]
features     = meta["feature_columns"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def img(filename, caption=""):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"Chart not found: {filename}. Run eda.py first.")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📡 Customer Churn Prediction")
    st.markdown("**IBM SkillsBuild**  \nData Analytics with AI Internship")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict Churn", "📊 EDA & Insights", "🤖 Model Information"],
    )
    st.markdown("---")
    st.caption(
        "Dataset: IBM Telco Customer Churn  \n"
        "7,043 customers · 20 features  \n"
        "After cleaning: 7,021 rows"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT CHURN
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Churn":

    st.title("📡 Customer Churn Prediction")
    st.markdown(
        """
        This application uses a machine-learning model trained on **7,021 IBM Telco customer records**
        to estimate whether a customer is likely to leave the service.

        Fill in the customer profile below and click **Predict Churn** to get the result.
        """
    )
    st.info(
        "⚠️ **Disclaimer:** Predictions are estimates produced by a machine-learning model "
        "and are not guarantees of future customer behaviour.",
        icon="ℹ️",
    )

    st.markdown("---")
    st.subheader("Customer Profile")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("churn_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics & Account**")
            gender   = st.selectbox("Gender", ["Male", "Female"])
            senior   = st.selectbox(
                "Senior Citizen", [0, 1],
                format_func=lambda x: "Yes (Senior)" if x else "No (Non-Senior)"
            )
            partner      = st.selectbox("Partner",    ["No", "Yes"])
            dependents   = st.selectbox("Dependents", ["No", "Yes"])
            tenure       = st.number_input(
                "Tenure (months)", min_value=0, max_value=72, value=12, step=1
            )
            contract     = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                help="Month-to-month customers churn most frequently (~43%)"
            )
            paperless    = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment      = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                help="Electronic check payers have the highest churn rate (~45%)"
            )

        with col2:
            st.markdown("**Phone Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines   = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            st.markdown("**Internet Services**")
            internet  = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                help="Fiber optic users churn at ~42%"
            )
            online_sec    = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                help="Customers without online security churn at ~42%"
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )
            device_prot   = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

        with col3:
            st.markdown("**Support & Streaming**")
            tech_support   = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                help="Customers without tech support churn at ~42%"
            )
            stream_tv      = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )
            stream_movies  = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

            st.markdown("**Charges**")
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0, max_value=200.0,
                value=65.0, step=0.01, format="%.2f"
            )
            # Auto-suggest TotalCharges based on tenure × monthly
            suggested_total = round(monthly_charges * max(tenure, 1), 2)
            total_charges   = st.number_input(
                "Total Charges ($)",
                min_value=0.0, max_value=15000.0,
                value=float(suggested_total),
                step=0.01, format="%.2f"
            )

        submitted = st.form_submit_button(
            "🔮  Predict Churn",
            type="primary",
            use_container_width=True,
        )

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        input_row = pd.DataFrame([{
            "tenure":           tenure,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            "SeniorCitizen":    senior,
            "Contract":         contract,
            "gender":           gender,
            "Partner":          partner,
            "Dependents":       dependents,
            "PhoneService":     phone_service,
            "PaperlessBilling": paperless,
            "MultipleLines":    multi_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_sec,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      stream_tv,
            "StreamingMovies":  stream_movies,
            "PaymentMethod":    payment,
        }])

        pred       = int(pipeline.predict(input_row)[0])
        probas     = pipeline.predict_proba(input_row)[0]
        churn_p    = float(probas[1])
        no_churn_p = float(probas[0])

        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2 = st.columns([1, 2])

        with r1:
            if pred == 1:
                st.error(
                    f"### 🚨 Likely to Churn\n\n"
                    f"**Churn Probability: {churn_p:.1%}**"
                )
            else:
                st.success(
                    f"### ✅ Likely to Stay\n\n"
                    f"**Churn Probability: {churn_p:.1%}**"
                )

            st.metric("Churn Probability",    f"{churn_p:.1%}")
            st.metric("No-Churn Probability", f"{no_churn_p:.1%}")

        with r2:
            # Probability bar
            prob_df = pd.DataFrame({
                "Outcome":     ["No Churn", "Churn"],
                "Probability": [no_churn_p, churn_p],
            })
            st.markdown("**Probability Breakdown**")
            # Manual coloured progress bars
            for label, prob, color in [
                ("No Churn", no_churn_p, "#4C72B0"),
                ("Churn",    churn_p,    "#DD8452"),
            ]:
                st.markdown(
                    f"**{label}** — {prob:.1%}  \n"
                    f'<div style="background:#e5e7eb;border-radius:6px;height:18px;margin-bottom:6px;">'
                    f'<div style="background:{color};width:{prob*100:.1f}%;height:18px;'
                    f'border-radius:6px;"></div></div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "⚠️ Predictions are estimates produced by a machine-learning model "
            "and are not guarantees of future customer behaviour."
        )

        # ── Risk factors for high-churn predictions ───────────────────────────
        if pred == 1 or churn_p > 0.35:
            st.markdown("---")
            st.subheader("📌 Likely Risk Factors")
            risks = []
            if contract == "Month-to-month":
                risks.append("**Month-to-month contract** — highest churn risk (~43%)")
            if internet == "Fiber optic":
                risks.append("**Fiber optic internet** — churn rate ~42%")
            if payment == "Electronic check":
                risks.append("**Electronic check payment** — churn rate ~45%")
            if tenure <= 12:
                risks.append(f"**Short tenure ({tenure} months)** — new customers are most at risk")
            if tech_support == "No" and internet != "No":
                risks.append("**No tech support** — churn rate ~42% without it")
            if online_sec == "No" and internet != "No":
                risks.append("**No online security** — churn rate ~42% without it")
            if senior == 1:
                risks.append("**Senior citizen** — churn rate ~42% vs ~24% for non-seniors")
            if monthly_charges > 80:
                risks.append(f"**High monthly charges** (${monthly_charges:.2f}) — positively correlated with churn")

            if risks:
                for r in risks:
                    st.markdown(f"- {r}")
            else:
                st.markdown("No specific high-risk factors detected for this profile.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":

    st.title("📊 Exploratory Data Analysis & Key Insights")
    st.markdown(
        "Analysis performed on **7,021 IBM Telco customer records** "
        "(7,043 raw rows; 11 TotalCharges blanks imputed; 22 duplicates removed)."
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers (cleaned)", "7,021")
    k2.metric("Churned Customers",         "1,857",  "26.4%")
    k3.metric("Overall Churn Rate",        "26.4%")
    k4.metric("Avg Monthly Charges",       "$64.76")

    st.markdown("---")

    # ── Churn distribution ────────────────────────────────────────────────────
    st.subheader("Overall Churn Distribution")
    img("01_churn_distribution.png", "73.6% No Churn / 26.4% Churn")

    st.markdown("---")
    st.subheader("Churn Rates by Key Features")

    ta, tb = st.columns(2)
    with ta:
        img("02_churn_by_contract.png",         "Churn by Contract Type")
        img("03_churn_by_internet_service.png",  "Churn by Internet Service")
        img("05_churn_by_tech_support.png",      "Churn by Tech Support")
        img("07_churn_by_senior_citizen.png",    "Churn by Senior Citizen Status")
    with tb:
        img("04_churn_by_payment_method.png",    "Churn by Payment Method")
        img("06_churn_by_online_security.png",   "Churn by Online Security")
        img("08_churn_by_paperless_billing.png", "Churn by Paperless Billing")
        img("09_churn_by_online_backup.png",     "Churn by Online Backup")

    st.markdown("---")
    st.subheader("Numeric Features by Churn")

    nc1, nc2 = st.columns(2)
    with nc1:
        img("12_tenure_by_churn.png",            "Tenure Distribution")
        img("14_tenure_boxplot_by_churn.png",     "Tenure Box-plot")
        img("18_churn_by_tenure_bucket.png",      "Churn Rate by Tenure Bucket")
    with nc2:
        img("13_monthly_charges_by_churn.png",   "Monthly Charges Distribution")
        img("15_monthly_charges_boxplot_by_churn.png", "Monthly Charges Box-plot")
        img("19_churn_by_monthly_charges_bucket.png",  "Churn by Charge Bracket")

    st.markdown("---")
    st.subheader("Correlation & Multi-panel Summary")
    ic1, ic2 = st.columns(2)
    with ic1:
        img("16_correlation_heatmap.png", "Pearson Correlation — Numeric Features")
    with ic2:
        img("17_categorical_churn_rates_panel.png", "Categorical Churn Rates Panel")

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔑 Key Insights (from actual data)")

    insights = [
        ("📋 Contract type is the strongest predictor",
         "Month-to-month: **42.7%** churn · One year: 11.3% · Two year: **2.8%**"),
        ("🌐 Fiber optic customers churn most",
         "Fiber optic: **41.9%** churn · DSL: 19.0% · No internet: 7.4%"),
        ("💳 Electronic check payers churn at nearly 3× the auto-pay rate",
         "Electronic check: **45.3%** · Bank transfer: 16.7% · Credit card: 15.2%"),
        ("⏱️ New customers (0–12 months) are most at risk",
         "Churners have a mean tenure of **18.0 months** vs 37.6 months for retained customers"),
        ("📈 Higher monthly charges correlate with churn",
         "Churners pay an average of **$74.44/mo** vs $61.27/mo for retained customers"),
        ("🛡️ Add-on services (Tech Support, Online Security) roughly halve churn",
         "Without: ~42% churn · With: ~15% churn"),
        ("👴 Senior citizens churn more",
         "Seniors: **41.7%** churn · Non-seniors: 23.6%"),
        ("⚖️ Class imbalance: 73.6% No Churn / 26.4% Churn",
         "Addressed by `class_weight='balanced'` in both models"),
        ("🚻 Gender has no meaningful effect",
         "Female: 26.9% · Male: 26.2% — effectively equal"),
    ]

    for title, detail in insights:
        with st.expander(title):
            st.markdown(detail)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFORMATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Information":

    st.title("🤖 Model Information")
    st.markdown(
        f"Final model: **{best_model}**  \n"
        "Selection criterion: highest **ROC-AUC** on a stratified 20% hold-out test set (n = 1,405)."
    )

    st.markdown("---")

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Model Comparison — Actual Test-Set Results")

    import pandas as _pd
    results_df = _pd.DataFrame(meta["all_results"]).set_index("Model")
    st.dataframe(
        results_df.style
            .highlight_max(axis=0, color="#d4edda")
            .highlight_min(axis=0, color="#f8d7da")
            .format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        img("cm_logistic_regression.png", "Logistic Regression")
    with cm2:
        img("cm_random_forest.png",       "Random Forest")

    # ── ROC & comparison charts ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ROC Curves & Metric Comparison")
    rc1, rc2 = st.columns(2)
    with rc1:
        img("roc_curves.png", "ROC Curves — both models")
    with rc2:
        img("model_comparison_bar.png", "All Metrics Side-by-Side")

    # ── Selected model detail ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Selected Model: {best_model}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{best_metrics['Accuracy']:.4f}")
    m2.metric("Precision", f"{best_metrics['Precision']:.4f}")
    m3.metric("Recall",    f"{best_metrics['Recall']:.4f}")
    m4.metric("F1-Score",  f"{best_metrics['F1']:.4f}")
    m5.metric("ROC-AUC",   f"{best_metrics['ROC_AUC']:.4f}")

    st.markdown("---")

    # ── Methodology ───────────────────────────────────────────────────────────
    st.subheader("Methodology")
    st.markdown(
        f"""
| Item | Details |
|------|---------|
| Dataset | IBM Telco Customer Churn (7,043 raw → 7,021 after cleaning) |
| Duplicates removed | {meta['duplicate_rows_removed']} exact duplicate rows |
| TotalCharges imputation | {meta['TotalCharges_imputation']} |
| Train / Test split | 80% / 20% — stratified by target |
| Class imbalance | `class_weight='balanced'` on both models |
| Contract encoding | Ordinal: Month-to-month=0, One year=1, Two year=2 |
| Binary categoricals | OneHotEncoder (drop='if\\_binary') |
| Multi-level categoricals | OneHotEncoder (drop='first') |
| Numeric features | StandardScaler |
| SeniorCitizen | Scaled as numeric (0/1) |
| **Selected model** | **{best_model}** |
| **ROC-AUC** | **{best_metrics['ROC_AUC']:.4f}** |
| **Recall (Churn)** | **{best_metrics['Recall']:.4f}** — catches ~78% of actual churners |
        """
    )

    st.markdown("---")
    st.subheader("Selection Rationale")
    other_model = [r for r in meta["all_results"] if r["Model"] != best_model][0]
    st.markdown(
        f"""
**{best_model}** was selected over **{other_model['Model']}** because:

1. **Higher ROC-AUC** ({best_metrics['ROC_AUC']:.4f} vs {other_model['ROC_AUC']:.4f})  
   ROC-AUC measures the model's ability to correctly rank a churner above a non-churner across *all* classification thresholds — the most robust metric for imbalanced binary problems.

2. **Higher Recall** ({best_metrics['Recall']:.4f} vs {other_model['Recall']:.4f})  
   In a customer retention context, missing a churner (false negative) is more costly than a false alarm. Logistic Regression catches ~78% of actual churners vs ~70% for Random Forest.

3. **Similar F1-Score** ({best_metrics['F1']:.4f} vs {other_model['F1']:.4f})  
   Both models achieve a near-identical F1, making the ROC-AUC and Recall the deciding factors.

> **Note:** Random Forest has higher Accuracy (0.7708 vs 0.7409) and Precision (0.553 vs 0.507),
> but this is partly because it predicts "No Churn" more often on this imbalanced dataset,
> which inflates those metrics without actually being better at the task.
        """
    )

    st.markdown("---")
    st.subheader("Limitations & Future Improvements")

    lim_col, fut_col = st.columns(2)
    with lim_col:
        st.markdown(
            """
**Limitations**

- No time-series features; `tenure` is a lifecycle proxy only
- No external signals (competitor pricing, market events)
- Class imbalance addressed via `class_weight` only — SMOTE not explored
- Default 0.5 threshold not optimised for business cost/benefit
- Static dataset; no automated retraining pipeline
            """
        )
    with fut_col:
        st.markdown(
            """
**Future Improvements**

- XGBoost / LightGBM with `Optuna` hyperparameter tuning
- SMOTE oversampling for improved minority-class recall
- SHAP explainability for per-customer feature attribution
- Threshold optimisation (maximise recall ≥ minimum precision)
- Real-time retraining pipeline on new customer data
- A/B testing framework for retention intervention evaluation
            """
        )

    st.markdown("---")
    st.caption(
        "⚠️ Predictions produced by this model are estimates from a statistical algorithm "
        "and are **not** a guarantee of future customer behaviour. "
        "They should be used as one input among many in a data-driven retention strategy."
    )
