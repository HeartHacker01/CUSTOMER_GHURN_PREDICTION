"""
train_model.py
==============
Machine Learning — Customer Churn Prediction
IBM SkillsBuild Data Analytics with AI Internship

Steps performed:
  1. Load and clean the raw CSV
  2. Remove duplicate rows (found in EDA: 22 exact duplicates)
  3. Build a leakage-free sklearn Pipeline / ColumnTransformer
  4. Stratified 80/20 train-test split
  5. Train Logistic Regression and Random Forest (both class-balanced)
  6. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC,
     Confusion Matrix, Classification Report
  7. Save comparison table  → outputs/model_results.csv
  8. Save visualisations    → outputs/
  9. Save the selected pipeline → model/churn_model.pkl

Environment note:
  On Python 3.14 / Windows, sklearn's _datasets_pair.pyd is blocked by
  Windows Smart App Control.  The stub below pre-registers empty modules
  so that sklearn.linear_model and sklearn.ensemble can import cleanly.
  It has zero effect on Logistic Regression or Random Forest execution.
"""

# ── sklearn DLL stub (must appear BEFORE any sklearn import) ─────────────────
import sys
import types as _t


def _patch_sklearn():
    class _Stub:
        pass
    _mods = [
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
    for mod_name in _mods:
        m = _t.ModuleType(mod_name)
        for attr in _attrs:
            setattr(m, attr, _Stub)
        sys.modules[mod_name] = m


_patch_sklearn()
# ── end stub ──────────────────────────────────────────────────────────────────

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
CSV    = os.path.join(BASE, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTDIR = os.path.join(BASE, "outputs")
MDLDIR = os.path.join(BASE, "model")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MDLDIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
BLUE   = "#4C72B0"
ORANGE = "#DD8452"

# ── Feature groups ────────────────────────────────────────────────────────────
NUMERIC  = ["tenure", "MonthlyCharges", "TotalCharges"]

ORDINAL  = ["Contract"]
ORD_CATS = [["Month-to-month", "One year", "Two year"]]   # 0 → 1 → 2

BINARY   = [
    "gender", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling",
]
# SeniorCitizen is already 0/1 integer → treated as numeric
NUMERIC_EXTRA = ["SeniorCitizen"]

MULTICAT = [
    "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC + NUMERIC_EXTRA + ORDINAL + BINARY + MULTICAT
TARGET = "Churn"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
def load_and_clean(path: str) -> tuple[pd.DataFrame, int]:
    """Returns (cleaned DataFrame, number of duplicate rows removed)."""
    print("=" * 64)
    print("  LOAD & CLEAN")
    print("=" * 64)

    df = pd.read_csv(path)
    print(f"\n[load]  Raw shape : {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Drop identifier
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges: whitespace → NaN → fill with MonthlyCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = \
        df.loc[df["TotalCharges"].isna(), "MonthlyCharges"]
    print(f"[clean] TotalCharges: {n_blank} whitespace rows → filled with MonthlyCharges")

    # Remove duplicate rows (identified in EDA: 22 after dropping customerID)
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dup = n_before - len(df)
    print(f"[clean] Duplicate rows removed : {n_dup}  ({n_before:,} → {len(df):,})")

    # Encode target: Yes → 1, No → 0
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    churn_n = df[TARGET].sum()
    print(f"\n[clean] Final shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"[clean] Churn = 1    : {churn_n:,}  ({churn_n/len(df)*100:.1f}%)")
    print(f"[clean] Churn = 0    : {len(df)-churn_n:,}  "
          f"({(len(df)-churn_n)/len(df)*100:.1f}%)")
    return df, n_dup


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
def build_preprocessor() -> ColumnTransformer:
    """
    All transformers are fitted on training data only → no data leakage.

    Column groups:
      numeric        : StandardScaler  (tenure, MonthlyCharges, TotalCharges,
                                        SeniorCitizen)
      ordinal        : OrdinalEncoder   Contract (0/1/2)
      binary cats    : OneHotEncoder(drop='if_binary')  → single column each
      multi-level cats: OneHotEncoder(drop='first')     → k-1 dummies
    """
    return ColumnTransformer(
        transformers=[
            ("num",
             StandardScaler(),
             NUMERIC + NUMERIC_EXTRA),

            ("ord",
             OrdinalEncoder(categories=ORD_CATS),
             ORDINAL),

            ("bin",
             OneHotEncoder(drop="if_binary", sparse_output=False),
             BINARY),

            ("cat",
             OneHotEncoder(drop="first", sparse_output=False),
             MULTICAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATE A FITTED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(pipeline, X_test: pd.DataFrame,
             y_test: pd.Series, name: str) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba)

    print(f"\n{'─'*64}")
    print(f"  {name}")
    print(f"{'─'*64}")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churned"]))
    print(f"  ROC-AUC : {roc:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    cm_path = os.path.join(OUTDIR, f"cm_{safe}.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {cm_path}")

    return {
        "Model":     name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "ROC_AUC":   round(roc,  4),
        "_pipeline": pipeline,
        "_y_proba":  y_proba,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN TRAINING ROUTINE
# ══════════════════════════════════════════════════════════════════════════════
def train():
    # ── 4.1  Load & clean ────────────────────────────────────────────────────
    df, n_dup = load_and_clean(CSV)

    # ── 4.2  Train / test split ──────────────────────────────────────────────
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[split] Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"[split] Test churn rate : {y_test.mean()*100:.1f}%  (stratified)")

    # ── 4.3  Define models ───────────────────────────────────────────────────
    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # ── 4.4  Train and evaluate both models ──────────────────────────────────
    print("\n" + "=" * 64)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 64)

    results   = []
    pipelines = {}

    for name, clf in models.items():
        print(f"\n[train] Fitting {name} …")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, name)
        results.append(metrics)
        pipelines[name] = pipe

    # ── 4.5  Comparison table ────────────────────────────────────────────────
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    results_df   = pd.DataFrame(results)[display_cols]

    print("\n" + "=" * 64)
    print("  MODEL COMPARISON TABLE")
    print("=" * 64)
    print(results_df.to_string(index=False))
    print("=" * 64)

    csv_path = os.path.join(OUTDIR, "model_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[saved] Comparison table → {csv_path}")

    # ── 4.6  Side-by-side bar chart ──────────────────────────────────────────
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x      = np.arange(len(metrics_to_plot))
    width  = 0.35
    colors = [BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(results):
        vals = [row[m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width,
                      label=row["Model"], color=colors[i],
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Evaluation Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    bar_path = os.path.join(OUTDIR, "model_comparison_bar.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] Comparison bar chart → {bar_path}")

    # ── 4.7  ROC curve overlay ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for row, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, row["_y_proba"])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{row['Model']}  (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    roc_path = os.path.join(OUTDIR, "roc_curves.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] ROC curve chart → {roc_path}")

    # ── 4.8  Select best model by ROC-AUC ────────────────────────────────────
    best_row  = max(results, key=lambda r: r["ROC_AUC"])
    best_name = best_row["Model"]
    best_pipe = pipelines[best_name]

    print(f"\n[select] Best model : {best_name}")
    print(f"         Reason     : highest ROC-AUC ({best_row['ROC_AUC']:.4f}), "
          f"which measures overall ranking ability across all thresholds — "
          f"the most robust single metric for an imbalanced binary problem.")

    # ── 4.9  Save the winning pipeline ───────────────────────────────────────
    model_path = os.path.join(MDLDIR, "churn_model.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"\n[saved] Pipeline → {model_path}")

    # ── 4.10 Save metadata alongside the model ───────────────────────────────
    meta = {
        "best_model":      best_name,
        "all_results":     [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ],
        "feature_columns": ALL_FEATURES,
        "target":          TARGET,
        "test_size":       0.20,
        "random_state":    42,
        "duplicate_rows_removed": n_dup,
        "TotalCharges_imputation": (
            "11 rows with tenure=0 had whitespace TotalCharges. "
            "Filled with MonthlyCharges (no charges accrued yet)."
        ),
    }
    meta_path = os.path.join(MDLDIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] Metadata    → {meta_path}")

    # ── 4.11 Smoke-test on a real held-out row ───────────────────────────────
    sample  = X_test.iloc[[0]]
    pred    = int(best_pipe.predict(sample)[0])
    proba   = best_pipe.predict_proba(sample)[0][1]
    actual  = int(y_test.iloc[0])
    print(f"\n[smoke] Row 0  actual={actual}  "
          f"predicted={'Churn' if pred else 'No Churn'}  "
          f"churn_prob={proba:.4f}")

    # ── 4.12 Print final summary ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  FINAL SUMMARY")
    print("=" * 64)
    for row in results:
        marker = " ← SELECTED" if row["Model"] == best_name else ""
        print(f"\n  {row['Model']}{marker}")
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
            print(f"    {metric:<12}: {row[metric]:.4f}")

    print(f"""
  Files saved
  ───────────
  model/churn_model.pkl
  model/model_meta.json
  outputs/model_results.csv
  outputs/cm_logistic_regression.png
  outputs/cm_random_forest.png
  outputs/model_comparison_bar.png
  outputs/roc_curves.png

✅  train_model.py complete.
""")


if __name__ == "__main__":
    train()
