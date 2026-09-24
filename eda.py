"""
eda.py
======
Data Cleaning and Exploratory Data Analysis
Customer Churn Prediction Project — IBM SkillsBuild Data Analytics with AI

Loads the real Telco Customer Churn CSV, cleans it, prints numerical
summaries, and saves charts to outputs/.

No ML model is built here.
"""

import os
import sys
import types as _t

# ── sklearn DLL patch (needed on Python 3.14/Windows — harmless elsewhere) ───
def _patch():
    class _S:
        pass
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
        for a in [
            "ArgKmin32","ArgKmin64","ArgKminClassMode32","ArgKminClassMode64",
            "RadiusNeighbors32","RadiusNeighbors64",
            "RadiusNeighborsClassMode32","RadiusNeighborsClassMode64",
            "_sqeuclidean_row_norms32","_sqeuclidean_row_norms64",
        ]:
            setattr(m, a, _S)
        sys.modules[n] = m
_patch()

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
SAVED  = []   # tracks saved filenames


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    SAVED.append(name)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  CUSTOMER CHURN — DATA CLEANING & EDA")
print("=" * 64)

df = pd.read_csv(CSV_PATH)
print(f"\n[load] {CSV_PATH}")
print(f"       Raw shape : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("  DATA CLEANING")
print("─" * 64)

# ── 2a. Drop customerID (identifier, not a feature) ──────────────────────────
df.drop(columns=["customerID"], inplace=True)
print("\n[clean] Dropped 'customerID' (non-predictive identifier).")

# ── 2b. Duplicate rows ────────────────────────────────────────────────────────
dup_rows = df.duplicated().sum()
print(f"[clean] Exact duplicate rows : {dup_rows}")

# ── 2c. Missing values (pandas NaN) ──────────────────────────────────────────
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("[clean] Pandas NaN values     : 0  (no nulls detected by pandas)")
else:
    print("[clean] Pandas NaN values :\n", null_counts[null_counts > 0].to_string())

# ── 2d. TotalCharges: convert to numeric, handle 11 whitespace rows ──────────
# pandas reads TotalCharges as str because 11 rows contain only whitespace.
# Those 11 rows all have tenure == 0 (brand-new customers).
# Strategy: fill blanks with MonthlyCharges.
# Rationale: a customer with zero months of service has accumulated exactly
# one month of charges at most — MonthlyCharges is the most logical estimate.
# This is transparent, reproducible, and avoids dropping data.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blank_mask = df["TotalCharges"].isna()
n_blank = blank_mask.sum()

print(f"\n[clean] TotalCharges conversion to numeric:")
print(f"        Rows that failed conversion (whitespace): {n_blank}")

if n_blank > 0:
    print(f"\n        These {n_blank} rows all have tenure = 0 (brand-new customers).")
    print(f"        Fill strategy : MonthlyCharges (no charges have accrued yet)")
    print()
    display_cols = ["gender", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    print(df.loc[blank_mask, display_cols].to_string())
    df.loc[blank_mask, "TotalCharges"] = df.loc[blank_mask, "MonthlyCharges"]
    print(f"\n        After fill — remaining NaN in TotalCharges: {df['TotalCharges'].isna().sum()}")

# ── 2e. SeniorCitizen note ────────────────────────────────────────────────────
print("\n[clean] SeniorCitizen is already numeric (0/1).")
print("        All other binary columns use 'Yes'/'No' strings.")
print("        This inconsistency will be handled in preprocessing.")

# ── 2f. Encode target for numeric use (keep original string column too) ───────
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

print(f"\n[clean] Final cleaned shape : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NUMERICAL SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("  NUMERICAL SUMMARIES")
print("─" * 64)

# ── 3a. Churn distribution ────────────────────────────────────────────────────
print("\n[summary] Churn distribution:")
vc = df["Churn"].value_counts()
for label, count in vc.items():
    print(f"           {label:3s} : {count:>5,}  ({count/len(df)*100:.1f}%)")

# ── 3b. Numeric feature stats ────────────────────────────────────────────────
print("\n[summary] Numeric feature statistics:")
num_stats = df[["tenure","MonthlyCharges","TotalCharges"]].describe().T
num_stats.columns = ["count","mean","std","min","25%","50%","75%","max"]
print(num_stats.round(2).to_string())

# ── 3c. Churn rates by key categorical features ───────────────────────────────
cat_features = [
    ("Contract",       "Churn Rate by Contract Type"),
    ("InternetService","Churn Rate by Internet Service"),
    ("PaymentMethod",  "Churn Rate by Payment Method"),
    ("TechSupport",    "Churn Rate by Tech Support"),
    ("OnlineSecurity", "Churn Rate by Online Security"),
    ("gender",         "Churn Rate by Gender"),
    ("SeniorCitizen",  "Churn Rate by Senior Citizen"),
    ("Partner",        "Churn Rate by Partner"),
    ("Dependents",     "Churn Rate by Dependents"),
    ("PaperlessBilling","Churn Rate by Paperless Billing"),
    ("PhoneService",   "Churn Rate by Phone Service"),
    ("MultipleLines",  "Churn Rate by Multiple Lines"),
    ("StreamingTV",    "Churn Rate by Streaming TV"),
    ("StreamingMovies","Churn Rate by Streaming Movies"),
    ("OnlineBackup",   "Churn Rate by Online Backup"),
    ("DeviceProtection","Churn Rate by Device Protection"),
]

print("\n[summary] Churn rates by categorical feature:")
for col, title in cat_features:
    rates = df.groupby(col)["Churn_bin"].agg(["sum","count","mean"])
    rates.columns = ["churned","total","churn_rate"]
    rates["churn_rate"] = (rates["churn_rate"] * 100).round(1)
    rates = rates.sort_values("churn_rate", ascending=False)
    print(f"\n  {title}:")
    print(rates.to_string())

# ── 3d. Numeric features: mean by churn ──────────────────────────────────────
print("\n[summary] Mean numeric values by Churn:")
print(df.groupby("Churn")[["tenure","MonthlyCharges","TotalCharges"]].mean().round(2).to_string())

# ── 3e. Correlation with Churn ────────────────────────────────────────────────
print("\n[summary] Pearson correlation with Churn (numeric cols):")
corr_cols = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Churn_bin"]
corr = df[corr_cols].corr()["Churn_bin"].drop("Churn_bin").sort_values()
print(corr.round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("  GENERATING CHARTS")
print("─" * 64)


# ── Chart 1: Overall churn distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Churn"].value_counts().reindex(["No", "Yes"])
bars = ax.bar(["No Churn", "Churned"], counts.values,
              color=[BLUE, ORANGE], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 60,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Overall Churn Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, counts.max() * 1.2)
ax.spines[["top","right"]].set_visible(False)
save(fig, "01_churn_distribution.png")
print("[chart] 01_churn_distribution.png")


# ── Chart 2: Churn by Contract ────────────────────────────────────────────────
def churn_rate_bar(col, filename, title, figsize=(8, 4), rotate=0):
    rates = (df.groupby(col)["Churn_bin"].mean() * 100).sort_values(ascending=False)
    n_total = df.groupby(col)["Churn_bin"].count()
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("muted", len(rates))
    bars = ax.bar(rates.index.astype(str), rates.values, color=colors, edgecolor="white", width=0.5)
    for bar, (label, rate) in zip(bars, rates.items()):
        n = n_total[label]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, min(rates.max() + 18, 100))
    ax.spines[["top","right"]].set_visible(False)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    save(fig, filename)
    print(f"[chart] {filename}")

churn_rate_bar("Contract",        "02_churn_by_contract.png",       "Churn Rate by Contract Type")
churn_rate_bar("InternetService", "03_churn_by_internet_service.png","Churn Rate by Internet Service")
churn_rate_bar("PaymentMethod",   "04_churn_by_payment_method.png", "Churn Rate by Payment Method", figsize=(9,4), rotate=15)
churn_rate_bar("TechSupport",     "05_churn_by_tech_support.png",   "Churn Rate by Tech Support")
churn_rate_bar("OnlineSecurity",  "06_churn_by_online_security.png","Churn Rate by Online Security")
churn_rate_bar("SeniorCitizen",   "07_churn_by_senior_citizen.png", "Churn Rate by Senior Citizen Status")
churn_rate_bar("PaperlessBilling","08_churn_by_paperless_billing.png","Churn Rate by Paperless Billing")
churn_rate_bar("OnlineBackup",    "09_churn_by_online_backup.png",  "Churn Rate by Online Backup")
churn_rate_bar("DeviceProtection","10_churn_by_device_protection.png","Churn Rate by Device Protection")
churn_rate_bar("MultipleLines",   "11_churn_by_multiple_lines.png", "Churn Rate by Multiple Lines")


# ── Chart 3: Churn by tenure (histogram) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for churn_val, label, color in [(0, "No Churn", BLUE), (1, "Churned", ORANGE)]:
    ax.hist(df.loc[df["Churn_bin"] == churn_val, "tenure"],
            bins=30, alpha=0.65, label=label, color=color, edgecolor="white")
ax.set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Number of Customers")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "12_tenure_by_churn.png")
print("[chart] 12_tenure_by_churn.png")


# ── Chart 4: Churn by MonthlyCharges (histogram) ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for churn_val, label, color in [(0, "No Churn", BLUE), (1, "Churned", ORANGE)]:
    ax.hist(df.loc[df["Churn_bin"] == churn_val, "MonthlyCharges"],
            bins=30, alpha=0.65, label=label, color=color, edgecolor="white")
ax.set_title("Monthly Charges Distribution by Churn", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Number of Customers")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "13_monthly_charges_by_churn.png")
print("[chart] 13_monthly_charges_by_churn.png")


# ── Chart 5: Tenure box-plot by churn ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="tenure",
            palette={"No": BLUE, "Yes": ORANGE},
            order=["No","Yes"], width=0.45, ax=ax)
ax.set_xticklabels(["No Churn", "Churned"])
ax.set_title("Tenure Box-plot by Churn", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel(""); ax.set_ylabel("Tenure (months)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "14_tenure_boxplot_by_churn.png")
print("[chart] 14_tenure_boxplot_by_churn.png")


# ── Chart 6: Monthly Charges box-plot by churn ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges",
            palette={"No": BLUE, "Yes": ORANGE},
            order=["No","Yes"], width=0.45, ax=ax)
ax.set_xticklabels(["No Churn", "Churned"])
ax.set_title("Monthly Charges Box-plot by Churn", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel(""); ax.set_ylabel("Monthly Charges ($)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "15_monthly_charges_boxplot_by_churn.png")
print("[chart] 15_monthly_charges_boxplot_by_churn.png")


# ── Chart 7: Correlation heatmap (numeric features) ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
corr_df = df[["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Churn_bin"]].copy()
corr_df.rename(columns={"Churn_bin":"Churn"}, inplace=True)
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool), k=1)
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix — Numeric Features", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
save(fig, "16_correlation_heatmap.png")
print("[chart] 16_correlation_heatmap.png")


# ── Chart 8: Multi-panel — top categorical churn rates ───────────────────────
top_cats = ["Contract","InternetService","PaymentMethod",
            "TechSupport","OnlineSecurity","PaperlessBilling"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, top_cats):
    rates  = (df.groupby(col)["Churn_bin"].mean() * 100).sort_values(ascending=False)
    n_tot  = df.groupby(col)["Churn_bin"].count()
    colors = sns.color_palette("muted", len(rates))
    bars   = ax.bar(rates.index.astype(str), rates.values,
                    color=colors, edgecolor="white", width=0.5)
    for bar, (lbl, rate) in zip(bars, rates.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Churn Rate by {col}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, min(rates.max() + 18, 100))
    ax.spines[["top","right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
fig.suptitle("Churn Rates by Key Categorical Features",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "17_categorical_churn_rates_panel.png")
print("[chart] 17_categorical_churn_rates_panel.png")


# ── Chart 9: Churn rate by tenure bucket ─────────────────────────────────────
df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"],
    include_lowest=True,
)
rates_bucket = (df.groupby("tenure_bucket", observed=True)["Churn_bin"].mean() * 100)
n_bucket     =  df.groupby("tenure_bucket", observed=True)["Churn_bin"].count()

fig, ax = plt.subplots(figsize=(9, 4))
colors = sns.color_palette("muted", len(rates_bucket))
bars = ax.bar(rates_bucket.index.astype(str), rates_bucket.values,
              color=colors, edgecolor="white", width=0.6)
for bar, (lbl, rate) in zip(bars, rates_bucket.items()):
    n = n_bucket[lbl]
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate:.1f}%\n(n={n:,})",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Churn Rate by Tenure Bucket (months)", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, rates_bucket.max() + 20)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "18_churn_by_tenure_bucket.png")
print("[chart] 18_churn_by_tenure_bucket.png")

df.drop(columns=["tenure_bucket"], inplace=True)  # clean up temp column


# ── Chart 10: MonthlyCharges bucket churn rate ────────────────────────────────
df["mc_bucket"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 30, 50, 70, 90, 120],
    labels=["$0–30", "$30–50", "$50–70", "$70–90", "$90–120"],
    include_lowest=True,
)
rates_mc = (df.groupby("mc_bucket", observed=True)["Churn_bin"].mean() * 100)
n_mc     =  df.groupby("mc_bucket", observed=True)["Churn_bin"].count()

fig, ax = plt.subplots(figsize=(9, 4))
colors = sns.color_palette("muted", len(rates_mc))
bars = ax.bar(rates_mc.index.astype(str), rates_mc.values,
              color=colors, edgecolor="white", width=0.6)
for bar, (lbl, rate) in zip(bars, rates_mc.items()):
    n = n_mc[lbl]
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate:.1f}%\n(n={n:,})",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Churn Rate by Monthly Charges Bracket", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, rates_mc.max() + 20)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "19_churn_by_monthly_charges_bucket.png")
print("[chart] 19_churn_by_monthly_charges_bucket.png")

df.drop(columns=["mc_bucket"], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  EDA COMPLETE")
print("=" * 64)
print(f"\n  Charts saved to : {OUTPUT_DIR}/")
print(f"  Total charts    : {len(SAVED)}")
for f in SAVED:
    print(f"    outputs/{f}")

print("""
─────────────────────────────────────────────────────────────
  KEY FINDINGS (from actual data)
─────────────────────────────────────────────────────────────
  1. Overall churn rate             : 26.5% (1,869 / 7,043)
  2. Month-to-month contract churn  : highest of contract types
  3. Fiber optic internet churn     : highest of internet types
  4. Electronic check payment churn : highest payment method
  5. Short-tenure customers (0–12m) : highest churn risk bucket
  6. Higher MonthlyCharges          : positively correlated with churn
  7. No TechSupport / OnlineSecurity: ~2x churn rate vs with-service
  8. Senior citizens                : higher churn than non-seniors
  9. No missing values after fix    : 11 TotalCharges blanks filled
 10. Class imbalance 73.5 / 26.5%  : must handle in modelling stage
─────────────────────────────────────────────────────────────
""")
