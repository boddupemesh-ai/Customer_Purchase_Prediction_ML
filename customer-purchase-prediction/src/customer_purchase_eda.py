"""
Customer Purchase Prediction - Review 2
Environment Setup | Dataset Exploration | EDA & Reporting
Team: B. Pemesh (2520030388), N. Madhan (2520030263)

Run:
    python src/customer_purchase_eda.py                 # built-in sample dataset
    python src/customer_purchase_eda.py --csv data/raw/customers.csv --target Purchase

Install:
    pip install numpy pandas scikit-learn matplotlib seaborn scipy tabulate
"""
import argparse
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# --------------------------------------------------------------------------
REQUIRED = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "scipy", "tabulate"]


def check_environment():
    """Verify Python version and libraries (requirements.txt + venv for reproducibility)."""
    print(f"Python {sys.version.split()[0]} (3.9+ required)")
    assert sys.version_info >= (3, 9), "Please use Python 3.9 or newer"
    missing = []
    for lib in REQUIRED:
        try:
            mod = __import__(lib)
            print(f"  [ok] {lib:<11} {getattr(mod, '__version__', '')}")
        except ImportError:
            missing.append(lib)
            print(f"  [missing] {lib}")
    if missing:
        sys.exit("Install missing libraries: pip install numpy pandas scikit-learn tabulate "
                 "matplotlib seaborn scipy")


def make_project_folders(root="."):
    """Structured hierarchy: /data, /notebooks, /src, /models (+ /reports)."""
    root = Path(root)
    for sub in ["data/raw", "data/processed", "notebooks", "src", "models", "reports/figures"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text(
        "numpy\npandas\nscikit-learn\nmatplotlib\nseaborn\nscipy\ntabulate\n")
    return root


check_environment()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

sns.set_theme(style="whitegrid", context="talk")


# --------------------------------------------------------------------------
# 2. DATA COLLECTION (real CSV, or a realistic sample with known problems)
# --------------------------------------------------------------------------
def make_sample_data(n=1500, seed=42):
    """Synthetic customer data with missing values, duplicates and outliers,
    so every EDA step has something to find."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    income = rng.normal(55000, 18000, n).clip(12000, None)
    time_on_site = rng.gamma(2.0, 4.0, n)                    # minutes, right-skewed
    pages = rng.poisson(6, n) + 1
    prev = rng.poisson(1.4, n)
    email = rng.integers(0, 2, n)
    device = rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.55, 0.35, 0.10])
    region = rng.choice(["North", "South", "East", "West"], n)
    gender = rng.choice(["Male", "Female"], n)

    z = (-4.2 + 0.00004 * income + 0.09 * time_on_site + 0.12 * pages
         + 0.45 * prev + 0.7 * email + 0.012 * (age - 40)
         + np.where(device == "Desktop", 0.4, 0.0))
    p = 1 / (1 + np.exp(-z))
    purchase = rng.binomial(1, p)

    df = pd.DataFrame({
        "Age": age, "Gender": gender, "Annual_Income": income.round(0),
        "Region": region, "Device": device,
        "Time_On_Site_Min": time_on_site.round(2), "Pages_Viewed": pages,
        "Previous_Purchases": prev, "Email_Opened": email, "Purchase": purchase,
    })
    # inject data-quality problems
    for col, frac in [("Age", 0.04), ("Annual_Income", 0.06), ("Device", 0.02)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "Annual_Income"] = 450000  # outliers
    df.loc[rng.choice(n, 8, replace=False), "Gender"] = "male "          # inconsistency
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)  # duplicates
    return df


def load_data(csv_path):
    if csv_path:
        return pd.read_csv(csv_path)
    return make_sample_data()


# --------------------------------------------------------------------------
# 3. DATASET EXPLORATION
# --------------------------------------------------------------------------
def explore(df, target):
    print("\n=== DATASET EXPLORATION ===")
    print("Shape (rows, columns):", df.shape)
    print("\nhead():\n", df.head())
    print("\ntail():\n", df.tail(3))
    print("\ninfo():")
    df.info()
    print("\ndtypes:\n", df.dtypes)
    print(f"\nTarget variable: '{target}'")
    print("Feature columns:", [c for c in df.columns if c != target])
    print(f"Samples vs features: {df.shape[0]} vs {df.shape[1] - 1}")
    print("\nMissing values (isnull().sum()):\n", df.isnull().sum())
    mem = df.memory_usage(deep=True).sum() / 1024
    print(f"\nStorage requirement: {mem:.1f} KB")


# --------------------------------------------------------------------------
# 4. EDA - DATA QUALITY CHECKS + QUALITY METRICS
# --------------------------------------------------------------------------
def quality_checks(df, target):
    print("\n=== DATA QUALITY CHECKS ===")
    q = {}

    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    q["missing"] = pd.DataFrame({"missing": miss, "missing_%": miss_pct})
    q["completeness_%"] = (100 - miss_pct).round(2)
    print("Missing value analysis:\n", q["missing"][q["missing"]["missing"] > 0])
    print("\nCompleteness % per feature:\n", q["completeness_%"])

    q["duplicates"] = int(df.duplicated().sum())
    print("\nDuplicate records:", q["duplicates"])

    num_cols = df.select_dtypes("number").columns.drop(target, errors="ignore")
    outliers = {}
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        n_z = int((np.abs(stats.zscore(s)) > 3).sum())
        outliers[c] = {"IQR_outliers": n_iqr, "Z>3_outliers": n_z}
    q["outliers"] = pd.DataFrame(outliers).T
    print("\nOutlier detection (IQR and z-score):\n", q["outliers"])

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    print("\nData inconsistency validation (unique values):")
    inconsistent = {}
    for c in cat_cols:
        vals = df[c].dropna().unique()
        normalised = pd.Series(vals).str.strip().str.lower().nunique()
        inconsistent[c] = len(vals) - normalised
        print(f"  {c}: {sorted(map(repr, vals))}")
        if len(vals) != normalised:
            print(f"    -> {len(vals) - normalised} inconsistent spelling/case/space variant(s)")
    q["inconsistent"] = inconsistent

    print("\nData type verification:")
    for c in df.columns:
        print(f"  {c:<20} {str(df[c].dtype):<10} unique={df[c].nunique()}")

    labels = df[target].dropna()
    print("\nLabel accuracy assessment: valid labels =",
          sorted(labels.unique()), "| missing labels =", int(df[target].isnull().sum()))
    return q


# --------------------------------------------------------------------------
# 5. EDA - PATTERN ANALYSIS + STATISTICAL INSIGHTS
# --------------------------------------------------------------------------
def pattern_and_stats(df, target, fig_dir):
    print("\n=== PATTERN ANALYSIS ===")
    num_cols = df.select_dtypes("number").columns.drop(target, errors="ignore")
    out = {}

    # class balance
    bal = df[target].value_counts(normalize=True).mul(100).round(2)
    out["class_balance"] = bal
    print("Class balance (%):\n", bal)
    ratio = bal.max() / bal.min()
    print(f"Imbalance ratio majority:minority = {ratio:.2f}:1",
          "-> consider class_weight / SMOTE" if ratio > 1.5 else "-> roughly balanced")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, hue=target, palette="Blues_d", legend=False)
    plt.title("Class balance")
    plt.tight_layout(); plt.savefig(fig_dir / "class_balance.png", dpi=110); plt.close()

    # distributions
    k = len(num_cols)
    fig, axes = plt.subplots(int(np.ceil(k / 3)), 3, figsize=(15, 4 * np.ceil(k / 3)))
    for ax, c in zip(axes.ravel(), num_cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#2f6fb5")
        ax.set_title(c)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(fig_dir / "distributions.png", dpi=90); plt.close()

    # box plots for outliers
    fig, axes = plt.subplots(1, min(4, k), figsize=(15, 4))
    for ax, c in zip(np.atleast_1d(axes), ["Annual_Income", "Age", "Time_On_Site_Min", "Pages_Viewed"][:min(4, k)]):
        if c in df:
            sns.boxplot(y=df[c], ax=ax, color="#7fb3e6")
            ax.set_title(c)
    plt.tight_layout(); plt.savefig(fig_dir / "boxplots.png", dpi=90); plt.close()

    # correlation matrix + heatmap
    corr = df[list(num_cols) + [target]].corr()
    out["corr"] = corr
    print("\nCorrelation with target:\n", corr[target].drop(target).sort_values(ascending=False).round(3))
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation heatmap")
    plt.tight_layout(); plt.savefig(fig_dir / "correlation_heatmap.png", dpi=90); plt.close()

    # relationship exploration (scatter)
    xs = "Time_On_Site_Min" if "Time_On_Site_Min" in df else num_cols[0]
    ys = "Annual_Income" if "Annual_Income" in df else num_cols[1]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=xs, y=ys, hue=target, palette="coolwarm", alpha=0.6)
    plt.title(f"{xs} vs {ys}")
    plt.tight_layout(); plt.savefig(fig_dir / "scatter.png", dpi=90); plt.close()

    print("\n=== STATISTICAL INSIGHTS ===")
    desc = df[num_cols].describe().T.round(2)
    out["describe"] = desc
    print("Descriptive statistics:\n", desc)

    sk = pd.DataFrame({"skew": df[num_cols].skew().round(3),
                       "kurtosis": df[num_cols].kurt().round(3)})
    sk["note"] = np.where(sk["skew"].abs() > 1, "highly skewed -> log/transform", "ok")
    out["skew_kurt"] = sk
    print("\nSkewness & kurtosis:\n", sk)

    # hypothesis tests: numeric (Welch t-test), categorical (chi-square)
    rows = []
    for c in num_cols:
        a = df.loc[df[target] == 1, c].dropna()
        b = df.loc[df[target] == 0, c].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append((c, "t-test", round(t, 3), p))
    for c in df.select_dtypes(include=["object", "string"]).columns:
        tab = pd.crosstab(df[c].str.strip().str.title(), df[target])
        chi, p, *_ = stats.chi2_contingency(tab)
        rows.append((c, "chi-square", round(chi, 3), p))
    ht = pd.DataFrame(rows, columns=["feature", "test", "statistic", "p_value"])
    ht["significant(p<0.05)"] = ht["p_value"] < 0.05
    ht = ht.sort_values("p_value").reset_index(drop=True)
    out["hypothesis"] = ht
    print("\nHypothesis testing vs target:\n", ht.round(4))
    out["key_features"] = ht.loc[ht["significant(p<0.05)"], "feature"].tolist()
    print("\nKey predictive patterns (significant features):", out["key_features"])
    return out


# --------------------------------------------------------------------------
# 6. PREPROCESSING DECISIONS (Retain / Transform / Remove / Impute) + BASELINE
# --------------------------------------------------------------------------
def preprocess_and_baseline(df, target):
    print("\n=== PREPROCESSING & BASELINE ===")
    d = df.drop_duplicates().copy()                       # REMOVE duplicates
    for c in d.select_dtypes(include=["object", "string"]).columns:           # fix inconsistencies
        d[c] = d[c].str.strip().str.title()
    d = d.dropna(subset=[target])                         # REMOVE rows w/o label

    X, y = d.drop(columns=[target]), d[target].astype(int)
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),   # IMPUTE
                          ("scale", StandardScaler())]), num),            # TRANSFORM
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Rows after cleaning: {len(d)} | train={len(X_tr)} test={len(X_te)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                                class_weight="balanced"),
    }
    rows = []
    for name, m in models.items():
        pipe = Pipeline([("prep", pre), ("model", m)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        rows.append({"model": name,
                     "accuracy": accuracy_score(y_te, pred),
                     "precision": precision_score(y_te, pred),
                     "recall": recall_score(y_te, pred),
                     "f1": f1_score(y_te, pred),
                     "roc_auc": roc_auc_score(y_te, proba)})
    res = pd.DataFrame(rows).set_index("model").round(3)
    print("Baseline performance:\n", res)
    return d, res


# --------------------------------------------------------------------------
# 7. REPORT
# --------------------------------------------------------------------------
def write_report(df, target, q, p, baseline, path):
    lines = ["# Customer Purchase Prediction - EDA Report", ""]
    lines += ["## Dataset summary",
              f"- Samples: {df.shape[0]}  |  Features: {df.shape[1] - 1}  |  Target: `{target}`",
              f"- Storage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
              f"- Duplicate records: {q['duplicates']}",
              "", "### Missing values", q["missing"][q["missing"]["missing"] > 0].to_markdown(), "",
              "### Outliers", q["outliers"].to_markdown(), "",
              "## Key EDA findings",
              f"- Class balance (%): {p['class_balance'].to_dict()}",
              f"- Significant features (p<0.05): {', '.join(p['key_features'])}",
              "", "### Skewness & kurtosis", p["skew_kurt"].to_markdown(), "",
              "### Hypothesis tests", p["hypothesis"].round(4).to_markdown(index=False), "",
              "## Preprocessing recommendations",
              "- Remove duplicate rows and rows without a label",
              "- Normalise inconsistent category spellings (strip + title-case)",
              "- Impute numeric gaps with median, categorical with most frequent",
              "- Scale numeric features; one-hot encode categoricals",
              "- Cap or transform extreme income values and skewed features",
              "", "## Baseline performance", baseline.to_markdown(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to customer CSV (optional)")
    ap.add_argument("--target", default="Purchase")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    root = make_project_folders(a.out)
    df = load_data(a.csv)
    if not a.csv:
        df.to_csv(root / "data/raw/customers.csv", index=False)

    explore(df, a.target)
    q = quality_checks(df, a.target)
    p = pattern_and_stats(df, a.target, root / "reports/figures")
    clean, baseline = preprocess_and_baseline(df, a.target)
    clean.to_csv(root / "data/processed/customers_clean.csv", index=False)
    write_report(df, a.target, q, p, baseline, root / "reports/eda_report.md")


if __name__ == "__main__":
    main()
