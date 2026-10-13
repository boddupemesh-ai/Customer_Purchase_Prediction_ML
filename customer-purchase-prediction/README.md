# Customer Purchase Prediction

Predict whether a customer is likely to make a purchase, using **Logistic Regression** and **Random Forest**.
This repository covers **Review 2: Environment Setup, Dataset Exploration, and EDA & Reporting**.

**Team:** B. Pemesh (2520030388), N. Madhan (2520030263)

## What the code does
`src/customer_purchase_eda.py` runs the whole Review 2 workflow:
1. Checks the environment (Python 3.9+, required libraries) and creates the project folders
2. Loads a CSV, or generates a sample dataset
3. Dataset exploration: `head`, `tail`, `info`, `shape`, `dtypes`, `isnull().sum()`
4. Data quality checks: missing values, duplicates, outliers (IQR and z-score), inconsistencies, data types, completeness %
5. Pattern analysis: distributions, correlation heatmap, class balance, scatter plots, box plots
6. Statistical insights: descriptive statistics, skewness and kurtosis, hypothesis tests (t-test, chi-square)
7. Preprocessing (remove duplicates, impute, scale, encode) and baseline Logistic Regression and Random Forest metrics
8. Writes `reports/eda_report.md` and figures in `reports/figures/`

## Folder structure
```
data/raw/         original dataset (customers.csv is a synthetic sample)
data/processed/   cleaned dataset
notebooks/        exploration notebooks
src/              reusable code
models/           saved trained models
reports/          EDA report and figures
docs/index.html   project website (GitHub Pages)
```

## Run it
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/customer_purchase_eda.py --csv data/raw/customers.csv --target Purchase
```
Use `--csv your_file.csv --target YourTargetColumn` for your own data. The target column must contain 0 and 1.

> `data/raw/customers.csv` is **synthetic sample data**, not real customer data. Results from it only demonstrate the pipeline.

## Website
Open `docs/index.html`, or enable GitHub Pages (Settings, Pages, branch `main`, folder `/docs`).
