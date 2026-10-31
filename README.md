# 🏦 Bank Credit Scoring & Intelligent Decision System (CreditIQ)

An end-to-end Machine Learning solution for credit default risk assessment and automated credit approval decisions, built with Python, `uv`, `scikit-learn`, `xgboost`, `lightgbm`, and `Streamlit`.

---

## 🚀 Key Features

1. **Multi-Model Machine Learning Pipeline**:
   - **Logistic Regression**: Linear baseline with balanced class weights.
   - **K-Nearest Neighbors (KNN)**: Distance-weighted non-parametric classifier.
   - **Decision Tree**: White-box rule-based classifier with tree pruning.
   - **Random Forest**: 150-estimator bagged ensemble.
   - **LightGBM & XGBoost**: State-of-the-art gradient boosted decision trees.
2. **Comprehensive Comparative Benchmarking**:
   - Model Leaderboard comparing **Accuracy**, **ROC-AUC**, **Balanced Accuracy**, **F1-Score**, **Precision**, **Recall**, and **Inference Latency**.
   - Interactive Receiver Operating Characteristic (**ROC Curves**) and Confusion Matrices.
   - Identification and recommendation of the **Top Performing Model**.
3. **Interactive Underwriting Simulator (`app.py`)**:
   - Real-time loan decision engine (**CREDIT APPROVED** vs. **CREDIT DECLINED**).
   - Dynamic risk tolerance cut-off slider.
   - Visual gauge meter (0–100% risk probability).
   - Human-readable explanation factors (favorable traits vs delinquency flags).
   - Pre-configured applicant profiles (Prime Applicant, Delinquent Risk, Young Graduate).
4. **Exploratory Data Analysis (EDA)**:
   - Visual distribution of defaults across credit limits, payment status history, and age demographics.

---

## 🛠️ Package Management with `uv`

All dependencies are strictly managed with [`uv`](https://docs.astral.sh/uv/).

### 1. Install & Sync Dependencies
```bash
uv sync
```

### 2. Run Data Pipeline & Train Models
```bash
uv run python src/train.py
```
This trains all 6 models, performs stratified 5-fold evaluation, logs metrics to `models/benchmark_results.json`, and exports model binaries (`.joblib`).

### 3. Launch Interactive Web Dashboard
```bash
uv run streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
Bank-ML-Project/
├── app.py                     # Interactive Streamlit Web Application
├── pyproject.toml             # uv project specification and dependencies
├── README.md                  # Project documentation & run guide
├── data/
│   └── default of credit card clients.xls # UCI Credit Card Dataset (30,000 records)
├── models/                    # Exported model pipelines & benchmark metrics (.json, .joblib)
└── src/
    ├── __init__.py
    ├── data_loader.py         # Data loading, cleaning, & stratified train/test splits
    ├── preprocessor.py        # ColumnTransformer, scaling, & encoding
    ├── train.py               # Multi-model training & benchmarking script
    └── inference.py           # Real-time credit risk scoring & explanation engine
```
