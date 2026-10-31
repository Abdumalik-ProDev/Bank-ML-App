"""
Data loader and preprocessing utilities for the Bank Credit Default dataset.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "default of credit card clients.xls")

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

TARGET_COLUMN = "DEFAULT_PAYMENT"

# Friendly labels for visualizations and forms
FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit (NT$)",
    "SEX": "Gender",
    "EDUCATION": "Education Level",
    "MARRIAGE": "Marital Status",
    "AGE": "Age (Years)",
    "PAY_0": "Repayment Status (Sep)",
    "PAY_2": "Repayment Status (Aug)",
    "PAY_3": "Repayment Status (Jul)",
    "PAY_4": "Repayment Status (Jun)",
    "PAY_5": "Repayment Status (May)",
    "PAY_6": "Repayment Status (Apr)",
    "BILL_AMT1": "Bill Statement Amount (Sep)",
    "BILL_AMT2": "Bill Statement Amount (Aug)",
    "BILL_AMT3": "Bill Statement Amount (Jul)",
    "BILL_AMT4": "Bill Statement Amount (Jun)",
    "BILL_AMT5": "Bill Statement Amount (May)",
    "BILL_AMT6": "Bill Statement Amount (Apr)",
    "PAY_AMT1": "Amount Paid (Sep)",
    "PAY_AMT2": "Amount Paid (Aug)",
    "PAY_AMT3": "Amount Paid (Jul)",
    "PAY_AMT4": "Amount Paid (Jun)",
    "PAY_AMT5": "Amount Paid (May)",
    "PAY_AMT6": "Amount Paid (Apr)",
}

EDUCATION_MAPPING = {
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Others",
    5: "Unknown",
    6: "Unknown",
    0: "Unknown"
}

MARRIAGE_MAPPING = {
    1: "Married",
    2: "Single",
    3: "Others",
    0: "Others"
}

SEX_MAPPING = {
    1: "Male",
    2: "Female"
}

PAY_STATUS_MAPPING = {
    -2: "No consumption / inactive",
    -1: "Paid in full",
    0: "Revolving credit (on-time minimum)",
    1: "Payment delayed by 1 month",
    2: "Payment delayed by 2 months",
    3: "Payment delayed by 3 months",
    4: "Payment delayed by 4 months",
    5: "Payment delayed by 5 months",
    6: "Payment delayed by 6 months",
    7: "Payment delayed by 7 months",
    8: "Payment delayed by 8 months"
}


def load_raw_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads raw Excel file and resolves headers if UCI nested format is used.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    # Inspect first few rows to check whether header is row 0 or 1
    df_sample = pd.read_excel(filepath, nrows=3)
    if "LIMIT_BAL" in df_sample.columns:
        df = pd.read_excel(filepath)
    else:
        # Standard UCI format has metadata in row 0 and headers in row 1
        df = pd.read_excel(filepath, header=1)

    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # In some versions, PAY_0 is named PAY_1
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})

    # Standardize target column
    target_candidates = ["default payment next month", "default_payment_next_month", "Y", "default"]
    for cand in target_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: TARGET_COLUMN})
            break

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Cast to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any nulls if present
    df = df.dropna()

    # Clean categorical groupings
    # EDUCATION: 0, 4, 5, 6 -> 4 (Others/Unknown)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x in [0, 4, 5, 6] else int(x))
    # MARRIAGE: 0, 3 -> 3 (Others)
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 3 if x in [0, 3] else int(x))

    return df


def get_train_test_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads clean data and splits into stratified train and test sets.
    """
    df = load_raw_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
