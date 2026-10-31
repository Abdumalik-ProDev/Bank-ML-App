"""
Preprocessing pipelines and transformations for Bank Credit Default features.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from typing import List

NUMERIC_FEATURES: List[str] = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

CATEGORICAL_FEATURES: List[str] = [
    "SEX", "EDUCATION", "MARRIAGE"
]

PAY_STATUS_FEATURES: List[str] = [
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
]


def create_preprocessor(scale_all: bool = True) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer that scales numeric features
    and handles categoricals and repayment status cleanly.
    """
    if scale_all:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES + PAY_STATUS_FEATURES),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ],
            remainder="drop"
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ("pay", "passthrough", PAY_STATUS_FEATURES),
            ],
            remainder="drop"
        )
    return preprocessor
