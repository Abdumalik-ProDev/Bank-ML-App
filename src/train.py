"""
Multi-Model Training, Evaluation, and Comparison Pipeline.
Trains:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- LightGBM Classifier
- XGBoost Classifier
Saves trained models, evaluation metrics, confusion matrices, and ROC curves.
"""

import os
import sys
import time
import json
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import get_train_test_data, FEATURE_COLUMNS, load_raw_data
from src.preprocessor import create_preprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_definitions():
    """
    Returns dictionary of models to be trained.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=25,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=9,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=3.5,  # accounts for ~1:3.5 class imbalance
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }


def train_and_evaluate_all():
    print("=" * 60)
    print("Starting Multi-Model Training and Benchmarking Pipeline...")
    print("=" * 60)

    # 1. Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    print(f"Default rate in train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

    preprocessor = create_preprocessor(scale_all=True)
    models = get_model_definitions()

    results = {}
    roc_data = {}
    feature_importances = {}

    best_score = -1.0
    recommended_model_name = None

    for name, model in models.items():
        print(f"\n---> Training: {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Measure training time
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        # Measure inference latency on test set
        t1 = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time = (time.time() - t1) / len(X_test) * 1000  # ms per sample

        # Predict probabilities
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute ROC curve coordinates
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        # Downsample ROC curve points to keep JSON lightweight
        stride = max(1, len(fpr) // 100)
        roc_data[name] = {
            "fpr": fpr[::stride].tolist() + [1.0],
            "tpr": tpr[::stride].tolist() + [1.0],
            "auc": float(roc_auc)
        }

        # Model model performance score (composite of ROC-AUC and Balanced Accuracy)
        composite_score = 0.6 * roc_auc + 0.4 * bal_acc
        if composite_score > best_score:
            best_score = composite_score
            recommended_model_name = name

        results[name] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "train_time_sec": round(float(train_time), 3),
            "latency_ms_per_sample": round(float(inference_time), 4),
            "confusion_matrix": cm,
            "composite_score": round(float(composite_score), 4)
        }

        print(f"  Accuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Train Time: {train_time:.2f}s")

        # Save pipeline
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_filepath = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
        joblib.dump(pipeline, model_filepath)

        # Extract feature importance if available
        clf = pipeline.named_steps["classifier"]
        prep = pipeline.named_steps["preprocessor"]
        try:
            cat_encoder = prep.named_transformers_["cat"]
            cat_cols = cat_encoder.get_feature_names_out(["SEX", "EDUCATION", "MARRIAGE"]).tolist()
            num_cols = prep.named_transformers_["num"].feature_names_in_.tolist()
            all_feature_names = num_cols + cat_cols
        except Exception:
            all_feature_names = FEATURE_COLUMNS

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            if len(importances) == len(all_feature_names):
                fi_dict = dict(sorted(zip(all_feature_names, map(float, importances)), key=lambda x: x[1], reverse=True)[:15])
                feature_importances[name] = fi_dict
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            if len(coefs) == len(all_feature_names):
                fi_dict = dict(sorted(zip(all_feature_names, map(float, coefs)), key=lambda x: x[1], reverse=True)[:15])
                feature_importances[name] = fi_dict

    # Save summary metadata
    benchmark_payload = {
        "recommended_model": recommended_model_name,
        "results": results,
        "roc_data": roc_data,
        "feature_importances": feature_importances,
        "feature_columns": FEATURE_COLUMNS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = os.path.join(MODELS_DIR, "benchmark_results.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED! Recommended Model: >>> {recommended_model_name} <<<")
    print(f"Saved artifacts and metrics to {MODELS_DIR}")
    print("=" * 60)
    return benchmark_payload


if __name__ == "__main__":
    train_and_evaluate_all()
