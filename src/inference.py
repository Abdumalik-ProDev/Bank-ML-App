"""
Credit Decision Inference Engine.
Loads trained models, executes risk scoring, renders Approve/Decline decisions,
and computes human-readable explanation factors.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from src.data_loader import FEATURE_COLUMNS, FEATURE_LABELS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
METRICS_PATH = os.path.join(MODELS_DIR, "benchmark_results.json")

_CACHED_MODELS: Dict[str, Any] = {}
_CACHED_BENCHMARK: Optional[Dict[str, Any]] = None


def load_benchmark_metadata() -> Dict[str, Any]:
    """
    Loads saved benchmark metadata.
    """
    global _CACHED_BENCHMARK
    if _CACHED_BENCHMARK is not None:
        return _CACHED_BENCHMARK

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            _CACHED_BENCHMARK = json.load(f)
            return _CACHED_BENCHMARK
    return {}


def load_model_pipeline(model_name: Optional[str] = None):
    """
    Loads requested model pipeline or the recommended model.
    """
    benchmark = load_benchmark_metadata()
    if model_name is None:
        model_name = benchmark.get("recommended_model", "LightGBM")

    clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filepath = os.path.join(MODELS_DIR, f"{clean_name}.joblib")

    if clean_name in _CACHED_MODELS:
        return _CACHED_MODELS[clean_name], model_name

    if not os.path.exists(filepath):
        # Fallback to any available joblib
        joblib_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
        if not joblib_files:
            raise FileNotFoundError("No trained models found. Please run src/train.py first.")
        filepath = os.path.join(MODELS_DIR, joblib_files[0])
        model_name = joblib_files[0].replace(".joblib", "").replace("_", " ").title()

    pipeline = joblib.load(filepath)
    _CACHED_MODELS[clean_name] = pipeline
    return pipeline, model_name


def evaluate_credit_application(
    applicant_data: Dict[str, Any],
    model_name: Optional[str] = None,
    risk_threshold: float = 0.30
) -> Dict[str, Any]:
    """
    Takes applicant data, scores default risk, determines Approve/Decline,
    and returns factors explaining the decision.
    """
    pipeline, resolved_model = load_model_pipeline(model_name)

    # Format input into single-row DataFrame
    row = {}
    for col in FEATURE_COLUMNS:
        row[col] = applicant_data.get(col, 0)
    df_applicant = pd.DataFrame([row])

    # Predict default probability
    if hasattr(pipeline, "predict_proba"):
        proba_default = float(pipeline.predict_proba(df_applicant)[0][1])
    else:
        # Logistic / SVM fallback
        decision_val = float(pipeline.decision_function(df_applicant)[0])
        proba_default = 1.0 / (1.0 + np.exp(-decision_val))

    proba_safe = 1.0 - proba_default
    is_approved = proba_default < risk_threshold

    # Risk tiers
    if proba_default < 0.15:
        risk_tier = "Minimal Risk"
        tier_color = "#10b981"  # green
    elif proba_default < 0.30:
        risk_tier = "Moderate Risk"
        tier_color = "#3b82f6"  # blue
    elif proba_default < 0.50:
        risk_tier = "Elevated Risk"
        tier_color = "#f59e0b"  # amber
    else:
        risk_tier = "Critical Risk"
        tier_color = "#ef4444"  # red

    # Compute key positive and negative risk factors for explainability
    positive_factors: List[str] = []
    risk_factors: List[str] = []

    # 1. Repayment history analysis (most critical predictor)
    recent_delinquency = max(applicant_data.get("PAY_0", 0), applicant_data.get("PAY_2", 0))
    if recent_delinquency <= 0:
        positive_factors.append("Consistent on-time repayment history over the past 2 months (No delays).")
    elif recent_delinquency == 1:
        risk_factors.append("Recent 1-month payment delay detected in the past 60 days.")
    else:
        risk_factors.append(f"Severe delinquency flag: {recent_delinquency} months consecutive payment delay.")

    past_delays = sum(1 for m in ["PAY_3", "PAY_4", "PAY_5", "PAY_6"] if applicant_data.get(m, 0) > 0)
    if past_delays == 0 and recent_delinquency <= 0:
        positive_factors.append("Flawless 6-month payment track record (zero past delinquency).")
    elif past_delays >= 2:
        risk_factors.append(f"Chronic late payments: {past_delays} late payment cycles between 3-6 months ago.")

    # 2. Credit line and utilization ratio
    limit_bal = applicant_data.get("LIMIT_BAL", 10000)
    bill_amt1 = applicant_data.get("BILL_AMT1", 0)
    utilization = (bill_amt1 / limit_bal) if limit_bal > 0 else 1.0

    if utilization < 0.30:
        positive_factors.append(f"Healthy credit utilization ratio ({utilization:.1%} of available credit line).")
    elif utilization > 0.85:
        risk_factors.append(f"High credit utilization warning ({utilization:.1%} of limit in current bill).")

    if limit_bal >= 200000:
        positive_factors.append(f"High credit limit capacity (NT$ {limit_bal:,.0f}).")
    elif limit_bal < 30000:
        risk_factors.append(f"Relatively low established credit limit (NT$ {limit_bal:,.0f}).")

    # 3. Pay vs Bill ratio
    pay_amt1 = applicant_data.get("PAY_AMT1", 0)
    if bill_amt1 > 0:
        pay_ratio = pay_amt1 / bill_amt1
        if pay_ratio >= 0.90:
            positive_factors.append("Paying off substantially all outstanding bill balances each billing cycle.")
        elif pay_ratio < 0.10:
            risk_factors.append("Recent payments cover under 10% of billed balances (high revolving debt).")

    if not positive_factors:
        positive_factors.append("Meets standard baseline criteria.")
    if not risk_factors:
        risk_factors.append("No critical risk red flags detected.")

    decision_text = "APPROVED" if is_approved else "DECLINED"
    action_summary = (
        f"Loan application is **{decision_text}** based on an estimated default probability of "
        f"{proba_default:.1%} (Risk threshold: {risk_threshold:.1%})."
    )

    return {
        "decision": decision_text,
        "is_approved": is_approved,
        "probability_default": round(proba_default, 4),
        "probability_safe": round(proba_safe, 4),
        "risk_percentage": round(proba_default * 100, 1),
        "risk_tier": risk_tier,
        "tier_color": tier_color,
        "risk_threshold": risk_threshold,
        "model_used": resolved_model,
        "summary": action_summary,
        "positive_factors": positive_factors,
        "risk_factors": risk_factors
    }
