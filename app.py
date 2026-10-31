"""
Bank Credit Scoring & Automated Decision Dashboard.
Interactive Streamlit application featuring model benchmarking,
exploratory data analysis, model explanations, and real-time loan approval simulator.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_raw_data,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    EDUCATION_MAPPING,
    MARRIAGE_MAPPING,
    SEX_MAPPING,
    PAY_STATUS_MAPPING
)
from src.inference import evaluate_credit_application, load_benchmark_metadata, load_model_pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="CreditIQ - Bank Credit Risk & Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-approved {
        background-color: #dcfce7;
        color: #15803d;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.8rem 1.6rem;
        border-radius: 12px;
        border: 2px solid #86efac;
        display: inline-block;
        text-align: center;
    }
    .badge-declined {
        background-color: #fee2e2;
        color: #b91c1c;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.8rem 1.6rem;
        border-radius: 12px;
        border: 2px solid #fca5a5;
        display: inline-block;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_raw_data():
    return load_raw_data()


def main():
    st.markdown('<div class="main-header">🏦 CreditIQ: Intelligent Bank Credit Scoring & Decision Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Model Comparative Benchmarking (KNN, Random Forest, Decision Tree, XGBoost, LightGBM) with Real-Time Credit Underwriting Simulator</div>', unsafe_allow_html=True)

    # Load metadata and benchmark
    benchmark = load_benchmark_metadata()
    has_benchmark = bool(benchmark and "results" in benchmark)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
        st.title("Credit Risk Center")
        st.write("Predict default risk on credit card accounts using machine learning.")
        st.divider()

        if has_benchmark:
            rec_model = benchmark.get("recommended_model", "N/A")
            best_acc = benchmark["results"].get(rec_model, {}).get("accuracy", 0.0)
            best_auc = benchmark["results"].get(rec_model, {}).get("roc_auc", 0.0)
            st.success(f"⭐ **Top Recommended Model**: {rec_model}")
            st.metric("Top Model Accuracy", f"{best_acc:.2%}")
            st.metric("Top Model ROC-AUC", f"{best_auc:.3f}")
        else:
            st.warning("Models not yet trained or benchmark not found.")

        st.divider()
        st.caption("UCI Default of Credit Card Clients Dataset (30,000 cases)")
        st.caption("Powered by `uv`, `scikit-learn`, `xgboost`, `lightgbm`, & `Streamlit`.")

    # Main Tabs
    tab_sim, tab_bench, tab_explain, tab_eda, tab_deploy = st.tabs([
        "💳 Live Credit Decision Simulator",
        "🏆 Model Comparison & Leaderboard",
        "🔍 Model Deep-Dives & Explanations",
        "📊 Exploratory Data Analysis (EDA)",
        "🌐 Share & Deployment Guide"
    ])

    # -------------------------------------------------------------
    # TAB 1: Live Credit Decision Simulator
    # -------------------------------------------------------------
    with tab_sim:
        st.subheader("Interactive Client Evaluation: Approve or Decline Credit")
        st.write("Enter applicant financial indicators or select a preset profile to test real-time underwriting decisions.")

        col_preset, col_model_sel, col_thresh = st.columns([1.5, 1.2, 1.3])
        with col_preset:
            preset = st.selectbox(
                "⚡ Quick Load Preset Profile:",
                [
                    "Custom Input",
                    "Profile A: Prime Applicant (Strong history, High limit)",
                    "Profile B: Delinquent / Risky Applicant (Past delays, High debt)",
                    "Profile C: Young Graduate (Moderate limit, Clean record)"
                ]
            )

        with col_model_sel:
            available_models = list(benchmark.get("results", {}).keys()) if has_benchmark else [
                "LightGBM", "Random Forest", "Decision Tree", "K-Nearest Neighbors (KNN)", "Logistic Regression", "XGBoost"
            ]
            default_ix = 0
            if has_benchmark and benchmark.get("recommended_model") in available_models:
                default_ix = available_models.index(benchmark.get("recommended_model"))
            selected_model = st.selectbox("🤖 Decision Model:", available_models, index=default_ix)

        with col_thresh:
            threshold = st.slider(
                "🎯 Risk Approval Cut-off (%):",
                min_value=10, max_value=60, value=30, step=5,
                help="If predicted default risk is below this threshold, the applicant is APPROVED. Otherwise, DECLINED."
            ) / 100.0

        # Preset Defaults
        defaults = {
            "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 32,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 25000, "BILL_AMT2": 24000, "BILL_AMT3": 23000, "BILL_AMT4": 20000, "BILL_AMT5": 19000, "BILL_AMT6": 18000,
            "PAY_AMT1": 25000, "PAY_AMT2": 24000, "PAY_AMT3": 23000, "PAY_AMT4": 20000, "PAY_AMT5": 19000, "PAY_AMT6": 18000
        }

        if preset.startswith("Profile A"):
            defaults.update({
                "LIMIT_BAL": 350000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 40,
                "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                "BILL_AMT1": 35000, "BILL_AMT2": 32000, "BILL_AMT3": 28000, "BILL_AMT4": 22000, "BILL_AMT5": 20000, "BILL_AMT6": 15000,
                "PAY_AMT1": 35000, "PAY_AMT2": 32000, "PAY_AMT3": 28000, "PAY_AMT4": 22000, "PAY_AMT5": 20000, "PAY_AMT6": 15000
            })
        elif preset.startswith("Profile B"):
            defaults.update({
                "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 28,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 48000, "BILL_AMT2": 49000, "BILL_AMT3": 47000, "BILL_AMT4": 46000, "BILL_AMT5": 44000, "BILL_AMT6": 42000,
                "PAY_AMT1": 1000, "PAY_AMT2": 0, "PAY_AMT3": 1500, "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
            })
        elif preset.startswith("Profile C"):
            defaults.update({
                "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 24,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 12000, "BILL_AMT4": 11000, "BILL_AMT5": 9000, "BILL_AMT6": 8000,
                "PAY_AMT1": 15000, "PAY_AMT2": 14000, "PAY_AMT3": 12000, "PAY_AMT4": 11000, "PAY_AMT5": 9000, "PAY_AMT6": 8000
            })

        # Form layout
        with st.form("applicant_evaluation_form"):
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=int(defaults["LIMIT_BAL"]), step=10000)
            with col_d2:
                age = st.slider("Age (Years)", min_value=18, max_value=80, value=int(defaults["AGE"]))
            with col_d3:
                gender_idx = 1 if defaults["SEX"] == 2 else 0
                gender = st.selectbox("Gender", ["Male", "Female"], index=gender_idx)
                sex_val = 2 if gender == "Female" else 1
            with col_d4:
                edu_map_inv = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
                default_edu_name = "University"
                for k, v in edu_map_inv.items():
                    if v == defaults["EDUCATION"]:
                        default_edu_name = k
                edu_sel = st.selectbox("Education Level", list(edu_map_inv.keys()), index=list(edu_map_inv.keys()).index(default_edu_name))
                education_val = edu_map_inv[edu_sel]

            col_m1, col_m2 = st.columns([1, 3])
            with col_m1:
                mar_map_inv = {"Married": 1, "Single": 2, "Others": 3}
                default_mar_name = "Single"
                for k, v in mar_map_inv.items():
                    if v == defaults["MARRIAGE"]:
                        default_mar_name = k
                mar_sel = st.selectbox("Marital Status", list(mar_map_inv.keys()), index=list(mar_map_inv.keys()).index(default_mar_name))
                marriage_val = mar_map_inv[mar_sel]

            st.write("---")
            st.markdown("##### 📅 Repayment History (Past 6 Months)")
            st.caption("Values: -1 = Paid in full, 0 = Revolving credit (minimum paid), 1 = 1 month delay, 2 = 2 months delay, etc.")

            col_p0, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
            with col_p0:
                pay_0 = st.slider("Month 1 (Sep)", -2, 8, int(defaults["PAY_0"]))
            with col_p2:
                pay_2 = st.slider("Month 2 (Aug)", -2, 8, int(defaults["PAY_2"]))
            with col_p3:
                pay_3 = st.slider("Month 3 (Jul)", -2, 8, int(defaults["PAY_3"]))
            with col_p4:
                pay_4 = st.slider("Month 4 (Jun)", -2, 8, int(defaults["PAY_4"]))
            with col_p5:
                pay_5 = st.slider("Month 5 (May)", -2, 8, int(defaults["PAY_5"]))
            with col_p6:
                pay_6 = st.slider("Month 6 (Apr)", -2, 8, int(defaults["PAY_6"]))

            st.write("---")
            st.markdown("##### 💵 Bill Statements & Payments (Recent 3 Months)")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                bill_1 = st.number_input("Bill Sept (NT$)", min_value=0, max_value=2000000, value=int(defaults["BILL_AMT1"]), step=1000)
                pay_1 = st.number_input("Paid Sept (NT$)", min_value=0, max_value=2000000, value=int(defaults["PAY_AMT1"]), step=1000)
            with col_b2:
                bill_2 = st.number_input("Bill Aug (NT$)", min_value=0, max_value=2000000, value=int(defaults["BILL_AMT2"]), step=1000)
                pay_2_val = st.number_input("Paid Aug (NT$)", min_value=0, max_value=2000000, value=int(defaults["PAY_AMT2"]), step=1000)
            with col_b3:
                bill_3 = st.number_input("Bill Jul (NT$)", min_value=0, max_value=2000000, value=int(defaults["BILL_AMT3"]), step=1000)
                pay_3_val = st.number_input("Paid Jul (NT$)", min_value=0, max_value=2000000, value=int(defaults["PAY_AMT3"]), step=1000)

            submitted = st.form_submit_button("🔍 Evaluate Credit Application", use_container_width=True)

        if submitted:
            applicant_payload = {
                "LIMIT_BAL": limit_bal, "SEX": sex_val, "EDUCATION": education_val, "MARRIAGE": marriage_val, "AGE": age,
                "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
                "BILL_AMT1": bill_1, "BILL_AMT2": bill_2, "BILL_AMT3": bill_3,
                "BILL_AMT4": defaults["BILL_AMT4"], "BILL_AMT5": defaults["BILL_AMT5"], "BILL_AMT6": defaults["BILL_AMT6"],
                "PAY_AMT1": pay_1, "PAY_AMT2": pay_2_val, "PAY_AMT3": pay_3_val,
                "PAY_AMT4": defaults["PAY_AMT4"], "PAY_AMT5": defaults["PAY_AMT5"], "PAY_AMT6": defaults["PAY_AMT6"]
            }

            try:
                res = evaluate_credit_application(applicant_payload, model_name=selected_model, risk_threshold=threshold)

                st.write("")
                st.write("---")
                st.subheader("📋 Decision Summary & Risk Report")

                col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 2])
                with col_res1:
                    st.write("**Automated Underwriting Decision:**")
                    if res["is_approved"]:
                        st.markdown('<div class="badge-approved">✅ CREDIT APPROVED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="badge-declined">❌ CREDIT DECLINED</div>', unsafe_allow_html=True)
                    st.caption(f"Evaluated with **{res['model_used']}**")

                with col_res2:
                    st.metric("Estimated Default Risk", f"{res['risk_percentage']}%", delta=f"{res['risk_percentage'] - threshold*100:+.1f}% vs Cut-off", delta_color="inverse")
                    st.markdown(f"**Risk Tier:** <span style='color:{res['tier_color']}; font-weight:bold; font-size:1.1rem;'>{res['risk_tier']}</span>", unsafe_allow_html=True)

                with col_res3:
                    # Risk Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=res["risk_percentage"],
                        title={'text': "Risk Score Meter (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1e293b"},
                            'steps': [
                                {'range': [0, 20], 'color': "#86efac"},
                                {'range': [20, 35], 'color': "#93c5fd"},
                                {'range': [35, 60], 'color': "#fde047"},
                                {'range': [60, 100], 'color': "#fca5a5"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.8,
                                'value': threshold * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=15, r=15, t=30, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Explanations
                st.write("---")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown("##### 🟢 Favorable Factors (Strengths):")
                    for factor in res["positive_factors"]:
                        st.markdown(f"- ✅ {factor}")

                with col_exp2:
                    st.markdown("##### 🔴 Risk Flags & Delinquency Drivers:")
                    for factor in res["risk_factors"]:
                        st.markdown(f"- ⚠️ {factor}")

            except Exception as e:
                st.error(f"Error evaluating applicant: {e}")
                st.info("Tip: Ensure model files are built by running the training pipeline.")

    # -------------------------------------------------------------
    # TAB 2: Model Comparison & Leaderboard
    # -------------------------------------------------------------
    with tab_bench:
        st.subheader("Model Benchmark Leaderboard & Comparative Evaluation")
        st.write("Compare performance metrics across all 6 evaluated machine learning models.")

        if not has_benchmark:
            st.info("Benchmark data will appear here once models are trained.")
        else:
            rec_model = benchmark.get("recommended_model", "LightGBM")
            st.info(f"🏆 **Recommended Model**: **{rec_model}**. It delivers the optimal trade-off between **High Accuracy**, **ROC-AUC discriminatory power**, and **Robust generalization** on imbalanced credit default data.")

            results = benchmark["results"]
            df_metrics = pd.DataFrame.from_dict(results, orient="index")
            df_metrics.index.name = "Model"
            df_metrics = df_metrics.reset_index()

            # Format metrics
            display_df = df_metrics.copy()
            display_df["accuracy"] = display_df["accuracy"].map(lambda x: f"{x:.2%}")
            display_df["roc_auc"] = display_df["roc_auc"].map(lambda x: f"{x:.4f}")
            display_df["balanced_accuracy"] = display_df["balanced_accuracy"].map(lambda x: f"{x:.2%}")
            display_df["precision"] = display_df["precision"].map(lambda x: f"{x:.2%}")
            display_df["recall"] = display_df["recall"].map(lambda x: f"{x:.2%}")
            display_df["f1_score"] = display_df["f1_score"].map(lambda x: f"{x:.4f}")
            display_df["train_time_sec"] = display_df["train_time_sec"].map(lambda x: f"{x:.2f}s")
            display_df["latency_ms_per_sample"] = display_df["latency_ms_per_sample"].map(lambda x: f"{x:.3f} ms")

            st.dataframe(
                display_df[[
                    "Model", "accuracy", "roc_auc", "balanced_accuracy", "f1_score",
                    "precision", "recall", "train_time_sec", "latency_ms_per_sample"
                ]].sort_values(by="roc_auc", ascending=False),
                use_container_width=True
            )

            # Interactive Plotly Comparison Bar Chart
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_acc = px.bar(
                    df_metrics,
                    x="Model",
                    y=["accuracy", "balanced_accuracy"],
                    barmode="group",
                    title="Accuracy vs Balanced Accuracy Across Models",
                    labels={"value": "Score (0 - 1.0)", "variable": "Metric"},
                    color_discrete_sequence=["#2563eb", "#60a5fa"]
                )
                fig_acc.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_chart2:
                fig_auc = px.bar(
                    df_metrics,
                    x="Model",
                    y="roc_auc",
                    color="roc_auc",
                    title="Discriminatory Power: ROC-AUC Ranking",
                    color_continuous_scale="Viridis",
                    labels={"roc_auc": "ROC-AUC Score"}
                )
                fig_auc.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_auc, use_container_width=True)

            # Combined ROC Curves
            st.write("---")
            st.markdown("##### 📈 Receiver Operating Characteristic (ROC) Curves")
            roc_data = benchmark.get("roc_data", {})
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random Chance (AUC=0.50)"))

            for m_name, m_roc in roc_data.items():
                fig_roc.add_trace(go.Scatter(
                    x=m_roc["fpr"],
                    y=m_roc["tpr"],
                    mode="lines",
                    name=f"{m_name} (AUC = {m_roc['auc']:.3f})"
                ))

            fig_roc.update_layout(
                title="ROC Curves Comparison (Higher curve = Superior risk separation)",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity / Recall)",
                height=450
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            # Confusion Matrix Inspector
            st.write("---")
            st.markdown("##### 🔲 Confusion Matrix Breakdown")
            sel_cm_model = st.selectbox("Inspect Confusion Matrix for Model:", list(results.keys()), index=0)
            cm_data = results[sel_cm_model]["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm_data,
                index=["Actual Non-Default (0)", "Actual Default (1)"],
                columns=["Predicted Non-Default (0)", "Predicted Default (1)"]
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix: {sel_cm_model}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Model Deep-Dives & Explanations
    # -------------------------------------------------------------
    with tab_explain:
        st.subheader("Model Architectural Deep-Dives & Feature Importances")
        st.write("Understand the mechanics, decision logic, and primary drivers behind each machine learning model.")

        models_available = list(benchmark.get("results", {}).keys()) if has_benchmark else ["LightGBM", "Random Forest"]
        dive_model = st.selectbox("Select Model to Inspect:", models_available)

        col_desc, col_fi = st.columns([1.2, 1.8])

        with col_desc:
            st.markdown(f"#### ⚙️ Architecture: {dive_model}")
            if "KNN" in dive_model:
                st.markdown("""
                - **Type**: Non-parametric instance-based algorithm.
                - **Mechanism**: Classifies an applicant based on the distance (Euclidean metric) to the *k* closest historical borrowers in feature space.
                - **Strengths**: Highly intuitive, makes zero assumptions about underlying data distribution.
                - **Weaknesses**: Sensitive to the curse of dimensionality and feature scaling; inference latency grows linearly with training size.
                """)
            elif "Decision Tree" in dive_model:
                st.markdown("""
                - **Type**: Single recursive binary partitioning tree.
                - **Mechanism**: Splits applicants iteratively on optimal feature thresholds (e.g., `PAY_0 > 1.5`) using Gini Impurity reduction.
                - **Strengths**: 100% white-box explainability; ideal for regulatory compliance and audit trails.
                - **Weaknesses**: Prone to high variance and overfitting without depth constraints.
                """)
            elif "Random Forest" in dive_model:
                st.markdown("""
                - **Type**: Bagged ensemble of decorrelated decision trees.
                - **Mechanism**: Aggregates predictions across 150 diverse trees trained on bootstrap samples and random feature subsets.
                - **Strengths**: Drastically reduces variance, captures non-linear interactions, resists overfitting.
                - **Weaknesses**: Slower training and larger model footprint than individual trees.
                """)
            elif "Logistic" in dive_model:
                st.markdown("""
                - **Type**: Generalized linear model with log-odds sigmoid link.
                - **Mechanism**: Fits linear hyperplanes estimating log-odds of credit default.
                - **Strengths**: Highly fast, stable, and widely established as banking standard baseline.
                - **Weaknesses**: Cannot easily capture non-linear interactions without manual feature engineering.
                """)
            elif "LightGBM" in dive_model or "XGBoost" in dive_model:
                st.markdown("""
                - **Type**: Gradient Boosted Decision Tree (GBDT) ensemble.
                - **Mechanism**: Sequentially trains shallow trees, where each subsequent tree corrects the residual errors of preceding trees using gradient descent on the loss function.
                - **Strengths**: Industry gold standard for tabular financial data, handles imbalanced classes with ease, superior accuracy and ROC-AUC.
                - **Weaknesses**: Requires careful hyperparameter tuning to prevent overfitting on noisy data.
                """)

        with col_fi:
            feature_imp_all = benchmark.get("feature_importances", {}) if has_benchmark else {}
            if dive_model in feature_imp_all:
                fi_data = feature_imp_all[dive_model]
                df_fi = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                fig_fi = px.bar(
                    df_fi,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Top 15 Most Influential Features ({dive_model})",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info(f"Feature importance visualization is not directly applicable for {dive_model} (distance-based instance model).")

    # -------------------------------------------------------------
    # TAB 4: Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    with tab_eda:
        st.subheader("Exploratory Data Analysis: Credit Card Default Patterns")
        try:
            df = get_cached_raw_data()
            st.write(f"Dataset contains **{len(df):,} credit card clients** with **{df.shape[1]} financial attributes**.")

            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                # Target distribution
                df_target = df["DEFAULT_PAYMENT"].value_counts().reset_index()
                df_target.columns = ["Default Status", "Count"]
                df_target["Status"] = df_target["Default Status"].map({0: "Non-Default (Paid)", 1: "Default (Unpaid)"})
                fig_target = px.pie(
                    df_target,
                    values="Count",
                    names="Status",
                    title="Class Distribution: Target Variable (Default)",
                    color="Status",
                    color_discrete_map={"Non-Default (Paid)": "#3b82f6", "Default (Unpaid)": "#ef4444"},
                    hole=0.4
                )
                st.plotly_chart(fig_target, use_container_width=True)

            with col_eda2:
                # Credit Limit vs Default
                fig_limit = px.histogram(
                    df,
                    x="LIMIT_BAL",
                    color="DEFAULT_PAYMENT",
                    barmode="overlay",
                    nbins=40,
                    title="Credit Limit Distribution by Default Outcome",
                    labels={"LIMIT_BAL": "Credit Limit (NT$)", "DEFAULT_PAYMENT": "Default"},
                    color_discrete_map={0: "#3b82f6", 1: "#ef4444"}
                )
                st.plotly_chart(fig_limit, use_container_width=True)

            col_eda3, col_eda4 = st.columns(2)
            with col_eda3:
                # Payment Status 0 vs Default Rate
                pay_risk = df.groupby("PAY_0")["DEFAULT_PAYMENT"].agg(["mean", "count"]).reset_index()
                pay_risk = pay_risk[pay_risk["count"] > 30]  # Filter small edge bins
                pay_risk["Default Rate (%)"] = pay_risk["mean"] * 100
                fig_pay = px.bar(
                    pay_risk,
                    x="PAY_0",
                    y="Default Rate (%)",
                    title="Default Rate vs. Recent Repayment Status (PAY_0)",
                    labels={"PAY_0": "Months Delayed (-1: On Time, 1+: Delayed)"},
                    color="Default Rate (%)",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_pay, use_container_width=True)

            with col_eda4:
                # Age distribution
                df["Age Group"] = pd.cut(df["AGE"], bins=[18, 25, 35, 50, 65, 90], labels=["18-25", "26-35", "36-50", "51-65", "65+"])
                age_risk = df.groupby("Age Group")["DEFAULT_PAYMENT"].mean().reset_index()
                age_risk["Default Rate (%)"] = age_risk["DEFAULT_PAYMENT"] * 100
                fig_age = px.bar(
                    age_risk,
                    x="Age Group",
                    y="Default Rate (%)",
                    title="Default Rate Across Age Demographics",
                    color="Default Rate (%)",
                    color_continuous_scale="Oranges"
                )
                st.plotly_chart(fig_age, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading EDA data: {e}")

    # -------------------------------------------------------------
    # TAB 5: Share & Deployment Guide
    # -------------------------------------------------------------
    with tab_deploy:
        st.subheader("Sharing & Hosting the Application")
        st.markdown("""
        ### How to Share This Interactive Application:

        #### Option 1: Streamlit Community Cloud (Recommended & Free)
        1. Push this repository to GitHub:
           ```bash
           git init && git add . && git commit -m "Bank ML Credit Scoring Project"
           git push origin main
           ```
        2. Visit [share.streamlit.io](https://share.streamlit.io).
        3. Connect your repository and select `app.py` as entry point.
        4. Your interactive credit scoring app will be live with a public shareable URL!

        #### Option 2: Run Locally or on Internal Network
        Launch the server accessible across your local Wi-Fi or office network:
        ```bash
        uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
        ```

        #### Option 3: Quick Public Tunnel (ngrok / Cloudflare Tunnel)
        To share with someone outside your network instantly without full deployment:
        ```bash
        # Terminal 1: Run Streamlit
        uv run streamlit run app.py --server.port=8501

        # Terminal 2: Tunnel port 8501
        npx localtunnel --port 8501
        # or: ngrok http 8501
        ```
        """)


if __name__ == "__main__":
    main()
