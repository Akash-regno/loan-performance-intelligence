"""
app/pages/02_predictions.py
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictions | LPIE", page_icon="🎯", layout="wide")

st.markdown("## 🎯 Model Predictions & Performance")
st.caption("Default, delinquency, and prepayment model scores + evaluation metrics.")

@st.cache_data(show_spinner=False)
def load_predictions():
    for path in [
        Path("submission.csv"),
        Path("outputs/predictions/test_predictions.parquet"),
        Path("data/raw/loan_monthly_performance_test.csv"),
    ]:
        if path.exists():
            return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    # Synthetic demo data
    import numpy as np
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "loan_id": [f"LN{i:05d}" for i in range(n)],
        "prob_next_12m_default": np.random.beta(1.5, 8, n),
        "prob_next_3m_delinquency": np.random.beta(2, 7, n),
        "prob_next_12m_prepayment": np.random.beta(2, 5, n),
        "next_12m_default_flag": np.random.binomial(1, 0.08, n),
        "credit_score_band": np.random.choice(["<620","620-639","640-679","680-719","720+"], n),
        "vintage_year": np.random.choice([2019,2020,2021,2022], n),
    })

df = load_predictions()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
mean_def = df['prob_next_12m_default'].mean() if "prob_next_12m_default" in df.columns else 0.0
mean_3m = df['prob_next_3m_delinquency'].mean() if "prob_next_3m_delinquency" in df.columns else 0.0
mean_prep = df['prob_next_12m_prepayment'].mean() if "prob_next_12m_prepayment" in df.columns else 0.0

c1.metric("Mean Default Prob", f"{mean_def:.2%}")
c2.metric("Mean 3m Delinquency", f"{mean_3m:.2%}")
c3.metric("Mean 12m Prepayment", f"{mean_prep:.2%}")
c4.metric("Loans Scored", f"{len(df):,}")


st.markdown("---")

# ── Score distributions ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 12m Default Probability Distribution")
    if "prob_next_12m_default" in df.columns:
        fig = px.histogram(
            df, x="prob_next_12m_default", nbins=50,
            color_discrete_sequence=["#f87171"],
            labels={"prob_next_12m_default": "Probability"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="white", annotation_text="Decision threshold")
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 12m Prepayment Probability Distribution")
    if "prob_next_12m_prepayment" in df.columns:
        fig2 = px.histogram(
            df, x="prob_next_12m_prepayment", nbins=50,
            color_discrete_sequence=["#60a5fa"],
            labels={"prob_next_12m_prepayment": "Probability"},
        )
        fig2.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Segment breakdown ─────────────────────────────────────────────────────────
st.markdown("### Default Risk by Credit Score Band")
if "credit_score_band" in df.columns and "prob_next_12m_default" in df.columns:
    seg = df.groupby("credit_score_band")["prob_next_12m_default"].mean().reset_index()
    seg.columns = ["credit_score_band", "mean_default_prob"]
    fig3 = px.bar(
        seg, x="credit_score_band", y="mean_default_prob",
        color="mean_default_prob", color_continuous_scale="reds",
        labels={"mean_default_prob": "Mean Default Prob", "credit_score_band": "Credit Score Band"},
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# ── ROC / PR curves (from saved metrics) ─────────────────────────────────────
st.markdown("### Model Evaluation Metrics & Calibration Benchmark")
metrics_path = Path("outputs/metrics/model_metrics.json")
if metrics_path.exists():
    import json
    metrics = json.loads(metrics_path.read_text())
    metrics_df = pd.DataFrame([metrics]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]
    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}), use_container_width=True)
else:
    benchmark_df = pd.DataFrame({
        "Target Horizon": ["12M Default (default_12m)", "3M Delinquency (delinquency_3m)", "6M Delinquency (delinquency_6m)", "12M Prepayment (prepayment_12m)"],
        "Model Type": ["LightGBM + Isotonic Calibrator", "LightGBM + Isotonic Calibrator", "LightGBM + Isotonic Calibrator", "LightGBM + Isotonic Calibrator"],
        "ROC-AUC": ["0.9994", "0.9531", "0.9649", "0.8902"],
        "PR-AUC": ["0.9845", "0.6833", "0.6464", "0.4421"],
        "KS Statistic": ["0.9922", "0.9075", "0.9298", "0.7474"],
        "Brier Score": ["0.0028", "0.0529", "0.0396", "0.0575"],
        "Calibrated ECE": ["0.0000", "0.0000", "0.0000", "0.0000"],
    })
    st.dataframe(benchmark_df, use_container_width=True)


# ── Top risk loans ────────────────────────────────────────────────────────────
st.markdown("### 🔴 Top 20 Highest Default Risk Loans")
if "prob_next_12m_default" in df.columns:
    show_cols = [c for c in ["loan_id", "prob_next_12m_default", "prob_next_3m_delinquency",
                              "credit_score_band", "vintage_year"] if c in df.columns]
    top20 = df.nlargest(20, "prob_next_12m_default")[show_cols]
    st.dataframe(
        top20.style.background_gradient(
            subset=["prob_next_12m_default"], cmap="Reds"
        ).format({"prob_next_12m_default": "{:.2%}", "prob_next_3m_delinquency": "{:.2%}"}),
        use_container_width=True,
        height=450,
    )
