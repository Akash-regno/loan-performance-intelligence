"""
app/pages/03_explainability.py
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Explainability | LPIE", page_icon="🔍", layout="wide")

st.markdown("## 🔍 Explainability — SHAP & FP/FN Analysis")
st.caption("Global feature importance, per-loan waterfall plots, and error analysis.")

@st.cache_data(show_spinner=False)
def load_shap():
    shap_path = Path("outputs/shap/shap_values.csv")
    if shap_path.exists():
        return pd.read_csv(shap_path)
    # Synthetic demo
    np.random.seed(0)
    features = [
        "days_past_due", "dpd_lag1", "dpd_trend_3m", "ltv_mid",
        "credit_score_mid", "pct_balance_remaining", "loan_age_months",
        "n_delinquencies_to_date", "interest_rate", "risk_composite",
        "dti_mid", "dpd_max_6m", "balance_change_1m", "is_seasoned_loan",
        "modification_flag", "reporting_year", "n_modifications_to_date",
        "ever_60dpd", "balance_dpd_risk", "dpd_acceleration",
    ]
    return pd.DataFrame(np.random.randn(200, len(features)) * 0.05, columns=features)

shap_df = load_shap()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌐 Global Importance", "🔬 Per-Loan Waterfall", "⚠️ FP/FN Analysis"])

# ── Tab 1: Global importance ──────────────────────────────────────────────────
with tab1:
    st.markdown("### Global Feature Importance (Mean |SHAP|)")
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(20)
    fig = go.Figure(go.Bar(
        x=mean_abs.values,
        y=mean_abs.index,
        orientation="h",
        marker_color=px.colors.sequential.Plasma[::-1][:len(mean_abs)],
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Top 20 Features by Mean |SHAP| Value",
        xaxis_title="Mean |SHAP|",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show static PNG if available
    png_path = Path("outputs/shap/default_global.png")
    if png_path.exists():
        st.image(str(png_path), caption="SHAP Global Summary (from training)")

# ── Tab 2: Per-loan waterfall ─────────────────────────────────────────────────
with tab2:
    st.markdown("### Per-Loan SHAP Waterfall")
    loan_idx = st.slider("Select loan index", 0, len(shap_df) - 1, 0)
    sv = shap_df.iloc[loan_idx]
    top_n = 12
    top_sv = sv.abs().nlargest(top_n)
    vals = sv[top_sv.index]

    fig2 = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(vals),
        x=vals.values,
        y=vals.index.tolist(),
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        decreasing={"marker": {"color": "#00c851"}},
        increasing={"marker": {"color": "#ff4444"}},
        totals={"marker": {"color": "#a78bfa"}},
    ))
    fig2.update_layout(
        template="plotly_dark",
        title=f"Loan #{loan_idx} — SHAP Contributions (Top {top_n} Features)",
        xaxis_title="SHAP Value (impact on log-odds)",
        height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "🟥 Positive SHAP = increases default probability  |  "
        "🟩 Negative SHAP = decreases default probability"
    )

# ── Tab 3: FP/FN analysis ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### False-Positive / False-Negative Analysis")
    fp_fn_path = Path("outputs/shap/fp_fn_report.xlsx")
    loaded_excel = False
    if fp_fn_path.exists():
        try:
            counts_df = pd.read_excel(fp_fn_path, sheet_name="Counts")
            st.dataframe(counts_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fp_df = pd.read_excel(fp_fn_path, sheet_name="FP_Drivers")
                st.markdown("**Top FP Driver Features**")
                st.dataframe(fp_df, use_container_width=True)
            with col2:
                fn_df = pd.read_excel(fp_fn_path, sheet_name="FN_Drivers")
                st.markdown("**Top FN Driver Features**")
                st.dataframe(fn_df, use_container_width=True)
            loaded_excel = True
        except Exception:
            loaded_excel = False

    if not loaded_excel:
        # Synthetic demo

        st.info("📁 Run training pipeline to generate FP/FN report. Showing demo data:")
        demo = pd.DataFrame({
            "Quadrant": ["TP", "TN", "FP", "FN"],
            "Count": [420, 6900, 280, 400],
            "Description": [
                "True default predicted correctly",
                "True non-default predicted correctly",
                "Non-default incorrectly flagged as default",
                "True default missed by model",
            ],
        })
        fig3 = px.bar(
            demo, x="Quadrant", y="Count", color="Quadrant",
            color_discrete_map={"TP": "#00c851", "TN": "#60a5fa", "FP": "#ffbb33", "FN": "#ff4444"},
            text="Count",
        )
        fig3.update_layout(template="plotly_dark", title="Confusion Quadrant Counts (Demo)")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(demo, use_container_width=True)
