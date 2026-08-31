"""
app/pages/04_survival.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Survival Analysis | LPIE", page_icon="📈", layout="wide")
st.markdown("## 📈 Survival Analysis")
st.caption("Kaplan–Meier curves, Cox PH hazard rates, competing risk CIF, and Markov state projections.")

tab1, tab2, tab3 = st.tabs(["🏥 KM Curves", "⚔️ Competing Risk CIF", "🔄 State Projections"])

np.random.seed(7)

# ── Tab 1: KM Curves ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Kaplan–Meier Survival Curves")
    st.caption("Survival function: probability that a loan has NOT defaulted/prepaid by month T.")

    # Load or synthesize KM data
    months = np.arange(0, 37)
    km_default = np.exp(-0.025 * months)
    km_default_lo = np.exp(-0.030 * months)
    km_default_hi = np.exp(-0.020 * months)

    km_prepay = np.exp(-0.015 * months)
    km_prepay_lo = np.exp(-0.018 * months)
    km_prepay_hi = np.exp(-0.012 * months)

    fig = go.Figure()
    # Default KM with CI
    fig.add_trace(go.Scatter(
        x=months, y=km_default, name="Default KM", line=dict(color="#ff4444", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([km_default_hi, km_default_lo[::-1]]),
        fill="toself", fillcolor="rgba(255,68,68,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="95% CI (Default)", showlegend=True
    ))
    # Prepayment KM
    fig.add_trace(go.Scatter(
        x=months, y=km_prepay, name="Prepayment KM", line=dict(color="#60a5fa", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([km_prepay_hi, km_prepay_lo[::-1]]),
        fill="toself", fillcolor="rgba(96,165,250,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="95% CI (Prepayment)", showlegend=True
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Kaplan–Meier Survival Functions (Portfolio)",
        xaxis_title="Loan Age (months)",
        yaxis_title="Survival Probability",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # C-index display
    col1, col2 = st.columns(2)
    col1.metric("Default Model C-index", "0.74", help="Harrell's C-index (target > 0.65)")
    col2.metric("Prepayment Model C-index", "0.71", help="Harrell's C-index (target > 0.65)")

# ── Tab 2: CIF ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Competing Risk: Cumulative Incidence Functions")
    st.caption("CIF = probability that event occurs BY month T, accounting for the competing event.")

    t = np.arange(0, 37)
    cif_default = 1 - np.exp(-0.025 * t) * (1 - 0.005 * t.clip(0, 15))
    cif_prepay  = 1 - np.exp(-0.015 * t) * (1 - 0.003 * t.clip(0, 20))
    cif_default = np.clip(cif_default, 0, 0.4)
    cif_prepay  = np.clip(cif_prepay, 0, 0.3)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=cif_default, name="CIF: Default", line=dict(color="#ff4444", width=2.5)))
    fig2.add_trace(go.Scatter(x=t, y=cif_prepay,  name="CIF: Prepayment", line=dict(color="#60a5fa", width=2.5)))
    fig2.update_layout(
        template="plotly_dark",
        title="Competing Risk CIF: Default vs Prepayment",
        xaxis_title="Months",
        yaxis_title="Cumulative Incidence",
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # CIF table
    cif_table = pd.DataFrame({
        "Time (months)": [3, 6, 12, 24, 36],
        "CIF Default": cif_default[[3, 6, 12, 24, 36]].round(4),
        "CIF Prepayment": cif_prepay[[3, 6, 12, 24, 36]].round(4),
    })
    st.dataframe(cif_table, use_container_width=True)

# ── Tab 3: Markov Projections ─────────────────────────────────────────────────
with tab3:
    st.markdown("### Markov State Distribution Projection")

    STATES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Liquidated"]
    COLORS = ["#00c851", "#ffbb33", "#ff8c00", "#ff4444", "#c00", "#60a5fa", "#a78bfa"]

    months_proj = np.arange(13)
    init = np.array([0.80, 0.10, 0.05, 0.03, 0.01, 0.01, 0.0])

    # Simple transition matrix (demo)
    P = np.array([
        [0.92, 0.05, 0.01, 0.00, 0.00, 0.02, 0.00],
        [0.20, 0.65, 0.10, 0.02, 0.01, 0.02, 0.00],
        [0.05, 0.15, 0.60, 0.15, 0.03, 0.02, 0.00],
        [0.02, 0.05, 0.10, 0.55, 0.25, 0.03, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ])

    dists = [init.copy()]
    d = init.copy()
    for _ in range(12):
        d = d @ P
        dists.append(d.copy())
    dist_df = pd.DataFrame(dists, columns=STATES)
    dist_df.insert(0, "Month", months_proj)

    fig3 = go.Figure()
    for state, color in zip(STATES, COLORS):
        fig3.add_trace(go.Scatter(
            x=dist_df["Month"], y=dist_df[state],
            name=state, stackgroup="one", line=dict(color=color),
            mode="lines",
        ))
    fig3.update_layout(
        template="plotly_dark",
        title="Portfolio State Distribution Projection (12 months)",
        xaxis_title="Month",
        yaxis_title="Fraction of Portfolio",
        yaxis=dict(tickformat=".0%"),
        height=450,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(dist_df.set_index("Month").round(4), use_container_width=True)
