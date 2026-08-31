"""
app/pages/05_scenarios.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Scenarios | LPIE", page_icon="🌡️", layout="wide")
st.markdown("## 🌡️ Scenario Simulation")
st.caption("Base / Adverse / High-Prepayment stress tests with portfolio Expected Loss.")

# ── Sidebar: scenario selector ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scenario Controls")
    scenario = st.selectbox("Scenario", ["Base", "Adverse", "High Prepayment", "Custom"])

    if scenario == "Custom":
        rate_delta = st.slider("Rate Δ (bp)", -300, 500, 0, step=25) / 100
        hpi_delta  = st.slider("HPI Δ (%)", -30, 20, 0) / 100
        unemp_delta = st.slider("Unemployment Δ (pp)", -3, 8, 0)
    else:
        params = {
            "Base":            (0.0,  0.0,  0.0),
            "Adverse":         (3.0, -0.15, 4.0),
            "High Prepayment": (-1.5, 0.10, 0.0),
        }
        rate_delta, hpi_delta, unemp_delta = params[scenario]
        st.info(f"Rate Δ: **{rate_delta:+.1f}%** | HPI Δ: **{hpi_delta:+.0%}** | Unemp Δ: **{unemp_delta:+.1f}pp**")

# ── Load scenario outputs ─────────────────────────────────────────────────────
def compute_scenario(rate_d, hpi_d, unemp_d):
    stress = max(rate_d * 0.03 + max(-hpi_d, 0) * 0.08 + unemp_d * 0.005, 0)
    refi   = max(-rate_d * 0.04, 0)
    return {
        "Default Rate 12m":   round(0.0306 + stress * 0.08, 4),
        "Prepayment Rate 12m": round(0.0939 + refi - stress * 0.05, 4),
        "Delinquency Rate 3m": round(0.1584 + stress * 0.25, 4),
        "Expected Loss ($M)":  round(52.08 + stress * 15, 2),
        "EL Rate (%)":         round(1.190 + stress * 0.4, 3),
    }

saved_scen_path = Path("outputs/scenarios/scenario_comparison.csv")
if saved_scen_path.exists():
    raw_scen = pd.read_csv(saved_scen_path).set_index("scenario")
    results = {
        "Base": {
            "Default Rate 12m": float(raw_scen.loc["base", "default_rate_12m_pct"]) / 100,
            "Prepayment Rate 12m": float(raw_scen.loc["base", "prepayment_rate_12m_pct"]) / 100,
            "Delinquency Rate 3m": float(raw_scen.loc["base", "delinquency_rate_3m_pct"]) / 100,
            "Expected Loss ($M)": round(float(raw_scen.loc["base", "portfolio_el_usd"]) / 1e6, 2),
            "EL Rate (%)": float(raw_scen.loc["base", "portfolio_el_rate_pct"]),
        },
        "Adverse": {
            "Default Rate 12m": float(raw_scen.loc["adverse", "default_rate_12m_pct"]) / 100,
            "Prepayment Rate 12m": float(raw_scen.loc["adverse", "prepayment_rate_12m_pct"]) / 100,
            "Delinquency Rate 3m": float(raw_scen.loc["adverse", "delinquency_rate_3m_pct"]) / 100,
            "Expected Loss ($M)": round(float(raw_scen.loc["adverse", "portfolio_el_usd"]) / 1e6, 2),
            "EL Rate (%)": float(raw_scen.loc["adverse", "portfolio_el_rate_pct"]),
        },
        "High Prepayment": {
            "Default Rate 12m": float(raw_scen.loc["high_prepayment", "default_rate_12m_pct"]) / 100,
            "Prepayment Rate 12m": float(raw_scen.loc["high_prepayment", "prepayment_rate_12m_pct"]) / 100,
            "Delinquency Rate 3m": float(raw_scen.loc["high_prepayment", "delinquency_rate_3m_pct"]) / 100,
            "Expected Loss ($M)": round(float(raw_scen.loc["high_prepayment", "portfolio_el_usd"]) / 1e6, 2),
            "EL Rate (%)": float(raw_scen.loc["high_prepayment", "portfolio_el_rate_pct"]),
        },
    }
else:
    results = {
        "Base":            compute_scenario(0.0,  0.0,  0.0),
        "Adverse":         compute_scenario(3.0, -0.15, 4.0),
        "High Prepayment": compute_scenario(-1.5, 0.10, 0.0),
    }

if scenario == "Custom":
    results["Custom"] = compute_scenario(rate_delta, hpi_delta, unemp_delta)

active = results.get(scenario, results["Base"])

# ── KPI row ───────────────────────────────────────────────────────────────────
base_el = results["Base"]["Expected Loss ($M)"]
delta_el = active["Expected Loss ($M)"] - base_el
delta_color = "inverse" if delta_el > 0 else "normal"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("12m Default Rate",    f"{active['Default Rate 12m']:.2%}", f"{(active['Default Rate 12m'] - results['Base']['Default Rate 12m']):.2%}", delta_color="inverse")
c2.metric("12m Prepayment Rate", f"{active['Prepayment Rate 12m']:.2%}")
c3.metric("3m Delinquency Rate", f"{active['Delinquency Rate 3m']:.2%}", f"{(active['Delinquency Rate 3m'] - results['Base']['Delinquency Rate 3m']):.2%}", delta_color="inverse")
c4.metric("Expected Loss",       f"${active['Expected Loss ($M)']:.1f}M", f"${delta_el:+.1f}M vs Base", delta_color="inverse")
c5.metric("EL Rate",             f"{active['EL Rate (%)']:.3f}%")

st.markdown("---")

# ── Scenario comparison table ─────────────────────────────────────────────────
st.markdown("### Scenario Comparison")
comp_df = pd.DataFrame(results).T
comp_df.index.name = "Scenario"

def color_cells(val):
    if isinstance(val, float) and val > 0.12:
        return "background-color: rgba(255,68,68,0.2)"
    return ""

st.dataframe(
    comp_df.style.map(color_cells),
    use_container_width=True,
)


# ── Portfolio EL waterfall ────────────────────────────────────────────────────
st.markdown("### Portfolio Expected Loss by Scenario")
fig = go.Figure(go.Bar(
    x=list(results.keys()),
    y=[v["Expected Loss ($M)"] for v in results.values()],
    marker_color=["#00c851", "#ff4444", "#60a5fa", "#a78bfa"][:len(results)],
    text=[f"${v['Expected Loss ($M)']:.1f}M" for v in results.values()],
    textposition="auto",
))
fig.update_layout(
    template="plotly_dark",
    title="Expected Loss by Scenario ($M)",
    yaxis_title="Expected Loss ($M)",
    height=380,
)
st.plotly_chart(fig, use_container_width=True)

# ── Transition matrix heatmap ─────────────────────────────────────────────────
st.markdown("### Markov Transition Matrix (Scenario-Shifted)")
STATES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Liquidated"]
P_base = np.array([
    [0.92, 0.05, 0.01, 0.00, 0.00, 0.02, 0.00],
    [0.20, 0.65, 0.10, 0.02, 0.01, 0.02, 0.00],
    [0.05, 0.15, 0.60, 0.15, 0.03, 0.02, 0.00],
    [0.02, 0.05, 0.10, 0.55, 0.25, 0.03, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
# Apply shift
stress = max(rate_delta * 0.03 + max(-hpi_delta, 0) * 0.08 + unemp_delta * 0.005, 0)
P_shifted = P_base.copy()
if stress > 0:
    P_shifted[0, 1] += stress * 0.03; P_shifted[0, 0] -= stress * 0.03
    P_shifted[1, 2] += stress * 0.05; P_shifted[1, 1] -= stress * 0.05
P_shifted = np.clip(P_shifted, 0, 1)
P_shifted /= P_shifted.sum(axis=1, keepdims=True)

fig2 = go.Figure(go.Heatmap(
    z=P_shifted, x=STATES, y=STATES,
    colorscale="Blues", text=P_shifted.round(3),
    texttemplate="%{text}", showscale=True,
))
fig2.update_layout(
    template="plotly_dark",
    title=f"Transition Probability Matrix — {scenario} Scenario",
    xaxis_title="To State", yaxis_title="From State",
    height=450,
)
st.plotly_chart(fig2, use_container_width=True)
