"""
app/streamlit_app.py
---------------------
Main Streamlit entry point for the Loan Performance Intelligence Engine.

Run with:
    streamlit run app/streamlit_app.py

Pages:
  01_data_quality.py
  02_predictions.py
  03_explainability.py
  04_survival.py
  05_scenarios.py
  06_anomalies.py
  07_llm_copilot.py
"""

import streamlit as st

st.set_page_config(
    page_title="Loan Performance Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark gradient background */
  .main {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
  }
  div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
  }

  /* Status badges */
  .badge-green  { background: #00c851; color: white; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
  .badge-yellow { background: #ffbb33; color: black; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
  .badge-red    { background: #ff4444; color: white; border-radius: 6px; padding: 2px 8px; font-size: 12px; }

  /* Section headers */
  .section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #a78bfa;
    border-bottom: 2px solid #a78bfa;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  /* Cards */
  .info-card {
    background: rgba(167, 139, 250, 0.08);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }

  /* LLM response box */
  .llm-response {
    background: rgba(0, 200, 81, 0.08);
    border: 1px solid rgba(0, 200, 81, 0.3);
    border-radius: 12px;
    padding: 20px;
    font-style: italic;
  }
  .llm-blocked {
    background: rgba(255, 68, 68, 0.08);
    border: 1px solid rgba(255, 68, 68, 0.5);
    border-radius: 12px;
    padding: 20px;
  }

  /* Streamlit buttons */
  .stButton button {
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
  }
  .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(167, 139, 250, 0.4);
  }

  /* Data tables */
  .dataframe { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(109,40,217,0.1));
  border: 1px solid rgba(167,139,250,0.4);
  border-radius: 16px;
  padding: 32px 40px;
  margin-bottom: 32px;
  backdrop-filter: blur(10px);
">
  <h1 style="
    font-family: Inter, sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  ">🏦 Loan Performance Intelligence Engine</h1>
  <p style="color: rgba(255,255,255,0.7); font-size: 1.05rem; margin: 0;">
    Intain Campus FinTech Challenge 2026 — AI Track
    &nbsp;|&nbsp; ML-First Loan Analytics Platform
  </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Guide ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="info-card">
      <h4 style="color:#a78bfa; margin:0 0 8px 0;">📊 Data & Models</h4>
      <p style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin:0;">
        Navigate to <b>01 Data Quality</b> and <b>02 Predictions</b> to explore
        profiling reports and model performance metrics.
      </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
      <h4 style="color:#60a5fa; margin:0 0 8px 0;">🔍 Explainability</h4>
      <p style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin:0;">
        <b>03 Explainability</b> — SHAP waterfall plots, LIME explanations,
        false-positive / false-negative deep-dives.
      </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
      <h4 style="color:#34d399; margin:0 0 8px 0;">📈 Survival & Scenarios</h4>
      <p style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin:0;">
        <b>04 Survival</b> — KM curves, hazard rates, CIF plots.<br>
        <b>05 Scenarios</b> — Base / Adverse / High-Prepay stress tests.
      </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="info-card">
      <h4 style="color:#f59e0b; margin:0 0 8px 0;">🤖 AI Copilot</h4>
      <p style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin:0;">
        <b>06 Anomalies</b> — Flagged records.<br>
        <b>07 LLM Copilot</b> — Grounded reviewer with HITL approve/reject.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("👈 Use the sidebar to navigate between pages. Data files must be placed in `data/raw/` before running.")
