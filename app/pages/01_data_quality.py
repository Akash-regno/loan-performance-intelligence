"""
app/pages/01_data_quality.py
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Data Quality | LPIE", page_icon="📊", layout="wide")

st.markdown("## 📊 Data Quality & Profiling")
st.caption("Missing values, outliers, DQ scores, and validation rule violations.")

@st.cache_data(show_spinner=False)
def load_data():
    for path in [
        Path("data/processed/loan_panel_clean.parquet"),
        Path("data/raw/loan_monthly_performance_train.csv"),
        Path("data/raw/loan_monthly_performance.csv"),
        Path("data/raw/loan_monthly_performance_test.csv"),
        Path("submission.csv"),
    ]:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, nrows=10000)
    # Synthetic demo data for standalone preview
    import numpy as np
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "loan_id": [f"LN{i:05d}" for i in range(n)],
        "month_index": np.random.randint(1, 36, n),
        "loan_age_months": np.random.randint(1, 48, n),
        "days_past_due": np.random.exponential(10, n).astype(int),
        "credit_score_band": np.random.choice(["<620","620-639","640-679","680-719","720+"], n),
        "ltv_band": np.random.choice(["0-60","60-70","70-80","80-90","90+"], n),
        "current_status": np.random.choice(["Current","30DPD","60DPD","Default","Prepaid"], n),
        "current_balance": np.random.uniform(50000, 450000, n),
        "dq_band": np.random.choice(["green","amber","red"], n, p=[0.85, 0.10, 0.05]),
    })

df = load_data()



# ── KPI row ──────────────────────────────────────────────────────────────────
total_rows = len(df)
missing_pct = round(100 * df.isna().mean().mean(), 2)
n_loans = df["loan_id"].nunique() if "loan_id" in df.columns else "—"
dq_green = round(100 * (df["dq_band"] == "green").mean(), 1) if "dq_band" in df.columns else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{total_rows:,}")
c2.metric("Unique Loans", f"{n_loans:,}" if isinstance(n_loans, int) else n_loans)
c3.metric("Overall Missing %", f"{missing_pct}%")
c4.metric("DQ Score Green %", f"{dq_green}%" if isinstance(dq_green, float) else dq_green)

st.markdown("---")

# ── Missing values ────────────────────────────────────────────────────────────
st.markdown("### Missing Values by Column")
missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]

if not missing.empty:
    fig = px.bar(
        x=missing.values * 100,
        y=missing.index,
        orientation="h",
        labels={"x": "Missing %", "y": "Column"},
        color=missing.values * 100,
        color_continuous_scale="reds",
        title="Missing Value Rate per Column",
    )
    fig.update_layout(template="plotly_dark", height=max(300, len(missing) * 22))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ No missing values detected.")

# ── DQ Score distribution ─────────────────────────────────────────────────────
if "dq_score" in df.columns:
    st.markdown("### DQ Score Distribution")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig2 = px.histogram(
            df, x="dq_score", nbins=50, color_discrete_sequence=["#a78bfa"],
            title="Per-Record DQ Score Distribution",
            labels={"dq_score": "DQ Score (0–100)"},
        )
        fig2.add_vline(x=70, line_dash="dash", line_color="#ffbb33", annotation_text="Yellow threshold")
        fig2.add_vline(x=50, line_dash="dash", line_color="#ff4444", annotation_text="Red threshold")
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        if "dq_band" in df.columns:
            band_counts = df["dq_band"].value_counts().reset_index()
            band_counts.columns = ["band", "count"]
            fig3 = px.pie(
                band_counts, names="band", values="count",
                color="band",
                color_discrete_map={"green": "#00c851", "yellow": "#ffbb33", "red": "#ff4444"},
                title="DQ Band Distribution",
            )
            fig3.update_layout(template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

# ── Validation flags ──────────────────────────────────────────────────────────
st.markdown("### Validation Rule Violations")
flag_cols = [c for c in df.columns if c.startswith("flag_")]
if flag_cols:
    flag_counts = df[flag_cols].sum().sort_values(ascending=False)
    fig4 = px.bar(
        x=flag_counts.values,
        y=flag_counts.index,
        orientation="h",
        title="Violation Count per Rule",
        color=flag_counts.values,
        color_continuous_scale="oranges",
        labels={"x": "Violation Count", "y": "Rule"},
    )
    fig4.update_layout(template="plotly_dark", height=max(300, len(flag_cols) * 22))
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No validation flag columns found. Run the validation pipeline first.")

# ── Column stats table ────────────────────────────────────────────────────────
with st.expander("📋 Full Column Statistics", expanded=False):
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty and len(numeric_df.columns) > 0:
        st.dataframe(numeric_df.describe().T.round(3), use_container_width=True)
    else:
        st.info("No numeric columns available to display summary statistics.")

