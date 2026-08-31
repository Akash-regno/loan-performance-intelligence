"""
app/pages/06_anomalies.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Anomalies | LPIE", page_icon="🚨", layout="wide")
st.markdown("## 🚨 Anomaly Detection & Exception Management")
st.caption("Ensemble anomaly scores (IF + LOF + HBOS) and rule-based exception flags.")

@st.cache_data(show_spinner=False)
def load_anomalies():
    for path in [
        Path("submission.csv"),
        Path("outputs/anomalies/top20_examples.csv"),
        Path("outputs/anomalies/anomaly_scores.parquet"),
    ]:
        if path.exists():
            df_loaded = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            if "anomaly_score" in df_loaded.columns:
                p95 = np.percentile(df_loaded["anomaly_score"].dropna(), 95) if len(df_loaded) > 0 else 0.5
                df_loaded["anomaly_flag"] = (df_loaded["anomaly_score"] >= p95).astype(int)
            return df_loaded
    # Synthetic demo
    np.random.seed(9)
    n = 500
    df = pd.DataFrame({
        "loan_id": [f"LN{i:05d}" for i in range(n)],
        "anomaly_score": np.clip(np.random.beta(1, 9, n), 0, 1),
        "anomaly_flag": 0,
        "exception_required": np.random.binomial(1, 0.07, n),
        "exception_type": np.random.choice(
            ["", "status_conflict", "balance_error", "servicer_dispute", "stale_record"],
            n, p=[0.93, 0.03, 0.02, 0.01, 0.01]
        ),
        "current_status": np.random.choice(["Current", "30DPD", "60DPD", "Default", "Prepaid"], n),
        "days_past_due": np.random.exponential(15, n).astype(int),
        "current_balance": np.random.uniform(50000, 500000, n).round(2),
        "credit_score_band": np.random.choice(["<620","620-639","640-679","680-719","720+"], n),
    })
    percentile_95 = np.percentile(df["anomaly_score"], 95)
    df["anomaly_flag"] = (df["anomaly_score"] >= percentile_95).astype(int)
    return df


df = load_anomalies()

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_total = len(df)
n_anomaly = int(df.get("anomaly_flag", pd.Series(0)).sum())
n_exception = int(df.get("exception_required", pd.Series(0)).sum())
mean_score = float(df["anomaly_score"].mean()) if "anomaly_score" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{n_total:,}")
c2.metric("Anomaly Flags", n_anomaly, f"{100*n_anomaly/n_total:.1f}%")
c3.metric("Exception Flags", n_exception, f"{100*n_exception/n_total:.1f}%")
c4.metric("Mean Anomaly Score", f"{mean_score:.4f}")

st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Anomaly Scores", "📋 Exception Review"])

# ── Tab 1: Anomaly scores ─────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Anomaly Score Distribution")
        fig = px.histogram(
            df, x="anomaly_score", nbins=60,
            color_discrete_sequence=["#f87171"],
            labels={"anomaly_score": "Anomaly Score"},
        )
        p95 = float(df["anomaly_score"].quantile(0.95))
        fig.add_vline(x=p95, line_dash="dash", line_color="#ffbb33",
                      annotation_text=f"95th pctl = {p95:.3f}")
        fig.update_layout(template="plotly_dark", title="Anomaly Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top 20 Anomalous Loans")
        top20_cols = [c for c in ["loan_id", "anomaly_score", "current_status",
                                    "days_past_due", "credit_score_band"] if c in df.columns]
        top20 = df.nlargest(20, "anomaly_score")[top20_cols]
        st.dataframe(
            top20.style.background_gradient(subset=["anomaly_score"], cmap="Reds")
                  .format({"anomaly_score": "{:.4f}"}),
            use_container_width=True, height=450,
        )

    # Scatter: score vs feature
    x_col = "days_past_due" if "days_past_due" in df.columns else ("prob_next_12m_default" if "prob_next_12m_default" in df.columns else df.columns[1])
    color_col = "current_status" if "current_status" in df.columns else ("next_state" if "next_state" in df.columns else None)
    st.markdown(f"### Anomaly Score vs {x_col.replace('_', ' ').title()}")
    fig2 = px.scatter(
        df.sample(min(500, len(df))),
        x=x_col, y="anomaly_score",
        color=color_col,
        opacity=0.7,
        title=f"Anomaly Score vs {x_col.replace('_', ' ').title()}",
        labels={x_col: x_col.replace('_', ' ').title(), "anomaly_score": "Anomaly Score"},
    )
    fig2.add_hline(y=p95, line_dash="dash", line_color="#ffbb33", annotation_text="Anomaly threshold")
    fig2.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Exception review ───────────────────────────────────────────────────
with tab2:
    st.markdown("### Exception-Flagged Records")

    exc_df = df[df.get("exception_required", pd.Series(0)) == 1].copy()

    if exc_df.empty:
        st.success("✅ No exception-flagged records found.")
    else:
        # Exception type breakdown
        if "exception_type" in exc_df.columns:
            type_counts = exc_df["exception_type"].value_counts().reset_index()
            type_counts.columns = ["exception_type", "count"]
            type_counts = type_counts[type_counts["exception_type"] != ""]

            fig3 = px.bar(
                type_counts, x="exception_type", y="count",
                color="exception_type", title="Exception Type Breakdown",
                labels={"count": "Count", "exception_type": "Type"},
            )
            fig3.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # Exception table
        exc_cols = [c for c in [
            "loan_id", "exception_type", "current_status",
            "days_past_due", "current_balance", "anomaly_score",
        ] if c in exc_df.columns]

        st.dataframe(
            exc_df[exc_cols].sort_values(
                "anomaly_score" if "anomaly_score" in exc_df.columns else exc_cols[0],
                ascending=False
            ).head(50),
            use_container_width=True,
            height=400,
        )

        csv = exc_df[exc_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Exception Report CSV",
            data=csv, file_name="exceptions.csv", mime="text/csv"
        )
