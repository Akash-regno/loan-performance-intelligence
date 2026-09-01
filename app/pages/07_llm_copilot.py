"""
app/pages/07_llm_copilot.py
----------------------------
LLM Copilot page: grounded reviewer summaries + HITL approve/reject panel.

Features:
  - Loan selector → auto-generate LLM explanation
  - Retrieved context accordion (shows what the LLM "saw")
  - Grounding status badge (✓ Grounded / ✗ Blocked)
  - HITL panel: Approve / Reject / Correct buttons
  - Audit log viewer (shows rejected/corrected examples)
  - Mock mode when LLM API key is not configured
"""

import json
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for Streamlit Cloud
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st


st.set_page_config(page_title="LLM Copilot | LPIE", page_icon="🤖", layout="wide")

# ── Shared styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.grounded-badge { background:#00c851; color:white; border-radius:6px; padding:4px 12px; font-size:13px; font-weight:600; }
.blocked-badge  { background:#ff4444; color:white; border-radius:6px; padding:4px 12px; font-size:13px; font-weight:600; }
.llm-box        { background:rgba(167,139,250,0.07); border:1px solid rgba(167,139,250,0.3);
                  border-radius:12px; padding:20px; font-size:0.95rem; line-height:1.7; }
.blocked-box    { background:rgba(255,68,68,0.07); border:1px solid rgba(255,68,68,0.4);
                  border-radius:12px; padding:20px; }
.hitl-approved  { background:rgba(0,200,81,0.1);  border-left:4px solid #00c851; padding:12px 16px; border-radius:0 8px 8px 0; }
.hitl-rejected  { background:rgba(255,68,68,0.1); border-left:4px solid #ff4444; padding:12px 16px; border-radius:0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🤖 LLM Reviewer Copilot")
st.caption(
    "Grounded AI-generated reviewer notes with human-in-the-loop approval. "
    "All outputs are labelled **RECOMMENDATION — NOT A DECISION**."
)

# ── Load predictions / test data (cached) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_predictions():
    """Load pre-scored predictions or raw loan performance if available."""
    try:
        for path in [
            Path("submission.csv"),
            Path("outputs/predictions/test_predictions.parquet"),
            Path("data/raw/loan_monthly_performance_train.csv"),
            Path("data/raw/loan_monthly_performance_test.csv"),
        ]:
            if path.exists():
                return pd.read_csv(path, nrows=1000) if path.suffix == ".csv" else pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_audit_log():
    """Load LLM audit log entries."""
    try:
        log_path = Path("outputs/llm_logs/audit.jsonl")
        if not log_path.exists():
            return []
        entries = []
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
    except Exception:
        return []

df = load_predictions()
audit_entries = load_audit_log()


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Copilot Settings")
    use_case = st.selectbox(
        "Use Case",
        ["Loan Explanation", "Exception Review", "Scenario Narrative",
         "Data Quality Review", "FP/FN Audit"],
    )
    provider_choice = st.selectbox("LLM Provider", ["Groq (Ultra-Fast LLaMA 3.3)", "OpenAI", "Mock Mode"])
    
    mock_mode = (provider_choice == "Mock Mode")
    if provider_choice == "Groq (Ultra-Fast LLaMA 3.3)":
        default_groq = os.environ.get("GROQ_API_KEY", "")
        if not default_groq:
            try:
                if "GROQ_API_KEY" in st.secrets:
                    default_groq = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        if not default_groq:
            try:
                from src.utils.config import get_config
                default_groq = get_config().get("llm", {}).get("groq_api_key", "")
            except Exception:
                pass
        groq_key = st.text_input("Groq API Key", type="password", value=default_groq, placeholder="Enter Groq key…")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

    elif provider_choice == "OpenAI":
        default_openai = os.environ.get("OPENAI_API_KEY", "")
        if not default_openai:
            try:
                if "OPENAI_API_KEY" in st.secrets:
                    default_openai = st.secrets["OPENAI_API_KEY"]
            except Exception:
                pass
        api_key = st.text_input("OpenAI API Key", type="password", value=default_openai, placeholder="Enter OpenAI key…")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key


# ── Main panel ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Generate Explanation", "📋 Audit Log", "📊 HITL Decisions"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Generate Explanation
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("#### 🔎 Select Loan")

        if not df.empty and "loan_id" in df.columns:
            loan_ids = df["loan_id"].astype(str).unique().tolist()[:200]
            selected_loan_id = st.selectbox("Loan ID", loan_ids)
            loan_row = df[df["loan_id"].astype(str) == selected_loan_id].iloc[0]

            # Show key loan fields
            st.markdown("**Key Fields:**")
            display_cols = [
                "current_status", "days_past_due", "credit_score_band",
                "ltv_band", "loan_age_months", "current_balance",
            ]
            for col in display_cols:
                if col in loan_row.index:
                    st.write(f"• **{col}**: `{loan_row[col]}`")

            # Show prediction if available
            if "prob_next_12m_default" in loan_row.index:
                default_prob = float(loan_row["prob_next_12m_default"])
                color = "#ff4444" if default_prob > 0.5 else "#ffbb33" if default_prob > 0.3 else "#00c851"
                st.markdown(
                    f"<div style='margin-top:12px;padding:12px;border-radius:8px;"
                    f"background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3)'>"
                    f"<b>12m Default Prob:</b> <span style='color:{color};font-size:1.5rem;font-weight:700'>"
                    f"{default_prob:.1%}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("📁 Place `loan_monthly_performance_test.csv` in `data/raw/` to enable loan selection.")
            selected_loan_id = "LN_DEMO_001"
            loan_row = pd.Series({
                "loan_id": "LN_DEMO_001",
                "current_status": "30DPD",
                "days_past_due": 35,
                "credit_score_band": "620-639",
                "ltv_band": "85-90",
                "loan_age_months": 14,
                "current_balance": 245000,
                "prob_next_12m_default": 0.67,
            })
            default_prob = 0.67
            for col, val in loan_row.items():
                if col not in ["loan_id", "prob_next_12m_default"]:
                    st.write(f"• **{col}**: `{val}`")
            st.markdown(
                "<div style='margin-top:12px;padding:12px;border-radius:8px;"
                "background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3)'>"
                "<b>12m Default Prob:</b> <span style='color:#ff4444;font-size:1.5rem;font-weight:700'>"
                "67.0%</span></div>",
                unsafe_allow_html=True,
            )

        generate_btn = st.button("🚀 Generate Explanation", use_container_width=True)

    with col_right:
        st.markdown("#### 🤖 Copilot Response")

        if "copilot_result" not in st.session_state:
            st.session_state["copilot_result"] = None

        if generate_btn:
            top_drivers = (
                str(loan_row.get("top_drivers", "days_past_due|ltv_band|credit_score_band"))
                if "top_drivers" in loan_row.index
                else "days_past_due|ltv_band|credit_score_band"
            )

            with st.spinner("Retrieving context and generating explanation…"):
                if mock_mode:
                    # ── Mock response (no API required) ──────────────────
                    response_text = (
                        f"This loan (ID: {selected_loan_id}) has an elevated 12-month default "
                        f"probability of {float(loan_row.get('prob_next_12m_default', 0.67)):.1%}, "
                        f"primarily driven by its current delinquency status "
                        f"({loan_row.get('current_status', 'N/A')}) and "
                        f"{loan_row.get('days_past_due', 'N/A')} days past due "
                        f"(data_dictionary_chunk_0). "
                        f"The loan's LTV band of {loan_row.get('ltv_band', 'N/A')} indicates "
                        f"limited equity cushion, increasing loss severity risk in a default event "
                        f"(data_dictionary_chunk_2). "
                        f"The credit score band of {loan_row.get('credit_score_band', 'N/A')} "
                        f"further compounds the risk profile, as historically lower-scoring borrowers "
                        f"show higher transition rates to 90+ DPD states (data_dictionary_chunk_4). "
                        f"\n\n[RECOMMENDATION — NOT A DECISION]"
                    )
                    grounding_passed = True
                    citations = ["data_dictionary_chunk_0", "data_dictionary_chunk_2", "data_dictionary_chunk_4"]
                    retrieved_chunks = [
                        {"chunk_id": "data_dictionary_chunk_0", "source": "data_dictionary.md",
                         "text": "days_past_due: Number of days the borrower is past due on their payment obligation."},
                        {"chunk_id": "data_dictionary_chunk_2", "source": "data_dictionary.md",
                         "text": "ltv_band: Loan-to-Value ratio band. Higher LTV = lower equity = higher loss severity."},
                        {"chunk_id": "data_dictionary_chunk_4", "source": "data_dictionary.md",
                         "text": "credit_score_band: Credit score range. Lower scores correlate with higher default rates."},
                    ]
                else:
                    try:
                        from src.llm.copilot import LLMCopilot
                        copilot = LLMCopilot()
                        result = copilot.explain_loan(
                            loan_data=loan_row.to_dict(),
                            default_prob=float(loan_row.get("prob_next_12m_default", 0.5)),
                            top_drivers=top_drivers,
                            loan_id=str(selected_loan_id),
                        )
                        response_text = result["response"]
                        grounding_passed = result["grounding_passed"]
                        citations = result.get("citations_found", [])
                        retrieved_chunks = result.get("retrieved_chunks", [])
                    except Exception as exc:
                        response_text = f"[ERROR: {exc}]"
                        grounding_passed = False
                        citations = []
                        retrieved_chunks = []

                st.session_state["copilot_result"] = {
                    "response_text": response_text,
                    "grounding_passed": grounding_passed,
                    "citations": citations,
                    "retrieved_chunks": retrieved_chunks,
                    "loan_id": str(selected_loan_id),
                }

        curr_res = st.session_state.get("copilot_result")
        if curr_res is not None:
            # ── Display grounding status ──────────────────────────────
            if curr_res["grounding_passed"]:
                st.markdown(
                    '<span class="grounded-badge">✓ Grounded Response</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="blocked-badge">✗ BLOCKED — Ungrounded</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Response box ──────────────────────────────────────────
            box_class = "llm-box" if curr_res["grounding_passed"] else "blocked-box"
            st.markdown(
                f'<div class="{box_class}">{curr_res["response_text"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Retrieved context ──────────────────────────────────────
            if curr_res["retrieved_chunks"]:
                with st.expander("📚 Retrieved Context Chunks", expanded=False):
                    for chunk in curr_res["retrieved_chunks"]:
                        st.markdown(
                            f"**{chunk.get('chunk_id', '')}** — *{chunk.get('source', '')}*"
                        )
                        st.text(chunk.get("text", "")[:400])
                        st.divider()

            # ── HITL Review Panel ──────────────────────────────────────
            st.markdown("#### 👤 Human Reviewer Decision")
            st.caption("Your decision is logged to `outputs/hitl_decisions.csv`")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("✅ Approve", key="btn_app", use_container_width=True):
                    _save_hitl_decision(curr_res["loan_id"], "approved", "")
                    st.success("Decision logged: **Approved**")
            with col_b:
                if st.button("❌ Reject", key="btn_rej", use_container_width=True):
                    _save_hitl_decision(curr_res["loan_id"], "rejected", "")
                    st.error("Decision logged: **Rejected**")
            with col_c:
                correction_text = st.text_area("Correction note", height=60, placeholder="Enter correction…")
                if st.button("✍️ Correct & Save", key="btn_corr", use_container_width=True):
                    _save_hitl_decision(curr_res["loan_id"], "corrected", correction_text)
                    st.warning("Decision logged: **Corrected**")
        else:
            st.info("👈 Select a loan and click **Generate Explanation** to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Audit Log
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📋 LLM Audit Log")

    if not audit_entries:
        st.info("No audit entries yet. Generate explanations to populate the log.")
    else:
        audit_df = pd.DataFrame(audit_entries)
        display_cols = ["timestamp", "use_case", "loan_id", "grounding_passed",
                        "reviewer_action", "model"]
        available = [c for c in display_cols if c in audit_df.columns]
        st.dataframe(audit_df[available], use_container_width=True, height=300)

        # Show ungrounded examples prominently
        ungrounded = [e for e in audit_entries if not e.get("grounding_passed", True)]
        if ungrounded:
            st.markdown(f"##### ⚠️ Ungrounded / Blocked Responses ({len(ungrounded)})")
            st.caption("Required by judging: examples where LLM output was wrong or blocked.")
            for entry in ungrounded[:3]:
                with st.expander(f"[{entry.get('use_case')}] {entry.get('timestamp', '')[:19]}"):
                    st.markdown(f"**Prompt (truncated):** {entry.get('prompt', '')[:300]}…")
                    st.markdown(f"**Response:** {entry.get('response', '')}")
                    st.markdown(f"**Grounding passed:** `{entry.get('grounding_passed')}`")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: HITL Decisions
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📊 Human-in-the-Loop Decisions")
    hitl_path = Path("outputs/hitl_decisions.csv")

    if hitl_path.exists():
        try:
            hitl_df = pd.read_csv(hitl_path)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviewed", len(hitl_df))
            with col2:
                approved = (hitl_df["action"] == "approved").sum() if "action" in hitl_df.columns else 0
                st.metric("Approved", approved)
            with col3:
                rejected = (hitl_df["action"] == "rejected").sum() if "action" in hitl_df.columns else 0
                st.metric("Rejected / Corrected", rejected)
            st.dataframe(hitl_df, use_container_width=True)
        except Exception:
            st.info("No HITL decisions recorded yet.")
    else:
        st.info("No HITL decisions recorded yet.")


# ── Helper ───────────────────────────────────────────────────────────────────
def _save_hitl_decision(loan_id: str, action: str, correction: str) -> None:
    """Append a HITL decision to the CSV."""
    import datetime

    try:
        hitl_path = Path("outputs/hitl_decisions.csv")
        hitl_path.parent.mkdir(parents=True, exist_ok=True)

        row = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "loan_id": loan_id,
            "action": action,
            "correction": correction,
        }])

        if hitl_path.exists():
            row.to_csv(hitl_path, mode="a", header=False, index=False)
        else:
            row.to_csv(hitl_path, index=False)
    except Exception:
        pass

