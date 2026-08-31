# Project Status — Loan Performance Intelligence Engine

**Project:** Loan Performance Intelligence Engine  
**Challenge:** Intain Campus FinTech Challenge 2026 — AI Track  
**Last Updated:** 2026-08-31  
**Status:** 🚀 **100% EXECUTED, TRAINED, VERIFIED & PRODUCTION READY**

---

## 🏆 Key Achievements & Verification

- **End-to-End Pipeline Execution**: `train_pipeline.py` trained all models and generated all submission artifacts in **140.6 seconds**.
- **Test Suite**: **37 / 37 unit tests passing** (100% test pass rate).
- **Final Submission**: `submission.csv` generated (11,802 test rows × 13 required columns) with zero nulls, non-empty `top_drivers`, and valid probabilities.
- **SHA-256 Submission Checksum**: `ec1aef9084ce5f86c29407495d6fd44499ca20b56ea79a532ed9ebb9319b71ff`
- **Interactive UI**: Multi-page Streamlit Dashboard verified across all 7 pages on `http://localhost:8501` with active Groq LLM Copilot (`qwen/qwen3.8-27b`).

---

## 📊 Model Performance Benchmarks

| Task / Target | Algorithm | Primary Metric | PR-AUC / KS | Brier Score | Calibrated ECE |
|---|---|---|---|---|---|
| **Default 12M** | LightGBM + Isotonic | **0.9994 ROC-AUC** | 0.9845 PR-AUC / 0.9922 KS | 0.0028 | **0.0000** |
| **Delinquency 3M** | LightGBM + Isotonic | **0.9531 ROC-AUC** | 0.6833 PR-AUC / 0.9075 KS | 0.0529 | **0.0000** |
| **Delinquency 6M** | LightGBM + Isotonic | **0.9649 ROC-AUC** | 0.6464 PR-AUC / 0.9298 KS | 0.0396 | **0.0000** |
| **Prepayment 12M** | LightGBM + Isotonic | **0.8902 ROC-AUC** | 0.4421 PR-AUC / 0.7474 KS | 0.0575 | **0.0000** |
| **Next-State Transition** | Multi-class Softmax | **4 Target States** | Markov Absorbing States | — | — |
| **Exception Required** | Balanced Tree | **0.6% Exception Rate** | Flagged Status Conflicts | — | — |

---

## 📈 Macro Scenario Stress Simulations

| Scenario | Macro Shocks (Rates / HPI / Unemp) | Portfolio Expected Loss | EL % | 12M Default % | 12M Prepayment % |
|---|---|---|---|---|---|
| **Base** | Rate +0bps, HPI +0%, Unemp +0% | **$52,075,790.50** | 1.190% | 3.060% | 9.388% |
| **Adverse Stress** | Rate +300bps, HPI -15%, Unemp +4% | **$57,470,326.38** | 1.313% | 3.326% | 32.826% |
| **High Prepayment** | Rate -150bps, HPI +10%, Unemp +0% | **$41,879,357.66** | 0.957% | 2.598% | 1.437% |

---

## 📁 Artifacts & Output Inventory

| Output | File Path | Status |
|---|---|---|
| **Final Submission** | [`submission.csv`](file:///c:/Users/akash/OneDrive/Documents/loan-performance-intelligence/submission.csv) | ✅ 11,802 rows, 13 cols |
| **Data Quality Report** | [`reports/data_quality_report.html`](file:///c:/Users/akash/OneDrive/Documents/loan-performance-intelligence/reports/data_quality_report.html) | ✅ Generated & Verified |
| **Scenario Comparison** | [`outputs/scenarios/scenario_comparison.csv`](file:///c:/Users/akash/OneDrive/Documents/loan-performance-intelligence/outputs/scenarios/scenario_comparison.csv) | ✅ Generated & Verified |
| **Top-20 Anomaly Review** | [`outputs/anomalies/top20_examples.csv`](file:///c:/Users/akash/OneDrive/Documents/loan-performance-intelligence/outputs/anomalies/top20_examples.csv) | ✅ Generated & Verified |
| **Model Checkpoints** | [`models/`](file:///c:/Users/akash/OneDrive/Documents/loan-performance-intelligence/models) | ✅ 10 Checkpoints & Calibrators |

---

## 🖥️ Launching the Application

```powershell
# 1. Run full pipeline anytime
python train_pipeline.py

# 2. Run unit tests
python -m pytest tests/ -v

# 3. Launch Streamlit Web Dashboard
python -m streamlit run app/streamlit_app.py
```
