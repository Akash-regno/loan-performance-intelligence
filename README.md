# 🏦 Loan Performance Intelligence Engine

> **Intain Campus FinTech Challenge 2026 — AI Track**
> ML-first system for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and grounded LLM-assisted review.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Pipeline](#running-the-pipeline)
6. [Running the Dashboard](#running-the-dashboard)
7. [Running Tests](#running-tests)
8. [Data Files Required](#data-files-required)
9. [Module Reference](#module-reference)
10. [Design Decisions](#design-decisions)
11. [Evaluation Metrics Targets](#evaluation-metrics-targets)

---

## Project Overview

The Loan Performance Intelligence Engine is a production-quality ML system that:

- **Profiles** raw loan panel data with per-column statistics, missingness analysis, outlier detection, and row-level DQ scores (0–100)
- **Predicts** loan performance across 6 horizons: 3m delinquency, 6m delinquency, 12m default, 12m prepayment, next state, and exception detection
- **Detects anomalies** using an ensemble of Isolation Forest + LOF + HBOS with 20+ reviewer-ready examples
- **Models survival** using Cox PH, Weibull AFT, competing risk Fine-Gray, and a 7-state Markov transition model
- **Simulates scenarios**: Base, Adverse (+300bp rates, −15% HPI, +4pp unemployment), High-Prepayment (−150bp rates, +10% HPI)
- **Explains predictions** with SHAP global/local plots, LIME, and FP/FN deep-dive analysis
- **Provides a grounded LLM copilot** with ChromaDB RAG, grounding checks, and JSONL audit logging — not an LLM wrapper

---

## Architecture

```
RAW DATA (4 CSV files)
       │
       ▼
┌─────────────────────┐
│  1. DATA LAYER      │  ingestion → validation → profiling → cleaning → drift
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. FEATURE ENG.    │  temporal → lag/rolling → encoding → interactions → macro
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. TEMPORAL SPLIT  │  TemporalSplitter → LeakageAuditor (blocks target in features)
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      ▼
┌──────────────┐                                  ┌──────────────────┐
│  ML MODELS   │  XGBoost / LightGBM              │ SURVIVAL MODELS  │  Cox PH + Markov
│  (6 models)  │  + Isotonic Calibration          │  (3 models)      │  + Competing Risk
└──────┬───────┘                                  └────────┬─────────┘
       │                                                   │
       ▼                                                   ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  ANOMALY ENGINE  │    │  EXPLAINABILITY  │    │  SCENARIO ENGINE     │
│  IF + LOF + HBOS │    │  SHAP + FP/FN   │    │  Base / Adv / Prepay │
└──────────────────┘    └──────────────────┘    └──────────────────────┘
       │                        │                          │
       └────────────────────────┴──────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  LLM COPILOT (RAG)    │
                    │  ChromaDB + Grounding │
                    │  + JSONL Audit Log    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  STREAMLIT DASHBOARD  │  7 pages
                    │  + HITL Review Panel  │
                    └───────────────────────┘
```

---

## Folder Structure

```
loan-performance-intelligence/
├── config/
│   └── config.yaml              # Central hyperparameters, paths, settings
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Load + merge 4 CSV files
│   │   ├── validation.py        # 14 domain validation rules
│   │   ├── profiling.py         # Column stats, DQ scores (0–100), HTML report
│   │   ├── cleaning.py          # Train-fitted imputers + IQR capping
│   │   └── drift.py             # PSI + KS drift detection, Evidently report
│   │
│   ├── features/
│   │   ├── temporal.py          # Vintage year/quarter, loan age bands
│   │   ├── lag_rolling.py       # DPD lags, rolling stats (shift(1) safety)
│   │   ├── encoding.py          # Ordinal / target / binary encoders
│   │   ├── interactions.py      # LTV×credit, DPD acceleration, risk_composite
│   │   ├── macro.py             # Macro scenario join, rate spread, refi incentive
│   │   ├── servicer.py          # Servicer conflict rate, source reliability
│   │   └── imbalance.py        # Class weights, SMOTE-NC, threshold tuning
│   │
│   ├── models/
│   │   ├── base_model.py        # Abstract base: train/predict/evaluate/save/load
│   │   ├── delinquency_3m.py    # LightGBM 3m + 6m delinquency
│   │   ├── delinquency_6m.py    # (alias)
│   │   ├── default_12m.py       # XGBoost 12m default
│   │   ├── prepayment_12m.py    # LightGBM 12m prepayment
│   │   ├── next_state.py        # LightGBM multi-class next state
│   │   ├── exception_model.py   # Binary + multi-class exception
│   │   └── calibration.py      # Isotonic calibration + ECE reporting
│   │
│   ├── survival/
│   │   ├── hazard.py            # Cox PH + Weibull AFT (lifelines)
│   │   ├── competing_risk.py    # Fine-Gray competing risk
│   │   └── transition.py       # 7-state Markov + scenario shifts
│   │
│   ├── anomaly/
│   │   ├── detector.py          # IF + LOF + HBOS ensemble
│   │   └── exception.py         # Rule-based exception aggregation
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py    # SHAP TreeExplainer, global + local plots
│   │   └── fp_fn_analysis.py    # TP/TN/FP/FN quadrant analysis
│   │
│   ├── scenarios/
│   │   └── engine.py            # Base/Adverse/High-Prepay + portfolio EL
│   │
│   ├── llm/
│   │   ├── rag_pipeline.py      # ChromaDB RAG index
│   │   ├── prompt_templates.py  # 5 versioned templates with guardrails
│   │   ├── audit_logger.py      # JSONL audit log + HITL updates
│   │   └── copilot.py          # Full LLM pipeline with grounding
│   │
│   └── utils/
│       ├── config.py            # Cached YAML loader
│       ├── logger.py            # Standardized logging
│       ├── metrics.py           # Full metrics suite
│       ├── seed.py              # Reproducibility
│       ├── time_split.py        # TemporalSplitter + LeakageAuditor
│       └── submission.py        # Submission format validator
│
├── app/
│   ├── streamlit_app.py         # Main entry point (dark glassmorphism UI)
│   └── pages/
│       ├── 01_data_quality.py
│       ├── 02_predictions.py
│       ├── 03_explainability.py
│       ├── 04_survival.py
│       ├── 05_scenarios.py
│       ├── 06_anomalies.py
│       └── 07_llm_copilot.py
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_survival.py
│
├── data/
│   ├── raw/                     # Place organizer CSV files here
│   ├── processed/
│   └── external/
│
├── models/                      # Saved .pkl model artifacts
├── outputs/                     # Predictions, SHAP, scenarios, anomalies
├── reports/                     # Data quality HTML reports
│
├── train_pipeline.py            # End-to-end 10-phase orchestrator
├── config/config.yaml
├── requirements.txt
├── model_card.md
├── AI_DEVELOPMENT_LOG.md
└── PROJECT_STATUS.md
```

---

## Setup & Installation

### 1. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place data files
Copy all 4 organizer-provided CSV files to `data/raw/`:
```
data/raw/loan_monthly_performance_train.csv
data/raw/loan_monthly_performance_test.csv
data/raw/loan_static_attributes.csv
data/raw/servicer_updates.csv
data/raw/macro_scenarios.csv
data/raw/data_dictionary.md
data/raw/validation_rules.json
data/raw/submission_template.csv
```

---

## Running the Pipeline

### Full training pipeline
```bash
python train_pipeline.py
```

### Dry run (ingestion only — verify data files are loaded correctly)
```bash
python train_pipeline.py --dry-run
```

### Custom config
```bash
python train_pipeline.py --config config/config.yaml
```

---

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard runs on `http://localhost:8501` by default.

> **Note:** The dashboard works in **demo mode** with synthetic data if the CSV files are not yet present. No data is required to explore the UI.

### LLM Copilot Setup (optional)
```bash
# Option A: OpenAI
export OPENAI_API_KEY=sk-...

# Option B: Ollama (local, free)
ollama pull mistral:7b
# Set llm.provider: ollama in config/config.yaml
```

---

## Running Tests

```bash
pytest tests/ -v
```

Individual test suites:
```bash
pytest tests/test_features.py -v    # Feature engineering
pytest tests/test_models.py -v      # ML models + metrics
pytest tests/test_survival.py -v    # Survival + anomaly
```

---

## Data Files Required

| File | Description |
|---|---|
| `loan_monthly_performance_train.csv` | Monthly loan panel — training set |
| `loan_monthly_performance_test.csv` | Monthly loan panel — test set (targets hidden) |
| `loan_static_attributes.csv` | Static loan origination attributes |
| `servicer_updates.csv` | Servicer-reported updates (may conflict with panel) |
| `macro_scenarios.csv` | Macro scenario parameters (base / adverse / high-prepay) |
| `data_dictionary.md` | Column definitions (used by RAG pipeline) |
| `validation_rules.json` | Rule definitions (used by validator + RAG) |
| `submission_template.csv` | Expected submission format |

---

## Module Reference

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `DataIngestion` | `src/data/ingestion.py` | Load + merge all 4 data files |
| `DataValidator` | `src/data/validation.py` | 14-rule domain validation |
| `DataProfiler` | `src/data/profiling.py` | DQ scores + HTML report |
| `DataCleaner` | `src/data/cleaning.py` | Train-fitted imputation |
| `DriftDetector` | `src/data/drift.py` | PSI + KS drift |
| `TemporalSplitter` | `src/utils/time_split.py` | Temporal train/val/test split |
| `LeakageAuditor` | `src/utils/time_split.py` | Blocks target-in-features leakage |
| `LagRollingFeatures` | `src/features/lag_rolling.py` | Lag features (shift(1)-safe) |
| `CategoricalEncoder` | `src/features/encoding.py` | Ordinal/target/binary encoding |
| `ImbalanceHandler` | `src/features/imbalance.py` | Class weights + SMOTE-NC |
| `DefaultModel` | `src/models/default_12m.py` | XGBoost 12m default |
| `Delinquency3mModel` | `src/models/delinquency_3m.py` | LightGBM 3m delinquency |
| `ModelCalibrator` | `src/models/calibration.py` | Isotonic calibration + ECE |
| `HazardModel` | `src/survival/hazard.py` | Cox PH / Weibull AFT |
| `MarkovTransitionModel` | `src/survival/transition.py` | 7-state monthly Markov |
| `AnomalyDetector` | `src/anomaly/detector.py` | IF+LOF+HBOS ensemble |
| `ExceptionEngine` | `src/anomaly/exception.py` | Rule-based exception flags |
| `SHAPExplainer` | `src/explainability/shap_explainer.py` | SHAP + top_drivers |
| `ScenarioEngine` | `src/scenarios/engine.py` | Portfolio stress testing |
| `RAGPipeline` | `src/llm/rag_pipeline.py` | ChromaDB vector index |
| `LLMCopilot` | `src/llm/copilot.py` | Grounded LLM reviewer |
| `AuditLogger` | `src/llm/audit_logger.py` | JSONL audit + HITL |

---

## Design Decisions

### No temporal leakage
All lag/rolling features use `shift(1)`. The `LeakageAuditor` raises `LeakageError` if any target column appears in the feature matrix before model training.

### No LLM wrappers
Core predictions are made by XGBoost/LightGBM only. The LLM copilot is restricted to **retrieval-augmented summarization** and is blocked from making new predictions.

### Grounding enforcement
Every LLM response must cite at least one chunk ID from the RAG retrieval. Responses that fail grounding are replaced with `⚠️ UNGROUNDED RESPONSE — BLOCKED`.

### Class imbalance
Handled via `scale_pos_weight` (class ratio) injected at model construction, plus optional SMOTE-NC oversampling. Decision threshold is tuned post-training on the validation set.

### Probability calibration
All binary classifiers are calibrated with isotonic regression on a held-out calibration partition. ECE target: < 0.05.

---

## Evaluation Metrics Targets

| Metric | Target |
|---|---|
| AUC-ROC (default_12m) | ≥ 0.75 |
| AUC-PR (default_12m) | ≥ 0.50 |
| KS Statistic | ≥ 0.35 |
| Brier Score | ≤ 0.10 |
| ECE (calibrated) | ≤ 0.05 |
| Harrell's C-index (survival) | ≥ 0.65 |
| Macro F1 (next_state) | ≥ 0.50 |

---

*Built for the Intain Campus FinTech Challenge 2026 — AI Track*
