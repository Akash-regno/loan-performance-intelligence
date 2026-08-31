# Model Card — Loan Performance Intelligence Engine

**Version:** 1.0.0
**Date:** 2026-08-31
**Challenge:** Intain Campus FinTech Challenge 2026 — AI Track

---

## Model Overview

| Property | Value |
|---|---|
| **Model family** | XGBoost (default_12m) + LightGBM (delinquency_3m/6m, prepayment_12m, next_state, exception) |
| **Task types** | Binary classification (×4) + Multi-class classification (×2) |
| **Calibration** | Isotonic regression on held-out calibration set |
| **Explainability** | SHAP TreeExplainer (global + local) |
| **Survival models** | Cox PH, Weibull AFT, Fine-Gray competing risk, 7-state Markov |
| **Anomaly detection** | Isolation Forest + LOF + HBOS ensemble |

---

## Intended Use

### Primary use cases
- **Delinquency risk scoring** (3m, 6m horizons): early warning for servicer outreach
- **Default prediction** (12m horizon): CECL expected credit loss estimation
- **Prepayment prediction** (12m horizon): portfolio prepayment speed modelling
- **Anomaly detection**: data quality exceptions for reviewer escalation
- **Scenario simulation**: stress-test portfolio EL under macro shocks
- **LLM-assisted review**: grounded plain-English summaries for human reviewers

### Primary users
- Loan analysts and risk managers reviewing flagged records
- Portfolio risk teams running stress tests
- Data quality teams auditing servicer reporting

### Out-of-scope uses
- **Making automated loan approval/denial decisions** — outputs are RECOMMENDATIONS ONLY
- **Replacing human underwriting judgment**
- **Regulatory capital calculations** without additional validation

---

## Training Data

| Property | Value |
|---|---|
| **Data type** | Monthly loan panel data (time series) |
| **Split** | Temporal — train on earlier months, validate/test on most recent months |
| **Leakage prevention** | `LeakageAuditor` blocks target columns in feature matrix |
| **Target construction** | Forward-looking: next_3m, next_6m, next_12m windows |
| **Class imbalance** | Handled via `scale_pos_weight` (class ratio) + optional SMOTE-NC |

> **Important:** No random row-level splitting was used. All splits are strictly temporal to prevent data leakage.

---

## Evaluation Metrics

### Binary classification targets

| Model | Metric | Target | Evaluation Set |
|---|---|---|---|
| `default_12m` | AUC-ROC | ≥ 0.75 | Temporal test split |
| `default_12m` | AUC-PR | ≥ 0.50 | Temporal test split |
| `default_12m` | KS Statistic | ≥ 0.35 | Temporal test split |
| `default_12m` | Brier Score | ≤ 0.10 | Temporal test split |
| `default_12m` | ECE (calibrated) | ≤ 0.05 | Calibration split |
| `delinquency_3m` | AUC-ROC | ≥ 0.72 | Temporal test split |
| `prepayment_12m` | AUC-ROC | ≥ 0.70 | Temporal test split |

### Survival model targets

| Model | Metric | Target |
|---|---|---|
| `hazard_default` | Harrell's C-index | ≥ 0.65 |
| `hazard_prepayment` | Harrell's C-index | ≥ 0.65 |

### Multi-class targets

| Model | Metric | Target |
|---|---|---|
| `next_state` | Macro F1 | ≥ 0.50 |

---

## Limitations

### Known limitations
1. **Cold-start problem**: Lag/rolling features (dpd_lag1, dpd_trend_3m) are zero for newly originated loans. Model accuracy is lower in the first 6 months of loan life.
2. **Geographic concentration**: If training data is concentrated in specific states/regions, performance may degrade in underrepresented geographies.
3. **Macro assumption sensitivity**: The scenario engine uses linear macro deltas. Non-linear macro effects (e.g., cliff-edge HPI crashes) are not modelled.
4. **LLM hallucination risk**: Despite grounding checks and chunk citation requirements, LLM responses may be imprecise. All LLM outputs are labelled `RECOMMENDATION — NOT A DECISION` and must be reviewed by a human.
5. **Calibration drift**: Model calibration may degrade as market conditions evolve. Recommend recalibration every 6–12 months.

### Data quality limitations
1. Missing servicer data is imputed using training-set medians/modes — this introduces uncertainty in fields with high missingness.
2. Validation rules may not capture all domain-specific exceptions. New rule types should be added to `validation_rules.json`.

---

## Fairness & Bias Considerations

### Protected attributes
The following sensitive attributes are **intentionally excluded** from the feature matrix to prevent disparate impact:
- Race / ethnicity
- Gender
- Age
- Religion
- National origin

### Indirect proxies to monitor
The following features may serve as proxies for protected characteristics and should be monitored for disparate impact:
- `state` (geographic redlining risk)
- `zip_code` / `property_type`
- `loan_purpose` (first-home buyers vs refinancers)

### Recommended fairness checks
Run `sklearn.inspection.permutation_importance` and segment performance metrics by `state` and `vintage_year` to detect unfair disparities.

---

## Model Governance

### Human-in-the-loop requirements
- All exception flags (`exception_required = 1`) require human reviewer sign-off before action
- LLM copilot outputs require `Approve` / `Reject` / `Correct` in the HITL panel
- Anomaly flags in the top-5% (anomaly_score ≥ 95th percentile) require escalation

### Audit trail
Every LLM invocation is logged to `outputs/llm_logs/audit.jsonl` with:
- Full prompt and response
- Retrieved chunk IDs
- Grounding pass/fail
- Reviewer action and correction

### Refresh schedule
- Model retraining: every 6 months, or when AUC-ROC drops > 3% from baseline
- Calibration refresh: every 3 months
- RAG index rebuild: when `data_dictionary.md` or `validation_rules.json` are updated

---

## Technical Specifications

| Property | Value |
|---|---|
| **Python version** | ≥ 3.10 |
| **Core ML** | XGBoost 2.0.3, LightGBM 4.3.0 |
| **Survival** | lifelines 0.29.0, scikit-survival 0.23.0 |
| **Anomaly** | PyOD 1.1.3 |
| **Explainability** | SHAP 0.44.0 |
| **RAG** | ChromaDB 0.4.24 |
| **Dashboard** | Streamlit 1.35.0 |
| **Random seed** | 42 (set in `src/utils/seed.py`) |

---

## Contact

*Submitted by: Intain Campus FinTech Challenge 2026 participant*
*Track: AI — Loan Performance Intelligence Engine*
