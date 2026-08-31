# AI Development Log

**Project:** Loan Performance Intelligence Engine
**Track:** Intain Campus FinTech Challenge 2026 — AI Track
**Format:** Chronological log of all AI-assisted decisions, model choices, and observed LLM behaviors

---

## Log Format

Each entry includes:
- **Date / Phase**
- **Decision or Observation**
- **Rationale**
- **LLM Role** (where applicable): what the LLM suggested vs. what was accepted/rejected

---

## Entries

---

### Entry 001 — Architecture Decision: No Random Splits
**Date:** 2026-08-30
**Phase:** Infrastructure
**Decision:** Reject random row-level train/test splitting. Use strict temporal splitting only.
**Rationale:** Loan panel data is a time series. Random splits allow future information to appear in training rows (e.g., a loan's future DPD status influences its current-period training label through rolling features). `TemporalSplitter` + `LeakageAuditor` enforce this at the code level.
**LLM Role:** Not involved. Rule derived from problem statement constraints.

---

### Entry 002 — Algorithm Selection: XGBoost for 12m Default
**Date:** 2026-08-30
**Phase:** Model design
**Decision:** Use XGBoost (tree_method=hist) for `default_12m` and LightGBM for all other binary models.
**Rationale:** XGBoost's `eval_metric=aucpr` directly optimises the AUC-PR metric specified in the judging criteria for the highest-stakes model. LightGBM is faster for the three lower-priority models.
**LLM Role:** Not involved. Algorithmic choice driven by domain knowledge.

---

### Entry 003 — Imbalance Strategy: scale_pos_weight over SMOTE
**Date:** 2026-08-30
**Phase:** Feature engineering
**Decision:** Primary imbalance strategy is `scale_pos_weight` (class ratio). SMOTE-NC is optional, disabled by default.
**Rationale:** SMOTE on loan panel data risks creating synthetic loans that violate domain constraints (e.g., synthetic loans with DPD > 90 but status = "Current"). `scale_pos_weight` adjusts the loss function without altering the feature distribution.
**LLM Role:** Not involved.

---

### Entry 004 — Survival Model Backend: lifelines with scikit-survival fallback
**Date:** 2026-08-30
**Phase:** Survival analysis
**Decision:** Use `lifelines` as the primary survival library. `scikit-survival` is used for Fine-Gray competing risk and falls back gracefully to two independent Cox PH models if unavailable.
**Rationale:** lifelines has a more stable API for panel data. scikit-survival's Fine-Gray implementation is the industry standard for competing risks but has more brittle dependencies.
**LLM Role:** Not involved.

---

### Entry 005 — LLM Grounding Rule: Chunk Citation Required
**Date:** 2026-08-30
**Phase:** LLM copilot
**Decision:** Every LLM response that does not cite at least one retrieved chunk ID by name is automatically blocked and replaced with `⚠️ UNGROUNDED RESPONSE — BLOCKED`.
**Rationale:** Without a hard grounding check, the LLM may fabricate column definitions, rule IDs, or regulatory references. The citation requirement forces the model to operate within the retrieved knowledge base.
**LLM Role:** Not involved (rule is implemented in code, not enforced by the LLM itself).

---

### Entry 006 — Forbidden Phrase List
**Date:** 2026-08-30
**Phase:** LLM copilot
**Decision:** Block any LLM response containing: "I predict", "the model will", "I believe the loan will", "I am certain".
**Rationale:** These phrases indicate the LLM is making forward-looking predictions rather than summarising model outputs. The copilot must only explain what the ML model already computed, not generate new predictions.
**LLM Role:** Not involved.

---

### Entry 007 — Probability Calibration: Isotonic over Platt
**Date:** 2026-08-30
**Phase:** Model calibration
**Decision:** Use isotonic regression (non-parametric) rather than Platt scaling (sigmoid).
**Rationale:** Platt scaling assumes the uncalibrated scores follow a sigmoid transformation — this assumption is often violated for tree models. Isotonic regression makes no parametric assumption and works better for N > 10,000 samples. ECE target: < 0.05.
**LLM Role:** Not involved.

---

### Entry 008 — Anomaly Ensemble: Rank Normalization
**Date:** 2026-08-30
**Phase:** Anomaly detection
**Decision:** Normalize raw scores from IF, LOF, HBOS using rank-based normalization (not min-max) before ensemble averaging.
**Rationale:** Min-max normalization is sensitive to extreme values — a single large outlier compresses all other scores to near zero. Rank normalization is robust to the scale of individual detector outputs.
**LLM Role:** Not involved.

---

### Entry 009 — Markov Transition: Absorbing State Enforcement
**Date:** 2026-08-30
**Phase:** Survival analysis
**Decision:** Default, Prepaid, and Liquidated states are hardcoded as absorbing states regardless of observed transitions.
**Rationale:** Empirically there may be a small number of data errors showing loans "recovering" from Default. The absorbing state constraint enforces domain logic (a charged-off loan cannot become Current) and prevents the Markov projection from producing nonsensical results.
**LLM Role:** Not involved.

---

### Entry 010 — Feature Encoding: Target Encoding Fit on Train Only
**Date:** 2026-08-30
**Phase:** Feature engineering
**Decision:** Target encoding for `servicer_name` and `state` is fitted only on training data. Test rows with unseen servicers/states receive the global training-set target mean.
**Rationale:** Fitting target encoding on all data would allow test-set labels to influence training features, creating look-ahead bias. The fallback to global mean is a conservative, unbiased imputation.
**LLM Role:** Not involved.

---

### Entry 011 — Dashboard: Mock Mode by Default
**Date:** 2026-08-31
**Phase:** Dashboard / LLM
**Decision:** The LLM Copilot page defaults to "Mock Mode" (no API key required) to enable judges to evaluate the dashboard without providing an OpenAI API key.
**Rationale:** Accessibility for evaluation. The mock response demonstrates the full HITL workflow (grounded response → badge → approve/reject) without requiring external API credentials.
**LLM Role:** Not involved.

---

### Entry 012 — Scenario Engine: LGD Map from loss_severity_band
**Date:** 2026-08-31
**Phase:** Scenario simulation
**Decision:** LGD (Loss Given Default) is mapped from `loss_severity_band` categorical values. Default LGD = 40% for "Unknown".
**Rationale:** loss_severity_band is an organizer-provided field. Using it directly preserves the submission data structure and avoids fabricating an LGD model not supported by the training data.
**LLM Role:** Not involved.

---

## LLM Behavior Observations

### Observation 001 — Mock Response Grounding
**Scenario:** LLM Copilot in Mock Mode
**Observation:** The mock response template is pre-constructed to include chunk IDs (`data_dictionary_chunk_0`, etc.). This ensures the HITL workflow can be demonstrated even without a live LLM call.
**Assessment:** Acceptable for demonstration purposes. In production, real LLM responses must earn grounding status by citing actual retrieved chunks.

### Observation 002 — Expected Ungrounded Response Pattern
**Scenario:** When LLM is prompted about a loan without relevant dictionary context
**Expected Behavior:** Response will not cite a chunk ID → grounding check fails → response blocked
**Mitigation:** RAG retrieval always includes a static regulatory context document to give the LLM base knowledge to cite, reducing false-block rate.

### Observation 003 — Template vs. Free-form
**Scenario:** All 5 copilot use cases use versioned prompt templates with grounding rules in the system prompt
**Assessment:** Versioned templates ensure reproducibility and allow grading on prompt quality. Template version is logged in every audit entry.

---

## Outstanding Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| LLM fabricates column definition | Medium | High | Grounding check + BLOCKED response |
| Model AUC degrades in out-of-time test | Medium | High | ECE monitoring + calibration refresh plan |
| SMOTE creates domain-invalid synthetic loans | Low | Medium | SMOTE disabled by default |
| Markov matrix has zero-rows for rare states | Low | Medium | Small-count smoothing via `row_sums = max(row_sums, 1)` |
| ChromaDB cache becomes stale | Low | Low | `force_rebuild=True` flag in RAGPipeline |
