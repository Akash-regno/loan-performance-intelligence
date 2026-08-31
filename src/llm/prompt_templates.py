"""
src/llm/prompt_templates.py
----------------------------
Versioned prompt templates for all 5 copilot use cases.

Every template:
  - Includes a grounding requirement (must cite retrieved chunk IDs)
  - Labels output as RECOMMENDATION — NOT A DECISION
  - Forbids fabricating column names or results
  - Is versioned (bump version when modifying)

Templates:
  - LOAN_EXPLANATION_v1      : explain why a loan has high default risk
  - EXCEPTION_REVIEW_v1      : summarize an exception flag
  - SCENARIO_NARRATIVE_v1    : 3-sentence scenario risk narrative
  - DATA_QUALITY_REVIEW_v1   : identify top data quality issues
  - FP_FN_AUDIT_v1           : summarize FP/FN patterns
"""

from __future__ import annotations


TEMPLATES: dict[str, dict] = {

    "loan_explanation": {
        "version": "v1",
        "use_case": "loan_explanation",
        "system": (
            "You are a loan performance analyst assistant. "
            "Your job is to explain ML model predictions in plain English. "
            "RULES: "
            "1. Only reference information from the CONTEXT section below. "
            "2. Do NOT fabricate column names, values, or results. "
            "3. Do NOT make predictions — explain what the model already predicted. "
            "4. End every response with: '[RECOMMENDATION — NOT A DECISION]' "
            "5. Cite the chunk ID(s) you used in parentheses, e.g. (data_dictionary_chunk_3)."
        ),
        "user": (
            "CONTEXT (retrieved from knowledge base):\n{context}\n\n"
            "LOAN DATA:\n{loan_data}\n\n"
            "MODEL PREDICTION:\n"
            "  - 12-month default probability: {default_prob:.1%}\n"
            "  - Top SHAP drivers: {top_drivers}\n\n"
            "TASK: In 3-5 sentences, explain in plain English why this loan has a "
            "{default_prob:.1%} default probability. Focus on the top SHAP drivers "
            "and reference their definitions from the CONTEXT. "
            "Cite the chunk IDs you used."
        ),
    },

    "exception_review": {
        "version": "v1",
        "use_case": "exception_review",
        "system": (
            "You are a loan data quality reviewer. "
            "RULES: "
            "1. Only use information from the CONTEXT below. "
            "2. Do NOT invent data or suggest exceptions not present in the data. "
            "3. Label your output as RECOMMENDATION — NOT A DECISION. "
            "4. Cite the rule ID(s) from the CONTEXT."
        ),
        "user": (
            "CONTEXT (validation rules and data dictionary):\n{context}\n\n"
            "EXCEPTION DETAILS:\n"
            "  - Loan ID: {loan_id}\n"
            "  - Exception type: {exception_type}\n"
            "  - Triggered rules: {exception_drivers}\n"
            "  - Key field values: {key_fields}\n\n"
            "TASK: In 2-3 sentences: (1) explain what this exception means, "
            "(2) state which rule was violated and why it matters, "
            "(3) recommend whether a human reviewer should approve, reject, or investigate further. "
            "Cite the rule ID from the CONTEXT. "
            "[RECOMMENDATION — NOT A DECISION]"
        ),
    },

    "scenario_narrative": {
        "version": "v1",
        "use_case": "scenario_narrative",
        "system": (
            "You are a risk analyst summarizing stress-test results. "
            "RULES: "
            "1. Only reference numbers from the SCENARIO RESULTS below — do not fabricate. "
            "2. Keep to exactly 3 sentences. "
            "3. Label your output as RECOMMENDATION — NOT A DECISION. "
            "4. Cite the chunk IDs from CONTEXT that support your narrative."
        ),
        "user": (
            "CONTEXT:\n{context}\n\n"
            "SCENARIO RESULTS ({scenario_name}):\n{scenario_table}\n\n"
            "TASK: Write exactly 3 sentences summarizing: "
            "(1) the portfolio impact under this scenario, "
            "(2) which segment is most affected, "
            "(3) what the primary risk driver is. "
            "Use only the numbers provided above. "
            "[RECOMMENDATION — NOT A DECISION]"
        ),
    },

    "data_quality_review": {
        "version": "v1",
        "use_case": "data_quality_review",
        "system": (
            "You are a data quality auditor for loan performance data. "
            "RULES: "
            "1. Only reference issues visible in the DATA QUALITY STATS below. "
            "2. Do NOT invent issues not present in the stats. "
            "3. Prioritize issues by business impact. "
            "4. Label as RECOMMENDATION — NOT A DECISION. "
            "5. Cite chunk IDs from CONTEXT."
        ),
        "user": (
            "CONTEXT:\n{context}\n\n"
            "DATA QUALITY STATS:\n{dq_stats}\n\n"
            "TOP VIOLATION COUNTS:\n{violation_counts}\n\n"
            "TASK: In 3-4 sentences, identify the most critical data quality issues "
            "in this batch and their potential impact on model reliability. "
            "Reference the field definitions from CONTEXT. "
            "[RECOMMENDATION — NOT A DECISION]"
        ),
    },

    "fp_fn_audit": {
        "version": "v1",
        "use_case": "fp_fn_audit",
        "system": (
            "You are an ML model auditor analyzing false-positive and false-negative patterns. "
            "RULES: "
            "1. Only reference statistics from the ANALYSIS TABLE below. "
            "2. Do NOT suggest model changes — only summarize patterns. "
            "3. Label as RECOMMENDATION — NOT A DECISION. "
            "4. Cite chunk IDs from CONTEXT."
        ),
        "user": (
            "CONTEXT:\n{context}\n\n"
            "FP/FN ANALYSIS ({model_name}):\n"
            "  Counts: TP={tp} TN={tn} FP={fp} FN={fn}\n"
            "  Top FP drivers: {top_fp_features}\n"
            "  Top FN drivers: {top_fn_features}\n\n"
            "TASK: In 3 sentences: "
            "(1) summarize the false-positive pattern (what confuses the model), "
            "(2) summarize the false-negative blind spot, "
            "(3) state which feature cluster appears most responsible. "
            "[RECOMMENDATION — NOT A DECISION]"
        ),
    },
}


def get_template(use_case: str) -> dict:
    """Return the prompt template dict for a given use case.

    Raises KeyError if use_case is not found.
    """
    if use_case not in TEMPLATES:
        raise KeyError(
            f"Unknown use_case '{use_case}'. "
            f"Available: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[use_case]


def format_prompt(
    use_case: str,
    role: str = "user",
    **kwargs,
) -> str:
    """Format a prompt template with the given keyword arguments.

    Parameters
    ----------
    use_case : str
        One of the keys in TEMPLATES.
    role : {'user', 'system'}
        Which template section to format.
    **kwargs :
        Template variables (e.g. context, loan_data, default_prob).

    Returns
    -------
    str
        Formatted prompt string.
    """
    tmpl = get_template(use_case)
    raw = tmpl.get(role, "")
    try:
        return raw.format(**kwargs)
    except KeyError as exc:
        raise KeyError(
            f"Missing template variable {exc} for use_case='{use_case}', role='{role}'"
        ) from exc
