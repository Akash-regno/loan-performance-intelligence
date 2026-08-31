"""
train_pipeline.py
-----------------
End-to-end training pipeline orchestrator.

Phases:
  1. Data ingestion + validation
  2. Feature engineering (temporal, lag/rolling, encoding, interactions, macro)
  3. Temporal split (no leakage)
  4. Model training (all 6 ML models)
  5. Calibration
  6. Evaluation
  7. SHAP explainability
  8. Anomaly detection
  9. Exception engine
  10. Survival models
  11. Scenario engine
  12. Submission export

Usage:
    python train_pipeline.py [--config config/config.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds
from src.utils.time_split import TemporalSplitter, LeakageAuditor

log = get_logger("train_pipeline")


def main(dry_run: bool = False) -> None:
    set_all_seeds()
    cfg = get_config()
    start = time.time()

    # Ensure required output directories exist
    for d in ["models", "outputs/shap", "outputs/anomalies", "outputs/scenarios", "outputs/llm_logs", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("LOAN PERFORMANCE INTELLIGENCE ENGINE — TRAINING PIPELINE")
    log.info("=" * 70)

    # ── Phase 1: Ingestion + Validation ──────────────────────────────────────
    log.info("[PHASE 1] Data ingestion…")
    from src.data.ingestion import DataIngestion
    from src.data.validation import DataValidator

    ingestion = DataIngestion()
    df = ingestion.load()

    validator = DataValidator()
    validation_report = validator.validate(df)
    log.info("Validation: %d rules checked | %d violations", len(validation_report), (validation_report["passed"] == False).sum())
    df = validator.add_violation_flags(df)

    if dry_run:
        log.info("DRY RUN: stopping after ingestion.")
        log.info("Dataset shape: %s", df.shape)
        return

    # ── Phase 2: Feature Engineering ─────────────────────────────────────────
    log.info("[PHASE 2] Feature engineering…")
    from src.data.profiling import DataProfiler
    from src.data.cleaning import DataCleaner
    from src.features.temporal import TemporalFeatures
    from src.features.lag_rolling import LagRollingFeatures
    from src.features.interactions import InteractionFeatures
    from src.features.macro import MacroFeatures

    # Temporal features (no fitting needed)
    df = TemporalFeatures().transform(df)
    df = LagRollingFeatures().transform(df)
    df = InteractionFeatures().transform(df)

    macro_fe = MacroFeatures()
    macro_fe.fit()
    df = macro_fe.transform(df)

    # ── Phase 3: Temporal Split ───────────────────────────────────────────────
    log.info("[PHASE 3] Temporal split (no leakage)…")
    splitter = TemporalSplitter(
        holdout_months=cfg["temporal_split"]["holdout_months"],
        validation_months=cfg["temporal_split"]["validation_months"],
    )
    train_df, val_df = splitter.split_train_val(df)

    # Load test split and compute identical features
    log.info("Loading test split for prediction…")
    test_df_raw = ingestion.load(split="test")
    test_df = validator.add_violation_flags(test_df_raw)
    test_df = TemporalFeatures().transform(test_df)
    test_df = LagRollingFeatures().transform(test_df)
    test_df = InteractionFeatures().transform(test_df)
    test_df = macro_fe.transform(test_df)


    log.info("Split sizes: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))

    # ── Data Cleaning (fit on train) ──────────────────────────────────────────
    cleaner = DataCleaner()
    train_df = cleaner.fit_transform(train_df)
    val_df   = cleaner.transform(val_df)
    test_df  = cleaner.transform(test_df)

    # ── Profiling ─────────────────────────────────────────────────────────────
    profiler = DataProfiler()
    train_df = profiler.add_dq_scores(train_df)
    profiler.export_html_report(train_df, "reports/data_quality_report.html")

    # ── Encoding (fit on train, transform all splits) ─────────────────────────
    from src.features.encoding import CategoricalEncoder
    from src.features.servicer import ServicerFeatures

    enc = CategoricalEncoder()
    enc.fit(train_df, target_col=cfg["targets"]["default_12m"])
    train_df = enc.transform(train_df)
    val_df   = enc.transform(val_df)
    test_df  = enc.transform(test_df)

    svc_fe = ServicerFeatures()
    svc_fe.fit(train_df)
    train_df = svc_fe.transform(train_df)
    val_df   = svc_fe.transform(val_df)
    test_df  = svc_fe.transform(test_df)

    # ── Build feature matrix ──────────────────────────────────────────────────
    EXCLUDE = set(cfg["targets"].values()) | {
        "loan_id", "month_index", "reporting_month", "origination_month",
        "last_updated_at", "dq_score", "dq_band",
    }
    feature_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c not in EXCLUDE
    ]

    # ── Leakage audit ─────────────────────────────────────────────────────────
    auditor = LeakageAuditor()
    for col in cfg["targets"].values():
        if col in feature_cols:
            auditor.raise_if_target_in_features(feature_cols, col)


    log.info("Feature matrix: %d features", len(feature_cols))

    # ── Phase 4: Model Training ───────────────────────────────────────────────
    log.info("[PHASE 4] Training ML models…")
    from src.features.imbalance import ImbalanceHandler
    from src.models.delinquency_3m import Delinquency3mModel
    from src.models.delinquency_3m import Delinquency6mModel
    from src.models.default_12m import DefaultModel
    from src.models.prepayment_12m import PrepaymentModel
    from src.models.next_state import NextStateModel
    from src.models.exception_model import ExceptionRequiredModel, ExceptionTypeModel

    imb = ImbalanceHandler()

    models = {}
    model_specs = [
        ("delinquency_3m",  Delinquency3mModel,     cfg["targets"]["delinquency_3m"]),
        ("delinquency_6m",  Delinquency6mModel,     cfg["targets"]["delinquency_6m"]),
        ("default_12m",     DefaultModel,            cfg["targets"]["default_12m"]),
        ("prepayment_12m",  PrepaymentModel,         cfg["targets"]["prepayment_12m"]),
    ]

    for model_key, ModelClass, target_col in model_specs:
        if target_col not in train_df.columns:
            log.warning("Target '%s' not found — skipping %s", target_col, model_key)
            continue

        y_train = train_df[target_col].dropna()
        X_train = train_df.loc[y_train.index, feature_cols]
        y_val   = val_df[target_col].dropna() if target_col in val_df.columns else None
        X_val   = val_df.loc[y_val.index, feature_cols] if y_val is not None else None

        spw = imb.compute_scale_pos_weight(y_train)
        model = ModelClass()
        model.train(X_train, y_train, X_val, y_val, feature_cols, scale_pos_weight=spw)
        model.save(f"models/{model_key}")
        models[model_key] = model

        if X_val is not None and y_val is not None:
            metrics = model.evaluate(X_val, y_val)
            log.info("%s metrics: %s", model_key, metrics)

    # Next state model
    next_state_col = cfg["targets"]["next_state"]
    if next_state_col in train_df.columns:
        y_train = train_df[next_state_col].dropna()
        X_train = train_df.loc[y_train.index, feature_cols]
        ns_model = NextStateModel()
        ns_model.train(X_train, y_train, feature_cols=feature_cols)
        ns_model.save("models/next_state")
        models["next_state"] = ns_model

    # Exception model
    exc_req_col = cfg["targets"]["exception_required"]
    if exc_req_col in train_df.columns:
        y_train = train_df[exc_req_col].fillna(0)
        X_train = train_df[feature_cols]
        spw = imb.compute_scale_pos_weight(y_train)
        exc_model = ExceptionRequiredModel()
        exc_model.train(X_train, y_train, scale_pos_weight=spw, feature_cols=feature_cols)
        exc_model.save("models/exception_required")
        models["exception_required"] = exc_model

    # ── Phase 5: Calibration ──────────────────────────────────────────────────
    log.info("[PHASE 5] Calibrating probabilities…")
    from src.models.calibration import ModelCalibrator

    calibrated_models = {}
    for model_key in ["delinquency_3m", "delinquency_6m", "default_12m", "prepayment_12m"]:
        if model_key not in models:
            continue
        target_col = dict(
            delinquency_3m=cfg["targets"]["delinquency_3m"],
            delinquency_6m=cfg["targets"]["delinquency_6m"],
            default_12m=cfg["targets"]["default_12m"],
            prepayment_12m=cfg["targets"]["prepayment_12m"],
        )[model_key]
        if target_col not in val_df.columns:
            continue
        y_val = val_df[target_col].dropna()
        X_val = val_df.loc[y_val.index, feature_cols]
        cal = ModelCalibrator(models[model_key])
        cal.fit(X_val, y_val)
        cal.save()
        calibrated_models[model_key] = cal

    # ── Phase 6: SHAP Explainability ──────────────────────────────────────────
    log.info("[PHASE 6] Computing SHAP values…")
    from src.explainability.shap_explainer import SHAPExplainer
    from src.explainability.fp_fn_analysis import FPFNAnalyzer

    if "default_12m" in models and cfg["targets"]["default_12m"] in test_df.columns:
        X_test = test_df[feature_cols]
        y_test = test_df[cfg["targets"]["default_12m"]].fillna(0)
        shap_exp = SHAPExplainer(models["default_12m"], feature_cols, "default_12m")
        shap_exp.compute(X_test, max_samples=3000)
        shap_exp.plot_global(save_path="outputs/shap/default_global.png")
        shap_exp.save_shap_csv("outputs/shap/shap_values.csv")

        proba = models["default_12m"].predict_proba(X_test)
        fp_fn = FPFNAnalyzer()
        result = fp_fn.analyze(y_test, proba, X_test, feature_cols, model_name="default_12m")
        fp_fn.export_report(result, "outputs/shap/fp_fn_report.csv")

    # ── Phase 7: Anomaly Detection ────────────────────────────────────────────
    log.info("[PHASE 7] Anomaly detection…")
    from src.anomaly.detector import AnomalyDetector
    from src.anomaly.exception import ExceptionEngine

    detector = AnomalyDetector()
    detector.fit(train_df[feature_cols], feature_cols)
    anomaly_scores = detector.predict(test_df[feature_cols])
    detector.save()

    test_df["anomaly_score"] = anomaly_scores.values
    test_df["anomaly_flag"] = detector.predict_binary(test_df[feature_cols]).values

    exc_engine = ExceptionEngine()
    test_df = exc_engine.run(test_df)
    exc_engine.export_top_examples(test_df, "outputs/anomalies/top20_examples.csv")

    # ── Phase 8: Survival Models ──────────────────────────────────────────────
    log.info("[PHASE 8] Survival models…")
    from src.survival.hazard import HazardModel
    from src.survival.transition import MarkovTransitionModel
    from src.survival.competing_risk import CompetingRiskModel

    hazard_default = HazardModel(event="default")
    hazard_default.fit(train_df)
    hazard_default.save()

    hazard_prepay = HazardModel(event="prepayment")
    hazard_prepay.fit(train_df)
    hazard_prepay.save()

    cr_model = CompetingRiskModel()
    cr_model.fit(train_df)

    transition_model = MarkovTransitionModel(segment_cols=["credit_score_band"])
    transition_model.fit(train_df)

    # ── Phase 9: Scenario Engine ──────────────────────────────────────────────
    log.info("[PHASE 9] Scenario simulations…")
    from src.scenarios.engine import ScenarioEngine

    scenario_engine = ScenarioEngine(
        models={k: calibrated_models.get(k, models.get(k)) for k in models if k != "next_state"},
        transition_model=transition_model,
        feature_cols=feature_cols,
    )
    all_results = scenario_engine.run_all(test_df)
    comparison = scenario_engine.compare(all_results)
    comparison.to_csv("outputs/scenarios/scenario_comparison.csv")
    log.info("Scenario comparison:\n%s", comparison)

    # ── Phase 10: Submission Export ───────────────────────────────────────────
    log.info("[PHASE 10] Generating submission.csv…")
    from src.utils.submission import SubmissionExporter

    top_drivers = []
    if "default_12m" in models:
        X_test = test_df[feature_cols]
        shap_exp2 = SHAPExplainer(models["default_12m"], feature_cols, "default_12m")
        shap_exp2.compute(X_test, max_samples=None)
        top_drivers = shap_exp2.get_top_drivers(n_top=3)

    predictions = pd.DataFrame()
    if "loan_id" in test_df.columns:
        predictions["loan_id"] = test_df["loan_id"].values
    if "month_index" in test_df.columns:
        predictions["month_index"] = test_df["month_index"].values

    for key, col in [
        ("delinquency_3m", "prob_next_3m_delinquency"),
        ("delinquency_6m", "prob_next_6m_delinquency"),
        ("default_12m", "prob_next_12m_default"),
        ("prepayment_12m", "prob_next_12m_prepayment"),
    ]:
        mdl = calibrated_models.get(key, models.get(key))
        if mdl:
            X_test_feat = test_df[feature_cols]
            predictions[col] = np.clip(mdl.predict_proba(X_test_feat), 0, 1)
        else:
            predictions[col] = 0.0

    if "next_state" in models:
        predictions["next_state"] = models["next_state"].predict(test_df[feature_cols])
    else:
        predictions["next_state"] = "Current"

    if "exception_required" in test_df.columns:
        predictions["exception_required"] = test_df["exception_required"].values
        predictions["exception_type"] = test_df.get("exception_type", pd.Series("")).values
    else:
        predictions["exception_required"] = 0
        predictions["exception_type"] = ""

    predictions["anomaly_score"] = test_df.get("anomaly_score", pd.Series(0.0)).values
    
    # Safe alignment of top_drivers length
    if top_drivers and len(top_drivers) == len(predictions):
        predictions["top_drivers"] = top_drivers
    elif top_drivers:
        drivers_list = list(top_drivers) + [""] * max(0, len(predictions) - len(top_drivers))
        predictions["top_drivers"] = drivers_list[:len(predictions)]
    else:
        predictions["top_drivers"] = [""] * len(predictions)

    predictions["action"] = "REVIEW"
    predictions["confidence"] = 0.8


    exporter = SubmissionExporter()
    exporter.export(predictions)

    elapsed = round(time.time() - start, 1)
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %.1f seconds.", elapsed)
    log.info("Submission saved to: %s", cfg["submission"]["output_file"])
    log.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Performance Training Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Stop after ingestion")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
