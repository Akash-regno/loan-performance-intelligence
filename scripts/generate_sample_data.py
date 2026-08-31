"""
scripts/generate_sample_data.py
--------------------------------
Generates a complete, realistic sample dataset pack conforming to the
Intain Campus FinTech Challenge 2026 specifications:

1. data/raw/data_dictionary.md
2. data/raw/validation_rules.json
3. data/raw/macro_scenarios.csv
4. data/raw/loan_static_attributes.csv
5. data/raw/servicer_updates.csv
6. data/raw/loan_monthly_performance_train.csv
7. data/raw/loan_monthly_performance_test.csv
8. data/raw/submission_template.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("1. Generating data_dictionary.md...")
data_dictionary_content = """# Loan Performance Intelligence Engine — Data Dictionary

## Identifiers & Temporal Fields
- `loan_id` (str): Unique loan identifier (e.g. LN000001).
- `month_index` (int): Sequential performance month index starting at 1.
- `reporting_month` (str): Monthly cycle date formatted as YYYY-MM.
- `origination_month` (str): Loan origination month formatted as YYYY-MM.
- `loan_age_months` (int): Elapsed loan age in months since origination.
- `remaining_term_months` (int): Remaining amortization term in months.

## Loan Balances & Financials
- `original_balance` (float): Disbursed loan amount at origination ($).
- `current_balance` (float): Unpaid principal balance (UPB) as of reporting month ($).
- `interest_rate` (float): Annual interest rate in percentage points (e.g. 4.5).
- `credit_score_band` (str): FICO score range at origination (<620, 620-639, ..., 780+).
- `ltv_band` (str): Loan-to-Value percentage band (0-60, 60-70, ..., 97+).
- `dti_band` (str): Debt-to-Income percentage band (0-20, 20-30, ..., 45+).

## Property & Servicing
- `state` (str): US State code (CA, TX, FL, NY, IL, etc.).
- `loan_purpose` (str): Purchase, Refinance, CashOut.
- `occupancy_type` (str): Primary, Secondary, Investment.
- `property_type` (str): Single Family, Condo, Multi-Family, Townhouse.
- `servicer_name` (str): Primary servicing entity name.
- `source_system` (str): Data origin (Servicer_Core, Subservicer_Portal, Direct_Feed).
- `document_status` (str): Complete, Pending_Review, Missing_Docs.
- `last_updated_at` (str): Timestamp of last tape update.

## Performance & Risk States
- `current_status` (str): Current delinquency state (Current, 30DPD, 60DPD, 90DPD, Default, Prepaid, Liquidated).
- `days_past_due` (int): Number of days payments are delinquent.
- `modification_flag` (int): 1 if loan has undergone modification, else 0.
- `prepayment_flag` (int): 1 if loan was prepaid in full, else 0.
- `default_flag` (int): 1 if loan entered default/charge-off, else 0.
- `loss_severity_band` (str): Expected loss severity band (0-10, 10-20, ..., 80+).

## Forward-Looking Target Labels (Train Only)
- `next_3m_delinquency_flag` (int): 1 if loan transitions to >=30 DPD within next 3 months.
- `next_6m_delinquency_flag` (int): 1 if loan transitions to >=60 DPD within next 6 months.
- `next_12m_default_flag` (int): 1 if loan defaults within next 12 months.
- `next_12m_prepayment_flag` (int): 1 if loan prepays within next 12 months.
- `next_state` (str): Most probable next state in next month.
- `exception_required` (int): 1 if record requires exception review.
- `exception_type` (str): Type of exception triggered (e.g. status_conflict, balance_error).
"""
(RAW_DIR / "data_dictionary.md").write_text(data_dictionary_content, encoding="utf-8")

print("2. Generating validation_rules.json...")
validation_rules_content = [
    {
        "rule_id": "VR01_BALANCE_LEQ_ORIGINAL",
        "description": "current_balance cannot exceed original_balance without modification_flag=1",
        "column": "current_balance",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR02_POSITIVE_BALANCES",
        "description": "original_balance must be > 0 and current_balance must be >= 0",
        "column": "current_balance",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR03_NON_NEGATIVE_AGE",
        "description": "loan_age_months must be >= 0",
        "column": "loan_age_months",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR04_REMAINING_TERM",
        "description": "remaining_term_months must be >= 0 and <= 360",
        "column": "remaining_term_months",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR05_STATUS_DPD_CONSISTENCY",
        "description": "If current_status is Current, days_past_due must be 0; if 30DPD, DPD in [30,59]",
        "column": "current_status",
        "severity": "HIGH"
    },
    {
        "rule_id": "VR06_NON_NEGATIVE_DPD",
        "description": "days_past_due must be >= 0",
        "column": "days_past_due",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR07_DEFAULT_HIGH_DPD",
        "description": "If default_flag=1 or status=Default, days_past_due should typically be >= 90",
        "column": "default_flag",
        "severity": "MEDIUM"
    },
    {
        "rule_id": "VR08_PREPAID_ZERO_BALANCE",
        "description": "If prepayment_flag=1 or status=Prepaid, current_balance must equal 0",
        "column": "prepayment_flag",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR09_PREPAID_XOR_DEFAULT",
        "description": "A loan cannot be simultaneously marked Prepaid and Default in the same cycle",
        "column": "prepayment_flag",
        "severity": "CRITICAL"
    },
    {
        "rule_id": "VR10_INTEREST_RATE_RANGE",
        "description": "interest_rate must be between 1.0 and 20.0 percent",
        "column": "interest_rate",
        "severity": "HIGH"
    }
]
with open(RAW_DIR / "validation_rules.json", "w", encoding="utf-8") as f:
    json.dump(validation_rules_content, f, indent=2)

print("3. Generating macro_scenarios.csv...")
macro_df = pd.DataFrame([
    {
        "scenario_name": "base",
        "interest_rate_shift_bps": 0,
        "hpi_growth_pct": 3.5,
        "unemployment_rate_pct": 3.8,
        "description": "Baseline macroeconomic outlook with stable growth"
    },
    {
        "scenario_name": "adverse",
        "interest_rate_shift_bps": 300,
        "hpi_growth_pct": -15.0,
        "unemployment_rate_pct": 7.8,
        "description": "Severe stagflation with sharp rate hike and home price contraction"
    },
    {
        "scenario_name": "high_prepayment",
        "interest_rate_shift_bps": -150,
        "hpi_growth_pct": 10.0,
        "unemployment_rate_pct": 3.5,
        "description": "Rapid monetary easing and strong housing appreciation prompting refinancing surge"
    }
])
macro_df.to_csv(RAW_DIR / "macro_scenarios.csv", index=False)

print("4. Generating static loan attributes and longitudinal panel data...")
N_LOANS_TRAIN = 3000
N_LOANS_TEST = 600
N_MONTHS = 24

CREDIT_BANDS = ["<620", "620-639", "640-659", "660-679", "680-699", "700-719", "720-739", "740-759", "760-779", "780+"]
LTV_BANDS = ["0-60", "60-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-97", "97+"]
DTI_BANDS = ["0-20", "20-30", "30-36", "36-43", "43-45", "45+"]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "WA", "AZ", "CO"]
SERVICERS = ["Apex_Servicing", "Beacon_Mortgage", "Crestline_Financial", "Delta_Servicing", "Evergreen_Loan_Ops"]
PROPERTY_TYPES = ["Single Family", "Condo", "Multi-Family", "Townhouse"]
PURPOSES = ["Purchase", "Refinance", "CashOut"]
OCCUPANCIES = ["Primary", "Secondary", "Investment"]

def generate_static_records(n_loans, start_id=1):
    records = []
    for i in range(n_loans):
        loan_id = f"LN{start_id + i:06d}"
        orig_year = np.random.choice([2019, 2020, 2021, 2022])
        orig_month = np.random.choice(range(1, 13))
        origination_month = f"{orig_year}-{orig_month:02d}"
        original_balance = round(float(np.random.choice([150000, 220000, 310000, 425000, 550000, 680000]) + np.random.uniform(-10000, 10000)), 2)
        credit_band = np.random.choice(CREDIT_BANDS, p=[0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05])
        ltv_band = np.random.choice(LTV_BANDS, p=[0.15, 0.15, 0.15, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02])
        dti_band = np.random.choice(DTI_BANDS, p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
        state = np.random.choice(STATES)
        property_type = np.random.choice(PROPERTY_TYPES, p=[0.7, 0.15, 0.05, 0.10])
        loan_purpose = np.random.choice(PURPOSES, p=[0.55, 0.30, 0.15])
        occupancy_type = np.random.choice(OCCUPANCIES, p=[0.80, 0.10, 0.10])
        servicer_name = np.random.choice(SERVICERS)
        base_rate = 3.25 + (2022 - orig_year) * 0.75 + (0.5 if credit_band in ["<620", "620-639"] else 0.0)
        interest_rate = round(float(base_rate + np.random.normal(0, 0.25)), 3)

        records.append({
            "loan_id": loan_id,
            "origination_month": origination_month,
            "original_balance": original_balance,
            "interest_rate": interest_rate,
            "credit_score_band": credit_band,
            "ltv_band": ltv_band,
            "dti_band": dti_band,
            "state": state,
            "property_type": property_type,
            "loan_purpose": loan_purpose,
            "occupancy_type": occupancy_type,
            "servicer_name": servicer_name,
            "loss_severity_band": np.random.choice(["10-20", "20-30", "30-40", "40-50", "50-60"]),
            "source_system": np.random.choice(["Servicer_Core", "Subservicer_Portal", "Direct_Feed"], p=[0.7, 0.2, 0.1]),
            "document_status": np.random.choice(["Complete", "Pending_Review", "Missing_Docs"], p=[0.90, 0.07, 0.03])
        })
    return pd.DataFrame(records)

static_train = generate_static_records(N_LOANS_TRAIN, start_id=1)
static_test = generate_static_records(N_LOANS_TEST, start_id=N_LOANS_TRAIN + 1)
static_all = pd.concat([static_train, static_test], ignore_index=True)
static_all.to_csv(RAW_DIR / "loan_static_attributes.csv", index=False)

def generate_panel(static_df, is_train=True):
    panel_rows = []
    
    for _, s in static_df.iterrows():
        loan_id = s["loan_id"]
        orig_y, orig_m = map(int, s["origination_month"].split("-"))
        balance = s["original_balance"]
        rate = s["interest_rate"]
        state_status = "Current"
        dpd = 0
        mod_flag = 0
        prepay_flag = 0
        default_flag = 0
        
        # Risk factor based on credit and LTV
        risk_level = 0.02
        if s["credit_score_band"] in ["<620", "620-639"]:
            risk_level += 0.08
        if s["ltv_band"] in ["90-95", "95-97", "97+"]:
            risk_level += 0.05
            
        months_to_simulate = N_MONTHS
        
        for m_idx in range(1, months_to_simulate + 1):
            cur_year = orig_y + (orig_m + m_idx - 2) // 12
            cur_month = (orig_m + m_idx - 2) % 12 + 1
            reporting_month = f"{cur_year}-{cur_month:02d}"
            loan_age = m_idx + (2022 - orig_y) * 12
            remaining_term = max(0, 360 - loan_age)
            
            # State transitions
            if state_status == "Current":
                if np.random.rand() < risk_level:
                    state_status = "30DPD"
                    dpd = int(np.random.choice([30, 35, 45]))
                elif np.random.rand() < 0.015:
                    state_status = "Prepaid"
                    prepay_flag = 1
                    balance = 0.0
                    dpd = 0
                else:
                    balance = max(0.0, balance - s["original_balance"] / 360.0)
                    dpd = 0
            elif state_status == "30DPD":
                roll = np.random.rand()
                if roll < 0.50:  # Cure
                    state_status = "Current"
                    dpd = 0
                elif roll < 0.80: # Stay
                    dpd = min(59, dpd + 30)
                else: # Degrade
                    state_status = "60DPD"
                    dpd = 60
            elif state_status == "60DPD":
                roll = np.random.rand()
                if roll < 0.30:  # Cure
                    state_status = "30DPD"
                    dpd = 30
                elif roll < 0.60:
                    dpd = min(89, dpd + 30)
                else:
                    state_status = "Default"
                    default_flag = 1
                    dpd = 90
            elif state_status in ["Default", "Prepaid", "Liquidated"]:
                dpd = 0 if state_status == "Prepaid" else dpd
                balance = 0.0 if state_status == "Prepaid" else balance
                
            # Random anomalies injection (small rate ~ 1.5%)
            if np.random.rand() < 0.015 and state_status not in ["Prepaid", "Default"]:
                anom_type = np.random.choice(["status_mismatch", "balance_spike"])
                if anom_type == "status_mismatch":
                    dpd = 65  # conflict with Current
                elif anom_type == "balance_spike":
                    balance = s["original_balance"] * 1.05

            row = {
                "loan_id": loan_id,
                "month_index": m_idx,
                "reporting_month": reporting_month,
                "origination_month": s["origination_month"],
                "loan_age_months": loan_age,
                "remaining_term_months": remaining_term,
                "original_balance": s["original_balance"],
                "current_balance": round(balance, 2),
                "interest_rate": rate,
                "credit_score_band": s["credit_score_band"],
                "ltv_band": s["ltv_band"],
                "dti_band": s["dti_band"],
                "state": s["state"],
                "loan_purpose": s["loan_purpose"],
                "occupancy_type": s["occupancy_type"],
                "property_type": s["property_type"],
                "servicer_name": s["servicer_name"],
                "current_status": state_status,
                "days_past_due": dpd,
                "modification_flag": mod_flag,
                "prepayment_flag": prepay_flag,
                "default_flag": default_flag,
                "loss_severity_band": s["loss_severity_band"],
                "last_updated_at": f"{reporting_month}-28 00:00:00",
                "source_system": s["source_system"],
                "document_status": s["document_status"]
            }
            
            # Forward targets for train
            if is_train:
                row["next_3m_delinquency_flag"] = 1 if (risk_level > 0.05 or dpd > 0) and np.random.rand() < 0.7 else 0
                row["next_6m_delinquency_flag"] = 1 if (risk_level > 0.08 or dpd >= 30) and np.random.rand() < 0.65 else 0
                row["next_12m_default_flag"] = 1 if (dpd >= 60 or (risk_level > 0.10 and np.random.rand() < 0.4)) else 0
                row["next_12m_prepayment_flag"] = 1 if prepay_flag == 1 or (rate > 5.0 and np.random.rand() < 0.25) else 0
                row["next_state"] = state_status if state_status in ["Default", "Prepaid"] else ("30DPD" if dpd > 0 else "Current")
                row["exception_required"] = 1 if (balance > s["original_balance"] or (state_status == "Current" and dpd > 0)) else 0
                row["exception_type"] = "status_conflict" if (state_status == "Current" and dpd > 0) else ("balance_error" if balance > s["original_balance"] else "")
                
            panel_rows.append(row)
            
            if state_status in ["Prepaid", "Default"]:
                break
                
    return pd.DataFrame(panel_rows)

print("5. Generating loan_monthly_performance_train.csv...")
train_panel = generate_panel(static_train, is_train=True)
train_panel.to_csv(RAW_DIR / "loan_monthly_performance_train.csv", index=False)
print(f"   Train rows: {len(train_panel):,}")

print("6. Generating loan_monthly_performance_test.csv...")
test_panel = generate_panel(static_test, is_train=False)
test_panel.to_csv(RAW_DIR / "loan_monthly_performance_test.csv", index=False)
print(f"   Test rows: {len(test_panel):,}")

print("7. Generating servicer_updates.csv...")
servicer_updates = []
for _, r in test_panel.sample(min(len(test_panel), 2000), random_state=42).iterrows():
    # Occasionally conflict with panel
    conflicting = np.random.rand() < 0.10
    reported_dpd = r["days_past_due"] + (30 if conflicting else 0)
    reported_balance = r["current_balance"] * (1.05 if conflicting else 1.0)
    
    servicer_updates.append({
        "loan_id": r["loan_id"],
        "reporting_month": r["reporting_month"],
        "servicer_name": r["servicer_name"],
        "reported_status": "30DPD" if reported_dpd >= 30 else "Current",
        "reported_dpd": int(reported_dpd),
        "reported_balance": round(float(reported_balance), 2),
        "servicer_conflict_flag": 1 if conflicting else 0,
        "stale_record_flag": 1 if np.random.rand() < 0.05 else 0,
        "update_timestamp": f"{r['reporting_month']}-27 18:00:00"
    })
pd.DataFrame(servicer_updates).to_csv(RAW_DIR / "servicer_updates.csv", index=False)

print("8. Generating submission_template.csv...")
sub_template = pd.DataFrame({
    "loan_id": test_panel["loan_id"],
    "month_index": test_panel["month_index"],
    "prob_next_3m_delinquency": 0.0,
    "prob_next_6m_delinquency": 0.0,
    "prob_next_12m_default": 0.0,
    "prob_next_12m_prepayment": 0.0,
    "next_state": "Current",
    "exception_required": 0,
    "exception_type": "",
    "anomaly_score": 0.0,
    "top_drivers": "",
    "action": "REVIEW",
    "confidence": 0.80
})
sub_template.to_csv(RAW_DIR / "submission_template.csv", index=False)

print("\nSUCCESS: All 8 data pack files successfully generated in data/raw/")
