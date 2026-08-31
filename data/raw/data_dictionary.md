# Loan Performance Intelligence Engine — Data Dictionary

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
