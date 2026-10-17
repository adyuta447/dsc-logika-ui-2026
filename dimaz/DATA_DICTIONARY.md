# Data dictionary — v2

All amounts are whole IDR. Timestamps use local Asia/Jakarta time.

## train_applications

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `application_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `application_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `decision_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `loan_amount` | integer | False | Whole Indonesian rupiah (IDR). |
| `term_months` | integer | False | Contractual tenor in months; selected loans have 12-month terms. |
| `annual_interest_rate` | number | False | Nominal annual decimal rate; monthly rate equals this divided by 12. |
| `proposed_monthly_installment` | integer | False | Whole Indonesian rupiah (IDR). |
| `loan_purpose` | string | False | Declared loan purpose. Values: education, home_improvement, household, working_capital, medical. |
| `target` | integer | False | 1 if any end-of-day loan DPD reaches 90 within 12 calendar months; otherwise 0. Values: 0, 1. |

## test_applications

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `application_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `application_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `decision_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `loan_amount` | integer | False | Whole Indonesian rupiah (IDR). |
| `term_months` | integer | False | Contractual tenor in months; selected loans have 12-month terms. |
| `annual_interest_rate` | number | False | Nominal annual decimal rate; monthly rate equals this divided by 12. |
| `proposed_monthly_installment` | integer | False | Whole Indonesian rupiah (IDR). |
| `loan_purpose` | string | False | Declared loan purpose. Values: education, home_improvement, household, working_capital, medical. |

## customer_profiles

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `age` | integer | False | Completed years of age at the selected decision. |
| `region` | string | False | Fictional region; no real geographic correspondence. Values: Region_A, Region_B, Region_C, Region_D, Region_E. |
| `employment_type` | string | False | Current income-generating activity. Values: salaried, self_employed, variable_income. |
| `employment_tenure_months` | integer | False | Months in the current occupation. |
| `declared_monthly_income` | integer | False | Whole Indonesian rupiah (IDR). |
| `dependents` | integer | False | Number of financial dependents. |
| `housing_status` | string | False | Current housing arrangement. Values: owned, rented, family. |

## previous_applications

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `application_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `application_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `decision_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `requested_amount` | integer | False | Whole Indonesian rupiah (IDR). |
| `term_months` | integer | False | Nonnegative count. |
| `decision` | string | False | Historical application outcome. Values: approved, rejected, cancelled. |
| `loan_id` | string | True | Originated loan; blank for rejected/cancelled requests. |
| `approved_amount` | integer | True | Approved principal, IDR; blank unless originated. |
| `annual_interest_rate` | number | True | Nominal annual decimal rate; blank unless originated. |
| `scheduled_monthly_installment` | integer | True | Regular amortizing installment, IDR; final installment adjusts rounding. Blank unless originated. |

## repayment_history

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `loan_id` | string | False | Opaque identifier; never a numerical model feature. |
| `installment_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `due_date` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `amount_due` | integer | False | Whole Indonesian rupiah (IDR). |
| `amount_paid` | integer | False | Whole Indonesian rupiah (IDR). |
| `last_payment_datetime` | datetime | True | Last allocation to this installment strictly before cutoff; blank when unpaid. |
| `days_past_due` | integer | False | Nonnegative count. |
| `max_days_past_due` | integer | False | Nonnegative count. |

## account_behavior

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `month` | date | False | First day of a complete observed calendar month. |
| `opening_balance` | integer | False | Whole Indonesian rupiah (IDR). |
| `closing_balance` | integer | False | Whole Indonesian rupiah (IDR). |
| `inflows` | integer | False | Whole Indonesian rupiah (IDR). |
| `outflows` | integer | False | Whole Indonesian rupiah (IDR). |
| `income_like_credits` | integer | False | Whole Indonesian rupiah (IDR). |
| `average_balance` | integer | False | Mean end-of-day balance, rounded to whole IDR. |
| `minimum_balance` | integer | False | Minimum end-of-day balance, IDR. |
| `low_balance_days` | integer | False | Days with end-of-day balance below IDR 250,000. |

## bureau_observations

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `bureau_observation_id` | string | False | Opaque identifier; never a numerical model feature. |
| `application_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `bureau_provider` | string | False | Fictional reporting provider. Values: Bureau_A, Bureau_B, Bureau_C. |
| `report_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `available_datetime` | datetime | False | ISO local Asia/Jakarta datetime; no daylight-saving adjustment. |
| `score` | number | True | Provider score, 300–850; higher is better. Missing for an unscored report. |
| `active_accounts` | integer | False | Open credit accounts covered by this provider. |
| `outstanding_balance` | integer | False | Reported debt on covered open accounts only, IDR. |
| `revolving_credit_limit` | integer | False | Whole Indonesian rupiah (IDR). |
| `revolving_outstanding_balance` | integer | False | Whole Indonesian rupiah (IDR). |
| `utilization` | number | True | Revolving balance / revolving limit; undefined without a positive limit. |
| `inquiries_last_6m` | integer | False | Nonnegative count. |
| `delinquent_accounts_last_12m` | integer | False | Accounts with delinquency in the report lookback, including now-closed accounts. |

## ecommerce_transactions

| Column | Type | Nullable | Definition |
|---|---|---|---|
| `order_id` | string | False | Opaque identifier; never a numerical model feature. |
| `order_item_id` | string | False | Opaque identifier; never a numerical model feature. |
| `customer_id` | string | False | Opaque identifier; never a numerical model feature. |
| `transaction_datetime` | datetime | False | Completed/cancelled order event time; status known at this event. |
| `product_category` | string | False | Item category. Values: groceries, fashion, electronics, home, health, education, personal_care. |
| `quantity` | integer | False | Nonnegative count. |
| `item_total` | integer | False | IDR value including quantity; cancelled basket values are not spending. |
| `order_total` | integer | False | Sum of item totals, repeated on each item. Deduplicate orders before aggregating. |
| `payment_method` | string | False | Order-level method. Values: COD, e_wallet, bank_transfer, debit_card, credit_card. |
| `order_status` | string | False | Status known at event time. Values: completed, cancelled. |
