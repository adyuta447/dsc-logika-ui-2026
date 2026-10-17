"""Train a leak-aware 12-month 90-DPD model from the supplied CSV files."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve, 
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CHUNK_SIZE = 200_000
TARGET = "target"
APP_ID = "application_id"
CUSTOMER_ID = "customer_id"


def _read_applications(path: Path, has_target: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["application_datetime", "decision_datetime"],
        low_memory=False,
    )
    if frame[APP_ID].isna().any() or frame[APP_ID].duplicated().any():
        raise ValueError(f"{path.name} must contain unique, non-null application IDs")
    if frame[CUSTOMER_ID].isna().any() or frame[CUSTOMER_ID].duplicated().any():
        raise ValueError(f"{path.name} must contain unique, non-null customer IDs")
    if has_target and not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")
    return frame


def _combine_partials(
    parts: list[pd.DataFrame],
    key: str,
    sum_columns: list[str],
    max_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=[key])
    combined = pd.concat(parts).reset_index()
    aggregations = {column: "sum" for column in sum_columns}
    aggregations.update({column: "max" for column in max_columns or []})
    return combined.groupby(key, as_index=False).agg(aggregations)


def build_features(data_dir: Path, applications: pd.DataFrame) -> pd.DataFrame:
    """Build features using only observations available before each application."""
    frame = applications.copy()
    lookup = frame[
        [APP_ID, CUSTOMER_ID, "application_datetime", "decision_datetime"]
    ].copy()
    if lookup[CUSTOMER_ID].duplicated().any():
        raise ValueError("Feature construction requires one application per customer")
    app_by_customer = lookup.set_index(CUSTOMER_ID)[APP_ID]
    app_time_by_customer = lookup.set_index(CUSTOMER_ID)["application_datetime"]

    profile = pd.read_csv(data_dir / "customer_profiles.csv", low_memory=False)
    frame = frame.merge(profile, on=CUSTOMER_ID, how="left", validate="one_to_one")
    if frame["declared_monthly_income"].isna().any():
        raise ValueError("Some applications have no matching customer profile")

    frame["payment_income_ratio"] = (
        frame["proposed_monthly_installment"] / frame["declared_monthly_income"]
    )
    frame["loan_income_ratio"] = (
        frame["loan_amount"] / frame["declared_monthly_income"]
    )
    frame["income_per_dependent"] = frame["declared_monthly_income"] / (
        frame["dependents"] + 1
    )
    frame["application_month"] = frame["application_datetime"].dt.month
    frame["application_weekday"] = frame["application_datetime"].dt.dayofweek

    bureau = pd.read_csv(
        data_dir / "bureau_observations.csv",
        parse_dates=["available_datetime"],
        low_memory=False,
    )
    bureau = bureau.merge(
        lookup[[APP_ID, "decision_datetime"]],
        on=APP_ID,
        how="inner",
        validate="many_to_one",
    )
    bureau = bureau[bureau["available_datetime"] <= bureau["decision_datetime"]]
    bureau_values = [
        "score",
        "active_accounts",
        "outstanding_balance",
        "revolving_credit_limit",
        "revolving_outstanding_balance",
        "utilization",
        "inquiries_last_6m",
        "delinquent_accounts_last_12m",
    ]
    bureau_features = bureau.groupby(APP_ID).agg(
        bureau_report_count=("bureau_provider", "nunique"),
        **{f"bureau_{col}_mean": (col, "mean") for col in bureau_values},
        **{
            f"bureau_{col}_max": (col, "max")
            for col in [
                "score",
                "active_accounts",
                "outstanding_balance",
                "utilization",
                "inquiries_last_6m",
                "delinquent_accounts_last_12m",
            ]
        },
    )
    frame = frame.merge(
        bureau_features, on=APP_ID, how="left", validate="one_to_one"
    )

    previous = pd.read_csv(
        data_dir / "previous_applications.csv",
        parse_dates=["application_datetime", "decision_datetime"],
        low_memory=False,
    )
    current = lookup.rename(
        columns={
            APP_ID: "current_application_id",
            "application_datetime": "current_application_datetime",
        }
    )
    previous = previous.merge(
        current[
            [
                CUSTOMER_ID,
                "current_application_id",
                "current_application_datetime",
            ]
        ],
        on=CUSTOMER_ID,
        how="inner",
        validate="many_to_many",
    )
    previous = previous[
        (previous["application_datetime"] < previous["current_application_datetime"])
        & (previous["decision_datetime"] < previous["current_application_datetime"])
    ]
    prior_features = previous.groupby("current_application_id").agg(
        prior_application_count=("application_id", "size"),
        prior_approved_count=("decision", lambda values: values.eq("approved").sum()),
        prior_rejected_count=("decision", lambda values: values.eq("rejected").sum()),
        prior_cancelled_count=("decision", lambda values: values.eq("cancelled").sum()),
        prior_requested_amount_mean=("requested_amount", "mean"),
        prior_approved_amount_mean=("approved_amount", "mean"),
        prior_installment_mean=("scheduled_monthly_installment", "mean"),
    )
    prior_features.index.name = APP_ID
    prior_features["prior_approval_rate"] = (
        prior_features["prior_approved_count"]
        / prior_features["prior_application_count"]
    )
    frame = frame.merge(
        prior_features, on=APP_ID, how="left", validate="one_to_one"
    )

    repayment_parts: list[pd.DataFrame] = []
    repayment_path = data_dir / "repayment_history.csv"
    repayment_columns = [
        CUSTOMER_ID,
        "due_date",
        "amount_due",
        "amount_paid",
        "last_payment_datetime",
    ]
    for part in pd.read_csv(
        repayment_path,
        usecols=repayment_columns,
        parse_dates=["due_date", "last_payment_datetime"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        part[APP_ID] = part[CUSTOMER_ID].map(app_by_customer)
        part["application_datetime"] = part[CUSTOMER_ID].map(app_time_by_customer)
        part = part[
            part[APP_ID].notna()
            & part["last_payment_datetime"].notna()
            & (part["due_date"] < part["application_datetime"])
            & (part["last_payment_datetime"] < part["application_datetime"])
        ].copy()
        if part.empty:
            continue
        part["paid_late_days"] = (
            part["last_payment_datetime"] - part["due_date"]
        ).dt.total_seconds().div(86_400).clip(lower=0)
        part["late_30_days"] = part["paid_late_days"].ge(30).astype("int8")
        part["late_90_days"] = part["paid_late_days"].ge(90).astype("int8")
        part["paid_fraction"] = (
            part["amount_paid"] / part["amount_due"].replace(0, np.nan)
        )
        repayment_parts.append(
            part.groupby(APP_ID).agg(
                past_installment_count=("due_date", "size"),
                past_amount_due=("amount_due", "sum"),
                past_amount_paid=("amount_paid", "sum"),
                past_paid_late_days_sum=("paid_late_days", "sum"),
                past_paid_late_days_max=("paid_late_days", "max"),
                past_paid_late_30_count=("late_30_days", "sum"),
                past_paid_late_90_count=("late_90_days", "sum"),
                past_paid_fraction_sum=("paid_fraction", "sum"),
            )
        )
    repayment_features = _combine_partials(
        repayment_parts,
        APP_ID,
        [
            "past_installment_count",
            "past_amount_due",
            "past_amount_paid",
            "past_paid_late_days_sum",
            "past_paid_late_30_count",
            "past_paid_late_90_count",
            "past_paid_fraction_sum",
        ],
        ["past_paid_late_days_max"],
    ).set_index(APP_ID)
    if not repayment_features.empty:
        repayment_features["past_paid_ratio"] = (
            repayment_features["past_amount_paid"]
            / repayment_features["past_amount_due"].replace(0, np.nan)
        )
        repayment_features["past_paid_late_days_mean"] = (
            repayment_features["past_paid_late_days_sum"]
            / repayment_features["past_installment_count"]
        )
        repayment_features["past_paid_fraction_mean"] = (
            repayment_features["past_paid_fraction_sum"]
            / repayment_features["past_installment_count"]
        )
        repayment_features["past_late_30_rate"] = (
            repayment_features["past_paid_late_30_count"]
            / repayment_features["past_installment_count"]
        )
        repayment_features["past_late_90_rate"] = (
            repayment_features["past_paid_late_90_count"]
            / repayment_features["past_installment_count"]
        )
    frame = frame.merge(
        repayment_features, on=APP_ID, how="left", validate="one_to_one"
    )

    account_parts: list[pd.DataFrame] = []
    account_path = data_dir / "account_behavior.csv"
    account_columns = [
        CUSTOMER_ID,
        "month",
        "inflows",
        "outflows",
        "income_like_credits",
        "closing_balance",
        "minimum_balance",
        "low_balance_days",
    ]
    for part in pd.read_csv(
        account_path,
        usecols=account_columns,
        parse_dates=["month"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        part[APP_ID] = part[CUSTOMER_ID].map(app_by_customer)
        part["application_datetime"] = part[CUSTOMER_ID].map(app_time_by_customer)
        application_month_start = (
            part["application_datetime"].dt.to_period("M").dt.to_timestamp()
        )
        part = part[
            part[APP_ID].notna() & (part["month"] < application_month_start)
        ]
        if part.empty:
            continue
        account_parts.append(
            part.groupby(APP_ID).agg(
                account_month_count=("month", "size"),
                account_inflows_sum=("inflows", "sum"),
                account_outflows_sum=("outflows", "sum"),
                account_income_credits_sum=("income_like_credits", "sum"),
                account_closing_balance_sum=("closing_balance", "sum"),
                account_minimum_balance_sum=("minimum_balance", "sum"),
                account_low_balance_days_sum=("low_balance_days", "sum"),
            )
        )
    account_features = _combine_partials(
        account_parts,
        APP_ID,
        [
            "account_month_count",
            "account_inflows_sum",
            "account_outflows_sum",
            "account_income_credits_sum",
            "account_closing_balance_sum",
            "account_minimum_balance_sum",
            "account_low_balance_days_sum",
        ],
    ).set_index(APP_ID)
    if not account_features.empty:
        months = account_features["account_month_count"].replace(0, np.nan)
        for source, result in [
            ("account_inflows_sum", "account_inflows_mean"),
            ("account_outflows_sum", "account_outflows_mean"),
            ("account_income_credits_sum", "account_income_credits_mean"),
            ("account_closing_balance_sum", "account_closing_balance_mean"),
            ("account_low_balance_days_sum", "account_low_balance_days_mean"),
            ("account_minimum_balance_sum", "account_minimum_balance_mean"),
        ]:
            account_features[result] = account_features[source] / months
        account_features["account_cashflow_ratio"] = (
            account_features["account_outflows_sum"]
            / account_features["account_inflows_sum"].replace(0, np.nan)
        )
    frame = frame.merge(
        account_features, on=APP_ID, how="left", validate="one_to_one"
    )

    ecommerce_parts: list[pd.DataFrame] = []
    ecommerce_path = data_dir / "ecommerce_transactions.csv"
    ecommerce_columns = [
        CUSTOMER_ID,
        "transaction_datetime",
        "order_item_id",
        "item_total",
        "quantity",
        "order_status",
    ]
    for part in pd.read_csv(
        ecommerce_path,
        usecols=ecommerce_columns,
        parse_dates=["transaction_datetime"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        part[APP_ID] = part[CUSTOMER_ID].map(app_by_customer)
        part["application_datetime"] = part[CUSTOMER_ID].map(app_time_by_customer)
        part = part[
            part[APP_ID].notna()
            & (part["transaction_datetime"] < part["application_datetime"])
            & part["order_status"].eq("completed")
        ]
        if part.empty:
            continue
        ecommerce_parts.append(
            part.groupby(APP_ID).agg(
                ecommerce_completed_item_count=("order_item_id", "size"),
                ecommerce_item_spend=("item_total", "sum"),
                ecommerce_quantity=("quantity", "sum"),
                ecommerce_last_transaction=("transaction_datetime", "max"),
            )
        )
    ecommerce_features = _combine_partials(
        ecommerce_parts,
        APP_ID,
        [
            "ecommerce_completed_item_count",
            "ecommerce_item_spend",
            "ecommerce_quantity",
        ],
        ["ecommerce_last_transaction"],
    ).set_index(APP_ID)
    if not ecommerce_features.empty:
        ecommerce_features["ecommerce_days_since_last_purchase"] = (
            frame.set_index(APP_ID)["application_datetime"]
            .sub(ecommerce_features["ecommerce_last_transaction"])
            .dt.total_seconds()
            .div(86_400)
        )
        ecommerce_features = ecommerce_features.drop(
            columns="ecommerce_last_transaction"
        )
    frame = frame.merge(
        ecommerce_features, on=APP_ID, how="left", validate="one_to_one"
    )

    return frame


def _make_model(features: pd.DataFrame) -> Pipeline:
    categorical = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    numeric = [column for column in features.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=True,
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _train_and_predict(
    data_dir: Path,
    output_dir: Path,
    as_of_date: pd.Timestamp,
    validation_months: int,
) -> dict[str, Any]:
    train = _read_applications(data_dir / "train_applications.csv", has_target=True)
    test = _read_applications(data_dir / "test_applications.csv", has_target=False)
    all_applications = pd.concat([train, test], ignore_index=True, sort=False)
    if all_applications[CUSTOMER_ID].duplicated().any():
        raise ValueError("Train/test customers overlap; feature joins would be ambiguous")

    print("Building as-of features for train and test applications...", flush=True)
    features = build_features(data_dir, all_applications)
    excluded = {
        APP_ID,
        CUSTOMER_ID,
        "application_datetime",
        "decision_datetime",
        TARGET,
    }
    feature_columns = [column for column in features.columns if column not in excluded]
    training_features = features.iloc[: len(train)][feature_columns].copy()
    test_features = features.iloc[len(train) :][feature_columns].copy()

    maturity_cutoff = as_of_date - pd.DateOffset(months=12)
    mature_mask = train["application_datetime"] <= maturity_cutoff
    mature_train = train.loc[mature_mask].copy()
    if mature_train.empty:
        raise ValueError(
            "No training labels are at least 12 months old for the supplied as-of date"
        )
    if mature_train[TARGET].nunique() != 2:
        raise ValueError("Mature training labels must contain both target classes")

    mature_positions = np.flatnonzero(mature_mask.to_numpy())
    latest_month = mature_train["application_datetime"].dt.to_period("M").max()
    validation_start = latest_month - (validation_months - 1)
    validation_mask = (
        mature_train["application_datetime"].dt.to_period("M") >= validation_start
    )
    validation_positions = mature_positions[validation_mask.to_numpy()]
    fitting_positions = mature_positions[~validation_mask.to_numpy()]
    if len(validation_positions) == 0 or len(fitting_positions) == 0:
        raise ValueError("Chronological validation split is empty")
    y_train = train[TARGET].astype(int)
    if y_train.iloc[fitting_positions].nunique() != 2:
        raise ValueError("Chronological training split must contain both target classes")
    if y_train.iloc[validation_positions].nunique() != 2:
        raise ValueError("Chronological validation split must contain both target classes")

    validation_model = _make_model(training_features)
    validation_model.fit(
        training_features.iloc[fitting_positions], y_train.iloc[fitting_positions]
    )
    validation_probabilities = validation_model.predict_proba(
        training_features.iloc[validation_positions]
    )[:, 1]
    validation_targets = y_train.iloc[validation_positions].to_numpy()
    fpr, tpr, thresholds = roc_curve(validation_targets, validation_probabilities)
    ks_score = float(np.max(tpr - fpr))
    metrics = {
        "as_of_date": as_of_date.date().isoformat(),
        "ks_score": ks_score,
        "label_maturity_cutoff": maturity_cutoff.isoformat(),
        "mature_training_rows": int(len(mature_positions)),
        "excluded_immature_training_rows": int((~mature_mask).sum()),
        "validation_start_month": str(validation_start),
        "validation_rows": int(len(validation_positions)),
        "validation_default_rate": float(validation_targets.mean()),
        "roc_auc": float(roc_auc_score(validation_targets, validation_probabilities)),
        "pr_auc": float(
            average_precision_score(validation_targets, validation_probabilities)
        ),
        "pr_auc_no_skill_baseline": float(validation_targets.mean()),
        "brier_score": float(
            brier_score_loss(validation_targets, validation_probabilities)
        ),
    }
    validation_frame = pd.DataFrame(
        {"target": validation_targets, "probability": validation_probabilities}
    )
    validation_frame["risk_decile"] = pd.qcut(
        validation_frame["probability"], 10, duplicates="drop"
    )
    risk_deciles = (
        validation_frame.groupby("risk_decile", observed=True)["target"]
        .agg(["size", "mean"])
        .rename(columns={"size": "applications", "mean": "default_rate"})
        .reset_index()
    )
    metrics["validation_risk_deciles"] = [
        {
            "applications": int(row.applications),
            "default_rate": float(row.default_rate),
        }
        for row in risk_deciles.itertuples(index=False)
    ]

    final_model = _make_model(training_features)
    final_model.fit(
        training_features.iloc[mature_positions], y_train.iloc[mature_positions]
    )
    default_probability = final_model.predict_proba(test_features)[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            APP_ID: test[APP_ID].to_numpy(),
            "default_probability": default_probability,
            "probability_no_90_dpd_within_12m": 1 - default_probability,
        }
    )
    predictions.to_csv(output_dir / "default_predictions.csv", index=False)
    predictions[[APP_ID, "default_probability"]].to_csv(
        output_dir / "submission_ready.csv", index=False
    )
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "target_definition": "1 if loan reaches at least 90 DPD within 12 calendar months",
            "training_as_of_date": as_of_date.date().isoformat(),
            "label_maturity_cutoff": maturity_cutoff.isoformat(),
        },
        output_dir / "default_model.joblib",
    )
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved predictions to {output_dir / 'default_predictions.csv'}")
    print(f"Saved model to {output_dir / 'default_model.joblib'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the supplied CSV files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model_output",
        help="Directory for model, predictions, and validation metrics",
    )
    parser.add_argument(
        "--as-of-date",
        type=pd.Timestamp,
        default=pd.Timestamp(date.today()),
        help="Only train on labels that have a full 12-month outcome window",
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=3,
        help="Number of latest mature calendar months reserved for validation",
    )
    args = parser.parse_args()
    if args.validation_months < 1:
        parser.error("--validation-months must be at least 1")
    _train_and_predict(
        args.data_dir,
        args.output_dir,
        pd.Timestamp(args.as_of_date).normalize(),
        args.validation_months,
    )


if __name__ == "__main__":
    main()
