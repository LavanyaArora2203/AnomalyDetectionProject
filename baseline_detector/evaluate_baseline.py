"""
evaluate_baseline.py

Evaluates the rule-based baseline detector against the injected
ground-truth fraud labels.

Inputs
------
outputs/baseline_predictions.csv
data/fraud_labels.csv

Outputs
-------
outputs/baseline_evaluation.csv
Console metrics
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------

OUTPUT_DIR = Path("outputs")

PREDICTIONS_FILE = OUTPUT_DIR / "baseline_predictions.csv"

LABELS_FILE = Path("data/fraud_labels.csv")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------


def load_data():

    predictions = pd.read_csv(PREDICTIONS_FILE)

    labels = pd.read_csv(LABELS_FILE)

    return predictions, labels


# ---------------------------------------------------------
# Merge Predictions with Ground Truth
# ---------------------------------------------------------


def merge_predictions_labels(predictions, labels):

    evaluation = predictions.merge(

        labels[["invoice_id", "fraud_label", "fraud_type"]],

        on="invoice_id",

        how="left",

    )

    # Any invoice not present in fraud_labels is normal
    evaluation["fraud_label"] = (
        evaluation["fraud_label"]
        .fillna(0)
        .astype(int)
    )

    evaluation["fraud_type"] = (
        evaluation["fraud_type"]
        .fillna("normal")
    )

    return evaluation


# ---------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------


def compute_metrics(df):

    y_true = df["fraud_label"]

    y_pred = df["final_prediction"]

    metrics = {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

    }

    return metrics


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------


def print_confusion_matrix(df):

    y_true = df["fraud_label"]

    y_pred = df["final_prediction"]

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix\n")

    print(pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Pred Normal", "Pred Fraud"],
    ))


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------


def print_classification_report(df):

    print("\nClassification Report\n")

    print(

        classification_report(

            df["fraud_label"],

            df["final_prediction"],

            digits=4,

            zero_division=0,

        )

    )


# ---------------------------------------------------------
# Fraud Type Breakdown
# ---------------------------------------------------------


def fraud_type_breakdown(df):

    print("\nFraud Type Detection\n")

    fraud_df = df[df["fraud_label"] == 1]

    if fraud_df.empty:

        print("No fraud labels found.")

        return

    summary = (

        fraud_df

        .groupby("fraud_type")

        .agg(

            Total_Fraud=("invoice_id", "count"),

            Detected=("final_prediction", "sum"),

        )

        .reset_index()

    )

    summary["Recall"] = (

        summary["Detected"]

        /

        summary["Total_Fraud"]

    ).round(3)

    print(summary)

    return summary


# ---------------------------------------------------------
# Save Evaluation File
# ---------------------------------------------------------


def save_results(df):

    output_file = OUTPUT_DIR / "baseline_evaluation.csv"

    df.to_csv(

        output_file,

        index=False,

    )

    print(f"\nSaved evaluation file -> {output_file}")


# ---------------------------------------------------------
# Print Overall Metrics
# ---------------------------------------------------------


def print_metrics(metrics):

    print("\n========== Baseline Evaluation ==========\n")

    for key, value in metrics.items():

        print(f"{key:<12}: {value:.4f}")

    print("\n=========================================")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    predictions, labels = load_data()

    evaluation = merge_predictions_labels(

        predictions,

        labels,

    )

    metrics = compute_metrics(

        evaluation,

    )

    print_metrics(metrics)

    print_confusion_matrix(

        evaluation,

    )

    print_classification_report(

        evaluation,

    )

    fraud_type_breakdown(

        evaluation,

    )

    save_results(

        evaluation,

    )


if __name__ == "__main__":

    main()