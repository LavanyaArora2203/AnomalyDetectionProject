"""
models/evaluate_lof.py

Evaluate Local Outlier Factor predictions against
ground truth fraud labels.
"""

from pathlib import Path

import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.mlflow_logger import (
    setup_experiment,
    log_metrics,
    log_artifact
)

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("outputs")

PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "lof_predictions.csv"
)

LABELS_FILE = Path(
    "data/fraud_labels.csv"
)

# ============================================================
# Load Data
# ============================================================

def load_data():

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    labels = pd.read_csv(
        LABELS_FILE
    )

    return predictions, labels


# ============================================================
# Merge Predictions + Labels
# ============================================================

def merge_data(
    predictions,
    labels
):

    evaluation = predictions.merge(

        labels[
            [
                "invoice_id",
                "fraud_label",
                "fraud_type"
            ]
        ],

        on="invoice_id",

        how="left"

    )

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


# ============================================================
# Metrics
# ============================================================

def compute_metrics(df):

    y_true = df["fraud_label"]

    y_pred = df["prediction"]

    metrics = {

        "Accuracy":

            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":

            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":

            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":

            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

    }

    return metrics


# ============================================================
# Print Metrics
# ============================================================

def print_metrics(metrics):

    print("\n==============================")
    print("LOF Evaluation")
    print("==============================\n")

    for key, value in metrics.items():

        print(

            f"{key:<12}: {value:.4f}"

        )

    print()


# ============================================================
# Confusion Matrix
# ============================================================

def print_confusion(df):

    cm = confusion_matrix(

        df["fraud_label"],

        df["prediction"]

    )

    cm = pd.DataFrame(

        cm,

        index=[

            "Actual Normal",

            "Actual Fraud"

        ],

        columns=[

            "Pred Normal",

            "Pred Fraud"

        ]

    )

    print("Confusion Matrix\n")

    print(cm)


# ============================================================
# Classification Report
# ============================================================

def print_report(df):

    print("\nClassification Report\n")

    print(

        classification_report(

            df["fraud_label"],

            df["prediction"],

            digits=4,

            zero_division=0

        )

    )


# ============================================================
# Fraud Type Recall
# ============================================================

def fraud_breakdown(df):

    frauds = df[

        df["fraud_label"] == 1

    ]

    summary = (

        frauds

        .groupby("fraud_type")

        .agg(

            total=(

                "invoice_id",

                "count"

            ),

            detected=(

                "prediction",

                "sum"

            )

        )

        .reset_index()

    )

    summary["recall"] = (

        summary["detected"]

        /

        summary["total"]

    ).round(3)

    print("\nFraud Type Recall\n")

    print(summary)

    return summary


# ============================================================
# Save Evaluation
# ============================================================

def save_results(df):

    output_file = (

        OUTPUT_DIR /

        "lof_evaluation.csv"

    )

    df.to_csv(

        output_file,

        index=False

    )

    print(

        f"\nSaved evaluation -> {output_file}"

    )


# ============================================================
# Main
# ============================================================

def main():

    predictions, labels = load_data()

    evaluation = merge_data(

        predictions,

        labels

    )

    metrics = compute_metrics(

        evaluation

    )

    setup_experiment()

    with mlflow.start_run(

        run_name="LOF_Evaluation"

    ):

        log_metrics(metrics)

        print_metrics(

            metrics

        )

        print_confusion(

            evaluation

        )

        print_report(

            evaluation

        )

        fraud_breakdown(

            evaluation

        )

        save_results(

            evaluation

        )

        log_artifact(

            OUTPUT_DIR /

            "lof_predictions.csv"

        )

        log_artifact(

            OUTPUT_DIR /

            "lof_evaluation.csv"

        )

    print("\nLOF Evaluation Complete.")


if __name__ == "__main__":
    main()

