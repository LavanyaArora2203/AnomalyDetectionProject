"""
Evaluate Autoencoder predictions against ground truth fraud labels.
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

from .config import *

# ==========================================================
# Load Data
# ==========================================================

def load_data():

    predictions = pd.read_csv(PREDICTION_FILE)

    labels = pd.read_csv("data/fraud_labels.csv")

    return predictions, labels


# ==========================================================
# Merge Labels
# ==========================================================

def merge_predictions(predictions, labels):

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


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(df):

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


# ==========================================================
# Confusion Matrix
# ==========================================================

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

    print()

    print("=" * 60)

    print("Confusion Matrix")

    print("=" * 60)

    print(cm)


# ==========================================================
# Classification Report
# ==========================================================

def print_report(df):

    print()

    print("=" * 60)

    print("Classification Report")

    print("=" * 60)

    print(

        classification_report(

            df["fraud_label"],

            df["prediction"],

            digits=4,

            zero_division=0

        )

    )


# ==========================================================
# Fraud Type Recall
# ==========================================================

def fraud_type_summary(df):

    frauds = df[

        df["fraud_label"] == 1

    ]

    summary = (

        frauds

        .groupby("fraud_type")

        .agg(

            total=("invoice_id", "count"),

            detected=("prediction", "sum")

        )

        .reset_index()

    )

    summary["recall"] = (

        summary["detected"]

        /

        summary["total"]

    ).round(3)

    print()

    print("=" * 60)

    print("Fraud Type Recall")

    print("=" * 60)

    print(summary)

    return summary


# ==========================================================
# Save Evaluation
# ==========================================================

def save_results(df):

    df.to_csv(

        EVALUATION_FILE,

        index=False

    )

    print()

    print(

        f"Saved evaluation -> {EVALUATION_FILE}"

    )


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(metrics):

    print()

    print("=" * 60)

    print("Autoencoder Evaluation")

    print("=" * 60)

    print()

    for key, value in metrics.items():

        print(

            f"{key:<12}: {value:.4f}"

        )


# ==========================================================
# Main
# ==========================================================

def main():

    predictions, labels = load_data()

    evaluation = merge_predictions(

        predictions,

        labels

    )

    metrics = calculate_metrics(

        evaluation

    )

    setup_experiment()

    with mlflow.start_run(

        run_name="Autoencoder_Evaluation"

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

        fraud_type_summary(

            evaluation

        )

        save_results(

            evaluation

        )

        log_artifact(

            PREDICTION_FILE

        )

        log_artifact(

            EVALUATION_FILE

        )

    print()

    print("Autoencoder evaluation completed successfully.")


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    main()