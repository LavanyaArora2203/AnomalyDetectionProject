## evaluate_isolation_forest.py
from pathlib import Path

import pandas as pd
from utils.mlflow_logger import *

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
OUTPUT_DIR = Path("data/outputs")

PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "isolation_forest_predictions.csv"
)

LABELS_FILE = Path(
    "data/fraud_labels.csv"
)
def load_data():

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    labels = pd.read_csv(
        LABELS_FILE
    )

    return predictions, labels
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

def print_metrics(metrics):

    print("\n========== Isolation Forest ==========\n")

    for key, value in metrics.items():

        print(f"{key:<12}: {value:.4f}")

    print("\n======================================")

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

    print("\nConfusion Matrix\n")

    print(cm)

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

def save_results(df):

    output_file = (

        OUTPUT_DIR /

        "isolation_forest_evaluation.csv"

    )

    df.to_csv(

        output_file,

        index=False

    )

    print(

        f"\nSaved {output_file}"

    )

def main():

    predictions, labels = load_data()

    evaluation = merge_data(

        predictions,

        labels

    )

    metrics = compute_metrics(

        evaluation

    )
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

    "isolation_forest_evaluation.csv"

)
    # setup_experiment()

    # with mlflow.start_run():

    #     model = train_model(X)
    #     log_parameters(

    # CONTAMINATION,

    # N_ESTIMATORS,

    # MAX_SAMPLES,

    # RANDOM_STATE

        
    #     )
    #     log_model(model)


if __name__ == "__main__":

    main()



# ========== Isolation Forest ==========

# Accuracy    : 0.9781
# Precision   : 0.0020
# Recall      : 0.0200
# F1          : 0.0036

# ======================================

# Confusion Matrix

#                Pred Normal  Pred Fraud
# Actual Normal        49000        1000
# Actual Fraud            98           2

# Classification Report

#               precision    recall  f1-score   support

#            0     0.9980    0.9800    0.9889     50000
#            1     0.0020    0.0200    0.0036       100

#     accuracy                         0.9781     50100
#    macro avg     0.5000    0.5000    0.4963     50100
# weighted avg     0.9960    0.9781    0.9870     50100


# Fraud Type Recall

#           fraud_type  total  detected  recall
# 0  Duplicate Invoice    100         2    0.02

# Saved data\outputs\isolation_forest_evaluation.csv