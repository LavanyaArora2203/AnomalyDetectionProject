# Isolation Forest vs LOF
# Isolation Forest	LOF
# Tree-based	Density-based
# Works well on global outliers	Works well on local outliers
# Fast	Slightly slower
# Good for large datasets	Good for clustered datasets

"""
models/lof_detector.py

Local Outlier Factor detector for invoice anomaly detection.
"""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from utils.mlflow_logger import (
    setup_experiment,
    log_parameters,
    log_model,
    log_artifact
)

# ============================================================
# Configuration
# ============================================================

ARTIFACT_DIR = Path("artifacts")
OUTPUT_DIR = Path("data/outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

N_NEIGHBORS = 20
CONTAMINATION = 0.02
METRIC = "euclidean"

# ============================================================
# Load Artifacts
# ============================================================

def load_artifacts():

    X = joblib.load(
        ARTIFACT_DIR / "feature_matrix.pkl"
    )

    metadata = pd.read_csv(
        ARTIFACT_DIR / "metadata.csv"
    )

    return X, metadata


# ============================================================
# Dataset Summary
# ============================================================

def inspect_dataset(X, metadata):

    print("\n===============================")
    print("Dataset Information")
    print("===============================")

    print(f"Feature Matrix : {X.shape}")
    print(f"Metadata       : {metadata.shape}")

    print("===============================\n")


# ============================================================
# Train LOF
# ============================================================

def train_model(X):

    model = LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        contamination=CONTAMINATION,
        metric=METRIC,
        novelty=False
    )

    predictions = model.fit_predict(X)

    scores = model.negative_outlier_factor_

    return model, predictions, scores


# ============================================================
# Convert Predictions
# ============================================================

def generate_predictions(predictions):

    predictions = (predictions == -1).astype(int)

    return predictions


# ============================================================
# Create Result DataFrame
# ============================================================

def create_results(
    metadata,
    scores,
    predictions
):

    results = metadata.copy()

    results["anomaly_score"] = scores

    results["prediction"] = predictions

    return results


# ============================================================
# Save Results
# ============================================================

def save_predictions(results):

    output_file = (
        OUTPUT_DIR /
        "lof_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved predictions -> {output_file}")


# ============================================================
# Summary
# ============================================================

def print_summary(results):

    print("\n===============================")
    print("LOF Summary")
    print("===============================\n")

    print(f"Total Invoices      : {len(results)}")

    print(
        f"Predicted Anomalies : "
        f"{results['prediction'].sum()}"
    )

    print(
        f"Normal Invoices     : "
        f"{len(results)-results['prediction'].sum()}"
    )

    print("\nTop 10 Most Suspicious\n")

    suspicious = (

        results

        .sort_values(
            by="anomaly_score"
        )

        .head(10)

    )
    print(suspicious.columns.tolist())

    print(

        suspicious[
            [
                "invoice_id",
                "vendor_id",
                "vendor_name",
                
                "anomaly_score"
            ]
        ]

    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Local Outlier Factor")
    print("=" * 60)

    X, metadata = load_artifacts()

    inspect_dataset(
        X,
        metadata
    )

    setup_experiment()

    with mlflow.start_run(run_name="Local_Outlier_Factor"):

        print("Training LOF...")

        model, raw_predictions, scores = train_model(X)

        predictions = generate_predictions(
            raw_predictions
        )

        results = create_results(
            metadata,
            scores,
            predictions
        )

        save_predictions(results)

        print_summary(results)

        log_parameters(
            contamination=CONTAMINATION,
            n_estimators=0,
            max_samples="NA",
            random_state=0
        )

        # LocalOutlierFactor cannot be logged
        # like Isolation Forest because novelty=False
        # stores no reusable prediction model.

        log_artifact(
            OUTPUT_DIR /
            "lof_predictions.csv"
        )

    print("\nLOF Finished Successfully.")


if __name__ == "__main__":
    main()


