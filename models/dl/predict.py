# python -m models.dl.predict
"""
Generate anomaly predictions using the trained Autoencoder.
"""

import mlflow
import numpy as np
import pandas as pd

from utils.mlflow_logger import (
    setup_experiment,
    log_artifact,
    log_parameters
)

from .config import *

from .utils import (
    load_feature_matrix,
    load_metadata,
    load_trained_model,
    save_threshold,
    load_threshold,
    save_predictions,
    save_reconstruction_errors,
    plot_reconstruction_errors
)


# ==========================================================
# Reconstruction
# ==========================================================

def reconstruct(model, X):

    reconstructed = model.predict(
        X,
        verbose=0
    )

    return reconstructed


# ==========================================================
# Reconstruction Error
# ==========================================================

def compute_reconstruction_error(
        X,
        reconstructed
):

    errors = np.mean(

        np.square(

            X - reconstructed

        ),

        axis=1

    )

    return errors


# ==========================================================
# Threshold
# ==========================================================

def calculate_threshold(errors):

    threshold = np.percentile(

        errors,

        THRESHOLD_PERCENTILE

    )

    return threshold


# ==========================================================
# Predictions
# ==========================================================

def generate_predictions(
        errors,
        threshold
):

    predictions = (

        errors > threshold

    ).astype(int)

    return predictions


# ==========================================================
# Create Results
# ==========================================================

def create_results(
        metadata,
        errors,
        predictions
):

    results = metadata.copy()

    results["reconstruction_error"] = errors

    results["prediction"] = predictions

    return results


# ==========================================================
# Print Summary
# ==========================================================

def print_summary(results):

    print()

    print("=" * 60)

    print("Prediction Summary")

    print("=" * 60)

    print(

        f"Total invoices : {len(results)}"

    )

    print(

        f"Detected anomalies : "

        f"{results['prediction'].sum()}"

    )

    print(

        f"Normal invoices : "

        f"{len(results)-results['prediction'].sum()}"

    )

    print()

    print("Top 10 Most Suspicious")

    print()

    print(

        results

        .sort_values(

            by="reconstruction_error",

            ascending=False

        )

        .head(10)[

            [

                "invoice_id",

                "vendor_id",

                "vendor_name",

                "amount",

                "reconstruction_error"

            ]

        ]

    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)

    print("Autoencoder Prediction")

    print("=" * 70)

    X = load_feature_matrix()

    metadata = load_metadata()

    model = load_trained_model()

    setup_experiment()

    with mlflow.start_run(

        run_name="Autoencoder_Prediction"

    ):

        reconstructed = reconstruct(

            model,

            X

        )

        errors = compute_reconstruction_error(

            X,

            reconstructed

        )

        threshold = calculate_threshold(

            errors

        )

        predictions = generate_predictions(

            errors,

            threshold

        )

        results = create_results(

            metadata,

            errors,

            predictions

        )

        save_reconstruction_errors(

            errors

        )

        save_threshold(

            threshold

        )

        save_predictions(

            results

        )

        log_parameters(

            threshold_percentile=THRESHOLD_PERCENTILE,

            threshold=float(threshold)

        )

        log_artifact(

            RECONSTRUCTION_ERROR_PATH

        )

        log_artifact(

            THRESHOLD_PATH

        )

        log_artifact(

            PREDICTION_FILE

        )

    print_summary(

        results

    )

    plot_reconstruction_errors(

        errors,

        threshold

    )

    print()

    print("Prediction completed successfully.")


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    main()
