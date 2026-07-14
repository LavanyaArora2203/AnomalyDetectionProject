import joblib
import pandas as pd
from utils.mlflow_logger import *

from pathlib import Path

from sklearn.ensemble import IsolationForest

ARTIFACT_DIR = Path("artifacts")

OUTPUT_DIR = Path("data/outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

N_ESTIMATORS = 200

CONTAMINATION = 0.02

RANDOM_STATE = 42

MAX_SAMPLES = "auto"

def load_artifacts():

    X = joblib.load(

        ARTIFACT_DIR /

        "feature_matrix.pkl"

    )

    metadata = pd.read_csv(

        ARTIFACT_DIR /

        "metadata.csv"

    )

    return X, metadata

def inspect_dataset(

    X,

    metadata

):

    print("\n========== Dataset ==========\n")

    print(

        "Feature Matrix Shape :",

        X.shape

    )

    print(

        "Metadata Shape :",

        metadata.shape

    )

    print("\n=============================\n")


def train_model(X):

    model = IsolationForest(

        n_estimators=N_ESTIMATORS,

        contamination=CONTAMINATION,

        max_samples=MAX_SAMPLES,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    model.fit(X)

    return model
def generate_scores(model, X):
    """
    Generate anomaly scores.

    Higher score  -> More normal
    Lower score   -> More anomalous
    """

    scores = model.decision_function(X)

    return scores

def generate_predictions(model, X):
    """
    Convert Isolation Forest predictions
    into binary fraud labels.

    0 = Normal
    1 = Fraud
    """

    predictions = model.predict(X)

    predictions = (predictions == -1).astype(int)

    return predictions

def create_results(
    metadata,
    scores,
    predictions
):

    results = metadata.copy()

    results["anomaly_score"] = scores

    results["prediction"] = predictions

    return results

def save_predictions(results):

    output_file = OUTPUT_DIR / "isolation_forest_predictions.csv"

    results.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved predictions to {output_file}")


def print_summary(results):

    print("\n========== Isolation Forest Summary ==========\n")

    print(
        "Total invoices :",
        len(results)
    )

    print(
        "Predicted anomalies :",
        results["prediction"].sum()
    )

    print(
        "Normal invoices :",
        len(results) -
        results["prediction"].sum()
    )

    print("\nTop 10 Most Suspicious Invoices\n")

    print(

        results

        .sort_values(
            "anomaly_score"
        )

        .head(10)

    )




def main():

    print("=" * 60)
    print("Isolation Forest")
    print("=" * 60)

    X, metadata = load_artifacts()

    inspect_dataset(
        X,
        metadata
    )



    print("\nTraining model...")
    setup_experiment()

    with mlflow.start_run():

        model = train_model(X)
        log_parameters(

    CONTAMINATION,

    N_ESTIMATORS,

    MAX_SAMPLES,

    RANDOM_STATE

        
        )
        log_model(model)

    print("Training complete.")

    scores = generate_scores(
        model,
        X
    )

    predictions = generate_predictions(
        model,
        X
    )

    results = create_results(
        metadata,
        scores,
        predictions
    )

    save_predictions(results)
    log_artifact(

    OUTPUT_DIR /

    "isolation_forest_predictions.csv"

)

    print_summary(results)

if __name__ == "__main__":
    main()
