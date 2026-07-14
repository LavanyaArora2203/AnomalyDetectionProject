"""
Configuration file for the Deep Learning
Autoencoder pipeline.
"""

from pathlib import Path

# ==========================================================
# Directories
# ==========================================================

ARTIFACT_DIR = Path("artifacts")

OUTPUT_DIR = Path("data/outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

ARTIFACT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Input Artifacts
# ==========================================================

FEATURE_MATRIX_PATH = (

    ARTIFACT_DIR /

    "feature_matrix.pkl"

)

METADATA_PATH = (

    ARTIFACT_DIR /

    "metadata.csv"

)

FEATURE_NAMES_PATH = (

    ARTIFACT_DIR /

    "feature_names.pkl"

)

# ==========================================================
# Saved Model
# ==========================================================

MODEL_PATH = (

    ARTIFACT_DIR /

    "autoencoder.keras"

)

HISTORY_PATH = (

    ARTIFACT_DIR /

    "training_history.pkl"

)

THRESHOLD_PATH = (

    ARTIFACT_DIR /

    "threshold.pkl"

)

RECONSTRUCTION_ERROR_PATH = (

    ARTIFACT_DIR /

    "reconstruction_errors.pkl"

)

# ==========================================================
# Output Files
# ==========================================================

PREDICTION_FILE = (

    OUTPUT_DIR /

    "autoencoder_predictions.csv"

)

EVALUATION_FILE = (

    OUTPUT_DIR /

    "autoencoder_evaluation.csv"

)

# ==========================================================
# Training Hyperparameters
# ==========================================================

EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 0.001

VALIDATION_SPLIT = 0.20

RANDOM_STATE = 42

# ==========================================================
# Autoencoder Architecture
# ==========================================================

HIDDEN_LAYER_1 = 64

HIDDEN_LAYER_2 = 32

LATENT_DIM = 16

DROPOUT_RATE = 0.10

# ==========================================================
# Optimizer
# ==========================================================

LOSS_FUNCTION = "mse"

OPTIMIZER = "adam"

# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING_PATIENCE = 5

RESTORE_BEST_WEIGHTS = True

# ==========================================================
# Anomaly Threshold
# ==========================================================

THRESHOLD_PERCENTILE = 98