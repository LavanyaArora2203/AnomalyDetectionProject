"""
Train Autoencoder for Invoice Anomaly Detection
"""

import mlflow
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from utils.mlflow_logger import (
    setup_experiment,
    log_parameters,
    log_artifact
)

from .config import *

from .utils import (
    load_feature_matrix,
    load_metadata,
    inspect_dataset,
    save_model,
    save_training_history,
    plot_training_loss
)

from .model import (
    build_autoencoder,
    print_model_summary
)


# ==========================================================
# Train / Validation Split
# ==========================================================

def prepare_data(X):

    X_train, X_val = train_test_split(

        X,

        test_size=VALIDATION_SPLIT,

        random_state=RANDOM_STATE,

        shuffle=True

    )

    return X_train, X_val


# ==========================================================
# Callbacks
# ==========================================================

def create_callbacks():

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=RESTORE_BEST_WEIGHTS,

        verbose=1

    )

    checkpoint = ModelCheckpoint(

        filepath=str(MODEL_PATH),

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )

    return [

        early_stopping,

        checkpoint

    ]


# ==========================================================
# Train Model
# ==========================================================

def train_model(

        model,

        X_train,

        X_val

):

    history = model.fit(

        X_train,

        X_train,

        validation_data=(

            X_val,

            X_val

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True,

        callbacks=create_callbacks(),

        verbose=1

    )

    return history


# ==========================================================
# Print Training Summary
# ==========================================================

def print_training_summary(history):

    print("\n")

    print("=" * 60)

    print("Training Complete")

    print("=" * 60)

    print(

        f"Epochs Trained : {len(history.history['loss'])}"

    )

    print(

        f"Final Training Loss : "

        f"{history.history['loss'][-1]:.6f}"

    )

    print(

        f"Final Validation Loss : "

        f"{history.history['val_loss'][-1]:.6f}"

    )

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)

    print("Invoice Autoencoder Training")

    print("=" * 70)

    # ------------------------------------------------------

    X = load_feature_matrix()

    metadata = load_metadata()

    inspect_dataset(

        X,

        metadata

    )

    # ------------------------------------------------------

    X_train, X_val = prepare_data(X)

    print(

        f"Training Samples : {len(X_train)}"

    )

    print(

        f"Validation Samples : {len(X_val)}"

    )

    # ------------------------------------------------------

    autoencoder, encoder, decoder = build_autoencoder(

        input_dim=X.shape[1]

    )

    print_model_summary(

        autoencoder

    )

    # ------------------------------------------------------

    setup_experiment()

    with mlflow.start_run(

        run_name="Autoencoder_Training"

    ):

        log_parameters(

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            learning_rate=LEARNING_RATE,

            validation_split=VALIDATION_SPLIT,

            hidden_layer_1=HIDDEN_LAYER_1,

            hidden_layer_2=HIDDEN_LAYER_2,

            latent_dimension=LATENT_DIM,

            dropout_rate=DROPOUT_RATE

        )

        history = train_model(

            autoencoder,

            X_train,

            X_val

        )

        save_model(

            autoencoder

        )

        save_training_history(

            history

        )

        log_artifact(

            MODEL_PATH

        )

        log_artifact(

            HISTORY_PATH

        )

    # ------------------------------------------------------

    print_training_summary(

        history

    )

    plot_training_loss(

        history.history

    )

    print(

        "\nAutoencoder training completed successfully."

    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
