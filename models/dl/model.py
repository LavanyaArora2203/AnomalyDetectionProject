"""
Autoencoder Model Architecture
"""

import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from .config import *


# ==========================================================
# Build Autoencoder
# ==========================================================

def build_autoencoder(input_dim):
    """
    Build a fully-connected Autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    autoencoder : tf.keras.Model
        Complete autoencoder model.

    encoder : tf.keras.Model
        Encoder network.

    decoder : tf.keras.Model
        Decoder network.
    """

    # ======================================================
    # Input
    # ======================================================

    inputs = Input(
        shape=(input_dim,),
        name="Input"
    )

    # ======================================================
    # Encoder
    # ======================================================

    x = Dense(
        HIDDEN_LAYER_1,
        activation="relu",
        name="Encoder_Dense_1"
    )(inputs)

    x = BatchNormalization(
        name="Encoder_BN_1"
    )(x)

    x = Dropout(
        DROPOUT_RATE,
        name="Encoder_Dropout_1"
    )(x)

    x = Dense(
        HIDDEN_LAYER_2,
        activation="relu",
        name="Encoder_Dense_2"
    )(x)

    encoded = Dense(
        LATENT_DIM,
        activation="relu",
        name="Latent_Space"
    )(x)

    # ======================================================
    # Decoder
    # ======================================================

    x = Dense(
        HIDDEN_LAYER_2,
        activation="relu",
        name="Decoder_Dense_1"
    )(encoded)

    x = Dense(
        HIDDEN_LAYER_1,
        activation="relu",
        name="Decoder_Dense_2"
    )(x)

    outputs = Dense(
        input_dim,
        activation="linear",
        name="Output"
    )(x)

    # ======================================================
    # Models
    # ======================================================

    autoencoder = Model(
        inputs,
        outputs,
        name="InvoiceAutoencoder"
    )

    encoder = Model(
        inputs,
        encoded,
        name="Encoder"
    )

    # ======================================================
    # Build Decoder Model
    # ======================================================

    latent_inputs = Input(
        shape=(LATENT_DIM,),
        name="Decoder_Input"
    )

    decoder_layer = autoencoder.get_layer(
        "Decoder_Dense_1"
    )(latent_inputs)

    decoder_layer = autoencoder.get_layer(
        "Decoder_Dense_2"
    )(decoder_layer)

    decoder_outputs = autoencoder.get_layer(
        "Output"
    )(decoder_layer)

    decoder = Model(
        latent_inputs,
        decoder_outputs,
        name="Decoder"
    )

    # ======================================================
    # Compile
    # ======================================================

    autoencoder.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss=LOSS_FUNCTION,

        metrics=["mae"]

    )

    return autoencoder, encoder, decoder


# ==========================================================
# Print Model Summary
# ==========================================================

def print_model_summary(model):

    print("\n")
    print("=" * 60)
    print("Autoencoder Architecture")
    print("=" * 60)

    model.summary()

    print("=" * 60)