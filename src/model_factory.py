# ==============================================================================
# Purpose:
# Dynamic construction of Deep Learning models (LSTM, GRU, SimpleRNN)
# for financial time-series prediction.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from config import FUTURE_STEPS

# ==============================================================================
# CONSTANTS
# ==============================================================================
CELL_TYPES = {
    "LSTM": LSTM,
    "GRU": GRU,
    "SIMPLERNN": SimpleRNN,
    "RNN": SimpleRNN,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_positive_integer(name: str, value: int) -> None:
    """Validate that a model hyperparameter is a positive integer."""
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def validate_dropout(dropout: float) -> None:
    """Validate dropout as a probability-like rate."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be between 0.0 and 1.0.")


# ==============================================================================
# DYNAMIC MODEL BUILDER
# ==============================================================================


def build_dl_model(
    lookback_steps: int,
    n_features: int,
    units: int = 128,
    cell_type: str = "LSTM",
    n_layers: int = 2,
    dropout: float = 0.3,
    loss: str = "huber",
    optimizer: str = "adam",
    bidirectional: bool = False,
    future_steps: int = FUTURE_STEPS,
) -> tf.keras.Model:
    """
    Constructs and compiles a deep recurrent neural network using TensorFlow/Keras.

    Args:
        lookback_steps: Number of historical days.
        n_features: Number of feature columns in input sequences.
        units: Number of hidden units in recurrent layers.
        cell_type: Recurrent cell architecture ('LSTM', 'GRU', 'SimpleRNN').
        n_layers: Total number of recurrent layers.
        dropout: Dropout fraction applied after each recurrent layer.
        loss: Loss function identifier (e.g. 'huber', 'mse', 'mae').
        optimizer: Optimizer identifier (e.g. 'adam', 'rmsprop').
        bidirectional: If True, wraps recurrent layers in Bidirectional wrappers.
        future_steps: Number of future steps/days to predict (default 1).

    Returns:
        Compiled Keras Sequential model.
    """
    # --------------------------------------------------------------------------
    # Step 0: Hyperparameter Validation
    # --------------------------------------------------------------------------
    positive_integer_params = {
        "lookback_steps": lookback_steps,
        "n_features": n_features,
        "units": units,
        "n_layers": n_layers,
        "future_steps": future_steps,
    }

    for name, value in positive_integer_params.items():
        validate_positive_integer(name, value)

    validate_dropout(dropout)

    # --------------------------------------------------------------------------
    # Step 1: Cell Type Resolution
    # --------------------------------------------------------------------------
    cell_type_upper = cell_type.upper()

    if cell_type_upper not in CELL_TYPES:
        raise ValueError(
            "Unsupported cell_type "
            f"'{cell_type}'. "
            "Available options are: "
            f"{list(CELL_TYPES.keys())}"
        )

    cell_class = CELL_TYPES[cell_type_upper]

    # --------------------------------------------------------------------------
    # Step 2: Assemble Layer Architecture
    # --------------------------------------------------------------------------
    model = Sequential(
        [
            tf.keras.Input(shape=(lookback_steps, n_features)),
        ]
    )

    for layer_index in range(n_layers):
        is_last_layer = layer_index == n_layers - 1

        recurrent_layer = cell_class(
            units=units,
            return_sequences=not is_last_layer,
        )

        if bidirectional:
            model.add(Bidirectional(recurrent_layer))
        else:
            model.add(recurrent_layer)

        if dropout > 0.0:
            model.add(Dropout(dropout))

    # --------------------------------------------------------------------------
    # Step 3: Compile Network with Targets
    # --------------------------------------------------------------------------
    # Output layer produces predictions for the specified future time steps
    model.add(Dense(future_steps, activation="linear"))

    # Compile the model with the configured loss function and evaluation metric.
    model.compile(loss=loss, metrics=["mae"], optimizer=optimizer)

    return model
