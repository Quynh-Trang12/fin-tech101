# ==============================================================================
# Purpose:
# Shared utilities for training, testing, and experiment runner scripts.
# ==============================================================================

import argparse
import tensorflow as tf
from config import SPLIT_METHOD, VALIDATION_RATIO

# ==============================================================================
# SEED DETERMINISM
# ==============================================================================


def set_seed(seed: int = 314) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducible experiments."""
    tf.keras.utils.set_random_seed(seed)
    print(f"[Seed Setup] Deterministic seed set to: {seed}")


# ==============================================================================
# COMMAND-LINE HELPERS
# ==============================================================================


def parse_feature_columns(feature_columns: str) -> list[str]:
    """Convert a comma-separated feature string into a cleaned feature list."""
    return [
        feature.strip() for feature in feature_columns.split(",") if feature.strip()
    ]


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared model architecture arguments to a command-line parser."""
    argument_specs = [
        ("--ticker", str, "CBA.AX", "Stock ticker symbol."),
        ("--cell_type", str, "LSTM", "Recurrent cell type: LSTM, GRU, or SimpleRNN."),
        ("--n_layers", int, 2, "Number of recurrent layers."),
        ("--units", int, 128, "Number of recurrent hidden units."),
        ("--dropout", float, 0.3, "Dropout rate."),
        ("--loss", str, "huber", "Loss function: huber, mse, or mae."),
        ("--model_name", str, "lstm_model", "Name used for model weights."),
        ("--subfolder", str, "", "Subfolder inside the results directory."),
        ("--lookback_steps", int, 50, "Number of past time steps to use."),
        ("--forecast_offset", int, 1, "Prediction offset from the input window."),
        ("--future_steps", int, 1, "Number of future steps to predict."),
        (
            "--validation_ratio",
            float,
            VALIDATION_RATIO,
            "Chronological validation split ratio relative to the training set.",
        ),
        (
            "--split_method",
            str,
            SPLIT_METHOD,
            "Train/test split strategy: date, ratio, or random.",
        ),
        (
            "--feature_columns",
            str,
            "adjclose,volume,open,high,low",
            "Comma-separated input feature list.",
        ),
    ]

    for flag, arg_type, default, help_text in argument_specs:
        parser.add_argument(flag, type=arg_type, default=default, help=help_text)

    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Wrap recurrent layers in Bidirectional wrappers.",
    )
