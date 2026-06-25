# ==============================================================================
# Purpose:
# Deep learning training pipeline with seed determinism.
# ==============================================================================

from pathlib import Path
import argparse
from typing import Any

# Import local data processing and model factory modules
from data_processing import load_and_process_data
from model_factory import build_dl_model
from utils.experiment_utils import (
    add_model_arguments,
    parse_feature_columns,
    set_seed,
)
from config import (
    RESULTS_DIR,
    DATA_DIR,
    TICKER,
    START_DATE,
    END_DATE,
    SPLIT_DATE,
    LOOKBACK_STEPS,
    FORECAST_OFFSET,
    FUTURE_STEPS,
)

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================


def train_model(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    split_date: str = SPLIT_DATE,
    lookback_steps: int = LOOKBACK_STEPS,
    forecast_offset: int = FORECAST_OFFSET,
    scale: bool = True,
    shuffle: bool = True,
    feature_columns: list[str] | None = None,
    units: int = 128,
    cell_type: str = "LSTM",
    n_layers: int = 2,
    dropout: float = 0.3,
    loss: str = "huber",
    optimizer: str = "adam",
    bidirectional: bool = False,
    epochs: int = 20,
    batch_size: int = 64,
    model_name: str = "lstm_model",
    subfolder: str = "",
    future_steps: int = FUTURE_STEPS,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """
    Orchestrates the entire training pipeline: loads/preprocesses data,
    creates the model from the factory, fits the model, and saves weights.

    Returns:
        dict: Training history and processed data dictionary for downstream use.
    """
    if feature_columns is None:
        feature_columns = ["adjclose", "volume", "open", "high", "low"]

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    # --------------------------------------------------------------------------
    # Step 1: Enforce Determinism
    # --------------------------------------------------------------------------
    set_seed(314)

    # --------------------------------------------------------------------------
    # Step 2: Load and Segment Dataset
    # --------------------------------------------------------------------------
    print(
        f"\n[Train Pipeline] Loading dataset for {ticker} from {start_date} to {end_date}..."
    )
    data = load_and_process_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback_steps=lookback_steps,
        scale=scale,
        shuffle=shuffle,
        forecast_offset=forecast_offset,
        split_by_date=True,
        split_date=split_date,
        feature_columns=feature_columns,
        cache_dir=DATA_DIR,
        future_steps=future_steps,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    print(f"[Train Pipeline] Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(
        f"[Train Pipeline] Test shapes  - X: {data['X_test'].shape}, y: {data['y_test'].shape}"
    )

    # --------------------------------------------------------------------------
    # Step 3: Instantiate Model Architecture
    # --------------------------------------------------------------------------
    n_features = len(feature_columns)
    print(
        f"[Train Pipeline] Building model architecture: {cell_type} | Units: {units} | Layers: {n_layers}"
    )

    model = build_dl_model(
        lookback_steps=lookback_steps,
        n_features=n_features,
        units=units,
        cell_type=cell_type,
        n_layers=n_layers,
        dropout=dropout,
        loss=loss,
        optimizer=optimizer,
        bidirectional=bidirectional,
        future_steps=future_steps,
    )

    # --------------------------------------------------------------------------
    # Step 4: Model Training (Fitting)
    # --------------------------------------------------------------------------
    print(
        f"[Train Pipeline] Training model '{model_name}' for {epochs} epochs with batch size {batch_size}..."
    )

    # Track performance using validation data to monitor overfitting
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data["X_test"], data["y_test"]),
        verbose=1,
    )

    # --------------------------------------------------------------------------
    # Step 5: Save Model Weights and Metadata
    # --------------------------------------------------------------------------
    weights_dir = results_dir / subfolder if subfolder else results_dir
    weights_path = weights_dir / f"{model_name}.weights.h5"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        print(
            "[Train Pipeline] Overwriting existing model weights at: "
            f"{weights_path.as_posix()}"
        )

    model.save_weights(weights_path)
    print(
        f"[Train Pipeline] Successfully saved model weights to: {weights_path.as_posix()}"
    )

    return {"history": history.history, "data": data, "model": model}


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the training script."""
    parser = argparse.ArgumentParser(
        description="Train deep recurrent stock prediction models."
    )
    add_model_arguments(parser)

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    return parser


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    train_model(
        ticker=args.ticker,
        cell_type=args.cell_type,
        n_layers=args.n_layers,
        units=args.units,
        dropout=args.dropout,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        subfolder=args.subfolder,
        bidirectional=args.bidirectional,
        lookback_steps=args.lookback_steps,
        forecast_offset=args.forecast_offset,
        future_steps=args.future_steps,
        feature_columns=parse_feature_columns(args.feature_columns),
    )
