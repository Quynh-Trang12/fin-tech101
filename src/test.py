# ==============================================================================
# Purpose:
# Deep learning model evaluation, metric calculations (correctly unscaled),
# and professional visualization.
# ==============================================================================

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Headless plotting backend
import matplotlib.pyplot as plt

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
    CSV_RESULTS_DIR,
    TICKER,
    START_DATE,
    END_DATE,
    SPLIT_DATE,
    LOOKBACK_STEPS,
    FORECAST_OFFSET,
    FUTURE_STEPS,
    FEATURE_COLUMNS,
)

# ==============================================================================
# EVALUATION & METRIC CALCULATION FUNCTIONS
# ==============================================================================


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_actual: np.ndarray,
    future_steps: int = FUTURE_STEPS,
) -> dict[str, float]:
    """
    Computes regression and classification metrics on unscaled stock prices.

    Args:
        y_true: Unscaled ground truth adjusted close prices.
        y_pred: Unscaled predicted adjusted close prices.
        prev_actual: Unscaled ground truth close prices for day t.
        future_steps: Forecast sequence horizon length.

    Returns:
        Standard evaluation metrics (MAE, RMSE, MAPE, Directional Accuracy).
    """
    # Ensure 2D arrays of shape (N, future_steps)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error (MAPE)
    safe_y_true = np.where(y_true == 0, 1e-5, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100

    # Directional Accuracy (DA)
    if future_steps > 1:
        # Broadcast prev_actual of shape (N,) to (N, future_steps)
        prev_actual_expanded = np.expand_dims(prev_actual, axis=1)
        actual_dir = np.sign(y_true - prev_actual_expanded)
        pred_dir = np.sign(y_pred - prev_actual_expanded)
    else:
        # Squeeze to 1D arrays for single-step directional accuracy
        y_true_1d = np.squeeze(y_true)
        y_pred_1d = np.squeeze(y_pred)
        actual_dir = np.sign(y_true_1d - prev_actual)
        pred_dir = np.sign(y_pred_1d - prev_actual)

    valid_mask = actual_dir != 0
    if np.sum(valid_mask) > 0:
        da = np.mean(actual_dir[valid_mask] == pred_dir[valid_mask]) * 100
    else:
        da = 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "DA": float(da),
    }


# ------------------------------------------------------------------------------
# TRADING PROFIT CALCULATIONS
# ------------------------------------------------------------------------------


def get_trading_profits(
    final_df: pd.DataFrame, forecast_offset: int, future_steps: int = FUTURE_STEPS
) -> dict[str, float]:
    """
    Simulates trading profits based on predicted price changes.

    If predicted future price is higher than current price, buy.
    If predicted future price is lower than current price, sell.

    Args:
        final_df: DataFrame containing actual and predicted prices.
        forecast_offset: Number of days ahead for the forecast.
        future_steps: Forecast sequence horizon length.

    Returns:
        Dictionary containing total buy/sell profits, total profit,
        profit per trade, and trading accuracy.
    """
    if future_steps > 1:
        # Use step 0 of the forecast sequence (which corresponds to t + forecast_offset)
        y_pred_col = "adjclose_future_0"
        y_true_col = "true_adjclose_future_0"
    else:
        y_pred_col = f"adjclose_{forecast_offset}"
        y_true_col = f"true_adjclose_{forecast_offset}"

    current_prices = final_df["adjclose"].to_numpy()
    predicted_prices = final_df[y_pred_col].to_numpy()
    true_prices = final_df[y_true_col].to_numpy()

    buy_profit = np.where(
        predicted_prices > current_prices,
        true_prices - current_prices,
        0.0,
    )

    sell_profit = np.where(
        predicted_prices < current_prices,
        current_prices - true_prices,
        0.0,
    )

    final_df["buy_profit"] = buy_profit
    final_df["sell_profit"] = sell_profit

    total_buy = final_df["buy_profit"].sum()
    total_sell = final_df["sell_profit"].sum()
    total_profit = total_buy + total_sell

    n_trades = len(final_df)
    profit_per_trade = total_profit / n_trades if n_trades > 0 else 0.0

    # Trading accuracy: fraction of trades that yielded a positive profit
    profitable_trades = (final_df["buy_profit"] > 0).sum() + (
        final_df["sell_profit"] > 0
    ).sum()
    trading_accuracy = (profitable_trades / n_trades) * 100 if n_trades > 0 else 0.0

    return {
        "total_buy_profit": float(total_buy),
        "total_sell_profit": float(total_sell),
        "total_profit": float(total_profit),
        "profit_per_trade": float(profit_per_trade),
        "trading_accuracy": float(trading_accuracy),
    }


# ==============================================================================
# VISUALIZATION PIPELINE
# ==============================================================================


def plot_prediction_chart(
    final_df: pd.DataFrame,
    forecast_offset: int,
    model_name: str,
    subfolder: str = "",
    future_steps: int = 1,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Saves a high-quality visualization comparing actual and predicted prices.

    Args:
        final_df: DataFrame containing actual and predicted prices.
        forecast_offset: Number of days ahead for the forecast.
        model_name: Name of the model for labeling the plot.
        subfolder: Optional subfolder for saving the plot.
        future_steps: Forecast sequence horizon length.

    Returns:
        None. Saves the plot to disk.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if future_steps > 1:
        # Plot step 0 of the forecast sequence (the next immediate day)
        ax.plot(
            final_df.index,
            final_df["true_adjclose_future_0"],
            label="Actual Price",
            color="#1f77b4",
            linewidth=2,
        )
        ax.plot(
            final_df.index,
            final_df["adjclose_future_0"],
            label="Predicted Price (t+1)",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
        )
        title_suffix = f" (Multi-Step: {future_steps} Days, showing Step 1)"
    else:
        ax.plot(
            final_df.index,
            final_df[f"true_adjclose_{forecast_offset}"],
            label="Actual Price",
            color="#1f77b4",
            linewidth=2,
        )
        ax.plot(
            final_df.index,
            final_df[f"adjclose_{forecast_offset}"],
            label="Predicted Price",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
        )
        title_suffix = f" (Forecast Offset: {forecast_offset} Days)"

    ax.set_title(
        f"Stock Price Prediction Comparison - {model_name}{title_suffix}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Date", fontsize=12, labelpad=10)
    ax.set_ylabel("Unscaled Price ($)", fontsize=12, labelpad=10)

    # Modern styling
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(
        loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # Save Plot
    results_dir = results_dir / subfolder if subfolder else results_dir
    plot_path = results_dir / f"{model_name}_prediction.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib overwrites the existing file; this message makes reruns explicit.
    if plot_path.exists():
        print(
            f"[Test Pipeline] Overwriting existing prediction plot at: {plot_path.as_posix()}"
        )

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] Saved prediction plot to: {plot_path.as_posix()}")


# ==============================================================================
# HELPER FUNCTIONS FOR TESTING PIPELINE
# ==============================================================================
def print_evaluation_summary(
    model_name: str,
    metrics: dict[str, float],
    trading_metrics: dict[str, float],
) -> None:
    """Print evaluation and trading results in a consistent terminal format."""
    print("\n" + "=" * 50)
    print(f" EVALUATION RESULTS - {model_name} ")
    print("=" * 50)
    print(f"Unscaled MAE:           ${metrics['MAE']:.4f}")
    print(f"Unscaled RMSE:          ${metrics['RMSE']:.4f}")
    print(f"Unscaled MAPE:          {metrics['MAPE']:.2f}%")
    print(f"Directional Accuracy:   {metrics['DA']:.2f}%")
    print(f"Trading Accuracy:       {trading_metrics['trading_accuracy']:.2f}%")
    print(f"Total Trading Profit:   ${trading_metrics['total_profit']:.2f}")
    print(f"Profit per Trade:       ${trading_metrics['profit_per_trade']:.2f}")
    print("=" * 50)


# ==============================================================================
# TESTING PIPELINE
# ==============================================================================


def test_model(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    split_date: str = SPLIT_DATE,
    lookback_steps: int = LOOKBACK_STEPS,
    forecast_offset: int = FORECAST_OFFSET,
    scale: bool = True,
    feature_columns: list[str] | None = None,
    units: int = 128,
    cell_type: str = "LSTM",
    n_layers: int = 2,
    dropout: float = 0.3,
    loss: str = "huber",
    optimizer: str = "adam",
    bidirectional: bool = False,
    model_name: str = "lstm_model",
    subfolder: str = "",
    future_steps: int = FUTURE_STEPS,
    results_dir: Path = RESULTS_DIR,
    csv_results_dir: Path = CSV_RESULTS_DIR,
) -> dict[str, float]:
    """
    Evaluates a trained model: loads weights, runs inference, inverts scaling,
    computes accurate metrics, runs trading simulations, and saves plots and CSV logs.

    Returns:
        dict: Evaluation metrics and trading performance dictionary.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    if future_steps < 1:
        raise ValueError("future_steps must be at least 1.")

    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least 1.")

    if forecast_offset < 1:
        raise ValueError("forecast_offset must be at least 1.")

    # --------------------------------------------------------------------------
    # Step 1: Enforce Determinism & Load Data
    # --------------------------------------------------------------------------
    set_seed(314)

    data = load_and_process_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback_steps=lookback_steps,
        scale=scale,
        shuffle=False,  # DO NOT shuffle for testing/plotting evaluation
        forecast_offset=forecast_offset,
        split_by_date=True,
        split_date=split_date,
        feature_columns=feature_columns,
        cache_dir=DATA_DIR,
        future_steps=future_steps,
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    # --------------------------------------------------------------------------
    # Step 2: Build Model and Load Weights
    # --------------------------------------------------------------------------
    n_features = len(feature_columns)
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

    weights_dir = results_dir / subfolder if subfolder else results_dir
    weights_path = weights_dir / f"{model_name}.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained model weights not found at: {weights_path.as_posix()}"
        )

    model.load_weights(weights_path)
    print(
        f"[Test Pipeline] Loaded trained model weights from: {weights_path.as_posix()}"
    )

    # --------------------------------------------------------------------------
    # Step 3: Run Inference & Inverse Transform Scaling
    # --------------------------------------------------------------------------
    print(f"[Test Pipeline] Running inference on {len(X_test)} samples...")
    y_pred = model.predict(X_test, verbose=0)

    # Correct unscaling of predictions and targets
    if scale:
        scaler = data["column_scaler"]["adjclose"]
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(
            y_test.shape
        )
        y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(
            y_pred.shape
        )
    else:
        y_test_unscaled = y_test
        y_pred_unscaled = y_pred

    # Get previous day's actual price to calculate direction
    test_df = data["test_df"].copy()
    test_df.sort_index(inplace=True)

    # Align shapes and insert into test dataframe
    if future_steps > 1:
        for i in range(future_steps):
            test_df[f"adjclose_future_{i}"] = y_pred_unscaled[:, i]
            test_df[f"true_adjclose_future_{i}"] = y_test_unscaled[:, i]
    else:
        y_test_unscaled_1d = np.squeeze(y_test_unscaled)
        y_pred_unscaled_1d = np.squeeze(y_pred_unscaled)
        test_df[f"adjclose_{forecast_offset}"] = y_pred_unscaled_1d
        test_df[f"true_adjclose_{forecast_offset}"] = y_test_unscaled_1d

    # Previous day's price is the raw 'adjclose' (which is index-aligned)
    prev_actual = test_df["adjclose"].values

    # --------------------------------------------------------------------------
    # Step 4: Calculate Metrics (Correctly Unscaled)
    # --------------------------------------------------------------------------
    metrics = calculate_metrics(
        y_test_unscaled, y_pred_unscaled, prev_actual, future_steps=future_steps
    )

    # Get trading simulation profit metrics
    trading_metrics = get_trading_profits(
        test_df, forecast_offset, future_steps=future_steps
    )

    print_evaluation_summary(
        model_name=model_name,
        metrics=metrics,
        trading_metrics=trading_metrics,
    )

    # --------------------------------------------------------------------------
    # Step 5: Save Plots and CSV Logs
    # --------------------------------------------------------------------------
    plot_prediction_chart(
        final_df=test_df,
        forecast_offset=forecast_offset,
        model_name=model_name,
        subfolder=subfolder,
        future_steps=future_steps,
        results_dir=results_dir,
    )

    csv_results_folder = csv_results_dir / subfolder if subfolder else csv_results_dir
    csv_path = csv_results_folder / f"{model_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # pandas overwrites the existing CSV; this message makes reruns explicit.
    if csv_path.exists():
        print(
            f"[Test Pipeline] Overwriting existing CSV results at: {csv_path.as_posix()}"
        )

    test_df.to_csv(csv_path)
    print(f"[Test Pipeline] Saved detailed predictions CSV to: {csv_path.as_posix()}")

    # Combine results
    all_results = {**metrics, **trading_metrics}

    # Return metrics
    return all_results


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the testing script."""
    parser = argparse.ArgumentParser(
        description="Evaluate deep recurrent stock prediction models."
    )
    add_model_arguments(parser)
    return parser


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    test_model(
        ticker=args.ticker,
        cell_type=args.cell_type,
        n_layers=args.n_layers,
        units=args.units,
        dropout=args.dropout,
        loss=args.loss,
        model_name=args.model_name,
        subfolder=args.subfolder,
        bidirectional=args.bidirectional,
        lookback_steps=args.lookback_steps,
        forecast_offset=args.forecast_offset,
        future_steps=args.future_steps,
        feature_columns=parse_feature_columns(args.feature_columns),
    )
