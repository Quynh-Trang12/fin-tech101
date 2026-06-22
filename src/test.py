# ==============================================================================
# Purpose: 
# Deep learning model evaluation, metric calculations (correctly unscaled),
# and professional visualization.
# ==============================================================================

import os
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Headless plotting backend
import matplotlib.pyplot as plt

# Import local data processing and model factory modules
from data_processing import load_and_process_data
from model_factory import build_dl_model

# ==============================================================================
# SEED DETERMINISM SETUP
# ==============================================================================

def set_seed(seed=314):
    """Enforces random seed determinism across standard runtime environments."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(314)

# ==============================================================================
# EVALUATION & METRIC CALCULATION FUNCTIONS
# ==============================================================================

def calculate_metrics(y_true, y_pred, prev_actual, future_steps=1):
    """
    Computes regression and classification metrics on unscaled stock prices.
    
    Args:
        y_true (np.array): Unscaled ground truth adjusted close prices.
        y_pred (np.array): Unscaled predicted adjusted close prices.
        prev_actual (np.array): Unscaled ground truth close prices for day t.
        future_steps (int): Forecast sequence horizon length.
        
    Returns:
        dict: Standard evaluation metrics (MAE, RMSE, MAPE, Directional Accuracy).
    """
    # Ensure 2D arrays of shape (N, future_steps)
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    if y_true.shape[0] == 1 and len(y_true[0]) > future_steps:
        y_true = y_true.T
        y_pred = y_pred.T
        
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
        "DA": float(da)
    }

# ------------------------------------------------------------------------------
# TRADING PROFIT CALCULATIONS
# ------------------------------------------------------------------------------

def get_trading_profits(final_df, forecast_offset, future_steps=1):
    """
    Simulates trading profits based on predicted price changes.
    
    If predicted future price is higher than current price, buy.
    If predicted future price is lower than current price, sell.
    """
    if future_steps > 1:
        # Use step 0 of the forecast sequence (which corresponds to t + forecast_offset)
        y_pred_col = "adjclose_future_0"
        y_true_col = "true_adjclose_future_0"
    else:
        y_pred_col = f"adjclose_{forecast_offset}"
        y_true_col = f"true_adjclose_{forecast_offset}"
    
    # Lambda functions to compute trade profits
    buy_profit  = lambda current, pred, true: true - current if pred > current else 0.0
    sell_profit = lambda current, pred, true: current - true if pred < current else 0.0
    
    final_df["buy_profit"] = list(map(
        buy_profit,
        final_df["adjclose"],
        final_df[y_pred_col],
        final_df[y_true_col]
    ))
    
    final_df["sell_profit"] = list(map(
        sell_profit,
        final_df["adjclose"],
        final_df[y_pred_col],
        final_df[y_true_col]
    ))
    
    total_buy = final_df["buy_profit"].sum()
    total_sell = final_df["sell_profit"].sum()
    total_profit = total_buy + total_sell
    
    n_trades = len(final_df)
    profit_per_trade = total_profit / n_trades if n_trades > 0 else 0.0
    
    # Trading accuracy: fraction of trades that yielded a positive profit
    profitable_trades = (final_df["buy_profit"] > 0).sum() + (final_df["sell_profit"] > 0).sum()
    trading_accuracy = (profitable_trades / n_trades) * 100 if n_trades > 0 else 0.0
    
    return {
        "total_buy_profit": float(total_buy),
        "total_sell_profit": float(total_sell),
        "total_profit": float(total_profit),
        "profit_per_trade": float(profit_per_trade),
        "trading_accuracy": float(trading_accuracy)
    }

# ==============================================================================
# VISUALIZATION PIPELINE
# ==============================================================================

def plot_prediction_chart(final_df, forecast_offset, model_name, subfolder="", future_steps=1):
    """
    Saves a high-quality visualization comparing actual and predicted prices.
    """
    plt.figure(figsize=(12, 6))
    
    if future_steps > 1:
        # Plot step 0 of the forecast sequence (the next immediate day)
        plt.plot(final_df.index, final_df['true_adjclose_future_0'], label="Actual Price", color="#1f77b4", linewidth=2)
        plt.plot(final_df.index, final_df['adjclose_future_0'], label="Predicted Price (t+1)", color="#ff7f0e", linestyle="--", linewidth=2)
        title_suffix = f" (Multi-Step: {future_steps} Days, showing Step 1)"
    else:
        plt.plot(final_df.index, final_df[f'true_adjclose_{forecast_offset}'], label="Actual Price", color="#1f77b4", linewidth=2)
        plt.plot(final_df.index, final_df[f'adjclose_{forecast_offset}'], label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=2)
        title_suffix = f" (Forecast Offset: {forecast_offset} Days)"
    
    plt.title(f"Stock Price Prediction Comparison - {model_name}{title_suffix}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12, labelpad=10)
    plt.ylabel("Unscaled Price ($)", fontsize=12, labelpad=10)
    
    # Modern styling
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    plt.xticks(rotation=45)
    # Save Plot
    results_dir = os.path.join("results", subfolder) if subfolder else "results"
    plot_path = os.path.join(results_dir, f"{model_name}_prediction.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # Explicitly remove existing plot to ensure a clean overwrite
    if os.path.exists(plot_path):
        os.remove(plot_path)
        print(f"[Test Pipeline] Overwriting existing prediction plot at: {plot_path}")
        
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[Visualization] Saved prediction plot to: {plot_path}")

# ==============================================================================
# TESTING PIPELINE
# ==============================================================================

def test_model(
    ticker="CBA.AX",
    start_date="2020-01-01",
    end_date="2024-07-02",
    split_date="2023-08-02",
    lookback_steps=50,
    forecast_offset=1,
    scale=True,
    feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
    units=128,
    cell_type="LSTM",
    n_layers=2,
    dropout=0.3,
    loss="huber",
    optimizer="adam",
    bidirectional=False,
    model_name="lstm_model",
    subfolder="",
    future_steps=1
):
    """
    Evaluates a trained model: loads weights, runs inference, inverts scaling,
    computes accurate metrics, runs trading simulations, and saves plots and CSV logs.
    
    Returns:
        dict: Evaluation metrics and trading performance dictionary.
    """
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
        cache_dir="data",
        future_steps=future_steps
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
        future_steps=future_steps
    )
    
    weights_dir = os.path.join("results", subfolder) if subfolder else "results"
    weights_path = os.path.join(weights_dir, f"{model_name}.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained model weights not found at: {weights_path}")
        
    model.load_weights(weights_path)
    print(f"[Test Pipeline] Loaded trained model weights from: {weights_path}")
    
    # --------------------------------------------------------------------------
    # Step 3: Run Inference & Inverse Transform Scaling
    # --------------------------------------------------------------------------
    print(f"[Test Pipeline] Running inference on {len(X_test)} samples...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Correct unscaling of predictions and targets
    if scale:
        scaler = data["column_scaler"]["adjclose"]
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
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
    metrics = calculate_metrics(y_test_unscaled, y_pred_unscaled, prev_actual, future_steps=future_steps)
    
    # Get trading simulation profit metrics
    trading_metrics = get_trading_profits(test_df, forecast_offset, future_steps=future_steps)
    
    print("\n" + "="*50)
    print(f" EVALUATION RESULTS - {model_name} ")
    print("="*50)
    print(f"Unscaled MAE:           ${metrics['MAE']:.4f}")
    print(f"Unscaled RMSE:          ${metrics['RMSE']:.4f}")
    print(f"Unscaled MAPE:          {metrics['MAPE']:.2f}%")
    print(f"Directional Accuracy:   {metrics['DA']:.2f}%")
    print(f"Trading Accuracy:       {trading_metrics['trading_accuracy']:.2f}%")
    print(f"Total Trading Profit:   ${trading_metrics['total_profit']:.2f}")
    print(f"Profit per Trade:       ${trading_metrics['profit_per_trade']:.2f}")
    print("="*50)
    
    # --------------------------------------------------------------------------
    # Step 5: Save Plots and CSV Logs
    # --------------------------------------------------------------------------
    plot_prediction_chart(test_df, forecast_offset, model_name, subfolder=subfolder, future_steps=future_steps)
    
    csv_results_folder = os.path.join("csv-results", subfolder) if subfolder else "csv-results"
    csv_path = os.path.join(csv_results_folder, f"{model_name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Explicitly remove existing CSV to ensure a clean overwrite
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[Test Pipeline] Overwriting existing CSV results at: {csv_path}")
        
    test_df.to_csv(csv_path)
    print(f"[Test Pipeline] Saved detailed predictions CSV to: {csv_path}")
    
    # Combine results
    all_results = {**metrics, **trading_metrics}
    
    # Return metrics
    return all_results

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deep recurrent stock prediction models.")
    
    parser.add_argument("--ticker", type=str, default="CBA.AX", help="Stock ticker symbol.")
    parser.add_argument("--cell_type", type=str, default="LSTM", help="Recurrent cell type (LSTM, GRU, SimpleRNN).")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of recurrent layers.")
    parser.add_argument("--units", type=int, default=128, help="Number of recurrent hidden units.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--loss", type=str, default="huber", help="Loss function (huber, mse, mae).")
    parser.add_argument("--model_name", type=str, default="lstm_model", help="Name of model weights file.")
    parser.add_argument("--subfolder", type=str, default="", help="Subfolder within results directory.")
    parser.add_argument("--bidirectional", action="store_true", help="Wrap recurrent layers in Bidirectional wrapper.")
    parser.add_argument("--lookback_steps", type=int, default=50, help="Number of past time steps to look back.")
    parser.add_argument("--forecast_offset", type=int, default=1, help="Offset to start forecast ahead.")
    parser.add_argument("--future_steps", type=int, default=1, help="Number of future steps to predict.")
    parser.add_argument("--feature_columns", type=str, default="adjclose,volume,open,high,low", help="Comma-separated feature list.")
    
    args = parser.parse_args()
    
    feature_list = [f.strip() for f in args.feature_columns.split(",") if f.strip()]
    
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
        feature_columns=feature_list
    )
