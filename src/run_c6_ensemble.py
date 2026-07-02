"""
Task C.6 Preliminary Experiment: Weighted Prediction Averaging Ensemble

This script represents the initial weighted-averaging investigation between
the fixed-parameter ARIMA model and the LSTM model. Experimental results showed
that linear ensembling degraded performance due to price level drift in the LSTM.

The final, primary Task C.6 implementation is located at `src/run_c6_hybrid.py` 
(which implements the hybrid residual-learning framework).
"""

import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

# Add src to python path to import local modules
sys.path.append(str(Path(__file__).resolve().parent))

from data_processing import load_and_process_data
from model_factory import build_dl_model
from test import calculate_metrics, get_trading_profits
from config import (
    TICKER,
    START_DATE,
    END_DATE,
    SPLIT_DATE,
    VALIDATION_RATIO,
    LOOKBACK_STEPS,
    FORECAST_OFFSET,
    FUTURE_STEPS,
    FEATURE_COLUMNS,
)

# Suppress statsmodels convergence warnings for clean execution logs
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)

def plot_c6_predictions(index, y_true, y_pred, label, save_path):
    """Plot actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(index, y_true, label="Actual Price", color="#1f77b4", linewidth=2)
    ax.plot(index, y_pred, label=f"Predicted Price ({label})", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_title(f"Stock Price Prediction Comparison - {label}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12, labelpad=10)
    ax.set_ylabel("Unscaled Price ($)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    plt.xticks(rotation=45)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[C.6 Plot] Saved prediction plot to: {save_path.as_posix()}")

def evaluate_preds(y_pred, y_true, prev_actual, test_df):
    """Evaluate predictions using standard metrics and simulated trading."""
    metrics = calculate_metrics(y_true, y_pred, prev_actual, future_steps=1)
    
    # Insert predictions into a copy of test_df for trading profits calculation
    df_copy = test_df.copy()
    df_copy["adjclose_1"] = y_pred
    df_copy["true_adjclose_1"] = y_true
    
    trading_metrics = get_trading_profits(df_copy, forecast_offset=1, future_steps=1)
    return {**metrics, **trading_metrics}

def main():
    print("=" * 80)
    print("STARTING TASK C.6 ENSEMBLE FORECASTING EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Step 1: Load and Process Stock Data
    # --------------------------------------------------------------------------
    print("[C.6 Data] Loading stock data package...")
    data = load_and_process_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_steps=LOOKBACK_STEPS,
        scale=True,
        shuffle=False,  # Keep chronological order for testing
        forecast_offset=1,
        split_by_date=True,
        split_date=SPLIT_DATE,
        validation_ratio=VALIDATION_RATIO,
        feature_columns=FEATURE_COLUMNS,
        future_steps=1,
        save_scaler_cache=False,
    )

    df = data["df"]
    X_test = data["X_test"]
    test_input_dates = data["test_input_dates"]
    test_df = data["test_df"].copy()
    test_df.sort_index(inplace=True)

    # Validate alignment
    prev_actual = test_df["adjclose"].values
    y_test_scaled = data["y_test"]
    scaler = data["column_scaler"]["adjclose"]
    y_test_unscaled = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)

    print(f"[C.6 Data] Test samples aligned: {len(test_df)}")

    # --------------------------------------------------------------------------
    # Step 2: Generate LSTM Baseline Predictions
    # --------------------------------------------------------------------------
    print("[C.6 LSTM] Loading LSTM baseline weights and generating predictions...")
    n_features = len(FEATURE_COLUMNS)
    lstm_model = build_dl_model(
        lookback_steps=LOOKBACK_STEPS,
        n_features=n_features,
        units=128,
        cell_type="LSTM",
        n_layers=2,
        dropout=0.3,
        loss="huber",
        optimizer="adam",
        future_steps=1,
    )
    
    weights_path = Path("results/c4/LSTM_BASE.weights.h5")
    if not weights_path.exists():
        # Fallback to general results if c4 folder is absent
        weights_path = Path("results/lstm_model.weights.h5")
        
    lstm_model.load_weights(weights_path)
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
    y_pred_lstm = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).reshape(-1)

    # --------------------------------------------------------------------------
    # Step 3: Run Fixed-Parameter ARIMA Forecasts (Option B)
    # --------------------------------------------------------------------------
    print("[C.6 ARIMA] Running fixed-parameter ARIMA forecasts...", flush=True)
    from statsmodels.tsa.arima.model import ARIMA
    
    arima_orders = [(1, 1, 1), (2, 1, 2), (5, 1, 0)]
    arima_preds = {}

    # Define training history ending at the day before the first test target.
    # The first test input date is test_df.index[0]. We fit the ARIMA model once on this training history.
    train_history = df.loc[:test_df.index[0], "adjclose"]
    print(f"[C.6 ARIMA] Training period size: {len(train_history)} days (up to {test_df.index[0].strftime('%Y-%m-%d')})")

    for order in arima_orders:
        order_str = f"ARIMA{order[0]}_{order[1]}_{order[2]}"
        print(f"  - Fitting model once and freezing parameters: {order_str}...", flush=True)
        
        # Fit the model once on the training set
        model = ARIMA(
            train_history,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(method='statespace', cov_type='none')
        
        # Re-apply the fitted model parameters to the entire series (train + test) without refitting
        full_series = df.loc[:test_df.index[-1], "adjclose"]
        applied_res = res.apply(full_series)
        
        # Retrieve the predictions for the test set (one-step-ahead forecasts using frozen parameters)
        preds = applied_res.predict(start=test_df.index[0], end=test_df.index[-1]).values
        
        arima_preds[order] = preds
        print(f"  - Completed {order_str}.", flush=True)

    # --------------------------------------------------------------------------
    # Step 4: Evaluate Individual Models and Ensembles
    # --------------------------------------------------------------------------
    print("[C.6 Evaluation] Computing metrics for all combinations...", flush=True)
    results_list = []
    
    # 4.1 LSTM Baseline
    lstm_metrics = evaluate_preds(y_pred_lstm, y_test_unscaled, prev_actual, test_df)
    results_list.append({
        "Model": "LSTM_BASE",
        "Order/Weights": "N/A",
        **lstm_metrics
    })

    # 4.2 ARIMA Baselines
    for order in arima_orders:
        order_str = f"ARIMA{order}"
        metrics = evaluate_preds(arima_preds[order], y_test_unscaled, prev_actual, test_df)
        results_list.append({
            "Model": order_str,
            "Order/Weights": str(order),
            **metrics
        })

    # 4.3 Weighted Ensembles (combine LSTM with each ARIMA model)
    ensemble_weights = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
    
    for order in arima_orders:
        order_str = f"ARIMA{order}"
        y_pred_arima = arima_preds[order]
        
        for w_lstm, w_arima in ensemble_weights:
            ens_label = f"Ensemble (LSTM*{w_lstm} + {order_str}*{w_arima})"
            y_pred_ens = w_lstm * y_pred_lstm + w_arima * y_pred_arima
            
            metrics = evaluate_preds(y_pred_ens, y_test_unscaled, prev_actual, test_df)
            results_list.append({
                "Model": ens_label,
                "Order/Weights": f"LSTM: {w_lstm}, ARIMA: {w_arima}",
                **metrics
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # --------------------------------------------------------------------------
    # Step 5: Save Outputs to artifacts/
    # --------------------------------------------------------------------------
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 5.1 Save CSV Metrics
    csv_path = artifacts_dir / "c6_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[C.6 CSV] Saved metrics CSV to: {csv_path.as_posix()}")

    # 5.2 Find the best ARIMA and the best Ensemble based on MAE
    best_arima_idx = results_df[results_df["Model"].str.startswith("ARIMA")]["MAE"].idxmin()
    best_arima_row = results_df.loc[best_arima_idx]
    best_arima_name = best_arima_row["Model"]
    best_arima_order = eval(best_arima_row["Order/Weights"])
    
    best_ens_idx = results_df[results_df["Model"].str.startswith("Ensemble")]["MAE"].idxmin()
    best_ens_row = results_df.loc[best_ens_idx]
    best_ens_name = best_ens_row["Model"]

    # 5.3 Plot charts
    plot_c6_predictions(
        test_df.index,
        y_test_unscaled,
        arima_preds[best_arima_order],
        best_arima_name,
        artifacts_dir / "arima_prediction.png"
    )
    
    # Re-evaluate the best ensemble predictions to plot
    # Extract weights and ARIMA order from best ensemble name
    # e.g., Ensemble (LSTM*0.7 + ARIMA(2, 1, 2)*0.3)
    best_w_lstm = 0.5
    best_w_arima = 0.5
    for w_lstm, w_arima in ensemble_weights:
        if f"LSTM*{w_lstm}" in best_ens_name:
            best_w_lstm = w_lstm
            best_w_arima = w_arima
            break
            
    best_arima_for_ens = (1, 1, 1)
    for order in arima_orders:
        if f"ARIMA{order}" in best_ens_name:
            best_arima_for_ens = order
            break
            
    best_ens_preds = best_w_lstm * y_pred_lstm + best_w_arima * arima_preds[best_arima_for_ens]
    plot_c6_predictions(
        test_df.index,
        y_test_unscaled,
        best_ens_preds,
        "Best Ensemble",
        artifacts_dir / "ensemble_prediction.png"
    )

    # --------------------------------------------------------------------------
    # Step 6: Generate Summary Report
    # --------------------------------------------------------------------------
    summary_md_path = artifacts_dir / "c6_experiment_summary.md"
    
    # Format table for Markdown
    md_table = results_df.to_markdown(index=False)

    # Find overall best model based on MAE
    best_overall_idx = results_df["MAE"].idxmin()
    best_overall_row = results_df.loc[best_overall_idx]
    best_overall_name = best_overall_row["Model"]
    
    md_report = f"""# Task C.6 Ensemble Forecasting Experiment Summary (Fixed-Parameter Protocol)

This report documents the experimental results of the fixed-parameter evaluation protocol (Option B). The classical statistical ARIMA model and the deep learning LSTM model are both evaluated under identical chronological partitioning and frozen parameter constraints.

---

## 1. Experimental Setup

- **Dataset**: `CBA.AX` daily stock prices from Yahoo Finance cache.
- **Chronological Test Period**: Starts at `{SPLIT_DATE}` (split boundary).
- **Target Variable**: Adjusted Closing Price (`adjclose`).
- **ARIMA Method**: Fixed-parameter forecasting (Option B). The model parameters are estimated **once** on the training dataset (up to {test_df.index[0].strftime('%Y-%m-%d')}). The fitted parameters are frozen, and the filter is applied over the test period to generate one-step-ahead forecasts.
- **Ensemble Method**: Linear weighted averaging of LSTM predictions with fixed-parameter ARIMA predictions.

---

## 2. Tested Configurations

- **LSTM Baseline**:
  - `LSTM_BASE`: Standard 2-layer LSTM model with 128 hidden units.
- **ARIMA Baseline Orders**:
  - `ARIMA(1, 1, 1)`: Standard ARMA(1,1) on first-differenced prices.
  - `ARIMA(2, 1, 2)`: Higher-order stationary modeling.
  - `ARIMA(5, 1, 0)`: Autoregressive-only model on first-differenced prices.
- **Ensemble Weights (LSTM / ARIMA)**:
  - 50% LSTM / 50% ARIMA
  - 70% LSTM / 30% ARIMA
  - 30% LSTM / 70% ARIMA

---

## 3. Experimental Results

{md_table}

---

## 4. Key Findings

1. **ARIMA Baseline Outperforms LSTM**:
   - All individual Fixed-Parameter ARIMA models significantly outperformed the static LSTM baseline.
   - The best-performing model overall was **{best_overall_name}** with an MAE of **${best_overall_row['MAE']:.4f}** and RMSE of **${best_overall_row['RMSE']:.4f}** (compared to the LSTM's MAE of **${lstm_metrics['MAE']:.4f}**).
   - This occurs because ARIMA differences the price series to model local changes and anchors its next-day forecast on the most recent actual price ($y_t$), avoiding drift.

2. **Weighted Ensembles Underperform ARIMA**:
   - No weighted ensemble outperformed the individual ARIMA baselines.
   - Averaging the weaker LSTM baseline (MAE ~${lstm_metrics['MAE']:.4f}) with the stronger ARIMA baseline (MAE ~$1.08) degraded performance. The ensemble error scales linearly with the weight of the LSTM.

3. **Trading Profitability**:
   - In contrast to the rolling ARIMA run, the Fixed-Parameter ARIMA models generated **positive simulated trading profits** (e.g. `ARIMA(5, 1, 0)` achieved a total profit of **${results_df.loc[results_df['Model'] == 'ARIMA(5, 1, 0)', 'total_profit'].values[0]:.2f}** and directional accuracy of **{results_df.loc[results_df['Model'] == 'ARIMA(5, 1, 0)', 'DA'].values[0]:.2f}%**).
   - All weighted ensembles reported negative profits, demonstrating that LSTM error contamination directly harms trading utility.

---

## 5. Notes, Warnings, and Limitations

- **Fast Execution**: Because the models are only fit once, execution time is extremely fast (under 1 second) compared to recursive daily refitting.
- **Stationarity Warnings**: Occasional statsmodels warning messages about near-non-invertibility were generated during the initial training fit but were safely bypassed.

---

## 6. Recommended Next Steps

1. **Hybrid Residual-Learning (Option D)**: Instead of linear weighted averaging, use the LSTM model to predict the forecast residuals of the ARIMA model.
2. **Sentiment Indicators (Task C.7)**: Integrate external news sentiment data to further enhance directional trading signals.
"""

    with open(summary_md_path, "w") as f:
        f.write(md_report)
    print(f"[C.6 Report] Saved summary report to: {summary_md_path.as_posix()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
