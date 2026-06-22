# ==============================================================================
# File: run_sweeps.py
# Purpose: Modular, production-grade experiment runner to orchestrate the 
#          hyperparameter sweeps, compile results, and output comparisons.
# ==============================================================================

import os
import pandas as pd

# Import global configurations and parameters
from config import SWEEP_CONFIGS, TICKER, START_DATE, END_DATE, SPLIT_DATE

# Import local training and evaluation modules
from train import train_model
from test import test_model

# ==============================================================================
# EXPERIMENT ORCHESTRATION PIPELINE
# ==============================================================================

def execute_all_sweeps():
    """
    Iteratively executes training and evaluation for each configuration,
    consolidates performance metrics, and saves the final result matrix.
    """
    results = []
    
    # --------------------------------------------------------------------------
    # Phase 1: Print experiment metadata header
    # --------------------------------------------------------------------------
    print("=" * 80)
    print(f"STARTING HYPERPARAMETER EXPERIMENT SWEEPS FOR {TICKER}")
    print(f"Data Interval:   {START_DATE} to {END_DATE}")
    print(f"Chronological Split Date: {SPLIT_DATE}")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # Phase 2: Run loop across configurations
    # --------------------------------------------------------------------------
    for config_id, params in SWEEP_CONFIGS.items():
        print("\n" + "#" * 70)
        print(f" EXPERIMENT: {config_id} - {params['description']}")
        print("#" * 70)
        
        # 1. Invoke modular training pipeline
        train_results = train_model(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            split_date=SPLIT_DATE,
            cell_type=params["cell_type"],
            n_layers=params["n_layers"],
            units=params["units"],
            loss=params["loss"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            model_name=config_id,
            dropout=0.3,
            shuffle=True
        )
        
        # 2. Invoke modular testing/evaluation pipeline (chronological split)
        test_results = test_model(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            split_date=SPLIT_DATE,
            cell_type=params["cell_type"],
            n_layers=params["n_layers"],
            units=params["units"],
            loss=params["loss"],
            model_name=config_id,
            dropout=0.3
        )
        
        # 3. Compile individual run row
        row = {
            "Config ID": config_id,
            "Cell Type": params["cell_type"],
            "Layers": params["n_layers"],
            "Units": params["units"],
            "Loss": params["loss"],
            "Epochs": params["epochs"],
            "Batch Size": params["batch_size"],
            "Unscaled MAE ($)": round(test_results["MAE"], 4),
            "Unscaled RMSE ($)": round(test_results["RMSE"], 4),
            "Unscaled MAPE (%)": round(test_results["MAPE"], 2),
            "Directional Acc (%)": round(test_results["DA"], 2),
            "Trading Acc (%)": round(test_results["trading_accuracy"], 2),
            "Total Trading Profit ($)": round(test_results["total_profit"], 2),
            "Profit per Trade ($)": round(test_results["profit_per_trade"], 2)
        }
        results.append(row)
        
    # --------------------------------------------------------------------------
    # Phase 3: Consolidate, Save, and Print Results
    # --------------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    
    # Save the consolidated experiment logs to results folder
    os.makedirs("results", exist_ok=True)
    results_csv_path = os.path.join("results", "c4_sweep_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER EXPERIMENT SWEEPS COMPLETED SUCCESSFULLY!")
    print(f"Consolidated results matrix saved to: {results_csv_path}")
    print("=" * 80)
    
    # Print formatted comparative table to terminal
    print("\nSummary Table:")
    try:
        print(df_results.to_markdown(index=False))
    except Exception:
        print(df_results.to_string(index=False))

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    execute_all_sweeps()
