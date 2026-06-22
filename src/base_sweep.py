# ==============================================================================
# Purpose: 
# Reusable, parameter-agnostic base experiment runner orchestrating
# hyperparameter sweeps, compiling results, and outputting comparisons.
# ==============================================================================

import os
import inspect
import pandas as pd

from train import train_model
from test import test_model


# ==============================================================================
# PARAMETER REFLECTION UTILITIES
# ==============================================================================

def get_valid_args(func, *args_dicts):
    """
    Combines multiple dictionaries and filters out keys that are not valid
    parameters of the target function.

    Args:
        func (callable): The target function to inspect.
        *args_dicts: One or more dictionaries containing potential parameters.

    Returns:
        dict: A dictionary containing only the keys that are valid for func.
    """
    combined = {}
    for d in args_dicts:
        if d:
            combined.update(d)
    sig = inspect.signature(func)
    return {k: v for k, v in combined.items() if k in sig.parameters}


# ==============================================================================
# ABSTRACT BASE RUNNER PIPELINE
# ==============================================================================

class BaseSweepRunner:
    """
    Abstract base class for running deep learning hyperparameter sweeps.
    Decoupled from specific metric columns and formatting requirements.
    """

    def __init__(self, ticker, start_date, end_date, split_date, dropout=0.3):
        """
        Initializes the base sweep runner with shared configuration parameters.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Dataset start date.
            end_date (str): Dataset end date.
            split_date (str): Training/testing chronological split date.
            dropout (float): Dropout regularization rate.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.dropout = dropout

    def get_global_params(self):
        """
        Returns a dictionary of shared global parameters to be passed to
        training and testing functions.

        Returns:
            dict: Global parameters dictionary.
        """
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "split_date": self.split_date,
            "dropout": self.dropout,
        }

    def print_header(self):
        """Prints a standardized header before starting the sweep."""
        print("=" * 80)
        print(f"STARTING SWEEP RUNNER: {self.__class__.__name__}")
        print(f"Ticker:                 {self.ticker}")
        print(f"Data Period:            {self.start_date} to {self.end_date}")
        print(f"Split Date:             {self.split_date}")
        print("=" * 80)

    def format_results_row(self, config_id, config_params, test_results):
        """
        Formats configuration parameters and test metrics into a row.
        Must be implemented by subclasses.

        Args:
            config_id (str): Unique configuration identifier.
            config_params (dict): Specific parameters of the configuration.
            test_results (dict): Output metrics dictionary from test_model.

        Returns:
            dict: Formatted CSV row.
        """
        raise NotImplementedError("Subclasses must implement format_results_row")

    def run_sweep(self, configs, output_csv_path):
        """
        Executes the sweep over all provided configurations.

        Args:
            configs (dict): Mapping of configuration IDs to parameter dicts.
            output_csv_path (str): File path to save consolidated CSV results.

        Returns:
            pd.DataFrame: The consolidated results matrix.
        """
        self.print_header()
        results = []

        for config_id, params in configs.items():
            print("\n" + "#" * 80)
            print(f" RUNNING CONFIGURATION: {config_id} - {params.get('description', '')}")
            print("#" * 80)

            # Combine global settings and configuration specific parameters
            merged_params = {
                **self.get_global_params(), 
                **params, 
                "model_name": config_id,
                "subfolder": getattr(self, "subfolder", "")
            }

            # 1. Invoke modular training pipeline
            train_args = get_valid_args(train_model, merged_params)
            print(f"[BaseSweepRunner] Invoking train_model for {config_id}...")
            train_model(**train_args)

            # 2. Invoke modular testing/evaluation pipeline
            test_args = get_valid_args(test_model, merged_params)
            print(f"[BaseSweepRunner] Invoking test_model for {config_id}...")
            test_results = test_model(**test_args)

            # 3. Format result row
            row = self.format_results_row(config_id, params, test_results)
            results.append(row)

        # Consolidate results into a DataFrame
        df_results = pd.DataFrame(results)

        # Ensure output directory exists and save to CSV
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Explicitly remove the old results file to guarantee overwrite
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            print(f"[BaseSweepRunner] Overwriting existing sweep results at: {output_csv_path}")
            
        df_results.to_csv(output_csv_path, index=False)

        print("\n" + "=" * 80)
        print(f"EXPERIMENT SWEEPS COMPLETED SUCCESSFULLY!")
        print(f"Consolidated results matrix saved to: {output_csv_path}")
        print("=" * 80)

        # Print formatted comparative table to terminal
        print("\nSummary Table:")
        try:
            print(df_results.to_markdown(index=False))
        except Exception:
            print(df_results.to_string(index=False))

        return df_results
