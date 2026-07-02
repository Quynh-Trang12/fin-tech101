# ==============================================================================
# Purpose:
# Reusable, parameter-agnostic base experiment runner orchestrating
# hyperparameter sweeps, compiling results, and outputting comparisons.
# ==============================================================================

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any
import inspect

import pandas as pd

from test import test_model
from train import train_model
from config import VALIDATION_RATIO

# ==============================================================================
# PARAMETER REFLECTION UTILITIES
# ==============================================================================


def get_valid_args(
    func: Callable[..., Any],
    *args_dicts: dict[str, Any],
) -> dict[str, Any]:
    """
    Return only keyword arguments accepted by the target function.

    Args:
        func: The target function to inspect.
        *args_dicts: One or more dictionaries containing potential keyword arguments.

    Returns:
        A dictionary containing only the valid keyword arguments for the target function.
    """
    valid_parameters = inspect.signature(func).parameters

    merged_args = {}
    for args in args_dicts:
        merged_args.update(args)

    return {
        name: value for name, value in merged_args.items() if name in valid_parameters
    }


# ==============================================================================
# ABSTRACT BASE RUNNER PIPELINE
# ==============================================================================


class BaseSweepRunner(ABC):
    """
    Abstract base class for running deep learning hyperparameter sweeps.
    Decoupled from specific metric columns and formatting requirements.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        split_date: str,
        dropout: float = 0.3,
        subfolder: str = "",
        validation_ratio: float = VALIDATION_RATIO,
    ) -> None:
        """
        Initializes the base sweep runner with shared configuration parameters.

        Args:
            ticker: Stock ticker symbol.
            start_date: Dataset start date.
            end_date: Dataset end date.
            split_date: Training/testing chronological split date.
            dropout: Dropout regularization rate.
            subfolder: Results output subfolder.
            validation_ratio: Chronological validation split ratio relative to training set.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.dropout = dropout
        self.subfolder = subfolder
        self.validation_ratio = validation_ratio

    def get_extra_global_params(self) -> dict[str, Any]:
        """Return subclass-specific parameters shared by all sweep configurations."""
        return {}

    def get_extra_header_lines(self) -> list[tuple[str, Any]]:
        """Return subclass-specific header lines for terminal output."""
        return []

    def get_global_params(self) -> dict[str, Any]:
        """Return shared parameters passed to training and testing functions."""
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "split_date": self.split_date,
            "dropout": self.dropout,
            "subfolder": self.subfolder,
            "validation_ratio": self.validation_ratio,
            **self.get_extra_global_params(),
        }

    def print_header(self) -> None:
        """Print a standardized sweep header."""
        print("=" * 80)
        print(f"STARTING SWEEP RUNNER:  {self.__class__.__name__}")
        print(f"Ticker:                 {self.ticker}")
        print(f"Data Period:            {self.start_date} to {self.end_date}")
        print(f"Split Date:             {self.split_date}")

        for label, value in self.get_extra_header_lines():
            print(f"{label}:".ljust(24) + f"{value}")

        print("=" * 80)

    @abstractmethod
    def format_results_row(
        self,
        config_id: str,
        config_params: dict[str, Any],
        test_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Format one sweep result row for the subclass-specific report."""
        raise NotImplementedError

    def run_single_config(
        self,
        config_id: str,
        config_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Train, test, and format one sweep configuration."""
        print("\n" + "#" * 80)
        print(
            " RUNNING CONFIGURATION: "
            f"{config_id} - {config_params.get('description', '')}"
        )
        print("#" * 80)

        # ----------------------------------------------------------------------
        # Phase 1: Merge global and configuration-specific parameters
        # ----------------------------------------------------------------------
        merged_params = {
            **self.get_global_params(),
            **config_params,
            "model_name": config_id,
        }

        # ----------------------------------------------------------------------
        # Phase 2: Train model for current configuration
        # ----------------------------------------------------------------------
        train_args = get_valid_args(train_model, merged_params)
        print(f"[BaseSweepRunner] Invoking train_model for {config_id}...")
        train_model(**train_args)

        # ----------------------------------------------------------------------
        # Phase 3: Test model and collect metrics
        # ----------------------------------------------------------------------
        test_args = get_valid_args(test_model, merged_params)
        print(f"[BaseSweepRunner] Invoking test_model for {config_id}...")
        test_results = test_model(**test_args)

        # ----------------------------------------------------------------------
        # Phase 4: Format subclass-specific result row
        # ----------------------------------------------------------------------
        return self.format_results_row(config_id, config_params, test_results)

    def save_results(
        self,
        df_results: pd.DataFrame,
        output_csv_path: Path,
    ) -> None:
        """Save sweep results to CSV and print a terminal summary."""
        # ----------------------------------------------------------------------
        # Phase 1: Save consolidated CSV
        # ----------------------------------------------------------------------
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if output_csv_path.exists():
            print(
                "[BaseSweepRunner] Overwriting existing sweep results at: "
                f"{output_csv_path.as_posix()}"
            )

        df_results.to_csv(output_csv_path, index=False)

        # ----------------------------------------------------------------------
        # Phase 2: Print completion summary
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("EXPERIMENT SWEEPS COMPLETED SUCCESSFULLY!")
        print(f"Consolidated results matrix saved to: {output_csv_path.as_posix()}")
        print("=" * 80)

        # ----------------------------------------------------------------------
        # Phase 3: Print terminal comparison table
        # ----------------------------------------------------------------------
        print("\nSummary Table:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.expand_frame_repr", False)

        rename_mapping = {
            "Model Name": "Model",
            "Cell Type": "Cell",
            "Layers": "L",
            "Units": "Units",
            "Batch Size": "Batch",
            "Unscaled MAE ($)": "MAE",
            "Unscaled RMSE ($)": "RMSE",
            "Unscaled MAPE (%)": "MAPE",
            "Directional Acc (%)": "Dir Acc",
            "Trading Acc (%)": "Trade Acc",
            "Total Trading Profit ($)": "Profit",
            "Profit per Trade ($)": "$/Trade",
        }
        display_df = df_results.rename(columns=rename_mapping)
        print(display_df.to_string(index=False))

    def run_sweep(
        self,
        configs: dict[str, dict[str, Any]],
        output_csv_path: Path,
    ) -> pd.DataFrame:
        """
        Executes the sweep over all provided configurations.

        Args:
            configs: Mapping of configuration IDs to parameter dicts.
            output_csv_path: File path to save consolidated CSV results.

        Returns:
            The consolidated results matrix.
        """
        if not configs:
            raise ValueError("configs must contain at least one sweep configuration.")

        self.print_header()
        results = [
            self.run_single_config(config_id, params)
            for config_id, params in configs.items()
        ]

        # Consolidate results into a DataFrame
        df_results = pd.DataFrame(results)
        self.save_results(df_results, output_csv_path)

        return df_results
