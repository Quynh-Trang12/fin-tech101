# ==============================================================================
# Purpose:
# C.4 deep learning hyperparameter experiment sweep orchestrator.
# ==============================================================================

from typing import Any

from base_sweep import BaseSweepRunner
from config import (
    C4_SWEEP_CONFIGS,
    END_DATE,
    RESULTS_DIR,
    SPLIT_DATE,
    START_DATE,
    TICKER,
)


# ==============================================================================
# TASK C.4 SWEEP RUNNER
# ==============================================================================


class C4SweepRunner(BaseSweepRunner):
    """Run Task C.4 sweeps for recurrent cell, depth, width, and loss variants."""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        split_date: str,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the C.4 sweep runner with shared experiment settings."""
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            split_date=split_date,
            dropout=dropout,
            subfolder="c4",
        )

    def format_results_row(
        self,
        config_id: str,
        config_params: dict[str, Any],
        test_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Format one C.4 sweep result row for the report-ready CSV."""
        return {
            "Model Name": config_id,
            "Cell Type": config_params["cell_type"],
            "Layers": config_params["n_layers"],
            "Units": config_params["units"],
            "Loss": config_params["loss"],
            "Epochs": config_params["epochs"],
            "Batch Size": config_params["batch_size"],
            "Unscaled MAE ($)": round(test_results["MAE"], 4),
            "Unscaled RMSE ($)": round(test_results["RMSE"], 4),
            "Unscaled MAPE (%)": round(test_results["MAPE"], 2),
            "Directional Acc (%)": round(test_results["DA"], 2),
            "Trading Acc (%)": round(test_results["trading_accuracy"], 2),
            "Total Trading Profit ($)": round(test_results["total_profit"], 2),
            "Profit per Trade ($)": round(test_results["profit_per_trade"], 2),
        }


# ==============================================================================
# MAIN RUNNER EXECUTION
# ==============================================================================


def run_c4_sweeps() -> None:
    """Run all Task C.4 hyperparameter sweeps and save consolidated results."""
    runner = C4SweepRunner(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        split_date=SPLIT_DATE,
        dropout=0.3,
    )

    runner.run_sweep(
        configs=C4_SWEEP_CONFIGS,
        output_csv_path=RESULTS_DIR / "c4" / "c4_sweep_results.csv",
    )


if __name__ == "__main__":
    run_c4_sweeps()
