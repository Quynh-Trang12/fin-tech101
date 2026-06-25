# ==============================================================================
# Purpose:
# C.5 Multivariate and Multistep Deep Learning experiment sweep orchestrator.
# ==============================================================================

from config import (
    C5_SWEEP_CONFIGS,
    END_DATE,
    FORECAST_OFFSET,
    LOOKBACK_STEPS,
    RESULTS_DIR,
    SPLIT_DATE,
    START_DATE,
    TICKER,
)
from base_sweep import BaseSweepRunner
from typing import Any


class C5SweepRunner(BaseSweepRunner):
    """
    Orchestrator class for Task C.5 experiment sweeps.
    Extends BaseSweepRunner to support multivariate features, lookback steps,
    and forecast offset parameters.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        split_date: str,
        lookback_steps: int,
        forecast_offset: int,
        dropout: float = 0.3,
    ):
        """
        Initializes the C5 sweep runner with window size and offset parameters.
        """
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            split_date=split_date,
            dropout=dropout,
            subfolder="c5",
        )
        self.lookback_steps = lookback_steps
        self.forecast_offset = forecast_offset

    def get_extra_global_params(self) -> dict[str, Any]:
        """Return C.5-specific parameters shared by all configurations."""
        return {
            "lookback_steps": self.lookback_steps,
            "forecast_offset": self.forecast_offset,
            "scale": True,
            "shuffle": True,
            "loss": "huber",
        }

    def get_extra_header_lines(self) -> list[tuple[str, Any]]:
        """Return C.5-specific terminal header lines."""
        return [
            ("Lookback Steps", self.lookback_steps),
            ("Forecast Offset", self.forecast_offset),
        ]

    def format_results_row(
        self,
        config_id: str,
        config_params: dict[str, Any],
        test_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Formats C5 configuration parameters and test metrics into a row matching
        the C5 report structure.
        """
        return {
            "Model Name": config_id,
            "Description": config_params["description"],
            "Features": ", ".join(config_params["feature_columns"]),
            "Future Steps": config_params["future_steps"],
            "MAE ($)": round(test_results["MAE"], 4),
            "RMSE ($)": round(test_results["RMSE"], 4),
            "MAPE (%)": round(test_results["MAPE"], 2),
            "DA (%)": round(test_results["DA"], 2),
            "Trading Acc (%)": round(test_results["trading_accuracy"], 2),
            "Total Profit ($)": round(test_results["total_profit"], 2),
            "Profit/Trade ($)": round(test_results["profit_per_trade"], 2),
        }


# ==============================================================================
# MAIN RUNNER EXECUTION
# ==============================================================================


def run_c5_sweeps() -> None:
    """
    Main entry point for running Task C.5 multivariate and multistep sweeps.
    """
    runner = C5SweepRunner(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        split_date=SPLIT_DATE,
        lookback_steps=LOOKBACK_STEPS,
        forecast_offset=FORECAST_OFFSET,
        dropout=0.3,
    )
    runner.run_sweep(
        configs=C5_SWEEP_CONFIGS,
        output_csv_path=RESULTS_DIR / "c5" / "c5_sweep_results.csv",
    )


if __name__ == "__main__":
    run_c5_sweeps()
