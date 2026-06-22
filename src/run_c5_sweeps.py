# ==============================================================================
# Purpose:
# C.5 Multivariate and Multistep Deep Learning experiment sweep orchestrator.
# ==============================================================================

from config import TICKER, START_DATE, END_DATE, SPLIT_DATE, LOOKBACK_STEPS, FORECAST_OFFSET
from base_sweep import BaseSweepRunner

# C.5 configurations are multivariate and/or multi-step models
C5_CONFIGS = {
    "lstm_uni_multistep": {
        "description": "Univariate Multistep (Close Only -> 5 Days Forecast)",
        "feature_columns": ["adjclose"],
        "future_steps": 5,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "LSTM"
    },
    "lstm_multi_singlestep": {
        "description": "Multivariate Single-Step (All Features -> 1 Day Forecast)",
        "feature_columns": ["adjclose", "volume", "open", "high", "low"],
        "future_steps": 1,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "LSTM"
    },
    "lstm_multi_multistep": {
        "description": "Multivariate Multistep Combined (All Features -> 5 Days Forecast)",
        "feature_columns": ["adjclose", "volume", "open", "high", "low"],
        "future_steps": 5,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "LSTM"
    }
}


class C5SweepRunner(BaseSweepRunner):
    """
    Orchestrator class for Task C.5 experiment sweeps.
    Extends BaseSweepRunner to support multivariate features, lookback steps,
    and forecast offset parameters.
    """

    def __init__(
        self,
        ticker,
        start_date,
        end_date,
        split_date,
        lookback_steps,
        forecast_offset,
        dropout=0.3
    ):
        """
        Initializes the C5 sweep runner with window size and offset parameters.
        """
        super().__init__(ticker, start_date, end_date, split_date, dropout)
        self.lookback_steps = lookback_steps
        self.forecast_offset = forecast_offset
        self.subfolder = "c5"

    def get_global_params(self):
        """
        Overrides parent method to include C.5 specific settings:
        lookback_steps, forecast_offset, loss="huber", scale=True, shuffle=True.
        """
        params = super().get_global_params()
        params.update({
            "lookback_steps": self.lookback_steps,
            "forecast_offset": self.forecast_offset,
            "scale": True,
            "shuffle": True,
            "loss": "huber"
        })
        return params

    def print_header(self):
        """
        Overrides parent method to output C.5 specific execution parameters.
        """
        print("=" * 80)
        print(f"STARTING SWEEP RUNNER: {self.__class__.__name__}")
        print(f"Ticker:                 {self.ticker}")
        print(f"Data Period:            {self.start_date} to {self.end_date}")
        print(f"Split Date:             {self.split_date}")
        print(f"Lookback Steps:         {self.lookback_steps}")
        print(f"Forecast Offset:        {self.forecast_offset}")
        print("=" * 80)

    def format_results_row(self, config_id, config_params, test_results):
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
            "Profit/Trade ($)": round(test_results["profit_per_trade"], 2)
        }


# ==============================================================================
# MAIN RUNNER EXECUTION
# ==============================================================================

def run_c5_sweeps():
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
        dropout=0.3
    )
    runner.run_sweep(
        configs=C5_CONFIGS,
        output_csv_path="results/c5/c5_sweep_results.csv"
    )


if __name__ == "__main__":
    run_c5_sweeps()

