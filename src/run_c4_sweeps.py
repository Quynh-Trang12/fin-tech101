# ==============================================================================
# Purpose: 
# C.4 Deep Learning hyperparameter experiment sweep orchestrator.
# ==============================================================================

from config import SWEEP_CONFIGS, TICKER, START_DATE, END_DATE, SPLIT_DATE
from base_sweep import BaseSweepRunner


class C4SweepRunner(BaseSweepRunner):
    """
    Orchestrator class for Task C.4 experiment sweeps.
    Compares recurrent cell types, model depths, model widths, and loss formulations.
    """

    def __init__(self, ticker, start_date, end_date, split_date, dropout=0.3):
        """
        Initializes the C4 sweep runner.
        """
        super().__init__(ticker, start_date, end_date, split_date, dropout)
        self.subfolder = "c4"

    def format_results_row(self, config_id, config_params, test_results):
        """
        Formats C4 configuration parameters and test metrics into a row matching
        the C4 report structure.
        """
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
            "Profit per Trade ($)": round(test_results["profit_per_trade"], 2)
        }


# ==============================================================================
# MAIN RUNNER EXECUTION
# ==============================================================================

def run_c4_sweeps():
    """
    Main entry point for running Task C.4 hyperparameter sweeps.
    """
    runner = C4SweepRunner(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        split_date=SPLIT_DATE,
        dropout=0.3
    )
    runner.run_sweep(
        configs=SWEEP_CONFIGS,
        output_csv_path="results/c4/c4_sweep_results.csv"
    )


if __name__ == "__main__":
    run_c4_sweeps()
