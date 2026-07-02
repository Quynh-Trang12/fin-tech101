import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("results/c4/c4_sweep_results.csv")
    if not csv_path.exists():
        csv_path = Path("csv-results/c4/c4_sweep_results.csv")

    if not csv_path.exists():
        print(f"Error: Could not find results CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Print using the exact print options and column mapping specified
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
    display_df = df.rename(columns=rename_mapping)
    print("\nSummary Table:")
    print(display_df.to_string(index=False))

if __name__ == "__main__":
    main()
