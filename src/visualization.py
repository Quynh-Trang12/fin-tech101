# ==============================================================================
# Purpose:
# Implementation of stock market visualizations (Candlestick & Boxplot)
# ==============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mplfinance as mpf

from config import TICKER, START_DATE, END_DATE, DATA_DIR
from data_processing import load_raw_stock_data, standardise_stock_dataframe

# ==============================================================================
# CANDLESTICK VISUALIZATION PIPELINE
# ==============================================================================


def plot_candlestick(
    df: pd.DataFrame, n_days: int, output_path: Path, title: str | None = None
) -> None:
    """
    Plots an n-day aggregated stock candlestick chart and saves it as an image.

    Args:
        df: Dataframe containing stock data with a datetime index
            and columns: 'open', 'high', 'low', 'close', 'volume'.
        n_days: Number of trading days to aggregate into each candle (n >= 1).
        output_path: File path where the plot will be saved.
        title: Title of the candlestick chart.

    Raises:
        ValueError: If n_days is less than 1 or if dataframe is empty.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Validation and Index Check
    # --------------------------------------------------------------------------
    if n_days < 1:
        raise ValueError("Parameter n_days must be greater than or equal to 1.")
    if df.empty:
        raise ValueError("Input dataframe is empty. Cannot generate plot.")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing candlestick columns: {sorted(missing_columns)}")

    # Copy dataframe and standardise index to DatetimeIndex
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    df_copy.sort_index(inplace=True)

    # --------------------------------------------------------------------------
    # Phase 2: n-Day Resampling / Aggregation
    # --------------------------------------------------------------------------
    # We group by integer division of row indices to group exactly n consecutive
    # trading days together (ignoring calendar gaps like weekends and holidays).
    row_count = len(df_copy)
    group_idx = np.arange(row_count) // n_days

    # Aggregate values for each group:
    # - Open: Open price of the first day in the group.
    # - High: Maximum High price of any day in the group.
    # - Low: Minimum Low price of any day in the group.
    # - Close: Close price of the last day in the group.
    # - Volume: Sum of volumes across all days in the group.
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Include 'adjclose' if present in dataset
    if "adjclose" in df_copy.columns:
        agg_rules["adjclose"] = "last"

    grouped = df_copy.groupby(group_idx, sort=False).agg(agg_rules)

    # Assign the date of the last day in each group as the representative index date
    grouped.index = pd.DatetimeIndex(
        df_copy.index.to_series().groupby(group_idx, sort=False).last().to_numpy()
    )
    grouped.index.name = "Date"

    # Rename columns to capitalized format required by mplfinance
    grouped.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # --------------------------------------------------------------------------
    # Phase 3: Plot Generation and Rendering
    # --------------------------------------------------------------------------
    # Set default title if none provided
    if title is None:
        title = f"{df_copy['ticker'].iloc[0] if 'ticker' in df_copy.columns else 'Stock'} - {n_days}-Day Candlestick Chart"

    # Ensure target output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure style and plot using mplfinance.
    # We specify:
    # - type='candle': standard candlestick rendering
    # - style='charles': standard green/red styling for up/down periods
    # - volume=True: adds a volume subplot below the price panel
    # - savefig: saves directly to disk to prevent GUI blocking in headless environments
    mpf.plot(
        grouped,
        type="candle",
        style="charles",
        title=title,
        ylabel="Price ($)",
        ylabel_lower="Volume",
        volume=True,
        savefig=output_path,
        figsize=(12, 8),
    )
    plt.close("all")
    print(f"[Visualization] Saved candlestick chart to: {output_path.as_posix()}")


# ==============================================================================
# MOVING BOXPLOT VISUALIZATION PIPELINE
# ==============================================================================


def plot_moving_boxplot(
    df: pd.DataFrame, n_days: int, output_path: Path, title: str | None = None
) -> None:
    """
    Plots moving boxplots showing price distribution over n consecutive trading days.

    Args:
        df: Dataframe containing stock data with datetime index
                           and 'adjclose' (or 'close') column.
        n_days: Number of trading days per boxplot window (n >= 1).
        output_path: File path where the plot will be saved.
        title: Title of the boxplot chart.

    Raises:
        ValueError: If n_days is less than 1 or if dataframe is empty.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Validation and Index Check
    # --------------------------------------------------------------------------
    if n_days < 1:
        raise ValueError("Parameter n_days must be greater than or equal to 1.")
    if df.empty:
        raise ValueError("Input dataframe is empty. Cannot generate plot.")

    if "adjclose" not in df.columns and "close" not in df.columns:
        raise ValueError("Boxplot requires either 'adjclose' or 'close' column.")

    # Copy dataframe and standardize index
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    df_copy.sort_index(inplace=True)

    # --------------------------------------------------------------------------
    # Phase 2: Data Resampling and Interval Labelling
    # --------------------------------------------------------------------------
    # Group by integer division of row indices to group exactly n consecutive
    # trading days together (ignoring calendar gaps like weekends and holidays).
    row_count = len(df_copy)
    group_idx = np.arange(row_count) // n_days
    groups = df_copy.groupby(group_idx)

    data_to_plot = []
    labels = []

    # Determine the target column (prefer 'adjclose', fallback to 'close')
    target_col = "adjclose" if "adjclose" in df_copy.columns else "close"

    for _, group in groups:
        # Collect prices within this window
        data_to_plot.append(group[target_col].values)

        # Label representing the start and end date of the window
        start_str = group.index[0].strftime("%Y-%m-%d")
        end_str = group.index[-1].strftime("%Y-%m-%d")
        labels.append(f"{start_str}\nto\n{end_str}")

    # --------------------------------------------------------------------------
    # Phase 3: Figure Customization & Matplotlib Rendering
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate the boxplots.
    # - patch_artist=True: allows box coloring.
    # - medianprops: styles the median line inside the box.
    # - boxprops: styles the outer box dimensions.
    # - flierprops: styles the outlier markers.
    ax.boxplot(
        data_to_plot,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=2),
        boxprops=dict(facecolor="#3498db", color="#2c3e50", alpha=0.7),
        whiskerprops=dict(color="#2c3e50", linewidth=1.5),
        capprops=dict(color="#2c3e50", linewidth=1.5),
        flierprops=dict(
            marker="o", markerfacecolor="#e74c3c", markersize=6, linestyle="none"
        ),
    )

    # Set tick labels and rotate them to avoid overlap
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # If there are too many boxes, thin out the x-axis tick labels for clarity
    max_visible_ticks = 15
    if len(labels) > max_visible_ticks:
        step = len(labels) // max_visible_ticks
        for idx, text in enumerate(ax.get_xticklabels()):
            if idx % step != 0:
                text.set_visible(False)

    # Labels and Grid customization
    ax.set_xlabel("Trading Date Windows", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Stock Price ($)", fontsize=12, fontweight="bold", labelpad=10)

    # Set default title if none provided
    if title is None:
        ticker_name = (
            df_copy["ticker"].iloc[0] if "ticker" in df_copy.columns else "Stock"
        )
        title = f"{ticker_name} - Moving Boxplot of {n_days}-Day Trading Windows"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Enable background grid lines on the y-axis for easy reference
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Ensure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved boxplot chart to: {output_path.as_posix()}")


# ==============================================================================
# MAIN EXECUTION FOR TASK C.3 (Standalone Visualizations)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(f"STARTING TASK C.3: VISUALIZATION EXPORTS FOR {TICKER}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Phase 1: Centralized Data Retrieval
    # --------------------------------------------------------------------------
    print("[Visualization] Loading and processing data using centralized pipeline...")

    # Utilize the modular loader to ensure consistent caching, scaling, and handling
    raw_df = load_raw_stock_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_dir=DATA_DIR,
    )

    df = standardise_stock_dataframe(
        raw_df=raw_df,
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # --------------------------------------------------------------------------
    # Phase 2: Output Directory Initialization
    # --------------------------------------------------------------------------
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "results" / "c3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Phase 3: Visualization Generation
    # --------------------------------------------------------------------------

    # 3.1: 5-Day Candlestick Chart (Weekly Aggregation)
    candle_path = output_dir / f"{TICKER}_5day_candlestick.png"
    print("\n[Task C.3] Generating 5-day aggregated candlestick chart...")
    plot_candlestick(df, n_days=5, output_path=candle_path)

    # 3.2: 10-Day Moving Boxplot (Monthly Aggregation)
    boxplot_path = output_dir / f"{TICKER}_10day_boxplot.png"
    print("\n[Task C.3] Generating 10-day moving boxplot distribution...")
    plot_moving_boxplot(df, n_days=10, output_path=boxplot_path)

    # --------------------------------------------------------------------------
    # Phase 4: Completion Validation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        "TASK C.3 COMPLETED: Visualizations successfully saved to "
        f"{output_dir.as_posix()}"
    )
    print("=" * 80)
