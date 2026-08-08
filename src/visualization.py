# ==============================================================================
# Purpose:
# Implementation of stock market visualisations for Task C.3:
# candlestick charts and windowed boxplots.
# ==============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mplfinance as mpf

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    SPLIT_DATE,
    DATA_DIR,
    C3_CANDLE_DAYS,
    C3_BOXPLOT_WINDOW,
    C3_BOXPLOT_STEP,
)
from data_processing import load_raw_stock_data, standardise_stock_dataframe


# ==============================================================================
# CANDLESTICK VISUALISATION PIPELINE
# ==============================================================================


def plot_candlestick(
    df: pd.DataFrame,
    n_days: int,
    output_path: Path,
    title: str | None = None,
    drop_incomplete: bool = True,
) -> None:
    """
    Plot an n-trading-day aggregated candlestick chart and save it as an image.

    Each candlestick represents exactly ``n_days`` consecutive trading rows:

    - Open: first opening price in the group
    - High: maximum high price in the group
    - Low: minimum low price in the group
    - Close: final closing price in the group
    - Volume: sum of trading volume in the group

    Args:
        df:
            DataFrame containing stock data with a datetime index and the
            columns ``open``, ``high``, ``low``, ``close``, and ``volume``.
        n_days:
            Number of consecutive trading days represented by each candle.
            Must be greater than or equal to 1.
        output_path:
            File path where the chart will be saved.
        title:
            Optional chart title.
        drop_incomplete:
            If True, remove the final partial group when the number of rows is
            not divisible by ``n_days``. This ensures that every candle
            represents exactly ``n_days`` trading days.

    Raises:
        ValueError:
            If the input is invalid, required columns are missing, or there
            are not enough rows to form one complete candle.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Validation and index preparation
    # --------------------------------------------------------------------------
    if n_days < 1:
        raise ValueError("Parameter n_days must be greater than or equal to 1.")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot generate plot.")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing candlestick columns: {sorted(missing_columns)}")

    df_copy = df.copy()

    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)

    df_copy.sort_index(inplace=True)

    if drop_incomplete:
        complete_rows = (len(df_copy) // n_days) * n_days
        df_copy = df_copy.iloc[:complete_rows]

    if len(df_copy) < n_days:
        raise ValueError(
            f"Not enough rows to form one complete {n_days}-day candlestick."
        )

    # --------------------------------------------------------------------------
    # Phase 2: Exact n-trading-day aggregation
    # --------------------------------------------------------------------------
    # Integer row grouping is used instead of calendar resampling so weekends
    # and public holidays do not create incomplete calendar-based groups.
    group_idx = np.arange(len(df_copy)) // n_days

    aggregation_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if "adjclose" in df_copy.columns:
        aggregation_rules["adjclose"] = "last"

    grouped = df_copy.groupby(group_idx, sort=False).agg(aggregation_rules)

    # The final trading date in each group is used as the candle date.
    grouped.index = pd.DatetimeIndex(
        df_copy.index.to_series().groupby(group_idx, sort=False).last().to_numpy()
    )
    grouped.index.name = "Date"

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
    # Phase 3: Plot generation
    # --------------------------------------------------------------------------
    if title is None:
        ticker_name = (
            df_copy["ticker"].iloc[0] if "ticker" in df_copy.columns else "Stock"
        )
        title = f"{ticker_name} - {n_days}-Trading-Day Candlestick Chart"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # style="charles" colours rising candles green and falling ones red.
    # savefig writes straight to file; bbox_inches="tight" trims whitespace.
    mpf.plot(
        grouped,
        type="candle",
        style="charles",
        title=title,
        ylabel="Price ($)",
        ylabel_lower="Volume",
        volume=True,
        savefig=dict(
            fname=str(output_path),
            dpi=150,
            bbox_inches="tight",
        ),
        figsize=(12, 8),
    )

    plt.close("all")
    print(f"[Visualisation] Saved candlestick chart to: {output_path.as_posix()}")


# ==============================================================================
# WINDOWED BOXPLOT VISUALISATION PIPELINE
# ==============================================================================


def plot_windowed_boxplot(
    df: pd.DataFrame,
    window_days: int,
    output_path: Path,
    step_days: int | None = None,
    title: str | None = None,
) -> None:
    """
    Plot stock-price distributions across consecutive trading-day windows.

    ``window_days`` controls the number of observations inside each boxplot.
    ``step_days`` controls how far the window advances:

    - ``step_days = 1`` creates a true moving window.
    - ``step_days = window_days`` creates non-overlapping windows.
    - Intermediate values create partially overlapping windows.

    Args:
        df:
            DataFrame containing a datetime index and either ``adjclose`` or
            ``close``.
        window_days:
            Number of consecutive trading days represented by each boxplot.
        output_path:
            File path where the chart will be saved.
        step_days:
            Number of rows by which the window advances. If omitted,
            non-overlapping windows are used.
        title:
            Optional chart title.

    Raises:
        ValueError:
            If the input is invalid or does not contain enough rows.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Validation and index preparation
    # --------------------------------------------------------------------------
    if window_days < 1:
        raise ValueError("Parameter window_days must be greater than or equal to 1.")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot generate plot.")

    if "adjclose" not in df.columns and "close" not in df.columns:
        raise ValueError("Boxplot requires either an 'adjclose' or 'close' column.")

    if step_days is None:
        step_days = window_days

    if step_days < 1:
        raise ValueError("Parameter step_days must be greater than or equal to 1.")

    df_copy = df.copy()

    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)

    df_copy.sort_index(inplace=True)

    if len(df_copy) < window_days:
        raise ValueError(
            f"At least {window_days} rows are required to create a window."
        )

    # --------------------------------------------------------------------------
    # Phase 2: Window construction and labelling
    # --------------------------------------------------------------------------
    target_column = "adjclose" if "adjclose" in df_copy.columns else "close"

    data_to_plot: list[np.ndarray] = []
    labels: list[str] = []

    for start_idx in range(
        0,
        len(df_copy) - window_days + 1,
        step_days,
    ):
        end_idx = start_idx + window_days
        window = df_copy.iloc[start_idx:end_idx]

        data_to_plot.append(window[target_column].to_numpy())

        start_date = window.index[0].strftime("%Y-%m-%d")
        end_date = window.index[-1].strftime("%Y-%m-%d")
        labels.append(f"{start_date}\nto\n{end_date}")

    # --------------------------------------------------------------------------
    # Phase 3: Figure rendering
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    # patch_artist=True fills the box bodies; without it boxprops facecolor
    # is ignored. The remaining *props arguments style one element each.
    ax.boxplot(
        data_to_plot,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=2),
        boxprops=dict(
            facecolor="#3498db",
            color="#2c3e50",
            alpha=0.7,
        ),
        whiskerprops=dict(color="#2c3e50", linewidth=1.5),
        capprops=dict(color="#2c3e50", linewidth=1.5),
        flierprops=dict(
            marker="o",
            markerfacecolor="#e74c3c",
            markersize=6,
            linestyle="none",
        ),
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    # Thin labels when many windows are displayed.
    max_visible_ticks = 15
    if len(labels) > max_visible_ticks:
        step = max(1, int(np.ceil(len(labels) / max_visible_ticks)))

        for index, text in enumerate(ax.get_xticklabels()):
            if index % step != 0:
                text.set_visible(False)

    ax.set_xlabel(
        "Trading-Date Windows",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Adjusted Closing Price ($)"
        if target_column == "adjclose"
        else "Closing Price ($)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    if title is None:
        ticker_name = (
            df_copy["ticker"].iloc[0] if "ticker" in df_copy.columns else "Stock"
        )

        if step_days == 1:
            window_description = f"{window_days}-Day Moving Windows"
        elif step_days == window_days:
            window_description = f"Non-Overlapping {window_days}-Day Windows"
        else:
            window_description = f"{window_days}-Day Windows, Step {step_days}"

        title = f"{ticker_name} - Price Distribution Across {window_description}"

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[Visualisation] Saved windowed boxplot to: {output_path.as_posix()}")


# ==============================================================================
# MAIN EXECUTION FOR TASK C.3
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(f"STARTING TASK C.3: VISUALISATION EXPORTS FOR {TICKER}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Phase 1: Centralised data retrieval
    # --------------------------------------------------------------------------
    print("[Visualisation] Loading data using the centralised data pipeline...")

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
    # Phase 2: Output directory
    # --------------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "results" / "c3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Phase 3: Visualisation generation
    # --------------------------------------------------------------------------

    # 3.1 Daily candlesticks over the test period provide detailed short-term
    # behaviour without overcrowding the full 2020-2024 dataset.
    test_period_df = df.loc[SPLIT_DATE:END_DATE]

    daily_candlestick_path = output_dir / f"{TICKER}_1day_candlestick_test_period.png"

    print("\n[Task C.3] Generating daily candlestick chart for the test period...")

    plot_candlestick(
        test_period_df,
        n_days=1,
        output_path=daily_candlestick_path,
        title=f"{TICKER} - Daily Candlestick Chart (Test Period)",
    )

    # 3.2 Aggregating several trading days per candle reduces daily noise while
    # preserving the broad trend across the complete dataset.
    aggregated_candlestick_path = (
        output_dir / f"{TICKER}_{C3_CANDLE_DAYS}day_candlestick.png"
    )

    print(
        f"\n[Task C.3] Generating {C3_CANDLE_DAYS}-trading-day "
        "aggregated candlestick chart..."
    )

    plot_candlestick(
        df,
        n_days=C3_CANDLE_DAYS,
        output_path=aggregated_candlestick_path,
        title=f"{TICKER} - {C3_CANDLE_DAYS}-Trading-Day Candlestick Chart",
    )

    # 3.3 Non-overlapping windows remain readable while showing how the price
    # distribution changes over time.
    boxplot_path = output_dir / f"{TICKER}_{C3_BOXPLOT_WINDOW}day_windowed_boxplot.png"

    print(
        f"\n[Task C.3] Generating {C3_BOXPLOT_WINDOW}-trading-day windowed boxplot..."
    )

    plot_windowed_boxplot(
        df,
        window_days=C3_BOXPLOT_WINDOW,
        step_days=C3_BOXPLOT_STEP,
        output_path=boxplot_path,
        title=(
            f"{TICKER} - Adjusted Close Distribution "
            f"by {C3_BOXPLOT_WINDOW}-Trading-Day Windows"
        ),
    )

    # --------------------------------------------------------------------------
    # Phase 4: Completion message
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TASK C.3 COMPLETED: Visualisations saved to {output_dir.as_posix()}")
    print("=" * 80)
