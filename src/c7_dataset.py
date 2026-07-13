# ==============================================================================
# Purpose:
# Construct the classification dataset for Task C.7 by merging daily stock prices
# with both aligned V2Tone and FinBERT news sentiment features.
#
# Generates the target label indicating next-trading-day price rise or fall.
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    C7_DATA_DIR,
    END_DATE,
    START_DATE,
    TICKER,
)
from data_processing import load_raw_stock_data, standardise_stock_dataframe


def validate_dataframe(df: pd.DataFrame, name: str, is_stock: bool = False):
    """
    Perform audit validation on a dataset before merging.
    
    Checks that the date column is present, unique, sorted, naive, and has no weekends.
    """
    if "date" not in df.columns:
        raise ValueError(f"{name} dataset is missing the 'date' column.")
        
    dates = df["date"]
    
    # Check date column dtype is datetime64[ns] and timezone naive
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError(f"{name} 'date' column must be a datetime dtype.")
    if dates.dt.tz is not None:
        raise ValueError(f"{name} 'date' column must be timezone-naive.")
        
    # Check uniqueness
    if not dates.is_unique:
        raise ValueError(f"{name} contains duplicate dates.")
        
    # Check sorting
    if not dates.is_monotonic_increasing:
        raise ValueError(f"{name} dates are not sorted chronologically.")
        
    # For stock dataset, verify no weekend trading sessions exist
    if is_stock:
        weekends = dates[dates.dt.dayofweek.isin([5, 6])]
        if not weekends.empty:
            raise ValueError(
                f"Stock dataset contains invalid weekend dates: {weekends.dt.strftime('%Y-%m-%d').tolist()}"
            )


def build_c7_dataset():
    """
    Build the Task C.7 dataset combining stock history, V2Tone news, and FinBERT news.
    """
    aligned_v2tone_path = C7_DATA_DIR / "gdelt_v2tone_aligned.parquet"
    aligned_finbert_path = C7_DATA_DIR / "gdelt_finbert_aligned.parquet"
    output_path = C7_DATA_DIR / "c7_dataset.parquet"
    
    # --------------------------------------------------------------------------
    # 1. Load Datasets
    # --------------------------------------------------------------------------
    print(f"Loading stock data for {TICKER} ({START_DATE} to {END_DATE})...")
    raw_stock = load_raw_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    stock_df = standardise_stock_dataframe(raw_stock, ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    
    print(f"Loading aligned daily GDELT V2Tone news features from: {aligned_v2tone_path.as_posix()}...")
    if not aligned_v2tone_path.exists():
        raise FileNotFoundError(f"Aligned V2Tone news parquet not found at: {aligned_v2tone_path}")
    v2tone_df = pd.read_parquet(aligned_v2tone_path)
    
    print(f"Loading aligned daily GDELT FinBERT news features from: {aligned_finbert_path.as_posix()}...")
    if not aligned_finbert_path.exists():
        raise FileNotFoundError(f"Aligned FinBERT news parquet not found at: {aligned_finbert_path}")
    finbert_df = pd.read_parquet(aligned_finbert_path)
    
    # --------------------------------------------------------------------------
    # 2. Validate Datasets Before Merge
    # --------------------------------------------------------------------------
    print("Validating datasets before merge...")
    validate_dataframe(stock_df, name="Stock", is_stock=True)
    validate_dataframe(v2tone_df, name="V2Tone News", is_stock=False)
    validate_dataframe(finbert_df, name="FinBERT News", is_stock=False)
    print("  - Pre-merge validation checks passed.")
    
    # --------------------------------------------------------------------------
    # 3. Create Classification Target Safely (on stock dataframe before merge)
    # --------------------------------------------------------------------------
    print("Creating binary target (next-trading-day rise or fall)...")
    # Shift adjclose by -1 to get the next trading day's price
    stock_df["next_adjclose"] = stock_df["adjclose"].shift(-1)
    
    # Drop the last row because its next-trading-day price is unknown
    stock_df_with_target = stock_df.dropna(subset=["next_adjclose"]).copy()
    
    # Binary classification target: 1 if price rises, 0 otherwise
    # Cast target directly to int8 to match requirements
    stock_df_with_target["target"] = (
        stock_df_with_target["next_adjclose"] > stock_df_with_target["adjclose"]
    ).astype(np.int8)
    
    # Remove the helper next_adjclose column to prevent lookahead leakage
    stock_df_with_target.drop(columns=["next_adjclose"], inplace=True)
    
    # --------------------------------------------------------------------------
    # 4. Rename FinBERT Columns and Perform Left Joins
    # --------------------------------------------------------------------------
    # Clean up redundant columns in FinBERT to avoid duplicates
    finbert_df_clean = finbert_df.drop(columns=["trading_date", "day_of_week"], errors="ignore").copy()
    
    # Rename FinBERT features with a prefix to prevent collisions with V2Tone fields
    finbert_rename_cols = {}
    for col in finbert_df_clean.columns:
        if col != "date":
            if col.startswith("finbert_"):
                finbert_rename_cols[col] = col
            else:
                finbert_rename_cols[col] = f"finbert_{col}"
    finbert_df_clean = finbert_df_clean.rename(columns=finbert_rename_cols)
    
    print("Merging stock data with aligned GDELT V2Tone news sentiment...")
    # Left join V2Tone news
    merged_df = pd.merge(stock_df_with_target, v2tone_df, on="date", how="left")
    merged_df.drop(columns=["trading_date"], errors="ignore", inplace=True)
    
    print("Merging stock data with aligned GDELT FinBERT news sentiment...")
    # Left join FinBERT news
    merged_df = pd.merge(merged_df, finbert_df_clean, on="date", how="left")
    
    # --------------------------------------------------------------------------
    # 5. Handle Missing News and Add Indicators
    # --------------------------------------------------------------------------
    # Zero-fill V2Tone features
    v2tone_feature_cols = [col for col in v2tone_df.columns if col not in ["date", "trading_date"]]
    merged_df[v2tone_feature_cols] = merged_df[v2tone_feature_cols].fillna(0.0)
    
    # Zero-fill FinBERT features
    finbert_feature_cols = [col for col in finbert_df_clean.columns if col != "date"]
    merged_df[finbert_feature_cols] = merged_df[finbert_feature_cols].fillna(0.0)
    
    # Add has_news indicators (1 when news is present, 0 otherwise)
    merged_df["has_news"] = (merged_df["article_count"] > 0).astype(np.int8)
    merged_df["has_finbert_news"] = (merged_df["finbert_article_count"] > 0).astype(np.int8)
    
    # --------------------------------------------------------------------------
    # 6. Final Sorting and Index Reset
    # --------------------------------------------------------------------------
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    
    # --------------------------------------------------------------------------
    # 7. Save Result
    # --------------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    
    # --------------------------------------------------------------------------
    # 8. Audit Summary Statistics
    # --------------------------------------------------------------------------
    total_rows = len(merged_df)
    first_date = merged_df["date"].min().strftime("%Y-%m-%d")
    last_date = merged_df["date"].max().strftime("%Y-%m-%d")
    num_cols = len(merged_df.columns)
    dup_dates = merged_df["date"].duplicated().sum()
    missing_values = merged_df.isna().sum().sum()
    
    # Class counts & percentages
    class_counts = merged_df["target"].value_counts()
    class_0_cnt = class_counts.get(0, 0)
    class_1_cnt = class_counts.get(1, 0)
    class_0_pct = (class_0_cnt / total_rows) * 100
    class_1_pct = (class_1_cnt / total_rows) * 100
    
    # News coverage count
    v2_news_count = (merged_df["has_news"] == 1).sum()
    fb_news_count = (merged_df["has_finbert_news"] == 1).sum()
    
    # First/Last 5 dates
    first_5_dates = merged_df["date"].head(5).dt.strftime("%Y-%m-%d").tolist()
    last_5_dates = merged_df["date"].tail(5).dt.strftime("%Y-%m-%d").tolist()
    
    print("\n" + "=" * 80)
    print("TASK C.7 DATASET AUDIT SUMMARY (ENHANCED)")
    print("=" * 80)
    print(f"Output Path:                      {output_path.as_posix()}")
    print(f"Total Rows (Trading Days):        {total_rows}")
    print(f"Date Range:                       {first_date} to {last_date}")
    print(f"Number of Columns:                {num_cols}")
    print(f"Duplicate Date Count:             {dup_dates}")
    print(f"Total Missing Values (NaN):       {missing_values}")
    print(f"Rows WITH V2Tone (has_news=1):    {v2_news_count} ({v2_news_count/total_rows*100:.2f}%)")
    print(f"Rows WITH FinBERT (has_fb_news=1):{fb_news_count} ({fb_news_count/total_rows*100:.2f}%)")
    print(f"Class 0 (Down/Flat/No Rise):      {class_0_cnt} ({class_0_pct:.2f}%)")
    print(f"Class 1 (Rise):                   {class_1_cnt} ({class_1_pct:.2f}%)")
    print(f"First 5 Dates:                    {first_5_dates}")
    print(f"Last 5 Dates:                     {last_5_dates}")
    print("=" * 80)
    print("TASK C.7 DATASET CONSTRUCTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    build_c7_dataset()
