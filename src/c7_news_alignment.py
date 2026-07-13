# ==============================================================================
# Purpose:
# Align daily news sentiment features to stock market trading days for Task C.7.
#
# Delivers weekend and holiday news to the next available trading day.
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    C7_DATA_DIR,
    GDELT_DAILY_V2TONE_PATH,
)
from data_processing import load_raw_stock_data, standardise_stock_dataframe


def align_news_to_trading_days(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Align calendar-day news features to stock market trading days.
    
    Forward weekend and holiday news to the next available trading day.
    
    Args:
        news_df: DataFrame containing daily GDELT features with a `published_date` column.
        stock_df: Cleaned stock DataFrame indexed by trading dates.
        
    Returns:
        tuple: (aligned_df, stats_dict)
            aligned_df: DataFrame containing daily news features aligned and aggregated
                        by trading days.
            stats_dict: Dictionary containing alignment statistics.
    """
    # 1. Prepare news dates
    news = news_df.copy()
    news["published_date"] = pd.to_datetime(news["published_date"]).dt.tz_localize(None)
    
    # 2. Get unique, sorted trading dates
    trading_dates = sorted(pd.to_datetime(stock_df.index).unique())
    trading_dates_arr = np.array(trading_dates)
    
    # 3. Align each calendar news date to the first trading date >= published_date
    # Using np.searchsorted to find the smallest trading date >= news date
    idx = np.searchsorted(trading_dates_arr, news["published_date"])
    
    # Detect unmatched dates (news dates after the last stock trading date)
    unmatched_mask = idx >= len(trading_dates_arr)
    unmatched_df = news[unmatched_mask]
    
    # Filter to valid mappings
    news_valid = news[~unmatched_mask].copy()
    valid_idx = idx[~unmatched_mask]
    
    # Map to trading date
    news_valid["trading_date"] = trading_dates_arr[valid_idx]
    
    # Identify weekend news that got mapped
    news_valid["day_of_week"] = news_valid["published_date"].dt.dayofweek
    weekend_mask = news_valid["day_of_week"].isin([5, 6]) # 5 = Saturday, 6 = Sunday
    weekend_mappings = news_valid[weekend_mask]
    
    # 4. Perform custom aggregation by trading day
    # Define a custom aggregation function to handle sum, min, max, and weighted means
    def custom_group_agg(group: pd.DataFrame) -> pd.Series:
        res = {}
        
        # Sum count-like columns
        sum_cols = [
            "article_count",
            "valid_v2tone_count",
            "word_count_total",
            "positive_article_count",
            "negative_article_count",
            "neutral_article_count",
        ]
        for col in sum_cols:
            if col in group.columns:
                res[col] = int(group[col].sum())
                
        # Min/max bounds
        if "tone_min" in group.columns:
            res["tone_min"] = group["tone_min"].min()
        if "tone_max" in group.columns:
            res["tone_max"] = group["tone_max"].max()
            
        # Weighted averages for article-level means (weighted by valid_v2tone_count)
        weight_col = "valid_v2tone_count"
        weights = group[weight_col]
        total_weight = weights.sum()
        
        mean_cols = [
            "tone_mean",
            "positive_score_mean",
            "negative_score_mean",
            "polarity_mean",
            "activity_reference_density_mean",
            "self_group_reference_density_mean",
            "word_count_mean",
        ]
        
        for col in mean_cols:
            if col in group.columns:
                if total_weight > 0:
                    res[col] = (group[col] * weights).sum() / total_weight
                else:
                    res[col] = np.nan
                    
        # Recalculate article share percentages based on summed counts
        v_count = res.get("valid_v2tone_count", 0)
        if v_count > 0:
            res["positive_article_share"] = res.get("positive_article_count", 0) / v_count
            res["negative_article_share"] = res.get("negative_article_count", 0) / v_count
            res["neutral_article_share"] = res.get("neutral_article_count", 0) / v_count
        else:
            res["positive_article_share"] = np.nan
            res["negative_article_share"] = np.nan
            res["neutral_article_share"] = np.nan
            
        return pd.Series(res)

    # Perform group aggregation using custom_group_agg
    # We pass include_groups=False to avoid future warnings
    aligned_df = (
        news_valid.groupby("trading_date")
        .apply(custom_group_agg, include_groups=False)
        .reset_index()
    )
    
    # 5. Validation Check: Total aligned article_count must equal total valid source article_count
    source_total = int(news_valid["article_count"].sum())
    aligned_total = int(aligned_df["article_count"].sum())
    
    print(f"\n[Validation] Source valid article count: {source_total}")
    print(f"[Validation] Aligned article count:      {aligned_total}")
    
    if source_total != aligned_total:
        raise ValueError(
            f"Validation failed: total aligned article count ({aligned_total}) does not match source ({source_total})."
        )
    
    # Add alias column for compatibility
    aligned_df["date"] = aligned_df["trading_date"]
    
    # Compile statistics
    stats = {
        "total_trading_days": len(trading_dates_arr),
        "unmatched_count": len(unmatched_df),
        "unmatched_dates": unmatched_df["published_date"].dt.strftime("%Y-%m-%d").tolist(),
        "weekend_mapped_count": len(weekend_mappings),
        "weekend_mappings_sample": [
            f"{row['published_date'].strftime('%Y-%m-%d')} ({row['published_date'].day_name()}) -> {row['trading_date'].strftime('%Y-%m-%d')}"
            for _, row in weekend_mappings.head(5).iterrows()
        ],
        "first_aligned_date": aligned_df["trading_date"].min().strftime("%Y-%m-%d") if not aligned_df.empty else None,
        "last_aligned_date": aligned_df["trading_date"].max().strftime("%Y-%m-%d") if not aligned_df.empty else None,
        "trading_days_with_news": len(aligned_df),
        "source_total_articles": source_total,
        "aligned_total_articles": aligned_total,
        "input_total_articles": int(news_df["article_count"].sum()),
    }
    
    return aligned_df, stats


def main():
    print("=" * 80)
    print("STARTING GDELT NEWS ALIGNMENT PROCESS (TASK C.7)")
    print("=" * 80)
    
    # Define output path
    output_path = C7_DATA_DIR / "gdelt_v2tone_aligned.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load GDELT daily news features
    print(f"Loading GDELT daily news features from: {GDELT_DAILY_V2TONE_PATH.as_posix()}...")
    if not GDELT_DAILY_V2TONE_PATH.exists():
        raise FileNotFoundError(f"GDELT daily news parquet file not found at: {GDELT_DAILY_V2TONE_PATH}")
        
    news_df = pd.read_parquet(GDELT_DAILY_V2TONE_PATH)
    
    # 2. Load raw stock data and clean it
    print(f"Loading and preprocessing stock data for {TICKER} ({START_DATE} to {END_DATE})...")
    raw_stock = load_raw_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    stock_df = standardise_stock_dataframe(raw_stock, ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    
    # 3. Align news to trading days
    print("Aligning GDELT calendar dates to market trading days...")
    aligned_df, stats = align_news_to_trading_days(news_df, stock_df)
    
    # 4. Save aligned features
    print(f"Saving aligned daily news features to: {output_path.as_posix()}...")
    aligned_df.to_parquet(output_path, index=False)
    
    # 5. Print statistics
    print("\n" + "-" * 50)
    print("ALIGNMENT STATISTICS SUMMARY")
    print("-" * 50)
    print(f"Total Stock Trading Days:         {stats['total_trading_days']}")
    print(f"Trading Days with Aligned News:   {stats['trading_days_with_news']} ({(stats['trading_days_with_news'] / stats['total_trading_days']) * 100:.2f}%)")
    print(f"First Aligned Trading Date:       {stats['first_aligned_date']}")
    print(f"Last Aligned Trading Date:        {stats['last_aligned_date']}")
    print(f"Unmatched News Dates (Future):    {stats['unmatched_count']}")
    print(f"Weekend News Days Forwarded:      {stats['weekend_mapped_count']}")
    
    if stats['unmatched_count'] > 0:
        print(f"Unmatched Dates Sample (First 5): {stats['unmatched_dates'][:5]}")
        
    if stats['weekend_mapped_count'] > 0:
        print("\nWeekend Mapping Samples:")
        for sample in stats['weekend_mappings_sample']:
            print(f"  - {sample}")
            
    print("=" * 80)
    print("ALIGNMENT PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
