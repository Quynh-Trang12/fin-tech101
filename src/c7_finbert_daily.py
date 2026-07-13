# ==============================================================================
# Purpose:
# Aggregate article-level FinBERT sentiment predictions into daily calendar-day
# features, and align them to stock market trading days for Task C.7.
#
# Reuses the timezone conversion and searchsorted date alignment strategy
# established in Task C.7 for V2Tone.
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    C7_DATA_DIR,
    END_DATE,
    GDELT_FINBERT_ARTICLE_PATH,
    MARKET_TIMEZONE,
    START_DATE,
    TICKER,
)
from data_processing import load_raw_stock_data, standardise_stock_dataframe


def aggregate_daily_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert article-level FinBERT predictions into daily calendar-day features.
    
    Args:
        df: Article-level FinBERT predictions DataFrame.
        
    Returns:
        DataFrame containing one row per Sydney calendar day.
    """
    data = df.copy()
    
    # 1. Convert published_at UTC timezone into Australia/Sydney local time
    # Reuses identical logic from c7_news_features.py
    data["published_date"] = (
        pd.to_datetime(
            data["published_at"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(MARKET_TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    
    # Drop rows with invalid dates (if any)
    data = data.dropna(subset=["published_date"]).copy()
    
    # 2. Group by published_date and compute daily sentiment statistics
    daily = (
        data.groupby("published_date")
        .agg(
            article_count=("DocumentIdentifier", "count"),
            finbert_positive_probability_mean=("finbert_positive_probability", "mean"),
            finbert_neutral_probability_mean=("finbert_neutral_probability", "mean"),
            finbert_negative_probability_mean=("finbert_negative_probability", "mean"),
            finbert_confidence_mean=("finbert_confidence", "mean"),
            finbert_confidence_max=("finbert_confidence", "max"),
            positive_article_count=("finbert_predicted_label", lambda s: int((s == "positive").sum())),
            neutral_article_count=("finbert_predicted_label", lambda s: int((s == "neutral").sum())),
            negative_article_count=("finbert_predicted_label", lambda s: int((s == "negative").sum())),
        )
        .sort_index()
        .reset_index()
    )
    
    # 3. Calculate predicted-label shares
    # Denominator is the number of valid FinBERT predictions (which equals article_count)
    daily["positive_article_share"] = daily["positive_article_count"] / daily["article_count"]
    daily["neutral_article_share"] = daily["neutral_article_count"] / daily["article_count"]
    daily["negative_article_share"] = daily["negative_article_count"] / daily["article_count"]
    
    return daily


def align_finbert_to_trading_days(
    daily_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Align calendar-day daily FinBERT features to stock trading days.
    
    Args:
        daily_df: Calendar-day daily FinBERT features DataFrame.
        stock_df: Standardised stock DataFrame indexed by trading date.
        
    Returns:
        tuple: (aligned_df, stats)
    """
    daily = daily_df.copy()
    daily["published_date"] = pd.to_datetime(daily["published_date"])
    
    # 1. Extract unique sorted trading dates
    trading_dates = sorted(pd.to_datetime(stock_df.index).unique())
    trading_dates_arr = np.array(trading_dates)
    
    # 2. Align calendar dates to nearest trading day (searchsorted logic)
    idx = np.searchsorted(trading_dates_arr, daily["published_date"])
    
    # Detect unmatched dates (news dates after the last stock trading date)
    unmatched_mask = idx >= len(trading_dates_arr)
    unmatched_df = daily[unmatched_mask]
    
    news_valid = daily[~unmatched_mask].copy()
    valid_idx = idx[~unmatched_mask]
    
    # Map each calendar news record to its correct trading day
    news_valid["trading_date"] = trading_dates_arr[valid_idx]
    
    # Identify weekend news that got mapped for alignment statistics
    news_valid["day_of_week"] = news_valid["published_date"].dt.dayofweek
    weekend_mask = news_valid["day_of_week"].isin([5, 6]) # 5 = Saturday, 6 = Sunday
    weekend_mappings = news_valid[weekend_mask]
    
    # 3. Group and aggregate multiple calendar days mapping onto the same trading day
    def custom_group_agg(group: pd.DataFrame) -> pd.Series:
        res = {}
        
        # Sum counts
        sum_cols = [
            "article_count",
            "positive_article_count",
            "neutral_article_count",
            "negative_article_count",
        ]
        for col in sum_cols:
            res[col] = int(group[col].sum())
            
        # Weighted averages using article_count as weights
        weights = group["article_count"]
        total_weight = weights.sum()
        
        mean_cols = [
            "finbert_positive_probability_mean",
            "finbert_neutral_probability_mean",
            "finbert_negative_probability_mean",
            "finbert_confidence_mean",
        ]
        for col in mean_cols:
            if total_weight > 0:
                res[col] = (group[col] * weights).sum() / total_weight
            else:
                res[col] = np.nan
                
        # Maximum confidence
        if total_weight > 0:
            res["finbert_confidence_max"] = group["finbert_confidence_max"].max()
        else:
            res["finbert_confidence_max"] = np.nan
            
        # Recalculate article shares based on summed counts (identical to weighted average)
        tot_articles = res["article_count"]
        if tot_articles > 0:
            res["positive_article_share"] = res["positive_article_count"] / tot_articles
            res["neutral_article_share"] = res["neutral_article_count"] / tot_articles
            res["negative_article_share"] = res["negative_article_count"] / tot_articles
        else:
            res["positive_article_share"] = np.nan
            res["neutral_article_share"] = np.nan
            res["negative_article_share"] = np.nan
            
        return pd.Series(res)
        
    # Apply custom aggregator by trading day
    aligned_df = (
        news_valid.groupby("trading_date")
        .apply(custom_group_agg, include_groups=False)
        .reset_index()
    )
    
    # Create date alias column for compatibility
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
        "source_valid_articles": int(news_valid["article_count"].sum()),
        "aligned_total_articles": int(aligned_df["article_count"].sum()),
    }
    
    return aligned_df, stats


def validate_daily_and_aligned_datasets(
    article_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
) -> None:
    """
    Perform validation checks on daily aggregated and aligned datasets.
    """
    print("[Validation] Running consistency and sanity checks...")
    
    # 1. Sum of article_count equals total raw valid headlines
    expected_articles = len(article_df)
    daily_articles = daily_df["article_count"].sum()
    if daily_articles != expected_articles:
        raise ValueError(
            f"Validation failed: Daily calendar article count sum ({daily_articles}) "
            f"does not match article-level source count ({expected_articles})."
        )
    print("  - Daily calendar total article count sum matches source exactly.")
    
    # 2. Aligned dataset represents exactly the same article population
    # (Checking if any articles were lost, allowing for unmatched articles at the very end)
    aligned_articles = aligned_df["article_count"].sum()
    unmatched_count = expected_articles - aligned_articles
    if unmatched_count < 0:
        raise ValueError(f"Validation failed: Aligned article count ({aligned_articles}) exceeds source.")
    print(f"  - Aligned article count: {aligned_articles} (Unmatched: {unmatched_count})")
    
    # 3. No trading day is duplicated in the aligned output
    if not aligned_df["trading_date"].is_unique:
        raise ValueError("Validation failed: Aligned dataset contains duplicate trading dates.")
    print("  - Aligned trading dates are unique.")
    
    # 4. No aligned trading date falls on a weekend
    aligned_df["day_of_week"] = aligned_df["trading_date"].dt.dayofweek
    weekend_rows = aligned_df[aligned_df["day_of_week"].isin([5, 6])]
    if not weekend_rows.empty:
        invalid_dates = weekend_rows["trading_date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Validation failed: Aligned dates contain weekend days: {invalid_dates}")
    print("  - Aligned trading dates contain no weekends.")
    
    # 5. Probability means remain within [0, 1]
    prob_cols = [
        "finbert_positive_probability_mean",
        "finbert_neutral_probability_mean",
        "finbert_negative_probability_mean",
        "finbert_confidence_mean",
        "finbert_confidence_max",
    ]
    for df_name, df_obj in [("Daily", daily_df), ("Aligned", aligned_df)]:
        for col in prob_cols:
            vals = df_obj[col].dropna()
            if not ((vals >= 0.0) & (vals <= 1.0)).all():
                raise ValueError(f"Validation failed: {df_name} probability values in column '{col}' are out of [0, 1] bounds.")
    print("  - All probability means and confidence scores are within [0, 1] bounds.")
    
    # 6. Shares remain within [0, 1]
    share_cols = [
        "positive_article_share",
        "neutral_article_share",
        "negative_article_share",
    ]
    for df_name, df_obj in [("Daily", daily_df), ("Aligned", aligned_df)]:
        for col in share_cols:
            vals = df_obj[col].dropna()
            if not ((vals >= 0.0) & (vals <= 1.0)).all():
                raise ValueError(f"Validation failed: {df_name} share values in column '{col}' are out of [0, 1] bounds.")
    print("  - All article share values are within [0, 1] bounds.")
    
    # 7. Share sum ≈ 1.0 (with floating point tolerance) for every row
    for df_name, df_obj in [("Daily", daily_df), ("Aligned", aligned_df)]:
        sums = df_obj["positive_article_share"] + df_obj["neutral_article_share"] + df_obj["negative_article_share"]
        sums = sums.dropna()
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise ValueError(f"Validation failed: {df_name} shares do not sum to approximately 1.0.")
    print("  - Daily and Aligned article share sums equal 1.0 (with floating-point tolerance).")
    print("[Validation] All dataset validation checks passed successfully!")


def main():
    print("=" * 80)
    print("STARTING DAILY FINBERT AGGREGATION & TRADING DAY ALIGNMENT")
    print("=" * 80)
    
    # Define output files
    daily_output_path = C7_DATA_DIR / "gdelt_daily_finbert.parquet"
    aligned_output_path = C7_DATA_DIR / "gdelt_finbert_aligned.parquet"
    
    # 1. Load article-level outputs
    print(f"Loading article-level FinBERT sentiment predictions from: {GDELT_FINBERT_ARTICLE_PATH.as_posix()}...")
    if not GDELT_FINBERT_ARTICLE_PATH.exists():
        raise FileNotFoundError(f"Article-level predictions not found at: {GDELT_FINBERT_ARTICLE_PATH}")
        
    article_df = pd.read_parquet(GDELT_FINBERT_ARTICLE_PATH)
    
    # 2. Run Daily calendar-day aggregation
    print("Aggregating articles into Sydney calendar-day features...")
    daily_df = aggregate_daily_finbert(article_df)
    
    # 3. Load stock market trading dates
    print(f"Loading and preprocessing stock data for {TICKER} ({START_DATE} to {END_DATE})...")
    raw_stock = load_raw_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    stock_df = standardise_stock_dataframe(raw_stock, ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    
    # 4. Run Trading-day alignment
    print("Aligning daily news features onto stock trading days...")
    aligned_df, stats = align_finbert_to_trading_days(daily_df, stock_df)
    
    # 5. Run validation checks
    validate_daily_and_aligned_datasets(article_df, daily_df, aligned_df)
    
    # 6. Save datasets to Parquet
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(daily_output_path, index=False)
    aligned_df.to_parquet(aligned_output_path, index=False)
    
    print(f"\n[Save] Saved daily calendar-day features to: {daily_output_path.as_posix()} (Rows: {len(daily_df)})")
    print(f"[Save] Saved aligned trading-day features to: {aligned_output_path.as_posix()} (Rows: {len(aligned_df)})")
    
    # Print alignment statistics
    print("\n" + "=" * 60)
    print("FINBERT DAILY ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"Total Stock Trading Days:     {stats['total_trading_days']}")
    print(f"Trading Days with news:       {stats['trading_days_with_news']}")
    print(f"Unmatched Calendar Dates:     {stats['unmatched_count']}")
    print(f"Weekend news mappings count:  {stats['weekend_mapped_count']}")
    print(f"First Aligned Date:           {stats['first_aligned_date']}")
    print(f"Last Aligned Date:            {stats['last_aligned_date']}")
    print(f"Total valid articles:         {stats['source_valid_articles']}")
    print(f"Total aligned articles:       {stats['aligned_total_articles']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
