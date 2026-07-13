# ==============================================================================
# Purpose:
# Parse article-level GDELT V2Tone values and aggregate them into daily news
# sentiment features for Task C.7.
#
# This version handles only V2Tone. Themes, GCAM, FinBERT, stock alignment,
# and classification are implemented separately in later stages.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GDELT_DAILY_V2TONE_PATH,
    GDELT_RAW_CACHE_PATH,
    MARKET_TIMEZONE,
)


# ==============================================================================
# V2TONE FIELD DEFINITIONS
# ==============================================================================

V2TONE_COLUMNS = [
    "tone",
    "positive_score",
    "negative_score",
    "polarity",
    "activity_reference_density",
    "self_group_reference_density",
    "word_count",
]


# ==============================================================================
# V2TONE PARSING
# ==============================================================================


def parse_v2tone_value(value: object) -> dict[str, float]:
    """
    Parse one GDELT V2Tone value into named numerical fields.

    GDELT stores V2Tone as seven comma-separated values:

        Tone,
        Positive Score,
        Negative Score,
        Polarity,
        Activity Reference Density,
        Self/Group Reference Density,
        Word Count

    Example:

        -1.25,2.40,3.65,6.05,18.20,0.30,542

    Args:
        value:
            Raw V2Tone value from one GDELT article.

    Returns:
        Dictionary containing the seven parsed V2Tone fields.

        If the value is missing, malformed, or does not contain exactly seven
        components, all returned fields contain NaN.
    """
    empty_result = {column: np.nan for column in V2TONE_COLUMNS}

    if value is None or pd.isna(value):
        return empty_result

    parts = str(value).strip().split(",")

    if len(parts) != len(V2TONE_COLUMNS):
        return empty_result

    try:
        parsed_values = [float(part.strip()) for part in parts]
    except (TypeError, ValueError):
        return empty_result

    return dict(zip(V2TONE_COLUMNS, parsed_values))


def parse_v2tone_column(news_data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the article-level V2Tone column into separate numerical columns.

    Args:
        news_data:
            Article-level GDELT DataFrame containing V2Tone and published_date.

    Returns:
        Copy of the input DataFrame with seven additional V2Tone columns.

    Raises:
        ValueError:
            If required columns are missing.
    """
    required_columns = {"published_at", "V2Tone"}
    missing_columns = required_columns.difference(news_data.columns)

    if missing_columns:
        raise ValueError(
            "News dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = news_data.copy()

    parsed_records = data["V2Tone"].apply(parse_v2tone_value)
    parsed_frame = pd.DataFrame(parsed_records.tolist(), index=data.index)

    for column in V2TONE_COLUMNS:
        data[column] = pd.to_numeric(parsed_frame[column], errors="coerce")

    return data


# ==============================================================================
# DAILY AGGREGATION
# ==============================================================================


def aggregate_daily_v2tone(news_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert article-level V2Tone values into one daily feature vector.

    Each calendar day may contain many articles. This function aggregates those
    articles so the resulting dataset can later be aligned with daily stock
    market records.

    Args:
        news_data:
            Article-level GDELT dataset.

    Returns:
        DataFrame containing one row per calendar day.
    """
    parsed_data = parse_v2tone_column(news_data)

    if "published_at" not in parsed_data.columns:
        raise ValueError(
            "News dataset must contain 'published_at' for timezone-safe aggregation."
        )

    parsed_data["published_date"] = (
        pd.to_datetime(
            parsed_data["published_at"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(MARKET_TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    parsed_data = parsed_data.dropna(subset=["published_date"]).copy()

    # A row is considered valid for sentiment aggregation when its main tone
    # score was parsed successfully.
    parsed_data["valid_v2tone"] = parsed_data["tone"].notna()

    daily = (
        parsed_data.groupby("published_date", as_index=False)
        .agg(
            article_count=("DocumentIdentifier", "count"),
            valid_v2tone_count=("valid_v2tone", "sum"),

            tone_mean=("tone", "mean"),
            tone_median=("tone", "median"),
            tone_std=("tone", "std"),
            tone_min=("tone", "min"),
            tone_max=("tone", "max"),

            positive_score_mean=("positive_score", "mean"),
            negative_score_mean=("negative_score", "mean"),
            polarity_mean=("polarity", "mean"),

            activity_reference_density_mean=(
                "activity_reference_density",
                "mean",
            ),
            self_group_reference_density_mean=(
                "self_group_reference_density",
                "mean",
            ),

            word_count_mean=("word_count", "mean"),
            word_count_total=("word_count", "sum"),
        )
        .sort_values("published_date")
        .reset_index(drop=True)
    )

    # Shares of articles whose overall tone was positive, negative, or neutral.
    tone_direction = (
        parsed_data.assign(
            positive_article=(parsed_data["tone"] > 0).astype(int),
            negative_article=(parsed_data["tone"] < 0).astype(int),
            neutral_article=(parsed_data["tone"] == 0).astype(int),
        )
        .groupby("published_date", as_index=False)
        .agg(
            positive_article_count=("positive_article", "sum"),
            negative_article_count=("negative_article", "sum"),
            neutral_article_count=("neutral_article", "sum"),
        )
    )

    daily = daily.merge(
        tone_direction,
        on="published_date",
        how="left",
        validate="one_to_one",
    )

    valid_denominator = daily["valid_v2tone_count"].replace(0, np.nan)

    daily["positive_article_share"] = (
        daily["positive_article_count"] / valid_denominator
    )
    daily["negative_article_share"] = (
        daily["negative_article_count"] / valid_denominator
    )
    daily["neutral_article_share"] = (
        daily["neutral_article_count"] / valid_denominator
    )

    # Standard deviation is undefined when only one valid article exists.
    # Filling it with zero means no within-day sentiment variation was observed.
    daily["tone_std"] = daily["tone_std"].fillna(0.0)

    return daily


# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================


def build_daily_v2tone_features(
    raw_cache_path: Path = GDELT_RAW_CACHE_PATH,
    output_path: Path = GDELT_DAILY_V2TONE_PATH,
) -> pd.DataFrame:
    """
    Load cached GDELT articles, build daily V2Tone features, and save them.

    Args:
        raw_cache_path:
            Article-level GDELT Parquet cache.
        output_path:
            Destination for daily V2Tone features.

    Returns:
        Daily V2Tone feature DataFrame.
    """
    if not raw_cache_path.exists():
        raise FileNotFoundError(
            f"Raw GDELT cache does not exist: {raw_cache_path.as_posix()}"
        )

    print(
        f"[C7 Features] Loading raw news cache: "
        f"{raw_cache_path.as_posix()}"
    )

    try:
        news_data = pd.read_parquet(raw_cache_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not read raw GDELT cache: {error}"
        ) from error

    print(f"[C7 Features] Loaded {len(news_data):,} articles.")
    print("[C7 Features] Parsing and aggregating V2Tone...")

    daily_features = aggregate_daily_v2tone(news_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        daily_features.to_parquet(output_path, index=False)
    except Exception as error:
        raise RuntimeError(
            f"Could not save daily V2Tone features: {error}"
        ) from error

    print(
        f"[C7 Features] Created {len(daily_features):,} daily records."
    )
    print(
        f"[C7 Features] Saved daily feature cache: "
        f"{output_path.as_posix()}"
    )

    return daily_features


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build daily GDELT V2Tone features for Task C.7."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=GDELT_RAW_CACHE_PATH,
        help="Path to the raw article-level GDELT Parquet cache.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=GDELT_DAILY_V2TONE_PATH,
        help="Path used to save daily V2Tone features.",
    )

    return parser.parse_args()


# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================


def main() -> None:
    """Build and display the daily V2Tone feature dataset."""
    arguments = parse_arguments()

    daily_features = build_daily_v2tone_features(
        raw_cache_path=arguments.input,
        output_path=arguments.output,
    )

    print("\n[C7 Features] Dataset summary")
    print(f"Rows: {len(daily_features):,}")
    print(f"Columns: {len(daily_features.columns)}")

    if not daily_features.empty:
        print(
            "Date range: "
            f"{daily_features['published_date'].min().date()} to "
            f"{daily_features['published_date'].max().date()}"
        )

        print(
            "Total articles represented: "
            f"{int(daily_features['article_count'].sum()):,}"
        )

        print(
            "Valid V2Tone records: "
            f"{int(daily_features['valid_v2tone_count'].sum()):,}"
        )

    print("\n[C7 Features] First five daily records:")
    print(daily_features.head().to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise RuntimeError(
            f"C7 news feature generation failed: {error}"
        ) from error