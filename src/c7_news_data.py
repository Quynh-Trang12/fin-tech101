# ==============================================================================
# Purpose:
# Download, cache, and load historical GDELT news records for Task C.7.
#
# This module is responsible only for raw news data acquisition.
# Sentiment extraction, daily aggregation, stock-data merging, and modelling
# are handled by separate modules.
# ==============================================================================

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

from config import (
    GDELT_COMPANY_QUERY,
    GDELT_END_DATE,
    GDELT_METADATA_PATH,
    GDELT_RAW_CACHE_PATH,
    GDELT_START_DATE,
    GDELT_TABLE,
)


# ==============================================================================
# QUERY CONSTRUCTION
# ==============================================================================


def build_gdelt_query(table_name: str = GDELT_TABLE) -> str:
    """
    Build the parameterised BigQuery SQL query used to collect GDELT records.

    The table name is inserted directly because BigQuery does not allow table
    identifiers to be supplied through scalar query parameters. All user-facing
    values, such as dates and the organisation search phrase, are supplied
    separately as query parameters.

    Args:
        table_name:
            Fully qualified BigQuery table name.

    Returns:
        SQL query string.
    """
    return f"""
        SELECT
            DATE,
            SourceCommonName,
            DocumentIdentifier,
            V2Organizations,
            V2Themes,
            V2Tone,
            GCAM
        FROM `{table_name}`
        WHERE
            _PARTITIONTIME >= @start_timestamp
            AND _PARTITIONTIME < @end_timestamp_exclusive
            AND LOWER(COALESCE(V2Organizations, ''))
                LIKE CONCAT('%', LOWER(@company_query), '%')
            AND DocumentIdentifier IS NOT NULL
        ORDER BY DATE
    """


# ==============================================================================
# VALIDATION AND CLEANING
# ==============================================================================


def validate_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and convert the requested inclusive date range.

    Args:
        start_date:
            First calendar date to collect, in YYYY-MM-DD format.
        end_date:
            Last calendar date to collect, in YYYY-MM-DD format.

    Returns:
        Tuple containing the converted start and end timestamps.

    Raises:
        ValueError:
            If either date is invalid or the start date is after the end date.
    """
    try:
        start_timestamp = pd.to_datetime(start_date, format="%Y-%m-%d")
        end_timestamp = pd.to_datetime(end_date, format="%Y-%m-%d")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "start_date and end_date must use YYYY-MM-DD format."
        ) from error

    if start_timestamp > end_timestamp:
        raise ValueError("start_date must not be later than end_date.")

    return start_timestamp, end_timestamp


def parse_gdelt_datetime(values: pd.Series) -> pd.Series:
    """
    Convert GDELT DATE values into pandas timestamps.

    GDELT normally stores DATE using the format YYYYMMDDHHMMSS, for example:

        20240115143000

    Invalid values are converted to pandas NaT rather than causing the complete
    collection process to fail.

    Args:
        values:
            Series containing raw GDELT DATE values.

    Returns:
        Parsed datetime series.
    """
    normalized_values = (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    return pd.to_datetime(
        normalized_values,
        format="%Y%m%d%H%M%S",
        errors="coerce",
        utc=True,
    )


def clean_gdelt_news(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Clean and deduplicate raw GDELT article records.

    Deduplication is based on DocumentIdentifier, which normally contains the
    original article URL. When multiple GDELT records refer to the same URL,
    only the first chronologically ordered record is retained.

    Args:
        raw_data:
            DataFrame returned by BigQuery.

    Returns:
        A tuple containing:
            - cleaned article-level DataFrame;
            - cleaning statistics.
    """
    required_columns = {
        "DATE",
        "SourceCommonName",
        "DocumentIdentifier",
        "V2Organizations",
        "V2Themes",
        "V2Tone",
        "GCAM",
    }

    missing_columns = required_columns.difference(raw_data.columns)

    if missing_columns:
        raise ValueError(
            "BigQuery result is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = raw_data.copy()
    rows_downloaded = len(data)

    # Remove records without a usable article identifier.
    data["DocumentIdentifier"] = (
        data["DocumentIdentifier"].astype("string").str.strip()
    )
    data = data[
        data["DocumentIdentifier"].notna()
        & data["DocumentIdentifier"].ne("")
    ].copy()

    # Parse the GDELT timestamp while retaining the original DATE field.
    data["published_at"] = parse_gdelt_datetime(data["DATE"])

    invalid_dates_removed = int(data["published_at"].isna().sum())
    data = data.dropna(subset=["published_at"]).copy()

    # Sort before deduplication so the earliest occurrence is retained.
    data = data.sort_values(
        ["published_at", "DocumentIdentifier"],
        kind="stable",
    )

    rows_before_deduplication = len(data)

    data = data.drop_duplicates(
        subset=["DocumentIdentifier"],
        keep="first",
    )

    duplicates_removed = rows_before_deduplication - len(data)

    # Daily aggregation will later use this normalized calendar date.
    data["published_date"] = data["published_at"].dt.tz_convert(None).dt.normalize()

    # Use predictable nullable string dtypes for textual GDELT fields.
    text_columns = [
        "SourceCommonName",
        "DocumentIdentifier",
        "V2Organizations",
        "V2Themes",
        "V2Tone",
        "GCAM",
    ]

    for column in text_columns:
        data[column] = data[column].astype("string")

    data = data.reset_index(drop=True)

    statistics = {
        "rows_downloaded": int(rows_downloaded),
        "invalid_dates_removed": int(invalid_dates_removed),
        "duplicates_removed": int(duplicates_removed),
        "rows_saved": int(len(data)),
    }

    return data, statistics


# ==============================================================================
# BIGQUERY DOWNLOAD
# ==============================================================================


def fetch_gdelt_news(
    start_date: str = GDELT_START_DATE,
    end_date: str = GDELT_END_DATE,
    company_query: str = GDELT_COMPANY_QUERY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Download relevant historical GDELT news records from Google BigQuery.

    The end date supplied by the project is treated as inclusive. BigQuery uses
    an exclusive upper partition boundary, so one calendar day is added when
    constructing the query parameter.

    Args:
        start_date:
            Inclusive collection start date.
        end_date:
            Inclusive collection end date.
        company_query:
            Organisation phrase searched inside V2Organizations.
        project_id:
            Optional Google Cloud project used to run the query. If omitted,
            Google Application Default Credentials determine the project.

    Returns:
        Tuple containing:
            - cleaned GDELT article DataFrame;
            - metadata describing the collection.
    """
    start_timestamp, end_timestamp = validate_date_range(start_date, end_date)

    if not company_query.strip():
        raise ValueError("company_query must not be empty.")

    end_timestamp_exclusive = end_timestamp + timedelta(days=1)

    print("[C7 News] Connecting to Google BigQuery...")

    try:
        client = bigquery.Client(project="project-a849e666-8b34-4dd7-baa")
    except Exception as error:
        raise RuntimeError(
            "Could not create the BigQuery client. Confirm that Google Cloud "
            "Application Default Credentials and a project are configured."
        ) from error

    query = build_gdelt_query()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_timestamp",
                "TIMESTAMP",
                start_timestamp.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter(
                "end_timestamp_exclusive",
                "TIMESTAMP",
                end_timestamp_exclusive.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter(
                "company_query",
                "STRING",
                company_query,
            ),
        ]
    )

    print(
        "[C7 News] Querying GDELT records for "
        f"'{company_query}' from {start_date} to {end_date}..."
    )

    try:
        query_job = client.query(query, job_config=job_config)
        raw_data = query_job.result().to_dataframe(
            create_bqstorage_client=False
        )
    except Exception as error:
        raise RuntimeError(f"GDELT BigQuery query failed: {error}") from error

    cleaned_data, cleaning_statistics = clean_gdelt_news(raw_data)

    metadata: dict[str, Any] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "google_cloud_project": client.project,
        "bigquery_table": GDELT_TABLE,
        "company_query": company_query,
        "start_date": start_date,
        "end_date": end_date,
        "query_job_id": query_job.job_id,
        **cleaning_statistics,
    }

    print(
        f"[C7 News] Downloaded {metadata['rows_downloaded']:,} records."
    )
    print(
        f"[C7 News] Removed {metadata['duplicates_removed']:,} duplicate articles."
    )
    print(
        f"[C7 News] Retained {metadata['rows_saved']:,} unique articles."
    )

    return cleaned_data, metadata


# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================


def save_gdelt_cache(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    cache_path: Path = GDELT_RAW_CACHE_PATH,
    metadata_path: Path = GDELT_METADATA_PATH,
) -> None:
    """
    Save cleaned GDELT records and collection metadata locally.

    Args:
        data:
            Cleaned article-level GDELT DataFrame.
        metadata:
            Dictionary describing the query and cleaning results.
        cache_path:
            Destination Parquet file.
        metadata_path:
            Destination JSON metadata file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data.to_parquet(cache_path, index=False)
    except Exception as error:
        raise RuntimeError(
            f"Could not write Parquet cache to '{cache_path}': {error}"
        ) from error

    try:
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    except Exception as error:
        raise RuntimeError(
            f"Could not write metadata to '{metadata_path}': {error}"
        ) from error

    print(f"[C7 News] Saved article cache: {cache_path.as_posix()}")
    print(f"[C7 News] Saved metadata: {metadata_path.as_posix()}")


def load_cached_gdelt_news(
    cache_path: Path = GDELT_RAW_CACHE_PATH,
) -> pd.DataFrame:
    """
    Load the locally cached GDELT article dataset.

    Args:
        cache_path:
            Location of the Parquet cache.

    Returns:
        Cached GDELT DataFrame.

    Raises:
        FileNotFoundError:
            If the cache does not exist.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"GDELT cache does not exist: {cache_path.as_posix()}"
        )

    try:
        data = pd.read_parquet(cache_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not read GDELT cache '{cache_path}': {error}"
        ) from error

    print(
        f"[C7 News] Loaded {len(data):,} cached articles from "
        f"{cache_path.as_posix()}"
    )

    return data


def get_gdelt_news(
    refresh: bool = False,
    start_date: str = GDELT_START_DATE,
    end_date: str = GDELT_END_DATE,
    company_query: str = GDELT_COMPANY_QUERY,
    cache_path: Path = GDELT_RAW_CACHE_PATH,
    metadata_path: Path = GDELT_METADATA_PATH,
) -> pd.DataFrame:
    """
    Load cached GDELT data or download a fresh copy when required.

    Behaviour:
        - If refresh is False and the cache exists, load the cache.
        - If refresh is True, download and overwrite the cache.
        - If no cache exists, download automatically.

    Args:
        refresh:
            Force a new BigQuery download.
        start_date:
            Inclusive collection start date.
        end_date:
            Inclusive collection end date.
        company_query:
            Organisation search phrase.
        project_id:
            Optional Google Cloud project ID.
        cache_path:
            Local Parquet cache path.
        metadata_path:
            Local JSON metadata path.

    Returns:
        Article-level GDELT DataFrame.
    """
    if cache_path.exists() and not refresh:
        return load_cached_gdelt_news(cache_path)

    if refresh:
        print("[C7 News] Refresh requested. Existing cache will be replaced.")
    else:
        print("[C7 News] No local cache found. Starting BigQuery download.")

    data, metadata = fetch_gdelt_news(
        start_date=start_date,
        end_date=end_date,
        company_query=company_query,
    )

    save_gdelt_cache(
        data=data,
        metadata=metadata,
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    return data


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone execution.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download or load historical GDELT news records for Task C.7."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a new BigQuery download and overwrite the local cache.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=GDELT_START_DATE,
        help="Inclusive collection start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=GDELT_END_DATE,
        help="Inclusive collection end date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--company-query",
        type=str,
        default=GDELT_COMPANY_QUERY,
        help="Organisation phrase searched inside GDELT V2Organizations.",
    )

    return parser.parse_args()


# ==============================================================================
# STANDALONE EXECUTION ENTRY POINT
# ==============================================================================


def main() -> None:
    """Run the GDELT cache workflow from the command line."""
    arguments = parse_arguments()

    data = get_gdelt_news(
        refresh=arguments.refresh,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        company_query=arguments.company_query,
    )

    print("\n[C7 News] Dataset summary")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")

    if not data.empty:
        print(
            "Publication range: "
            f"{data['published_at'].min()} to {data['published_at'].max()}"
        )
        print(f"Unique sources: {data['SourceCommonName'].nunique():,}")

    print("\n[C7 News] First five records:")
    print(
        data[
            [
                "published_at",
                "SourceCommonName",
                "DocumentIdentifier",
            ]
        ].head()
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise RuntimeError(f"C7 news data execution failed: {error}") from error