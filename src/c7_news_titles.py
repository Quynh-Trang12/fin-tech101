# ==============================================================================
# Purpose:
# Download GDELT article titles from gdg_partitioned, cache them locally,
# and enrich the existing Task C.7 raw GKG article dataset.
#
# This module does not perform FinBERT inference, daily aggregation,
# stock-date alignment, or modelling.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

from config import (
    GDELT_COMPANY_QUERY,
    GDELT_END_DATE,
    GDELT_RAW_CACHE_PATH,
    GDELT_START_DATE,
    GDELT_TABLE,
)
from c7_news_data import load_cached_gdelt_news, validate_date_range


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BIGQUERY_PROJECT_ID = "project-a849e666-8b34-4dd7-baa"

# GKG contains the existing article metadata.
# GDG contains page titles and page-change information.
GDELT_GDG_TABLE = "gdelt-bq.gdeltv2.gdg_partitioned"

C7_DATA_DIR = GDELT_RAW_CACHE_PATH.parent

GDELT_TITLES_CACHE_PATH = C7_DATA_DIR / "gdelt_cba_titles.parquet"
GDELT_ENRICHED_CACHE_PATH = C7_DATA_DIR / "gdelt_cba_enriched.parquet"
GDELT_TITLES_METADATA_PATH = C7_DATA_DIR / "gdelt_cba_titles_metadata.json"

# These titles usually describe error pages, navigation pages, or access barriers
# rather than genuine news headlines.
INVALID_TITLE_PATTERN = re.compile(
    r"\b("
    r"404|"
    r"not found|"
    r"access denied|"
    r"cookie policy|"
    r"privacy policy|"
    r"home page|"
    r"page unavailable|"
    r"server error|"
    r"internal server error"
    r")\b",
    flags=re.IGNORECASE,
)


# ==============================================================================
# QUERY CONSTRUCTION
# ==============================================================================


def build_title_query(
    gkg_table: str = GDELT_TABLE,
    gdg_table: str = GDELT_GDG_TABLE,
) -> str:
    """
    Build the BigQuery query used to retrieve one deterministic English title
    for each Commonwealth Bank article URL.

    Selection rules:
        1. Identify relevant GKG article URLs using the same date range and
           organisation filter as c7_news_data.py.
        2. Join GKG DocumentIdentifier to GDG page_url.
        3. Keep English records where page_lang is either "en" or "english".
        4. Prefer page_title; use title_new only when page_title is missing.
        5. Select the earliest usable GDG title record for each URL.

    Args:
        gkg_table:
            Fully qualified GKG table name.
        gdg_table:
            Fully qualified GDG table name.

    Returns:
        Parameterised SQL query string.
    """
    return f"""
        WITH cba_articles AS (
            SELECT DISTINCT
                DocumentIdentifier
            FROM `{gkg_table}`
            WHERE
                _PARTITIONTIME >= @start_timestamp
                AND _PARTITIONTIME < @end_timestamp_exclusive
                AND LOWER(COALESCE(V2Organizations, ''))
                    LIKE CONCAT('%', LOWER(@company_query), '%')
                AND DocumentIdentifier IS NOT NULL
        ),

        title_candidates AS (
            SELECT
                gdg.page_url,
                NULLIF(TRIM(gdg.page_title), '') AS page_title,
                NULLIF(TRIM(gdg.title_new), '') AS title_new,
                gdg.page_lang,
                gdg.status,
                gdg.fetchdate_orig,
                gdg.fetchdate_check,

                ROW_NUMBER() OVER (
                    PARTITION BY gdg.page_url
                    ORDER BY
                        COALESCE(
                            gdg.fetchdate_orig,
                            gdg.fetchdate_check,
                            TIMESTAMP('9999-12-31')
                        ) ASC,
                        gdg.fetchdate_check ASC
                ) AS title_rank

            FROM `{gdg_table}` AS gdg

            INNER JOIN cba_articles AS cba
                ON cba.DocumentIdentifier = gdg.page_url

            WHERE
                LOWER(TRIM(COALESCE(gdg.page_lang, '')))
                    IN ('en', 'english')

                AND COALESCE(
                    NULLIF(TRIM(gdg.page_title), ''),
                    NULLIF(TRIM(gdg.title_new), '')
                ) IS NOT NULL
        )

        SELECT
            page_url AS DocumentIdentifier,
            page_title,
            title_new,
            COALESCE(page_title, title_new) AS headline,
            page_lang,
            status,
            fetchdate_orig,
            fetchdate_check
        FROM title_candidates
        WHERE title_rank = 1
        ORDER BY DocumentIdentifier
    """


# ==============================================================================
# TITLE CLEANING AND VALIDATION
# ==============================================================================


def clean_title_text(value: object) -> str | pd.NA:
    """
    Normalize one title while preserving its textual content.

    Args:
        value:
            Raw title value.

    Returns:
        Cleaned title string or pandas.NA when unusable.
    """
    if pd.isna(value):
        return pd.NA

    title = str(value)

    # Normalize repeated whitespace and surrounding quotation marks.
    title = re.sub(r"\s+", " ", title).strip()
    title = title.strip("\"' ")

    if len(title) < 5:
        return pd.NA

    if INVALID_TITLE_PATTERN.search(title):
        return pd.NA

    return title


def clean_gdelt_titles(
    raw_titles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Clean and validate downloaded GDG title records.

    Args:
        raw_titles:
            DataFrame returned by the BigQuery title query.

    Returns:
        Tuple containing:
            - cleaned title DataFrame;
            - title-cleaning statistics.
    """
    required_columns = {
        "DocumentIdentifier",
        "page_title",
        "title_new",
        "headline",
        "page_lang",
        "status",
        "fetchdate_orig",
        "fetchdate_check",
    }

    missing_columns = required_columns.difference(raw_titles.columns)

    if missing_columns:
        raise ValueError(
            "Title query result is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = raw_titles.copy()
    rows_downloaded = len(data)

    data["DocumentIdentifier"] = (
        data["DocumentIdentifier"]
        .astype("string")
        .str.strip()
    )

    data = data[
        data["DocumentIdentifier"].notna()
        & data["DocumentIdentifier"].ne("")
    ].copy()

    for column in ["page_title", "title_new", "headline"]:
        data[column] = data[column].map(clean_title_text).astype("string")

    # Reconstruct the preferred headline after cleaning.
    # page_title remains the preferred source; title_new is only a fallback.
    data["headline"] = data["page_title"].fillna(data["title_new"])

    invalid_titles_removed = int(data["headline"].isna().sum())
    data = data.dropna(subset=["headline"]).copy()

    # The SQL query should already return one row per URL. This defensive
    # deduplication preserves a deterministic one-to-one merge contract.
    rows_before_deduplication = len(data)

    data = data.sort_values(
        [
            "DocumentIdentifier",
            "fetchdate_orig",
            "fetchdate_check",
        ],
        kind="stable",
        na_position="last",
    )

    data = data.drop_duplicates(
        subset=["DocumentIdentifier"],
        keep="first",
    )

    duplicates_removed = rows_before_deduplication - len(data)

    for column in [
        "DocumentIdentifier",
        "page_title",
        "title_new",
        "headline",
        "page_lang",
        "status",
    ]:
        data[column] = data[column].astype("string")

    data = data.reset_index(drop=True)

    if not data["DocumentIdentifier"].is_unique:
        raise ValueError(
            "Cleaned title dataset still contains duplicate article URLs."
        )

    statistics = {
        "rows_downloaded": int(rows_downloaded),
        "invalid_titles_removed": int(invalid_titles_removed),
        "duplicates_removed": int(duplicates_removed),
        "rows_saved": int(len(data)),
    }

    return data, statistics


# ==============================================================================
# BIGQUERY DOWNLOAD
# ==============================================================================


def fetch_gdelt_titles(
    start_date: str = GDELT_START_DATE,
    end_date: str = GDELT_END_DATE,
    company_query: str = GDELT_COMPANY_QUERY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Download one usable English title per matching GKG article URL.

    Args:
        start_date:
            Inclusive collection start date.
        end_date:
            Inclusive collection end date.
        company_query:
            Organisation phrase searched in GKG V2Organizations.

    Returns:
        Tuple containing:
            - cleaned title DataFrame;
            - collection metadata.
    """
    start_timestamp, end_timestamp = validate_date_range(
        start_date,
        end_date,
    )

    if not company_query.strip():
        raise ValueError("company_query must not be empty.")

    end_timestamp_exclusive = end_timestamp + timedelta(days=1)

    print("[C7 Titles] Connecting to Google BigQuery...")

    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    except Exception as error:
        raise RuntimeError(
            "Could not create the BigQuery client. Confirm that Google Cloud "
            "Application Default Credentials are configured."
        ) from error

    query = build_title_query()

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
        "[C7 Titles] Querying English GDELT titles for "
        f"'{company_query}' from {start_date} to {end_date}..."
    )

    try:
        query_job = client.query(
            query,
            job_config=job_config,
        )

        raw_titles = query_job.result().to_dataframe(
            create_bqstorage_client=False
        )

    except Exception as error:
        raise RuntimeError(
            f"GDELT title query failed: {error}"
        ) from error

    cleaned_titles, cleaning_statistics = clean_gdelt_titles(
        raw_titles
    )

    metadata: dict[str, Any] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "google_cloud_project": client.project,
        "gkg_table": GDELT_TABLE,
        "gdg_table": GDELT_GDG_TABLE,
        "company_query": company_query,
        "start_date": start_date,
        "end_date": end_date,
        "query_job_id": query_job.job_id,
        "title_selection_rule": (
            "Earliest English GDG record per page_url; page_title preferred; "
            "title_new used only as fallback."
        ),
        **cleaning_statistics,
    }

    print(
        f"[C7 Titles] Downloaded {metadata['rows_downloaded']:,} title rows."
    )
    print(
        "[C7 Titles] Removed "
        f"{metadata['invalid_titles_removed']:,} invalid titles."
    )
    print(
        "[C7 Titles] Retained "
        f"{metadata['rows_saved']:,} unique usable English headlines."
    )

    return cleaned_titles, metadata


# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================


def save_title_cache(
    titles: pd.DataFrame,
    metadata: dict[str, Any],
    cache_path: Path = GDELT_TITLES_CACHE_PATH,
    metadata_path: Path = GDELT_TITLES_METADATA_PATH,
) -> None:
    """
    Save the title cache and its metadata.

    Args:
        titles:
            Cleaned one-row-per-URL title DataFrame.
        metadata:
            Collection and cleaning metadata.
        cache_path:
            Destination title Parquet path.
        metadata_path:
            Destination metadata JSON path.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        titles.to_parquet(cache_path, index=False)
    except Exception as error:
        raise RuntimeError(
            f"Could not save title cache '{cache_path}': {error}"
        ) from error

    try:
        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as error:
        raise RuntimeError(
            f"Could not save title metadata '{metadata_path}': {error}"
        ) from error

    print(
        f"[C7 Titles] Saved title cache: {cache_path.as_posix()}"
    )
    print(
        f"[C7 Titles] Saved title metadata: {metadata_path.as_posix()}"
    )


def load_cached_titles(
    cache_path: Path = GDELT_TITLES_CACHE_PATH,
) -> pd.DataFrame:
    """
    Load the cached GDG title dataset.

    Args:
        cache_path:
            Title cache path.

    Returns:
        Cached title DataFrame.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Title cache does not exist: {cache_path.as_posix()}"
        )

    try:
        titles = pd.read_parquet(cache_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not read title cache '{cache_path}': {error}"
        ) from error

    if not titles["DocumentIdentifier"].is_unique:
        raise ValueError(
            "Cached title dataset contains duplicate DocumentIdentifier values."
        )

    print(
        f"[C7 Titles] Loaded {len(titles):,} cached English headlines from "
        f"{cache_path.as_posix()}"
    )

    return titles


# ==============================================================================
# DATASET ENRICHMENT
# ==============================================================================


def enrich_gdelt_news_with_titles(
    raw_news: pd.DataFrame,
    titles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Left-join article titles onto the existing raw GKG article cache.

    The raw GKG dataset remains authoritative for publication timestamps,
    V2Tone, Themes, GCAM, source information, and article membership.

    Args:
        raw_news:
            Existing cleaned GKG article cache.
        titles:
            One-row-per-URL English title cache.

    Returns:
        Tuple containing:
            - enriched article DataFrame;
            - merge statistics.
    """
    required_raw_columns = {
        "DocumentIdentifier",
        "published_at",
    }

    missing_raw_columns = required_raw_columns.difference(
        raw_news.columns
    )

    if missing_raw_columns:
        raise ValueError(
            "Raw GKG cache is missing required columns: "
            + ", ".join(sorted(missing_raw_columns))
        )

    if not raw_news["DocumentIdentifier"].is_unique:
        raise ValueError(
            "Raw GKG article cache must contain unique DocumentIdentifier values."
        )

    if not titles["DocumentIdentifier"].is_unique:
        raise ValueError(
            "Title cache must contain unique DocumentIdentifier values."
        )

    title_columns = [
        "DocumentIdentifier",
        "page_title",
        "title_new",
        "headline",
        "page_lang",
        "status",
        "fetchdate_orig",
        "fetchdate_check",
    ]

    enriched = raw_news.merge(
        titles[title_columns],
        on="DocumentIdentifier",
        how="left",
        validate="one_to_one",
    )

    if len(enriched) != len(raw_news):
        raise ValueError(
            "Enrichment changed the number of raw GKG article rows."
        )

    enriched["has_english_headline"] = (
        enriched["headline"].notna()
    ).astype("int8")

    rows_with_headline = int(
        enriched["has_english_headline"].sum()
    )

    total_rows = len(enriched)

    coverage_percent = (
        100.0 * rows_with_headline / total_rows
        if total_rows
        else 0.0
    )

    enriched = enriched.sort_values(
        ["published_at", "DocumentIdentifier"],
        kind="stable",
    ).reset_index(drop=True)

    merge_statistics: dict[str, Any] = {
        "raw_article_rows": int(total_rows),
        "rows_with_english_headline": int(rows_with_headline),
        "rows_without_english_headline": int(
            total_rows - rows_with_headline
        ),
        "english_headline_coverage_percent": float(
            coverage_percent
        ),
    }

    return enriched, merge_statistics


def save_enriched_cache(
    enriched_data: pd.DataFrame,
    cache_path: Path = GDELT_ENRICHED_CACHE_PATH,
) -> None:
    """
    Save the enriched article-level GKG + GDG dataset.

    Args:
        enriched_data:
            GKG article records with optional English headlines.
        cache_path:
            Destination Parquet path.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        enriched_data.to_parquet(
            cache_path,
            index=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not save enriched cache '{cache_path}': {error}"
        ) from error

    print(
        f"[C7 Titles] Saved enriched article cache: "
        f"{cache_path.as_posix()}"
    )


# ==============================================================================
# PIPELINE
# ==============================================================================


def get_gdelt_titles(
    refresh: bool = False,
    start_date: str = GDELT_START_DATE,
    end_date: str = GDELT_END_DATE,
    company_query: str = GDELT_COMPANY_QUERY,
) -> pd.DataFrame:
    """
    Load cached titles or download them from BigQuery.

    Args:
        refresh:
            Force title re-download.
        start_date:
            Inclusive title-query start date.
        end_date:
            Inclusive title-query end date.
        company_query:
            GKG organisation search phrase.

    Returns:
        Cleaned title DataFrame.
    """
    if GDELT_TITLES_CACHE_PATH.exists() and not refresh:
        return load_cached_titles()

    if refresh:
        print(
            "[C7 Titles] Refresh requested. Existing title cache "
            "will be replaced."
        )
    else:
        print(
            "[C7 Titles] No title cache found. Starting BigQuery query."
        )

    titles, metadata = fetch_gdelt_titles(
        start_date=start_date,
        end_date=end_date,
        company_query=company_query,
    )

    save_title_cache(
        titles=titles,
        metadata=metadata,
    )

    return titles


def build_enriched_gdelt_cache(
    refresh_titles: bool = False,
    start_date: str = GDELT_START_DATE,
    end_date: str = GDELT_END_DATE,
    company_query: str = GDELT_COMPANY_QUERY,
) -> pd.DataFrame:
    """
    Build the enriched GKG article cache.

    Args:
        refresh_titles:
            Force a fresh GDG title query.
        start_date:
            Inclusive query start date.
        end_date:
            Inclusive query end date.
        company_query:
            Organisation phrase used in GKG filtering.

    Returns:
        Enriched article-level DataFrame.
    """
    print(
        f"[C7 Titles] Loading raw GKG article cache: "
        f"{GDELT_RAW_CACHE_PATH.as_posix()}"
    )

    raw_news = load_cached_gdelt_news(
        GDELT_RAW_CACHE_PATH
    )

    titles = get_gdelt_titles(
        refresh=refresh_titles,
        start_date=start_date,
        end_date=end_date,
        company_query=company_query,
    )

    enriched, merge_statistics = enrich_gdelt_news_with_titles(
        raw_news=raw_news,
        titles=titles,
    )

    save_enriched_cache(enriched)

    print("\n[C7 Titles] Enrichment summary")
    print(
        f"Raw GKG articles:          "
        f"{merge_statistics['raw_article_rows']:,}"
    )
    print(
        f"With English headline:     "
        f"{merge_statistics['rows_with_english_headline']:,}"
    )
    print(
        f"Without English headline:  "
        f"{merge_statistics['rows_without_english_headline']:,}"
    )
    print(
        f"Headline coverage:         "
        f"{merge_statistics['english_headline_coverage_percent']:.2f}%"
    )

    return enriched


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download GDELT English article titles and enrich the existing "
            "Task C.7 GKG article cache."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Force a fresh BigQuery title query and overwrite the title cache."
        ),
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=GDELT_START_DATE,
        help="Inclusive start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=GDELT_END_DATE,
        help="Inclusive end date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--company-query",
        type=str,
        default=GDELT_COMPANY_QUERY,
        help="Organisation phrase searched in GKG V2Organizations.",
    )

    return parser.parse_args()


# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================


def main() -> None:
    """Run the GDELT title-download and enrichment workflow."""
    arguments = parse_arguments()

    enriched = build_enriched_gdelt_cache(
        refresh_titles=arguments.refresh,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        company_query=arguments.company_query,
    )

    print("\n[C7 Titles] Enriched dataset summary")
    print(f"Rows:       {len(enriched):,}")
    print(f"Columns:    {len(enriched.columns):,}")

    if not enriched.empty:
        print(
            "Date range: "
            f"{enriched['published_at'].min()} to "
            f"{enriched['published_at'].max()}"
        )

        print("\n[C7 Titles] First five matched headlines:")

        matched = enriched[
            enriched["has_english_headline"].eq(1)
        ]

        print(
            matched[
                [
                    "published_at",
                    "SourceCommonName",
                    "headline",
                ]
            ].head()
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise RuntimeError(
            f"C7 title enrichment execution failed: {error}"
        ) from error