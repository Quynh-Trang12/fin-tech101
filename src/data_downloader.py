# ==============================================================================
# Purpose:
# Direct API stock data downloader utilizing query2.finance.yahoo.com
# ==============================================================================

import json
import urllib.request
import pandas as pd

from config import TICKER, START_DATE, END_DATE, DATA_DIR

# ==============================================================================
# CORE DOWNLOAD FUNCTION
# ==============================================================================


def download_real_stock_data(
    ticker: str = TICKER, start_date: str = START_DATE, end_date: str = END_DATE
) -> pd.DataFrame:
    """
    Downloads real daily stock history directly from query2.finance.yahoo.com JSON chart API.
    Bypasses standard yfinance crumb/cookie blocks.

    Args:
        ticker: The stock ticker (default 'CBA.AX').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        Cleaned stock historical dataframe.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Timestamp Conversion and Request Setup
    # --------------------------------------------------------------------------
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
    print(f"[Downloader] Querying URL: {url}")

    # Configure request with a standard browser User-Agent to prevent 401/403 blocks
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    # --------------------------------------------------------------------------
    # Phase 2: API Call and JSON Parsing
    # --------------------------------------------------------------------------
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            js = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"HTTP request to query2 API failed: {error}") from error

    # Verify the JSON response contains results
    if "chart" not in js or "result" not in js["chart"] or not js["chart"]["result"]:
        raise ValueError(f"Invalid API response structure for ticker '{ticker}'.")

    result = js["chart"]["result"][0]

    # Extract timestamps and indicators
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError(
            f"No price history found in the requested range for '{ticker}'."
        )

    indicators = result.get("indicators", {})
    quotes = indicators.get("quote", [])
    adjclose_entries = indicators.get("adjclose", [])

    if not quotes or not adjclose_entries:
        raise ValueError(f"Missing quote or adjusted-close data for '{ticker}'.")

    quote = quotes[0]
    adjclose_list = adjclose_entries[0].get("adjclose", [])

    # Convert timestamps to local exchange dates
    # We retrieve the exchange timezone from the Yahoo response metadata to prevent
    # calendar shift issues (e.g., Australian trading sessions shifting backward to Sunday when using UTC).
    exchange_timezone = result.get("meta", {}).get("exchangeTimezoneName")

    if not exchange_timezone:
        raise ValueError(
            f"Yahoo Finance did not provide an exchange timezone for '{ticker}'."
        )

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(exchange_timezone)
        .tz_localize(None)
        .normalize()
    )

    # --------------------------------------------------------------------------
    # Phase 3: DataFrame Construction & Cleaning
    # --------------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "adjclose": adjclose_list,
            "volume": quote.get("volume", []),
            "ticker": ticker,
        },
        index=dates,
    )

    # Clean the dataset by handling any missing intervals or NaN values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.index.name = ""

    return df


# ==============================================================================
# STANDALONE EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        print(f"[Downloader] Initiating direct download for {TICKER}...")
        df = download_real_stock_data(TICKER, START_DATE, END_DATE)

        # Save to data directory
        data_dir = DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        cache_path = data_dir / f"{TICKER}_cache.csv"

        df.to_csv(cache_path)
        print(
            f"[Downloader] Successfully saved real stock data to: {cache_path.as_posix()}"
        )

    except Exception as error:
        raise RuntimeError(f"Downloader execution failed: {error}") from error
