# ==============================================================================
# Purpose: 
# Direct API stock data downloader utilizing query2.finance.yahoo.com
# ==============================================================================

import os
import json
import urllib.request
import datetime as dt
import pandas as pd

# ==============================================================================
# CORE DOWNLOAD FUNCTION
# ==============================================================================

def download_real_stock_data(ticker="CBA.AX", start_date="2020-01-01", end_date="2024-07-04"):
    """
    Downloads real daily stock history directly from query2.finance.yahoo.com JSON chart API.
    Bypasses standard yfinance crumb/cookie blocks.
    
    Args:
        ticker (str): The stock ticker (default 'CBA.AX').
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: Cleaned stock historical dataframe.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Timestamp Conversion and Request Setup
    # --------------------------------------------------------------------------
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts = int(pd.to_datetime(end_date).timestamp())
    
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
    print(f"[Downloader] Querying URL: {url}")
    
    # Configure request with a standard browser User-Agent to prevent 401/403 blocks
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    # --------------------------------------------------------------------------
    # Phase 2: API Call and JSON Parsing
    # --------------------------------------------------------------------------
    try:
        with urllib.request.urlopen(req) as response:
            js = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"HTTP request to query2 API failed: {e}")
        
    # Verify the JSON response contains results
    if 'chart' not in js or 'result' not in js['chart'] or not js['chart']['result']:
        raise ValueError(f"Invalid API response structure for ticker '{ticker}'.")
        
    result = js['chart']['result'][0]
    
    # Extract timestamps and indicators
    timestamps = result.get('timestamp', [])
    if not timestamps:
        raise ValueError(f"No price history found in the requested range for '{ticker}'.")
        
    quote = result['indicators']['quote'][0]
    adjclose_list = result['indicators']['adjclose'][0]['adjclose']
    
    # Convert timestamps to string dates
    dates = [dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
    
    # --------------------------------------------------------------------------
    # Phase 3: DataFrame Construction & Cleaning
    # --------------------------------------------------------------------------
    df = pd.DataFrame({
        'open': quote.get('open', []),
        'high': quote.get('high', []),
        'low': quote.get('low', []),
        'close': quote.get('close', []),
        'adjclose': adjclose_list,
        'volume': quote.get('volume', []),
        'ticker': ticker
    }, index=pd.to_datetime(dates))
    
    # Clean the dataset by handling any missing intervals or NaN values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.index.name = ''
    
    return df

# ==============================================================================
# STANDALONE EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    ticker = "CBA.AX"
    start_date = "2020-01-01"
    end_date = "2024-07-04"
    
    try:
        print(f"[Downloader] Initiating direct download for {ticker}...")
        df = download_real_stock_data(ticker, start_date, end_date)
        
        # Save to data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        cache_path = os.path.join(data_dir, f"{ticker}_cache.csv")
        
        df.to_csv(cache_path)
        print(f"[Downloader] Successfully saved real stock data to: {cache_path}")
        
    except Exception as e:
        print(f"[Downloader] Execution error: {e}")
