# ==============================================================================
# File: data_processing.py
# Purpose: Modular time-series data loader, cache, split, and scaler pipelines.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==============================================================================
# CORE DATA PROCESSING FUNCTION
# ==============================================================================

def load_and_process_data(
    ticker,
    start_date,
    end_date,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    split_ratio=0.8,
    split_date=None,
    feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
    cache_dir="data"
):
    """
    Loads daily stock market data, applies scaling, handles missing values (NaNs),
    constructs sliding lookback sequences, and splits into train/test datasets.
    
    Args:
        ticker (str): The stock symbol (e.g. 'CBA.AX').
        start_date (str): Start date for data range (YYYY-MM-DD).
        end_date (str): End date for data range (YYYY-MM-DD).
        n_steps (int): The lookback sequence length (number of historical days).
        scale (bool): Whether to scale feature columns to [0, 1].
        shuffle (bool): Whether to shuffle the train dataset.
        lookup_step (int): Number of days ahead to forecast.
        split_by_date (bool): If True, splits chronologically. If False, splits randomly.
        split_ratio (float): Ratio of training samples (e.g., 0.8 for 80% train).
        split_date (str): Optional. Split by a specific date instead of ratio.
        feature_columns (list): Columns to extract as features.
        cache_dir (str): Location to save and load CSV data caches.
        
    Returns:
        dict: Processed data dictionary containing:
            - X_train, y_train: Training sequences and targets.
            - X_test, y_test: Testing sequences and targets.
            - test_df: Sliced test dataframe with actual unscaled values.
            - column_scaler: Dictionary of MinMaxScaler objects per feature.
            - df: Cleaned and processed base dataframe.
            - last_sequence: The tail sequence for next-day predicting.
    """
    # --------------------------------------------------------------------------
    # Phase 1: Initialize Cache Directories
    # --------------------------------------------------------------------------
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_cache.csv")
    legacy_cache_path = os.path.join(cache_dir, f"{ticker}_2026-06-08.csv")
    
    # --------------------------------------------------------------------------
    # Phase 2: Stock Data Retrieval (Cache Check or yfinance Download)
    # --------------------------------------------------------------------------
    raw_df = None
    if os.path.exists(cache_path):
        print(f"[Data Cache] Loading local cache: {cache_path}")
        raw_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    elif os.path.exists(legacy_cache_path):
        print(f"[Data Cache] Loading local legacy cache: {legacy_cache_path}")
        raw_df = pd.read_csv(legacy_cache_path, index_col=0, parse_dates=True)
        
    if raw_df is not None:
        pass
    else:
        print(f"[Data Download] Downloading live data for {ticker} from yfinance...")
        raw_df = yf.download(ticker, start="2000-01-01")
        if raw_df.empty:
            raise ValueError(
                f"Failed to download live stock data for {ticker} and no cached CSV file was found.\n"
                f"If you are running in an offline or sandboxed environment, please download the real stock data manually:\n"
                f"1. Navigate to: https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1577836800&period2=1719964800&interval=1d\n"
                f"2. Save the returned JSON data or convert/download it as a CSV file.\n"
                f"3. Place the CSV file in the 'data/' directory as '{ticker}_cache.csv' to continue execution.\n"
                f"Alternatively, you can run the download helper script: python src/data_downloader.py"
            )
        raw_df.to_csv(cache_path)
        print(f"[Data Cache] Saved downloaded data to: {cache_path}")
        
    # --------------------------------------------------------------------------
    # Phase 3: Column Standardization & Date Slicing
    # --------------------------------------------------------------------------
    # Standardise column names to lowercase to support both yfinance and cached files
    raw_df.columns = [col.lower() for col in raw_df.columns]
    
    df = pd.DataFrame(index=raw_df.index)
    df["open"] = raw_df["open"]
    df["high"] = raw_df["high"]
    df["low"] = raw_df["low"]
    df["close"] = raw_df["close"]
    if "adj close" in raw_df.columns:
        df["adjclose"] = raw_df["adj close"]
    else:
        df["adjclose"] = raw_df["adjclose"]
    df["volume"] = raw_df["volume"]
    df["ticker"] = ticker
    
    # Filter dataset strictly to the user-specified start and end dates
    df = df.loc[start_date:end_date].copy()
    
    # --------------------------------------------------------------------------
    # Phase 4: Missing Value Handling (NaN Imputation)
    # --------------------------------------------------------------------------
    # Forward-fill first to propagate the last known price, then backward-fill
    # to handle any remaining NaNs at the beginning of the sequence.
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Verify that all target features are in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"Required feature column '{col}' is missing."
        
    # Store the original date index as a column to track sequence timelines
    df["date"] = df.index
    
    result = {}
    result["df"] = df.copy()
    
    # --------------------------------------------------------------------------
    # Phase 5: Column-Specific Feature Scaling
    # --------------------------------------------------------------------------
    column_scaler = {}
    if scale:
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]].values)
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
        
    # --------------------------------------------------------------------------
    # Phase 6: Temporal Sequence & Target Construction
    # --------------------------------------------------------------------------
    # Create the prediction targets by shifting Adj Close forward by lookup_step
    df['future'] = df['adjclose'].shift(-lookup_step)
    
    # Settle the last sequence before dropping NaNs (used for future forecasts)
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # Drop rows containing NaNs introduced by the future target shift
    df.dropna(subset=['future'], inplace=True)
    
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    
    # Create sliding windows of historical observations
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
            
    # Append the last sequences to generate predictions beyond the dataset range
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence
    
    # Separate features (X) and targets (y)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X, y = np.array(X), np.array(y)
    
    # --------------------------------------------------------------------------
    # Phase 7: Train / Test Dataset Splitting Strategies
    # --------------------------------------------------------------------------
    if split_by_date:
        # Chronological Split: Keeps time-series structure intact
        if split_date is not None:
            # Split by a specific calendar date boundary
            split_date_parsed = pd.to_datetime(split_date)
            # Find the sample index that corresponds to the split date
            dates = X[:, -1, -1]
            train_mask = dates < split_date_parsed
            
            result["X_train"] = X[train_mask]
            result["y_train"] = y[train_mask]
            result["X_test"]  = X[~train_mask]
            result["y_test"]  = y[~train_mask]
        else:
            # Split by a specific percentage ratio chronologically
            train_samples = int(split_ratio * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"]  = X[train_samples:]
            result["y_test"]  = y[train_samples:]
            
        # Optional: Shuffle the training set only (does not cause lookahead leakage)
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            
    else:
        # Random Split: Splitting data points randomly across time (Data Leakage)
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, train_size=split_ratio, shuffle=shuffle
        )
        
    # Extract test set dates to map back to the original index
    test_dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[test_dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    
    # Strip date columns from neural network input arrays and cast to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    
    return result

# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def shuffle_in_unison(a, b):
    """Shuffles two numpy arrays in unison using the same permutation state."""
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
