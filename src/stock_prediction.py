import os
import random
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility across runs
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
    """
    Shuffles two numpy arrays in unison to preserve the row-wise matching
    between training features (a) and target outputs (b).
    """
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def cached_file_has_required_columns(csv_path, required_cols):
    """
    Verifies if a cached CSV file contains all the required columns for model features.
    
    Args:
        csv_path (Path): Path to the CSV file.
        required_cols (list): List of required column names.
        
    Returns:
        bool: True if all columns exist, False otherwise.
    """
    try:
        # Read only the header row to inspect columns without loading the whole file
        columns = pd.read_csv(csv_path, nrows=0).columns
    except Exception:
        return False
    
    normalized = {str(column).lower() for column in columns}
    # Map possible alternative names returned by yfinance
    if "adj close" in normalized:
        normalized.add("adjclose")
    if "close" in normalized:
        normalized.add("adjclose")
        
    # Check if required columns form a subset of normalized CSV column names
    return set(required_cols).issubset(normalized)

def find_cached_data(ticker, required_cols):
    """
    Traverses the directory structure upwards to locate a cached stock CSV file in data/.
    
    Args:
        ticker (str): The company ticker name.
        required_cols (list): Features needed to verify schema suitability.
        
    Returns:
        Path or None: The path to the latest valid cached CSV if found, else None.
    """
    # Search in current directory, parent directory, and parent's parent directory
    for root in [Path.cwd(), *Path.cwd().parents]:
        data_dir = root / "data"
        # Match files like 'CBA.AX_*.csv'
        matches = sorted(data_dir.glob(f"{ticker}_*.csv"))
        if matches:
            return matches[-1]
            
        # Fallback to any CSV file in data_dir that fits the schema
        matches = [path for path in sorted(data_dir.glob("*.csv")) if cached_file_has_required_columns(path, required_cols)]
        if matches:
            return matches[-1]
    return None

def read_cached_data(csv_path):
    """Reads a CSV file, parses dates, and sets the date column as the DataFrame index."""
    columns = pd.read_csv(csv_path, nrows=0).columns
    if "Date" in columns:
        return pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return pd.read_csv(csv_path, parse_dates=[0], index_col=0)

def load_ticker_frame(ticker, start_date, end_date, required_cols):
    """
    Loads stock data by attempting to download from Yahoo Finance. If the download fails
    (e.g., due to API rate-limiting), it falls back to the local cached CSV file.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Training start date limit.
        end_date (str): Testing end date limit.
        required_cols (list): Required features to check local schema.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    import yfinance as yf

    df = pd.DataFrame()
    download_failed = False
    try:
        # Download from yfinance (suppressing progress bar to run cleanly)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"yfinance download failed with exception: {e}")
        download_failed = True

    if df.empty or download_failed:
        print("Yahoo Finance download failed or returned empty. Searching for cached CSV fallback...")
        cached_file = find_cached_data(ticker, required_cols)
        if cached_file is not None:
            print(f"Found local cache: {cached_file}. Loading...")
            df = read_cached_data(cached_file)
        else:
            raise FileNotFoundError(f"Yahoo Finance data unavailable and no valid cached CSV file found for {ticker}")

    # Standardize column headers
    # yfinance download sometimes returns a MultiIndex column index (e.g. (Close, CBA.AX))
    # We collapse MultiIndex columns by extracting the top level names (Adj Close, Close, etc.)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert column headers to lowercase strings
    df.columns = [str(column).lower() for column in df.columns]
    
    # Map different closing price name representations to 'adjclose'
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adjclose"})
    elif "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]
        
    return df

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              start_date="2020-01-01", end_date="2024-07-02", cache_dir="data", handle_nans=True):
    """
    Loads, cleans, scales, splits, and prepares sliding window sequences for the model.
    Fulfills all Task C.2 requirements.
    
    Args:
        ticker (str): Company ticker symbol.
        n_steps (int): Sequence lookback window length.
        scale (bool): Whether to scale features to [0, 1] range.
        shuffle (bool): Whether to shuffle dataset rows.
        lookup_step (int): Predict y from x at time step (t + lookup_step).
        split_by_date (bool): Split chronologically (True) or randomly (False).
        test_size (float): Testing dataset ratio.
        feature_columns (list): Features to feed into the LSTM.
        start_date (str): Start boundary for data filtering.
        end_date (str): End boundary for data filtering.
        cache_dir (str): Data directory path.
        handle_nans (bool): Clear missing NaN values.
        
    Returns:
        dict: Processed data arrays, scalers, and dataframes.
    """
    # Load DataFrame from yfinance or cache fallback (C.2 - Requirement d)
    df = load_ticker_frame(ticker, start_date, end_date, feature_columns)
    
    # Filter dataset according to custom start and end dates (C.2 - Requirement a)
    # Ensure index is datetime index to allow date slicing
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_date:end_date]
    
    result = {}
    result['df'] = df.copy()

    # Ensure required features exist in DataFrame
    for col in feature_columns:
        assert col in df.columns, f"Required feature '{col}' does not exist in loaded dataset."

    # Handle NaN missing values (C.2 - Requirement b)
    if handle_nans and df.isnull().values.any():
        print(f"Dataset contains {df.isnull().sum().sum()} missing values. Cleaning NaNs...")
        # Drop rows that are entirely empty
        df.dropna(how='all', inplace=True)
        # Linearly interpolate missing intermediate values
        df.interpolate(method='linear', inplace=True)
        # Forward-fill and backward-fill remaining boundary NaNs
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    # Add date index as a column for date tracking inside numpy arrays
    if "date" not in df.columns:
        df["date"] = df.index

    # Scale feature columns individually and store scalers in a dict (C.2 - Requirement e)
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            # scikit-learn preprocessing requires a 2D array: (samples, 1)
            scaled_col = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            df[column] = scaled_col[:, 0]
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # Add the target future label column by shifting backward by lookup_step
    # For index t, future closing price will be from index t + lookup_step
    df['future'] = df['adjclose'].shift(-lookup_step)

    # Retrieve the last sequence window corresponding to dates without future labels
    # This represents the latest lookup_step days used to predict tomorrow's price
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # Drop rows that do not have future labels (the final lookup_step rows)
    df.dropna(subset=['future'], inplace=True)

    # Construct sliding window sequences of length n_steps
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    # Iterate through entries to create windows: x = [t-n_steps:t], y = [t + lookup_step]
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Append latest historical steps to last_sequence to form input of shape (1, n_steps, features)
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence
    
    # Separate feature windows (X) from label values (y)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split dataset (C.2 - Requirement c)
    if split_by_date:
        # Chronological Split (v0.1)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # Shuffle training and testing samples independently in unison
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # Random Split (P1)
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )

    # Get testing dates
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    
    # Strip date column from final arrays to feed only numeric features to the LSTM
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """
    Constructs a stacked LSTM model using Keras.
    
    Args:
        sequence_length (int): Lookback sequence length (number of time steps).
        n_features (int): Number of input features.
        units (int): Neurons per LSTM layer.
        cell (Layer): LSTM, GRU or SimpleRNN layer.
        n_layers (int): Layer stacking count.
        dropout (float): Dropout rate.
        loss (str): Loss function.
        optimizer (str): Optimizer.
        bidirectional (bool): Wrap layers in Bidirectional wrapper.
        
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential()
    # Explicit Input layer specifying time steps and features shape
    model.add(Input(shape=(sequence_length, n_features)))
    
    for i in range(n_layers):
        is_last_layer = (i == n_layers - 1)
        return_sequences = not is_last_layer
        
        # Instantiate layer
        layer = cell(units, return_sequences=return_sequences)
        
        # Add Bidirectional wrapper if configured
        if bidirectional:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)
            
        # Add Dropout to prevent overfitting
        model.add(Dropout(dropout))
        
    # Dense linear output layer for regression target
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model
