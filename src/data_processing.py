# ==============================================================================
# Purpose:
# Leakage-aware stock data loading, preprocessing, splitting, scaling, and
# sequence construction for FinTech101 time-series forecasting experiments.
# ==============================================================================

import os
import pickle

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
TARGET_COLUMN = "adjclose"
RANDOM_SEED = 314


# ==============================================================================
# RAW DATA LOADING
# ==============================================================================


def load_raw_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """Load stock data from a local cache, or download it when no cache exists.

    The cache is treated as the source of truth once it exists. This avoids
    repeated API calls and keeps later experiments reproducible.

    Args:
        ticker: Market symbol to load, for example ``"CBA.AX"``.
        start_date: First date required by the experiment, in ``YYYY-MM-DD`` form.
        end_date: Last date required by the experiment, in ``YYYY-MM-DD`` form.
        cache_dir: Directory that stores cached stock CSV files.

    Returns:
        A raw dataframe loaded either from disk or Yahoo Finance.

    Raises:
        ValueError: If live download fails and no cache file is available.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{ticker}_cache.csv")

    if os.path.exists(cache_path):
        print(f"[Data Cache] Loading local cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[Data Download] Downloading live data for {ticker} from yfinance...")
    raw_df = yf.download(ticker, start=start_date, end=end_date)

    if raw_df.empty:
        raise ValueError(
            f"No cached CSV was found for {ticker}, and yfinance returned an "
            f"empty dataframe. Run src/data_downloader.py first or place "
            f"{ticker}_cache.csv inside the '{cache_dir}' directory."
        )

    raw_df.to_csv(cache_path)
    print(f"[Data Cache] Saved downloaded data to: {cache_path}")

    return raw_df


# ==============================================================================
# DATAFRAME CLEANING
# ==============================================================================


def standardise_stock_dataframe(
    raw_df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Convert raw Yahoo/cache data into the internal dataframe format.

    The returned dataframe uses lowercase OHLCV column names, contains a ticker
    column for downstream visualisation, and keeps the date index sorted.

    Args:
        raw_df: Raw dataframe from cache or Yahoo Finance.
        ticker: Market symbol attached to each row.
        start_date: First date retained for this experiment.
        end_date: Last date retained for this experiment.

    Returns:
        A cleaned dataframe with ``open``, ``high``, ``low``, ``close``,
        ``adjclose``, ``volume``, ``ticker``, and ``date`` columns.

    Raises:
        ValueError: If required price/volume columns are missing or the date
            slice contains no rows.
    """
    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().strip() for col in df.columns]
    else:
        df.columns = [str(col).lower().strip() for col in df.columns]

    df.rename(columns={"adj close": "adjclose"}, inplace=True)

    required_columns = ["open", "high", "low", "close", "adjclose", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required stock columns: {missing_columns}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df[required_columns].copy()
    df["ticker"] = ticker

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.loc[start_date:end_date].copy()

    if df.empty:
        raise ValueError(
            f"No stock rows found for {ticker} between {start_date} and {end_date}."
        )

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["date"] = df.index

    return df


# ==============================================================================
# SUPERVISED TARGET CONSTRUCTION
# ==============================================================================


def add_future_targets(
    df: pd.DataFrame,
    forecast_offset: int,
    future_steps: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Add future price labels and their corresponding calendar dates.

    The target date columns are needed because leakage-safe splitting must be
    based on the date being predicted, not the final date inside the input window.

    Args:
        df: Cleaned stock dataframe.
        forecast_offset: Number of trading rows ahead for the first prediction.
            For example, ``1`` means next trading day.
        future_steps: Number of consecutive future prices to predict.

    Returns:
        A tuple containing the supervised dataframe, target price column names,
        and target date column names.

    Raises:
        ValueError: If the forecast horizon is invalid.
    """
    if forecast_offset < 1:
        raise ValueError("forecast_offset must be at least 1.")

    if future_steps < 1:
        raise ValueError("future_steps must be at least 1.")

    df = df.copy()
    target_columns = []
    target_date_columns = []

    for step in range(future_steps):
        shift_amount = forecast_offset + step
        target_col = f"future_{step}"
        target_date_col = f"future_date_{step}"

        df[target_col] = df[TARGET_COLUMN].shift(-shift_amount)
        df[target_date_col] = df["date"].shift(-shift_amount)

        target_columns.append(target_col)
        target_date_columns.append(target_date_col)

    df.dropna(subset=target_columns + target_date_columns, inplace=True)

    return df, target_columns, target_date_columns


# ==============================================================================
# SEQUENCE CONSTRUCTION
# ==============================================================================


def build_sequences(
    df: pd.DataFrame,
    lookback_steps: int,
    feature_columns: List[str],
    target_columns: List[str],
    target_date_columns: List[str],
) -> Dict[str, np.ndarray]:
    """Build unscaled sliding-window samples for model training and testing.

    Each sample contains a historical input window, its future target value(s),
    the date where the input window ends, and the date(s) being predicted.

    Args:
        df: Supervised dataframe containing feature and target columns.
        lookback_steps: Number of past trading rows in each input window.
        feature_columns: Market features used as model inputs.
        target_columns: Future price columns used as labels.
        target_date_columns: Future date columns matching the labels.

    Returns:
        A dictionary containing unscaled ``X``, ``y``, input-end dates, and
        target dates.

    Raises:
        ValueError: If feature columns are missing or there is insufficient data.
    """
    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least 1.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if len(df) < lookback_steps:
        raise ValueError(
            f"Cannot build {lookback_steps}-step windows from only {len(df)} rows."
        )

    X, y = [], []
    input_end_dates = []
    target_dates = []

    for end_idx in range(lookback_steps - 1, len(df)):
        start_idx = end_idx - lookback_steps + 1

        X.append(df[feature_columns].iloc[start_idx : end_idx + 1].values)
        y.append(df[target_columns].iloc[end_idx].values)
        input_end_dates.append(df.index[end_idx])
        target_dates.append(df[target_date_columns].iloc[end_idx].values)

    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.float32),
        "input_end_dates": np.array(input_end_dates),
        "target_dates": np.array(target_dates),
    }


# ==============================================================================
# TRAIN / TEST SPLITTING
# ==============================================================================


def split_sequences(
    sequence_data: Dict[str, np.ndarray],
    split_by_date: bool = True,
    split_ratio: float = 0.8,
    split_date: Optional[str] = None,
    shuffle: bool = True,
) -> Dict[str, np.ndarray]:
    """Split samples into train and test sets.

    Chronological splitting uses target dates, so a sample is placed in the test
    set when the value being predicted belongs to the test period.

    Args:
        sequence_data: Dictionary returned by ``build_sequences``.
        split_by_date: Use chronological splitting when true; otherwise use
            random splitting for controlled comparison experiments.
        split_ratio: Fraction of samples assigned to training when no explicit
            split date is provided.
        split_date: First target date that should belong to the test set.
        shuffle: Shuffle only the training samples for chronological splits.

    Returns:
        A dictionary containing train/test arrays and their input-end dates.

    Raises:
        ValueError: If the split configuration creates an empty train or test set.
    """
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1.")

    X = sequence_data["X"]
    y = sequence_data["y"]
    input_end_dates = sequence_data["input_end_dates"]
    target_dates = sequence_data["target_dates"]

    if split_by_date:
        split_data = split_sequences_by_target_date(
            X=X,
            y=y,
            input_end_dates=input_end_dates,
            target_dates=target_dates,
            split_date=split_date,
            split_ratio=split_ratio,
        )

        if shuffle:
            (
                split_data["X_train"],
                split_data["y_train"],
                split_data["train_input_dates"],
            ) = shuffle_training_samples(
                split_data["X_train"],
                split_data["y_train"],
                split_data["train_input_dates"],
            )
    else:
        split_data = split_sequences_randomly(
            X=X,
            y=y,
            input_end_dates=input_end_dates,
            split_ratio=split_ratio,
            shuffle=shuffle,
        )

    if len(split_data["X_train"]) == 0 or len(split_data["X_test"]) == 0:
        raise ValueError(
            "Train/test split produced an empty dataset. Check split_date, "
            "split_ratio, lookback_steps, forecast_offset, and future_steps."
        )

    return split_data


def split_sequences_by_target_date(
    X: np.ndarray,
    y: np.ndarray,
    input_end_dates: np.ndarray,
    target_dates: np.ndarray,
    split_date: Optional[str],
    split_ratio: float,
) -> Dict[str, np.ndarray]:
    """Split time-series samples without leaking test targets into training."""
    if split_date is None:
        train_size = int(split_ratio * len(X))

        return {
            "X_train": X[:train_size],
            "y_train": y[:train_size],
            "X_test": X[train_size:],
            "y_test": y[train_size:],
            "train_input_dates": input_end_dates[:train_size],
            "test_input_dates": input_end_dates[train_size:],
        }

    split_timestamp = pd.to_datetime(split_date)

    # For single-step forecasting, first_target_dates and last_target_dates are
    # identical. For multi-step forecasting, they define the whole target horizon.
    first_target_dates = pd.to_datetime(target_dates[:, 0])
    last_target_dates = pd.to_datetime(target_dates[:, -1])

    train_mask = last_target_dates < split_timestamp
    test_mask = first_target_dates >= split_timestamp

    dropped_count = len(X) - int(train_mask.sum()) - int(test_mask.sum())
    if dropped_count > 0:
        print(
            "[Data Split] Dropped "
            f"{dropped_count} boundary sample(s) because their multi-step "
            "target horizon crosses the split date."
        )

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "train_input_dates": input_end_dates[train_mask],
        "test_input_dates": input_end_dates[test_mask],
    }


def split_sequences_randomly(
    X: np.ndarray,
    y: np.ndarray,
    input_end_dates: np.ndarray,
    split_ratio: float,
    shuffle: bool,
) -> Dict[str, np.ndarray]:
    """Split samples randomly for comparison experiments, not live forecasting."""
    indices = np.arange(len(X))

    train_indices, test_indices = train_test_split(
        indices,
        train_size=split_ratio,
        shuffle=shuffle,
        random_state=RANDOM_SEED,
    )

    print(
        "[Data Split] Random split selected. This is not leakage-safe for "
        "realistic time-series forecasting."
    )

    return {
        "X_train": X[train_indices],
        "y_train": y[train_indices],
        "X_test": X[test_indices],
        "y_test": y[test_indices],
        "train_input_dates": input_end_dates[train_indices],
        "test_input_dates": input_end_dates[test_indices],
    }


def shuffle_training_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_input_dates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle training arrays while preserving sample alignment."""
    rng = np.random.default_rng(RANDOM_SEED)
    permutation = rng.permutation(len(X_train))

    return (
        X_train[permutation],
        y_train[permutation],
        train_input_dates[permutation],
    )


# ==============================================================================
# TRAINING-ONLY SCALING
# ==============================================================================


def fit_training_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: List[str],
) -> Dict[str, MinMaxScaler]:
    """Fit feature scalers using training samples only.

    The target scaler is fitted using both historical training values and
    training labels for ``adjclose``. Test values are intentionally excluded.

    Args:
        X_train: Unscaled training windows with shape
            ``(samples, lookback_steps, features)``.
        y_train: Unscaled future target prices with shape
            ``(samples, future_steps)``.
        feature_columns: Names of the feature columns in the same order as the
            last dimension of ``X_train``.

    Returns:
        A dictionary mapping each feature name to a fitted ``MinMaxScaler``.
    """
    scalers = {}

    for feature_index, column in enumerate(feature_columns):
        historical_values = X_train[:, :, feature_index].reshape(-1, 1)

        if column == TARGET_COLUMN:
            target_values = y_train.reshape(-1, 1)
            fit_values = np.vstack([historical_values, target_values])
        else:
            fit_values = historical_values

        scaler = MinMaxScaler()
        scaler.fit(fit_values)
        scalers[column] = scaler

    return scalers


def scale_sequences(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: List[str],
    scalers: Dict[str, MinMaxScaler],
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale input windows and targets with training-fitted scalers."""
    X_scaled = X.copy().astype(np.float32)

    # --------------------------------------------------------------------------
    # Phase 1: Scale each feature channel independently
    # --------------------------------------------------------------------------
    for feature_index, column in enumerate(feature_columns):
        scaler = scalers[column]
        feature_slice = X_scaled[:, :, feature_index]
        original_shape = feature_slice.shape

        X_scaled[:, :, feature_index] = scaler.transform(
            feature_slice.reshape(-1, 1)
        ).reshape(original_shape)

    # --------------------------------------------------------------------------
    # Phase 2: Scale future adjclose labels with the adjclose scaler
    # --------------------------------------------------------------------------
    target_scaler = scalers[TARGET_COLUMN]
    y_scaled = target_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


def scale_last_sequence(
    last_sequence: np.ndarray,
    feature_columns: List[str],
    scalers: Dict[str, MinMaxScaler],
) -> np.ndarray:
    """Scale the latest lookback window for future inference."""
    scaled_sequence = last_sequence.copy().astype(np.float32)

    for feature_index, column in enumerate(feature_columns):
        scaled_sequence[:, feature_index] = (
            scalers[column]
            .transform(scaled_sequence[:, feature_index].reshape(-1, 1))
            .reshape(-1)
        )

    return scaled_sequence.astype(np.float32)


def save_scalers(
    scalers: Dict[str, MinMaxScaler],
    ticker: str,
    output_dir: str,
    metadata: Dict[str, Any],
) -> str:
    """Persist training-fitted scalers and metadata for later inspection.

    Args:
        scalers: Fitted scaler objects keyed by feature name.
        ticker: Market symbol used to name the scaler file.
        output_dir: Directory where the scaler artefact is written.
        metadata: Configuration values explaining how the scalers were fitted.

    Returns:
        Path to the saved pickle file.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_ticker = ticker.replace(".", "_")
    scaler_path = os.path.join(output_dir, f"{safe_ticker}_scalers.pkl")

    with open(scaler_path, "wb") as file:
        pickle.dump(
            {
                "scalers": scalers,
                "metadata": metadata,
            },
            file,
        )

    print(f"[Scaler Cache] Saved training-fitted scalers to: {scaler_path}")

    return scaler_path


# ==============================================================================
# MAIN DATA PROCESSING PIPELINE
# ==============================================================================


def load_and_process_data(
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_steps: int = 50,
    scale: bool = True,
    shuffle: bool = True,
    forecast_offset: int = 1,
    split_by_date: bool = True,
    split_ratio: float = 0.8,
    split_date: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    cache_dir: str = "data",
    future_steps: int = 1,
    save_scaler_cache: bool = True,
    results_dir: str = "results/c2",
) -> Dict[str, Any]:
    """Load, clean, split, scale, and package stock data for model pipelines.

    This is the public data-processing entry point used by training, testing,
    visualisation, and sweep scripts.

    Args:
        ticker: Market symbol to process.
        start_date: First date included in the raw experiment period.
        end_date: Last date included in the raw experiment period.
        lookback_steps: Number of historical trading rows per input sample.
        scale: Apply MinMax scaling when true.
        shuffle: Shuffle training samples after chronological splitting.
        forecast_offset: Number of trading rows between the current input end
            and the first predicted target.
        split_by_date: Use chronological target-date split when true.
        split_ratio: Training fraction used when no explicit split date is given.
        split_date: First target date assigned to the test set.
        feature_columns: Input feature names. Defaults to OHLCV-style features.
        cache_dir: Directory for raw CSV cache and scaler cache.
        future_steps: Number of future target prices per sample.
        save_scaler_cache: Save fitted scalers to disk when scaling is enabled.
        results_dir: Directory for generated Task C.2 preprocessing artefacts,
            including fitted scaler files.

    Returns:
        A dictionary containing model-ready arrays, unscaled arrays, scalers,
        date mappings, the cleaned dataframe, and the test dataframe.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS.copy()

    # --------------------------------------------------------------------------
    # Phase 1: Load and clean raw market data
    # --------------------------------------------------------------------------
    raw_df = load_raw_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )

    df = standardise_stock_dataframe(
        raw_df=raw_df,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    # --------------------------------------------------------------------------
    # Phase 2: Create supervised future targets before sequence construction
    # --------------------------------------------------------------------------
    supervised_df, target_columns, target_date_columns = add_future_targets(
        df=df,
        forecast_offset=forecast_offset,
        future_steps=future_steps,
    )

    # --------------------------------------------------------------------------
    # Phase 3: Build unscaled sliding-window samples
    # --------------------------------------------------------------------------
    sequence_data = build_sequences(
        df=supervised_df,
        lookback_steps=lookback_steps,
        feature_columns=feature_columns,
        target_columns=target_columns,
        target_date_columns=target_date_columns,
    )

    # --------------------------------------------------------------------------
    # Phase 4: Split samples before fitting any scaler
    # --------------------------------------------------------------------------
    split_data = split_sequences(
        sequence_data=sequence_data,
        split_by_date=split_by_date,
        split_ratio=split_ratio,
        split_date=split_date,
        shuffle=shuffle,
    )

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # --------------------------------------------------------------------------
    # Phase 5: Fit scalers on training data only
    # --------------------------------------------------------------------------
    result = {
        "df": df.copy(),
        "X_train_unscaled": X_train.copy(),
        "y_train_unscaled": y_train.copy(),
        "X_test_unscaled": X_test.copy(),
        "y_test_unscaled": y_test.copy(),
        "train_input_dates": split_data["train_input_dates"],
        "test_input_dates": split_data["test_input_dates"],
    }

    last_sequence_unscaled = df[feature_columns].tail(lookback_steps).values

    if scale:
        column_scaler = fit_training_scalers(
            X_train=X_train,
            y_train=y_train,
            feature_columns=feature_columns,
        )

        X_train, y_train = scale_sequences(
            X=X_train,
            y=y_train,
            feature_columns=feature_columns,
            scalers=column_scaler,
        )

        X_test, y_test = scale_sequences(
            X=X_test,
            y=y_test,
            feature_columns=feature_columns,
            scalers=column_scaler,
        )

        last_sequence = scale_last_sequence(
            last_sequence=last_sequence_unscaled,
            feature_columns=feature_columns,
            scalers=column_scaler,
        )

        result["column_scaler"] = column_scaler

        if save_scaler_cache:
            result["scaler_path"] = save_scalers(
                scalers=column_scaler,
                ticker=ticker,
                output_dir=results_dir,
                metadata={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "split_date": split_date,
                    "split_by_date": split_by_date,
                    "split_ratio": split_ratio,
                    "lookback_steps": lookback_steps,
                    "forecast_offset": forecast_offset,
                    "future_steps": future_steps,
                    "feature_columns": feature_columns,
                    "target_column": TARGET_COLUMN,
                    "fitted_on": "training samples only",
                },
            )
    else:
        last_sequence = last_sequence_unscaled.astype(np.float32)

    # --------------------------------------------------------------------------
    # Phase 6: Build test dataframe aligned with prediction input dates
    # --------------------------------------------------------------------------
    test_df = df.loc[split_data["test_input_dates"]].copy()
    test_df = test_df[~test_df.index.duplicated(keep="first")]

    # --------------------------------------------------------------------------
    # Phase 7: Return model-ready data package
    # --------------------------------------------------------------------------
    result.update(
        {
            "X_train": X_train.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "y_test": y_test.astype(np.float32),
            "test_df": test_df,
            "last_sequence": last_sequence.astype(np.float32),
            "target_columns": target_columns,
            "target_date_columns": target_date_columns,
        }
    )

    print("[Data Pipeline] Completed leakage-aware data processing.")
    print(f"[Data Pipeline] X_train: {result['X_train'].shape}")
    print(f"[Data Pipeline] y_train: {result['y_train'].shape}")
    print(f"[Data Pipeline] X_test:  {result['X_test'].shape}")
    print(f"[Data Pipeline] y_test:  {result['y_test'].shape}")

    return result
