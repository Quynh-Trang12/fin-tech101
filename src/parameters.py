import os
import time

# ==============================================================================
# Stock Dataset Parameters
# ==============================================================================
TICKER = "CBA.AX"                  # Ticker symbol of company
TRAIN_START = "2020-01-01"         # Start date for loading training data
TRAIN_END = "2023-08-01"           # End date for training data
TEST_START = "2023-08-02"          # Start date for test data
TEST_END = "2024-07-02"            # End date for test data

# ==============================================================================
# Preprocessing and Splitting Settings
# ==============================================================================
N_STEPS = 50                       # Sequence length / lookback window size (days)
LOOKUP_STEP = 1                    # Forecast horizon (number of days in future)

SCALE = True                       # Scale feature columns to [0, 1] range
SHUFFLE = True                     # Shuffle datasets during processing
SPLIT_BY_DATE = False              # False: random split (P1); True: chronological split (v0.1)
TEST_SIZE = 0.2                    # Ratio of test set size

# Input feature columns and the target output column
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
PRICE_VALUE = "adjclose"           # Main target column to predict/evaluate

# ==============================================================================
# LSTM Model Hyperparameters
# ==============================================================================
N_LAYERS = 2                       # Number of stacked LSTM layers
LSTM_UNITS = 256                   # Neurons per LSTM layer
DROPOUT = 0.4                      # Dropout rate for regularization
BIDIRECTIONAL = False              # Use bidirectional LSTM layers

# ==============================================================================
# Training Parameters
# ==============================================================================
LOSS = "huber"                     # Loss function (e.g. 'huber', 'mean_squared_error')
OPTIMIZER = "adam"                 # Optimizer type
BATCH_SIZE = 64
EPOCHS = 40                        # Training epochs

# ==============================================================================
# Output Directories & Naming
# ==============================================================================
DATA_DIR = "data"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
CSV_RESULTS_DIR = "csv-results"

# Generate a unique name for the model based on parameters
date_now = time.strftime("%Y-%m-%d")
shuffle_str = f"sh-{int(SHUFFLE)}"
scale_str = f"sc-{int(SCALE)}"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

model_name = (
    f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-"
    f"{LOSS}-{OPTIMIZER}-LSTM-seq-{N_STEPS}-step-{LOOKUP_STEP}-"
    f"layers-{N_LAYERS}-units-{LSTM_UNITS}"
)
if BIDIRECTIONAL:
    model_name += "-b"
