# ==============================================================================
# File: config.py
# Purpose: Global configurations, feature settings, and experiment sweep
#          parameters for FinTech101 stock forecasting.
# ==============================================================================

# ==============================================================================
# GLOBAL DATA & PREPROCESSING CONFIGURATIONS
# ==============================================================================
TICKER = "CBA.AX"
START_DATE = "2020-01-01"
END_DATE = "2024-07-02"
SPLIT_DATE = "2023-08-02"  # Chronological split boundary date (train ends 2023-08-01, test starts 2023-08-02)

# ----- Sliding Window Parameters -----
N_STEPS = 50       # Lookback sequence window size (days)
LOOKUP_STEP = 1    # Forecast horizon (days ahead)

# ----- Feature Definition -----
FEATURE_COLUMNS = ['adjclose', 'volume', 'open', 'high', 'low']

# ==============================================================================
# TASK C.4 HYPERPARAMETER EXPERIMENT CONFIGURATIONS
# ==============================================================================
SWEEP_CONFIGS = {
    # ---- Cell Type Comparison ----
    "LSTM_BASE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base LSTM — Standard 2-layer LSTM control benchmark"
    },
    "GRU_BASE": {
        "cell_type": "GRU",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base GRU — GRU cell vs LSTM (parameter efficiency, no separate cell state)"
    },
    "RNN_BASE": {
        "cell_type": "SimpleRNN",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base RNN — Vanilla RNN, tests vanishing gradient on 50-day windows"
    },
    # ---- Depth Comparison ----
    "LSTM_STACKED": {
        "cell_type": "LSTM",
        "n_layers": 3,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Stacked LSTM — Deep 3-layer representational capacity vs overfitting"
    },
    "LSTM_SHALLOW": {
        "cell_type": "LSTM",
        "n_layers": 1,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Shallow LSTM — Minimal 1-layer model (Occam's Razor regularizer)"
    },
    # ---- Width Comparison ----
    "LSTM_WIDE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 256,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Wide LSTM — 256-unit state capacity expansion vs overfitting noise"
    },
    "LSTM_NARROW": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 64,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Narrow LSTM — 64-unit bottleneck compression as regularizer"
    },
    # ---- Loss Function Comparison ----
    "LSTM_MSE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "mse",
        "epochs": 20,
        "batch_size": 64,
        "description": "LSTM MSE Loss — Outlier-sensitive quadratic loss vs robust Huber baseline"
    }
}
