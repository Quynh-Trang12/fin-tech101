# ================================================================
# Purpose:
# Global configurations, feature settings, and experiment sweep
# parameters for FinTech101 stock forecasting.
# ================================================================

from pathlib import Path

# ==============================================================================
# GLOBAL DATA & PREPROCESSING CONFIGURATIONS
# ==============================================================================
TICKER = "CBA.AX"
START_DATE = "2020-01-01"
END_DATE = "2024-07-02"
SPLIT_DATE = "2023-08-02"  # Chronological split boundary date (train ends 2023-08-01, test starts 2023-08-02)
SPLIT_METHOD = "date"  # Train/test split strategy: "date" | "ratio" | "random"
VALIDATION_RATIO = 0.15  # Chronological validation split ratio (relative to the training set)

# ----- Sliding Window Parameters -----
LOOKBACK_STEPS = 50  # Lookback sequence window size (days)
FORECAST_OFFSET = 1  # Days ahead to start forecast (offset)
FUTURE_STEPS = 1  # Number of future steps/days to predict

# ----- Feature Definition -----
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# ----- Task C.3 Visualisation Parameters -----
C3_CANDLE_DAYS = 5  # Trading days aggregated per candle in the summary chart
C3_BOXPLOT_WINDOW = 10  # Trading days represented by each boxplot
C3_BOXPLOT_STEP = 10  # Window advance; equal to the window size = non-overlapping

# ----- Experiment Sweep Directories -----
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
CSV_RESULTS_DIR = Path("csv-results")

# ==============================================================================
# TASK C.4 HYPERPARAMETER EXPERIMENT CONFIGURATIONS
# ==============================================================================
C4_SWEEP_CONFIGS = {
    # ---- Cell Type Comparison ----
    "LSTM_BASE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base LSTM — Standard 2-layer LSTM control benchmark",
    },
    "GRU_BASE": {
        "cell_type": "GRU",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base GRU — GRU cell vs LSTM (parameter efficiency, no separate cell state)",
    },
    "RNN_BASE": {
        "cell_type": "SimpleRNN",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Base RNN — Vanilla RNN, tests vanishing gradient on 50-day windows",
    },
    # ---- Depth Comparison ----
    "LSTM_STACKED": {
        "cell_type": "LSTM",
        "n_layers": 3,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Stacked LSTM — Deep 3-layer representational capacity vs overfitting",
    },
    "LSTM_SHALLOW": {
        "cell_type": "LSTM",
        "n_layers": 1,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Shallow LSTM — Minimal 1-layer model (Occam's Razor regularizer)",
    },
    # ---- Width Comparison ----
    "LSTM_WIDE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 256,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Wide LSTM — 256-unit state capacity expansion vs overfitting noise",
    },
    "LSTM_NARROW": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 64,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 64,
        "description": "Narrow LSTM — 64-unit bottleneck compression as regularizer",
    },
    # ---- Loss Function Comparison ----
    "LSTM_MSE": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "mse",
        "epochs": 20,
        "batch_size": 64,
        "description": "LSTM MSE Loss — Outlier-sensitive quadratic loss vs robust Huber baseline",
    },
    # ---- Epoch Count Comparison ----
    "LSTM_LONGTRAIN": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 40,
        "batch_size": 64,
        "description": "Long-Train LSTM — Doubled epoch budget vs baseline, tests overfitting onset",
    },
    # ---- Batch Size Comparison ----
    "LSTM_SMALLBATCH": {
        "cell_type": "LSTM",
        "n_layers": 2,
        "units": 128,
        "loss": "huber",
        "epochs": 20,
        "batch_size": 16,
        "description": "Small-Batch LSTM — Smaller batch size vs baseline, tests gradient noise/generalisation",
    },
}

# ==============================================================================
# TASK C.5 MULTIVARIATE & MULTISTEP EXPERIMENT CONFIGURATIONS
# ==============================================================================
# Keep 6 features named in the C.5 brief from FEATURE_COLUMNS 
# so C.4's already-verified sweep results are unaffected.
C5_FEATURE_COLUMNS = ["adjclose", "close", "volume", "open", "high", "low"]

C5_SWEEP_CONFIGS = {
    "gru_uni_multistep": {
        "description": "Univariate Multistep (Close Only -> 5 Days Forecast)",
        "feature_columns": ["adjclose"],
        "future_steps": 5,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "GRU",
    },
    "gru_multi_singlestep": {
        "description": "Multivariate Single-Step (All Features -> 1 Day Forecast)",
        "feature_columns": C5_FEATURE_COLUMNS,
        "future_steps": 1,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "GRU",
    },
    "gru_multi_multistep": {
        "description": "Multivariate Multistep Combined (All Features -> 5 Days Forecast)",
        "feature_columns": C5_FEATURE_COLUMNS,
        "future_steps": 5,
        "epochs": 20,
        "batch_size": 64,
        "units": 128,
        "n_layers": 2,
        "cell_type": "GRU",
    },
}

# ==============================================================================
# TASK C.7 NEWS CONFIGURATION
# ==============================================================================
C7_DATA_DIR = DATA_DIR / "c7"

BIGQUERY_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

COMPANY_QUERY = "commonwealth bank"

NEWS_COLUMNS = [
    "DATE",
    "SourceCommonName",
    "DocumentIdentifier",
    "V2Organizations",
    "V2Themes",
    "V2Tone",
    "GCAM",
]

GDELT_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
GDELT_COMPANY_QUERY = "commonwealth bank"

GDELT_START_DATE = START_DATE
GDELT_END_DATE = END_DATE

GDELT_RAW_CACHE_PATH = C7_DATA_DIR / "gdelt_cba_raw.parquet"
GDELT_METADATA_PATH = C7_DATA_DIR / "gdelt_cba_metadata.json"

GDELT_DAILY_V2TONE_PATH = C7_DATA_DIR / "gdelt_daily_v2tone.parquet"

MARKET_TIMEZONE = "Australia/Sydney"

# ----- FinBERT Settings -----
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 64
FINBERT_MAX_LENGTH = 128
GDELT_ENRICHED_CACHE_PATH = C7_DATA_DIR / "gdelt_cba_enriched.parquet"
GDELT_FINBERT_ARTICLE_PATH = C7_DATA_DIR / "gdelt_finbert_article.parquet"
GDELT_FINBERT_CHECKPOINT_PATH = C7_DATA_DIR / "gdelt_finbert_checkpoint.parquet"




