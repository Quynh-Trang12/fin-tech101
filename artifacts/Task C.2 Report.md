# Option C - Task C.2 Data Processing 1 Report

## Project Details
- **Project:** FinTech101 Stock Price Prediction System
- **Subject:** COS30018 - Intelligent Systems
- **Task:** Option C - Task 2: Modular Data Loading and Preprocessing
- **Target Ticker:** Commonwealth Bank of Australia (`CBA.AX`)
- **Report Date:** 22 June 2026

---

## Introduction
This report documents the design, implementation, and verification of the modular data loading and processing module (`src/data_processing.py`) completed for Task C.2. The module implements a robust data pipeline supporting multi-feature datasets, local caching, missing value (NaN) imputation, multiple train/test data splitting strategies (chronological ratio, chronological date, and random splits), and column-specific MinMaxScaler scaling. The module is fully verified to run successfully on the target `CBA.AX` stock dataset.

---

## 1. Data Pipeline Architecture & Implementation

The module is implemented in [src/data_processing.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/data_processing.py). It contains a primary function, `load_and_process_data`, structured into five distinct phases separated by professional comment dividers (`# =====` and `# -----`):

### 1.1 Phase 1: Local Caching & Data Loading
To prevent API rate-limiting issues (HTTP 429) and enable offline sandboxed execution, the pipeline implements a local caching mechanism:
* The loader looks for `{ticker}_cache.csv` or the legacy cache `{ticker}_2026-06-08.csv` in the `data/` directory.
* If a cache is found, the file is loaded locally via `pandas.read_csv`.
* If no cache exists, the downloader fetches live daily historical price data directly from Yahoo Finance via the `yfinance` library, saves it to the cache folder, and parses it.
* Column headers are converted to lowercase (`open`, `high`, `low`, `close`, `adjclose`, `volume`) to standardize remapping between different sources.
* Slices the dataframe based on the user-specified `start_date` and `end_date` bounds.

### 1.2 Phase 2: Missing Value (NaN) Imputation
Financial time-series data can contain missing values due to market holidays or data collection anomalies. The pipeline handles NaNs using:
1. **Forward Fill (`ffill`)**: Propagates the last observed valid price forward. This represents the most logical economic assumption (if the market was closed or data was missing, the asset price remains unchanged from the last transaction).
2. **Backward Fill (`bfill`)**: Used as a fallback to handle any NaNs present at the beginning of the series (before any valid prices have been observed).

### 1.3 Phase 3: Column-Specific Feature Scaling
Instead of applying a single scaler to the entire dataset (which would collapse the variance of high-volume columns or distort price ranges), features are scaled independently:
* Each column in the specified `feature_columns` is scaled to the range `[0, 1]` using a separate `MinMaxScaler` object.
* The MinMaxScaler objects are stored in a dictionary `column_scaler` keyed by column name (e.g. `column_scaler["adjclose"]`).
* Storing the scalers in a dictionary ensures they are preserved for future use, allowing unscaled predictions to be reconstructed during model evaluation without scaling leakage.

### 1.4 Phase 4: Sequence Construction
To feed data into recurrent neural networks (RNNs/LSTMs), 2D time-series indices must be restructured into 3D temporal arrays:
* Slices sliding sequences of length $N$ (e.g. 50 lookback days).
* Settle the prediction target by shifting `adjclose` forward by `lookup_step` days to represent future targets.
* Slices the tail sequence (`last_sequence`) containing the latest $N$ days of price data (used for future forecasts) before dropping target NaNs.
* Discards the final `lookup_step` rows (which contain NaNs in the shifted `future` column) to clean up training sequences.

### 1.5 Phase 5: Train/Test Splitting Strategies
To support rigorous model evaluation and prevent data leakage, the pipeline implements three split methods:
1. **Chronological split by ratio (`split_by_date=True, split_date=None`)**: Slices the first $X\%$ of temporal samples (e.g. 80%) for training and the remaining for testing.
2. **Chronological split by date (`split_by_date=True, split_date="YYYY-MM-DD"`)**: Slices all sequences before the split date for training, and all sequences after for testing.
3. **Random split (`split_by_date=False`)**: Uses scikit-learn's `train_test_split` to randomly partition samples.

---

## 2. Methodological Analysis & Justification

### 2.1 The Danger of Shuffled Random Splits (Data Leakage)
In standard machine learning tasks, a random split of training and testing data is preferred to ensure that both sets represent the same statistical distribution. However, in time-series forecasting, **random splitting is highly inappropriate and violates SOLID principles**:
* **Lookahead Bias / Temporal Leakage**: A random split destroys the chronological order of time-series data. If a sequence from day $t$ (consisting of observations from $t-49$ to $t$) is randomly assigned to the training set, and a sequence from day $t+1$ is assigned to the test set, the model will train on day $t+1$'s target during the training phase. When evaluated on the test set for day $t$, the model has already memorized the future price action, inflating the test performance.
* **Economic Fallacy**: In real-world trading, models must predict future prices using only past information. A model trained on a random split cannot be deployed in practice because it relies on looking ahead to future data points that do not yet exist.
* **Chronological split** is the only correct method for time-series validation. It trains the model strictly on past history and evaluates it on unseen future data.

### 2.2 Feature-Specific Scaling vs. Global Scaling
Applying a global scaler across all feature columns (e.g. scaling Close and Volume together) leads to numerical stability issues:
* Stock prices are valued in hundreds of dollars, while trading volume is measured in millions of shares.
* A global scaler would scale volume values to `[0, 1]`, which would squash all price values close to zero.
* Standardizing scaling per feature using independent `MinMaxScaler` objects guarantees that all variables reside within identical bounds `[0, 1]`, allowing the LSTM network to learn relationships across columns without gradients being dominated by high-magnitude variables.

---

## 3. Verification Log

To verify the correctness of the modular data processor, we ran a Python verification script that loads the cached `CBA.AX` dataset, processes it using a 50-day lookback sequence, and splits it chronologically with an 80/20 ratio:

```powershell
.venv\Scripts\python.exe -c "from src.data_processing import load_and_process_data; data = load_and_process_data('CBA.AX', '2020-01-01', '2024-07-02'); print(data['X_train'].shape); print(data['X_test'].shape); print(data['column_scaler'].keys())"
```

### Verification Console Output
```text
[Data Cache] Loading local cache: data\CBA.AX_cache.csv
(870, 50, 5)
(218, 50, 5)
dict_keys(['adjclose', 'volume', 'open', 'high', 'low'])
```

#### Terminal Execution Screenshot
![Terminal Execution Screenshot](screenshots/c2_terminal.png)

### Output Interpretation
1. **Cache Loading**: The pipeline successfully detected and loaded the cached file [CBA.AX_cache.csv](file:///s:/COS30018-Intelligent-System/fin-tech101/data/CBA.AX_cache.csv), avoiding API network requests.
2. **Train/Test Slices**: 
   * Slices 870 training sequences of shape `(50, 5)` (50 lookback days, 5 feature columns).
   * Slices 218 test sequences.
   * This matches the 80/20 chronological ratio split (total 1088 sequences).
3. **Scaler Retention**: The `column_scaler` dictionary contains five distinct `MinMaxScaler` objects, confirming that the pipeline scaled each column independently and stored the parameters for subsequent inverse scaling.

---

## 4. Conclusion
Task C.2 has been successfully completed. We implemented a robust, modular, and non-overengineered time-series data processing pipeline in `src/data_processing.py`. By isolating scaling parameters per feature and supporting strict chronological splits, the pipeline prevents data leakage and satisfies all requirements. The codebase is clean and ready for transition to the advanced visualizations of Task C.3.
