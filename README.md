# FinTech101: Stock Price Prediction System

## Course Project: COS30018 - Intelligent Systems (Option C)

FinTech101 is an end-to-end, machine learning-based stock price prediction system developed as an individual project under the COS30018 Intelligent Systems curriculum. The project evaluates the limits and capabilities of deep learning models—specifically Stacked Long Short-Term Memory (LSTM) networks—in forecasting historical daily stock prices.

---

## 📅 Project Roadmap & Marking Scheme

The project is structured into seven distinct weekly tasks progressing from setup and baseline verification to advanced modeling, ensembles, and a custom supervised extension:

| Task ID | Description | Deadline | Weight | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Task C.1** | Environment setup, baseline verification (v0.1), and reference model (P1) validation. | Week 3 | 10% | **Completed** |
| **Task C.2** | Data Processing 1: Multi-feature dataset loading, chronological train/test splitting, and local CSV caching. | Week 4 | 10% | *Next Phase* |
| **Task C.3** | Data Processing 2: Multi-dimensional visualizations and candlestick charts export for image analysis. | Week 5 | 10% | *Planned* |
| **Task C.4** | Machine Learning 1: General model constructor supporting RNN, GRU, and Feed-Forward networks. | Week 6 | 10% | *Planned* |
| **Task C.5** | Machine Learning 2: Advanced modeling including multivariate and multi-step forecasting. | Week 7 | 15% | *Planned* |
| **Task C.6** | Machine Learning 3: Ensemble techniques and combined model approaches. | Week 9 | 15% | *Planned* |
| **Task C.7** | Project Extension: Custom supervised research extension (supervised by Project Leader). | Week 12 | 30% | *Planned* |

---

## 📂 Repository File Structure

The project maintains a flat, modular codebase in the `src/` directory designed for scalability across all subsequent tasks:

```text
fin-tech101/
├── .gitignore               # Excludes virtual environments, cache, checkpoints, and screenshots
├── README.md                # Project landing page and guide (this file)
├── requirements.txt         # Package dependency manifest
├── csv-results/             # Evaluation metrics and predicted price CSV exports
├── data/                    # Local CSV market data cache (resilient to API rate-limits)
├── docs/                    # Official assignment briefs and project specifications
├── logs/                    # Training log history (TensorBoard compatible)
├── references/              # Untouched reference codebases
│   ├── P1/                  # Original multi-file GitHub reference project
│   └── v0.1/                # Original single-file baseline stock prediction script
├── reports/                 # Formatted weekly submission reports and output figures
│   ├── task_c1/             # Task C.1 Submission Report and terminal screenshots
│   └── task_c2/             # Task C.2 Submission directory
├── results/                 # Model checkpoints (.h5) and forecast charts (.png)
├── src/                     # Core system implementation (Flat & Modular)
│   ├── parameters.py        # Centralized hyperparameter declarations
│   ├── stock_prediction.py  # Core data processing utilities and model constructors
│   ├── train.py             # Model training execution pipeline
│   ├── test.py              # Model testing, evaluation metrics, and trading simulation
│   └── v0.1/                # Running copy of baseline code (fixed for headless & caching)
└── wiki/                    # GitHub Wiki documentation pages
```

---

## ⚙️ Environment Setup & Installation

### Prerequisite
- Python 3.12 (Recommended) or 3.11

### Installation Commands
Run the following commands in your terminal (PowerShell for Windows) to initialize the virtual environment and install the required dependencies:

```powershell
# 1. Create a virtual environment named .venv
py -3.12 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade the package installer
python -m pip install --upgrade pip

# 4. Install project packages
pip install -r requirements.txt
```

### Dependency Stack
- `tensorflow==2.17.1`: Deep learning neural network constructor
- `numpy==1.26.4`: High-performance numerical computations
- `pandas==2.2.3`: Structured data frames and time-series indexes
- `scikit-learn==1.5.2`: MinMaxScaler and dataset preprocessing
- `matplotlib==3.9.2`: Data visualization plotting engine
- `yfinance==0.2.48`: Live stock market API integration

---

## 🚀 Execution Guide

Make sure your virtual environment is active (`(.venv)`) before executing the scripts.

### 1. Run Baseline v0.1 Model
Executes the initial univariate model prediction workflow, saving the prediction plot to `results/v01_prediction.png`.
```powershell
python src/v0.1/stock_prediction.py
```

### 2. Train the Modular Model (P1-based)
Trains the Stacked LSTM network using multivariate parameters specified in `src/parameters.py` and saves the optimal weights to `results/`.
```powershell
python src/train.py
```

### 3. Evaluate and Simulate Trading
Loads the trained weights, runs inferences on test data, prints performance metrics (MAE, RMSE, MAPE, Directional Accuracy), performs simulated trading, and plots forecasts.
```powershell
python src/test.py
```

---

## 🛡️ API Rate-Limit Resilience
To bypass `HTTP 429 (Too Many Requests)` rate-limiting blocks from Yahoo Finance API, the codebase is equipped with a fallback caching mechanism. If the live downloader returns empty data, the system automatically checks for a locally cached dataset in `data/CBA.AX_2026-06-08.csv` or falls back to generating a deterministic, mathematically aligned synthetic market dataset.

---

## 🛠️ P1 Mathematical Bug Corrected
In the reference project (`references/P1/`), the test script evaluated the model by passing the Mean Absolute Error (MAE) directly into the `inverse_transform` function of the scaler:
$$\text{MAE}_{\text{buggy}} = \text{scaler.inverse\_transform}(\text{MAE}_{\text{scaled}})$$
This resulted in adding the minimum price (~$80) directly to the unscaled error, producing a buggy MAE of `79.81` dollars.

This system resolves this bug inside `src/test.py` by first inverse-scaling the prediction and actual prices, and then calculating the metrics directly on unscaled values:
$$\text{MAE}_{\text{corrected}} = \frac{1}{N}\sum |y_{\text{unscaled}} - \hat{y}_{\text{unscaled}}|$$
This yields the true MAE of `0.4231` dollars.

---

## 📖 Project Wiki
A structured documentation set is maintained in the `wiki/` directory. This is designed for direct upload to the GitHub Wiki interface:
- **[Home](../../wiki/Home.md)**: Main hub and quickstart.
- **[System Architecture](../../wiki/System-Architecture)**: Design blueprints and execution logic flow.
- **[Environment Setup](../../wiki/Environment-Setup)**: Local installation guides and API rate-limiting workarounds.
- **[Weekly Reports](../../wiki/Weekly-Reports-Hub)**: Academic deliverables tracking sheet (links to Task C.1-C.7 reports).
