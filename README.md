# FinTech101: Stock Price Prediction System

## Course Project: COS30018 - Intelligent Systems (Option C)

FinTech101 is an end-to-end, machine learning-based stock price prediction system developed as an individual project under the COS30018 Intelligent Systems curriculum. The project evaluates the limits and capabilities of deep learning models—specifically Stacked Long Short-Term Memory (LSTM) networks—in forecasting historical daily stock prices.

---

## 📅 Project Overview

This project is structured as a progression from initial environment setup and baseline verification, through data processing and exploratory data analysis, to advanced modeling techniques, ensembles, and custom supervised research extensions. 

For detailed information on individual tasks, weekly progress, and in-depth technical specifications, please refer to the **[Project Wiki](#project-wiki)**.

---

## 📂 Repository File Structure

The project maintains a flat, modular codebase designed for scalability:

```text
fin-tech101/
├── .gitignore               # Git exclusions
├── README.md                # You are here
├── requirements.txt         # Dependencies
├── baselines/               # Task C.1: Executable baselines
├── csv-results/             # Task outputs: CSV evaluations
├── data/                    # Local data cache for data processing
├── docs/                    # Assignment briefs
├── references/              # Untouched, original baseline codebases
├── reports/                 # Task C.1-C.7: Submission reports
├── results/                 # Model checkpoints & charts
├── src/                     
│   ├── base_sweep.py        # Task C.4/C.5: Sweep logic
│   ├── config.py            # Global hyperparameters
│   ├── data_downloader.py   # Task C.2: Data fetcher
│   ├── data_processing.py   # Task C.2: Dataset processing
│   ├── model_factory.py     # Task C.4: Network constructors
│   ├── run_c4_sweeps.py     # Task C.4: Baseline sweeps
│   ├── run_c5_sweeps.py     # Task C.5: Advanced sweeps
│   ├── test.py              # Task C.5/C.6: Evaluation
│   ├── train.py             # Task C.5/C.6: Training
│   └── visualization.py     # Task C.3: EDA charting
└── wiki/                    # Wiki documentation
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

### 1. Run Baseline Models
Execute the initial reference models to establish performance baselines:
```powershell
# v0.1 Baseline
python baselines/v0.1/stock_prediction.py

# P1 Baseline
python baselines/p1/train.py
python baselines/p1/test.py
```

### 2. Train the Active Modular Model
Trains the networks using settings specified in `src/config.py` and saves weights to `results/`.
```powershell
python src/train.py
```

### 3. Evaluate and Simulate Trading
Loads the trained weights, runs inferences on test data, prints unscaled performance metrics, performs simulated trading, and plots forecasts.
```powershell
python src/test.py
```

### 4. Run Hyperparameter Sweeps
Execute comprehensive parameter sweeps for advanced tasks:
```powershell
python src/run_c4_sweeps.py
python src/run_c5_sweeps.py
```

---

## 🛡️ API Rate-Limit Resilience
To bypass `HTTP 429 (Too Many Requests)` rate-limiting blocks from Yahoo Finance API, the codebase is equipped with a fallback caching mechanism. If the live downloader returns empty data, the system automatically checks for a locally cached dataset in the `data/` directory.

---

## 📖 Project Wiki

For task breakdowns, architectural blueprints, error-handling methodologies, and weekly academic reports, please consult the documentation set in the `wiki/` directory. This is designed for direct upload to the GitHub Wiki interface:

- **[Home](../../wiki/Home)**: Main hub and quickstart.
- **[System Architecture](../../wiki/System-Architecture)**: Design blueprints and execution logic flow.
- **[Environment Setup](../../wiki/Environment-Setup)**: Local installation guides and API rate-limiting workarounds.
- **[Weekly Reports](../../wiki/Weekly-Reports-Hub)**: Academic deliverables tracking sheet.
