# FinTech101: A Modular Machine Learning Framework for Stock Price Prediction

## 📌 Overview

FinTech101 is a modular machine learning framework for stock price prediction that combines historical market data with external financial news and financial analysis. The project provides reusable pipelines for data acquisition, preprocessing, visualization, deep learning, statistical forecasting, hybrid models, and sentiment-enhanced classification.

The project was developed for **COS30018 – Intelligent Systems (Project Option C)** by progressively extending the original **v0.1** and **P1** implementations into a maintainable and extensible machine learning codebase.

---

## 🎯 Objectives and Key Capabilities

The objective of FinTech101 is to investigate how different statistical and deep learning forecasting approaches influence stock price prediction while building a reusable project that supports fair, reproducible, and extensible machine learning experiments.

### 1. Reusable Experiment Pipeline

Provides a consistent workflow for downloading, preparing, and reusing historical market data across every experiment.

### 2. Configurable Models

Builds recurrent neural networks from configurable parameters instead of maintaining separate implementations for each model architecture.

### 3. Experiment Automation

Runs repeatable training, evaluation, and experiment sweeps through a shared execution pipeline.

### 4. Comprehensive Performance Evaluation

Compares forecasting models using prediction accuracy, price movement direction, and trading-oriented performance metrics.

### 5. Modular Architecture

Organises the forecasting workflow into reusable components that can be maintained and extended independently.

### 6. External Sentiment Integration

Incorporates external financial news from GDELT, employing rule-based V2Tone sentiment and transformer-based FinBERT classifications to improve stock price direction prediction.

---

## 📂 Repository Structure

The repository separates implementation code, datasets, generated outputs, reference code, and documentation so that each part of the project has a clearly defined responsibility.

```text
fin-tech101/
├── baselines/                  # Modified, executable Option C codebases
├── references/                 # Provided, untouched Option C codebases
├── data/                       # Cached datasets
│   └── c7/                     # C.7 intermediate and aligned datasets
├── results/                    # Generated artefacts (weights/plots)
├── csv-results/                # Evaluation outputs (CSVs)
├── src/                        # Core pipeline
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py         # Package marker
│   │   └── experiment_utils.py # Pipeline helpers
│   ├── config.py               # Shared settings
│   ├── data_downloader.py      # Data download
│   ├── data_processing.py      # Data preparation
│   ├── visualization.py        # Data visualisation
│   ├── model_factory.py        # Model builder
│   ├── train.py                # Model training
│   ├── test.py                 # Model evaluation
│   ├── base_sweep.py           # Sweep framework
│   ├── run_c4_sweeps.py        # Hyperparameter sweeps
│   ├── run_c5_sweeps.py        # Advanced forecasting
│   ├── run_c6.py               # Hybrid residual-learning and ensemble sweeps
│   ├── run_c7.py               # C.7 workflow orchestrator
│   ├── c7_news_data.py         # GDELT GKG news downloader
│   ├── c7_news_titles.py       # GDELT GDG headline enrichment joiner
│   ├── c7_news_features.py     # V2Tone daily sentiment aggregator
│   ├── c7_news_alignment.py    # V2Tone timezone searchsorted aligner
│   ├── c7_finbert_features.py  # FinBERT article headlines sentiment inference
│   ├── c7_finbert_daily.py     # FinBERT timezone daily aggregation & alignment
│   ├── c7_dataset.py           # Final classification dataset builder
│   ├── c7_preprocessing.py     # Chronological train/val/test data splitter
│   ├── c7_baseline.py          # Logistic Regression experiments sweep
│   ├── c7_feature_audit.py     # Diagnostics collinearity & VIF audit
│   ├── c7_v2tone_experiment.py # Baseline V2Tone classification experiment
│   └── c7_reduced_v2tone_experiment.py # Validation-set reduced V2Tone experiment
├── requirements.txt            # Project dependencies
└── README.md                   # You are here
```

---

## ⚙️ Getting Started

FinTech101 is designed to run entirely on a local machine. Before running any experiments, clone the repository, create a Python virtual environment, and install the project dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/Quynh-Trang12/fin-tech101.git
cd fin-tech101
```

### 2. Create and Activate Virtual Environment

**Create a virtual enviroment:**
```bash
python -3.12 -m venv .venv
```

**Activate the virtual environment:**

*Windows*
```bash
.venv\Scripts\activate
```

*macOS / Linux*
```bash
source .venv/bin/activate
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Project

A typical experiment follows the execution pipeline below. Each stage produces the primary inputs for the following stage of the stock forecasting pipeline.

### 4. Download Historical Market Data
```bash
python src/data_downloader.py
```
**Primary Output**
`data/CBA.AX_cache.csv`: Cached historical market dataset.


### 5. Prepare the Dataset
```bash
python src/data_processing.py
```
**Primary Output**
`results/c2/CBA_AX_scalers.pkl`: Training-fitted feature scalers reused throughout the forecasting pipeline.


### 6. Generate Market Visualisations
```bash
python src/visualization.py
```
**Primary Output**
`results/c3/`: Candlestick charts and moving boxplots for historical market analysis.


### 7. Train a Stock Forecasting Model
```bash
python src/train.py
```
**Primary Output**
`results/lstm_model.weights.h5`: Trained model weights, prediction outputs, and training history.


### 8. Evaluate Model Performance
```bash
python src/test.py
```
**Primary Output**
`csv-results/lstm_model.csv` and `results/lstm_model_prediction.png`: Prediction CSV files, evaluation metrics, and prediction plots.


### 9. Compare Recurrent Model Architectures and Hyperparameter Configurations
```bash
python src/run_c4_sweeps.py
```
**Primary Output**
`results/c4/`: Hyperparameter sweep summaries and model comparison results.


### 10. Run Multivariate and Multi-step Forecasting Experiments
```bash
python src/run_c5_sweeps.py
```
**Primary Output**
`results/c5/`: Multivariate and multi-step forecasting experiment results.


### 11. Run Hybrid Residual-Learning Experiments

```bash
python src/run_c6.py
```

**Primary Outputs**

- `results/c6/`
  - Residual-learning LSTM model weights
  - ARIMA baseline prediction plots
  - Hybrid residual-learning prediction plots
  - Experiment summary (`c6_hybrid_summary.md`)

- `csv-results/c6/`
  - Consolidated forecasting evaluation metrics
  - Simulated trading performance metrics


### 12. Run Task C.7 Sentiment-Enhanced Classification Experiments

```bash
python src/run_c7.py
```

**Primary Outputs**

- `data/c7/`
  - Final classification dataset (`c7_dataset.parquet`)
  - Aligned daily sentiment features (`gdelt_v2tone_aligned.parquet`, `gdelt_finbert_aligned.parquet`)

- `csv-results/c7/`
  - Consolidated comparative metrics (`c7_logistic_comparison.csv`)

- `results/c7/`
  - Test-set evaluation confusion matrices (PNGs)


---

## 📔 Option C Task Mapping

The table below maps each Project Option C task to its primary implementation and the corresponding outputs produced by the project.

| Task                         | Primary Implementation                                                                             | Primary Outputs                                                                    |
| :--------------------------- | :------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **C.1 Environment Setup**    | `baselines/`, `references/`, `requirements.txt`, `README.md`                                       | Local development environment, GitHub repository, and project documentation        |
| **C.2 Data Processing**      | `src/data_downloader.py`, `src/data_processing.py`, `src/config.py`                                | Cached datasets, fitted feature scalers, and preprocessing artefacts               |
| **C.3 Data Visualisation**   | `src/visualization.py`                                                                             | Candlestick charts and moving boxplots                                             |
| **C.4 Deep Learning Models** | `src/model_factory.py`, `src/train.py`, `src/test.py`, `src/base_sweep.py`, `src/run_c4_sweeps.py` | Trained model weights, prediction results, evaluation metrics, and sweep summaries |
| **C.5 Advanced Forecasting** | `src/data_processing.py`, `src/test.py`, `src/run_c5_sweeps.py`                                    | Multivariate and multi-step forecasting experiment results                         |
| **C.6 Ensemble Learning**    | `src/run_c6.py`                                                                                    | Hybrid residual-learning forecasting pipeline, prediction plots, model weights, and consolidated evaluation metrics.              |
| **C.7 Independent Research** | `src/run_c7.py`, `src/c7_news_data.py`, `src/c7_news_titles.py`, `src/c7_news_features.py`, `src/c7_news_alignment.py`, `src/c7_finbert_features.py`, `src/c7_finbert_daily.py`, `src/c7_dataset.py`, `src/c7_preprocessing.py`, `src/c7_baseline.py` | Parsed daily and aligned news datasets, FinBERT prediction caches, standardized Logistic Regression classifiers, metrics comparisons, and confusion matrices. |

---

## 📖 Documentation

Comprehensive technical documentation is maintained separately through the project's GitHub Wiki. The Wiki expands on the implementation, design decisions, and experimental work summarised throughout this repository.

Available documentation includes:

* **[Home](https://github.com/Quynh-Trang12/fin-tech101/wiki/)** — Project background, scope, and objectives.
* **[Repository Structure](https://github.com/Quynh-Trang12/fin-tech101/wiki/Repository-Structure)** — Responsibilities of each project directory and source module.
* **[System Architecture](https://github.com/Quynh-Trang12/fin-tech101/wiki/FinTech101-System-Architecture)** — High-level architecture and module interactions.
* **[Environment Setup](https://github.com/Quynh-Trang12/fin-tech101/wiki/Environment-Setup)** — Development environment and dependency rationale.
* **[Running Individual Components](https://github.com/Quynh-Trang12/fin-tech101/wiki/Running-Individual-Components)** — Usage guide for each executable script.
* **[Experimental Pipeline](https://github.com/Quynh-Trang12/fin-tech101/wiki/Experimental-Pipeline)** — End-to-end workflow from data acquisition to model evaluation.
* **[Evaluation Metrics](https://github.com/Quynh-Trang12/fin-tech101/wiki/Evaluation-Metrics)** — Definitions and interpretation of all reported metrics.
* **[Option C Task Mapping](https://github.com/Quynh-Trang12/fin-tech101/wiki/Option-C-Task-Mapping)** — Traceability between project requirements and implementation.
* **[Weekly Reports](https://github.com/Quynh-Trang12/fin-tech101/wiki/Weekly-Reports)** — Archived reports for Tasks C.1–C.7 documenting weekly progress.
