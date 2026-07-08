# FinTech101: A Modular Deep Learning Framework for Stock Price Forecasting

## 📌 Overview

FinTech101 is a modular deep learning project for forecasting stock prices from historical market data. It provides a reusable workflow for preparing datasets, developing forecasting models, evaluating prediction performance, and comparing experimental results.

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

---

## 📂 Repository Structure

The repository separates implementation code, datasets, generated outputs, reference code, and documentation so that each part of the project has a clearly defined responsibility.

```text
fin-tech101/
├── baselines/                  # Modified, executable Option C codebases
├── references/                 # Original Option C codebases
├── data/                       # Cached datasets
├── docs/                       # Assignment resources
├── results/                    # Generated artefacts (c1-c6 weights/plots)
├── csv-results/                # Evaluation outputs (c1, c4, c5, c6 CSVs)
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
│   ├── run_c6_ensemble.py      # Preliminary ensembling experiments
│   └── run_c6_hybrid.py        # Hybrid residual-learning sweeps
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
python -m venv .venv
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
python src/run_c6_hybrid.py
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
| **C.6 Ensemble Learning**    | `src/run_c6_hybrid.py`, `src/run_c6_ensemble.py`                                                   | Hybrid residual-learning forecasting pipeline, prediction plots, model weights, and consolidated evaluation metrics.              |
| **C.7 Independent Research** | *(Planned)*                                                                                        | Research extension implementation and supporting artefacts                         |

---

## 📖 Documentation

Comprehensive technical documentation is maintained separately through the project's GitHub Wiki. The Wiki expands on the implementation, design decisions, and experimental work summarised throughout this repository.

Available documentation includes:

* **Home** — Project background, scope, and objectives.
* **Repository Structure** — Responsibilities of each project directory and source module.
* **System Architecture** — High-level architecture and module interactions.
* **Environment Setup** — Development environment and dependency rationale.
* **Running Individual Components** — Usage guide for each executable script.
* **Experimental Pipeline** — End-to-end workflow from data acquisition to model evaluation.
* **Evaluation Metrics** — Definitions and interpretation of all reported metrics.
* **Option C Task Mapping** — Traceability between project requirements and implementation.
* **Weekly Reports** — Archived reports for Tasks C.1–C.7 documenting weekly progress.
