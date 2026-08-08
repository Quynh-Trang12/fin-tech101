# Option C Task Mapping

## Purpose

This page traces each Project Option C task requirement to its primary implementation in the repository and the outputs that demonstrate it was completed. It is the reference point for checking that every graded task has corresponding code and evidence.

---

## Task-to-Implementation Traceability

| Task | Primary Implementation | Primary Outputs |
| :--- | :---------------------- | :--------------- |
| **C.1 Environment Setup** | `baselines/`, `references/`, `requirements.txt`, `README.md` | Local development environment, GitHub repository, and project documentation |
| **C.2 Data Processing** | `src/data_downloader.py`, `src/data_processing.py`, `src/config.py` | Cached datasets, fitted feature scalers, and preprocessing artefacts |
| **C.3 Data Visualisation** | `src/visualization.py` | Candlestick charts and moving boxplots |
| **C.4 Deep Learning Models** | `src/model_factory.py`, `src/train.py`, `src/test.py`, `src/base_sweep.py`, `src/run_c4_sweeps.py` | Trained model weights, prediction results, evaluation metrics, and sweep summaries |
| **C.5 Advanced Forecasting** | `src/data_processing.py`, `src/test.py`, `src/run_c5_sweeps.py` | Multivariate and multi-step forecasting experiment results |
| **C.6 Ensemble Learning** | `src/run_c6.py` | Hybrid residual-learning forecasting pipeline, prediction plots, model weights, and consolidated evaluation metrics |
| **C.7 Independent Research** | `src/run_c7.py`, `src/c7_news_data.py`, `src/c7_news_titles.py`, `src/c7_news_features.py`, `src/c7_news_alignment.py`, `src/c7_finbert_features.py`, `src/c7_finbert_daily.py`, `src/c7_dataset.py`, `src/c7_preprocessing.py`, `src/c7_baseline.py` | Parsed daily and aligned news datasets, FinBERT prediction caches, standardised Logistic Regression classifiers, metrics comparisons, and confusion matrices |

---

## Reading This Table Alongside the Architecture

**System Architecture** groups these same modules by responsibility (data processing, model construction, training, evaluation) rather than by task number, and explains how C.6 and C.7 reuse pieces of the core forecasting workflow without following its train-then-evaluate shape. Use this page to find where a task lives in the codebase; use System Architecture to understand why it is organised that way.

Each task's detailed implementation notes, screenshots, and evaluation discussion are documented in its corresponding entry on **Weekly Reports**.

---

## Continue Exploring

- **Home** — Project background, scope, and objectives.
- **System Architecture** — High-level architecture and module interactions.
- **Experiment Pipeline** — End-to-end workflow from data acquisition to model evaluation.
- **Weekly Reports** — Archived reports for Tasks C.1–C.7 documenting weekly progress.
