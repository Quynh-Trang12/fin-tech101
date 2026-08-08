# Repository Structure

## Purpose

FinTech101 separates implementation code, datasets, generated outputs, reference material, and documentation so that each part of the project has a clearly defined responsibility. This page explains what belongs in each top-level directory and why, so that new code lands in the right place and existing code is easy to find.

---

## Top-Level Layout

| Directory | Responsibility |
| :--------- | :-------------- |
| `src/` | All executable pipeline code: data acquisition, preprocessing, model construction, training, evaluation, experiment sweeps, and the C.6/C.7 extensions |
| `data/` | Cached datasets, including GDELT and FinBERT intermediate files under `data/c7/` |
| `results/` | Generated artefacts such as trained model weights and prediction plots, organised into task subfolders (`results/c4/`, `results/c6/`, etc.) |
| `csv-results/` | Evaluation outputs in CSV form, mirroring the same task subfolder structure as `results/` |
| `reports/` | The Task C.1–C.7 weekly reports and their supporting screenshots |
| `wiki/` | Project documentation, mirrored to the GitHub Wiki |
| `baselines/` | Modified, executable versions of the original Option C code bases (v0.1, P1) — reference only |
| `references/` | The original, unmodified Option C code bases — reference only |

`baselines/` and `references/` are kept untouched deliberately: they document the starting point of the project so that improvements made throughout C.1–C.7 can be compared against it.

---

## Source Code Organisation

`src/` is organised by pipeline responsibility rather than by task number, so that a component built for one task can be reused by a later one without duplication.

| Module | Responsibility |
| :------ | :--------------- |
| `config.py` | Shared settings: ticker, date ranges, split configuration, feature columns, and sweep configurations |
| `data_downloader.py` | Downloads and caches historical market data |
| `data_processing.py` | Leakage-safe preprocessing: cleaning, sliding-window construction, chronological splitting, and scaling |
| `visualization.py` | Candlestick and boxplot chart generation |
| `model_factory.py` | Builds and compiles LSTM/GRU/SimpleRNN models from hyperparameters |
| `train.py` | Trains a model and saves its weights |
| `test.py` | Loads trained weights, runs inference, and computes evaluation metrics |
| `base_sweep.py` | Reusable base class for running experiment sweeps |
| `run_c4_sweeps.py`, `run_c5_sweeps.py` | Task C.4/C.5 sweep orchestrators, built on `base_sweep.py` |
| `run_c6.py` | Statistical (ARIMA) and residual deep learning hybrid forecasting pipeline |
| `run_c7.py`, `c7_*.py` | News sentiment data acquisition, feature extraction, dataset construction, and classification pipeline |
| `utils/experiment_utils.py` | Shared helpers: seeding, CLI argument parsing, feature column parsing |

The `c7_*.py` modules are split by pipeline stage (news download, title enrichment, V2Tone aggregation, trading-day alignment, FinBERT inference, dataset merging, preprocessing, and modelling) rather than combined into one script, so each stage can be re-run and cached independently during a long-running data collection process.

---

## Continue Exploring

- **Home** — Project background, scope, and objectives.
- **System Architecture** — High-level architecture and module interactions.
- **Experiment Pipeline** — End-to-end workflow from data acquisition to model evaluation.
- **Option C Task Mapping** — Traceability between project requirements and implementation.
