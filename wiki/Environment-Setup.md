# Environment Setup

## Purpose

This page documents the software environment FinTech101 requires and why each major dependency is included. It exists so the project can be reproduced on a new machine without re-deriving these choices from `requirements.txt` alone.

---

## Requirements

FinTech101 targets **Python 3.12**, run inside an isolated virtual environment so its dependencies do not conflict with other projects on the same machine.

| Package | Version | Why it is needed |
| :------- | :------- | :----------------- |
| `tensorflow` | 2.17.1 | Builds and trains the LSTM/GRU/SimpleRNN models in `model_factory.py` |
| `numpy` | 1.26.4 | Pinned to match TensorFlow 2.17.1's supported NumPy range |
| `pandas` | 2.2.3 | Dataframe handling throughout data processing, evaluation, and CSV export |
| `scikit-learn` | 1.5.2 | `MinMaxScaler`/`StandardScaler`, chronological/random splitting, and the C.7 Logistic Regression classifier |
| `matplotlib` | 3.9.2 | Prediction charts, candlestick summaries, and confusion matrix plots |
| `mplfinance` | ≥0.12.10b0 | Candlestick chart rendering for Task C.3 |
| `yfinance` | 0.2.48 | Historical market data download |
| `yahoo-fin` | 0.8.9.1 | Supplementary market data utilities |
| `google-cloud-bigquery`, `pyarrow`, `db-dtypes` | latest | Query and cache GDELT news records from BigQuery for Task C.7 |
| `transformers`, `torch` | latest | Run FinBERT sentiment inference for Task C.7 |

The full pinned list is maintained in `requirements.txt` at the repository root; it is the source of truth if this table and the file ever diverge.

---

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Quynh-Trang12/fin-tech101.git
cd fin-tech101
```

### 2. Create and Activate a Virtual Environment

```bash
python -3.12 -m venv .venv
```

*Windows*
```bash
.venv\Scripts\activate
```

*macOS / Linux*
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Why a Single Shared Environment

`v0.1` and `P1`, the two Task C.1 baseline code bases, were deliberately set up to share one virtual environment rather than two separate ones. Both use overlapping libraries (TensorFlow, scikit-learn, yfinance), and a shared environment made it possible to compare their behaviour on the same dataset under the same library versions — removing library version differences as a confound when judging which baseline produced a "better" prediction.

The same principle carries forward through the rest of the project: every task from C.2 onward runs inside this one environment, so a change in results between tasks reflects a change in the pipeline or model, not a change in the underlying software stack.

---

## Continue Exploring

- **Home** — Project background, scope, and objectives.
- **Repository Structure** — Responsibilities of each project directory and source module.
- **Running Individual Components** — Usage guide for each executable script.
- **System Architecture** — High-level architecture and module interactions.
