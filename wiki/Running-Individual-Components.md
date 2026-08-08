# Running Individual Components

## Purpose

Every script in `src/` can be run on its own during development, without executing the full pipeline. This page documents how to run each stage individually, what it produces, and which command-line flags are available for ad-hoc experiments outside the predefined sweep configurations.

All commands are run from the `src/` directory, since the project's modules import each other without a package prefix.

---

## Core Pipeline Stages

| Step | Command | Primary Output |
| :--- | :------- | :--------------- |
| Download market data | `python data_downloader.py` | `data/CBA.AX_cache.csv` |
| Prepare and scale the dataset | `python data_processing.py` | `results/c2/CBA_AX_scalers.pkl` |
| Generate visualisations | `python visualization.py` | `results/c3/` candlestick and boxplot charts |
| Train a model | `python train.py` | `results/<model_name>.weights.h5` |
| Evaluate a trained model | `python test.py` | `csv-results/<model_name>.csv`, `results/<model_name>_prediction.png` |
| Run the C.4 hyperparameter sweep | `python run_c4_sweeps.py` | `results/c4/c4_sweep_results.csv` |
| Run the C.5 multivariate/multistep sweep | `python run_c5_sweeps.py` | `results/c5/c5_sweep_results.csv` |
| Run the C.6 hybrid forecasting pipeline | `python run_c6.py` | `results/c6/`, `csv-results/c6/` |
| Run the full C.7 news sentiment workflow | `python run_c7.py` | `data/c7/`, `results/c7/`, `csv-results/c7/` |

`train.py` and `test.py` must be run with matching hyperparameter flags — the same `cell_type`, `n_layers`, `units`, and `feature_columns` used to train a model are required to reconstruct its architecture before its saved weights can be loaded in `test.py`.

---

## Ad-Hoc Training and Evaluation Flags

`train.py` and `test.py` share the same command-line arguments, defined once in `utils/experiment_utils.py` so both scripts stay in sync.

| Flag | Default | Description |
| :---- | :------ | :----------- |
| `--ticker` | `CBA.AX` | Stock ticker symbol |
| `--cell_type` | `LSTM` | Recurrent cell: `LSTM`, `GRU`, or `SimpleRNN` |
| `--n_layers` | `2` | Number of recurrent layers |
| `--units` | `128` | Hidden units per recurrent layer |
| `--dropout` | `0.3` | Dropout rate applied after each recurrent layer |
| `--loss` | `huber` | Loss function: `huber`, `mse`, or `mae` |
| `--lookback_steps` | `50` | Number of past trading days used as input |
| `--forecast_offset` | `1` | Days ahead the forecast targets |
| `--future_steps` | `1` | Number of future days predicted per sample |
| `--feature_columns` | `adjclose,volume,open,high,low` | Comma-separated input feature list |
| `--split_method` | `date` | Train/test split strategy: `date`, `ratio`, or `random` |
| `--validation_ratio` | `0.15` | Chronological validation split, relative to the training set |
| `--bidirectional` | off | Wraps recurrent layers in a `Bidirectional` wrapper when set |
| `--model_name` | `lstm_model` | Name used for saved weights, plots, and CSV output |
| `--subfolder` | *(none)* | Optional subfolder under `results/`/`csv-results/` |

`train.py` additionally accepts `--epochs` (default `20`) and `--batch_size` (default `64`).

**Example — train and evaluate a 3-layer GRU:**

```bash
python train.py --cell_type GRU --n_layers 3 --units 256 --epochs 30 --model_name my_gru
python test.py  --cell_type GRU --n_layers 3 --units 256 --model_name my_gru
```

---

## C.7 Scripts

The C.7 pipeline is split into ten independently runnable stages so that expensive steps (GDELT download, FinBERT inference) can be cached and re-run selectively rather than repeated on every run. `run_c7.py` runs them in order automatically; each can also be run standalone from `src/`:

`c7_news_data.py → c7_news_titles.py → c7_news_features.py → c7_news_alignment.py → c7_finbert_features.py --resume → c7_finbert_daily.py → c7_dataset.py → c7_preprocessing.py → c7_baseline.py`

`c7_finbert_features.py` supports a `--resume` flag so that a long-running FinBERT inference batch can be interrupted and continued from its last saved checkpoint rather than restarted from the beginning.

---

## Continue Exploring

- **Home** — Project background, scope, and objectives.
- **Environment Setup** — Development environment and dependency rationale.
- **Repository Structure** — Responsibilities of each project directory and source module.
- **Experiment Pipeline** — End-to-end workflow from data acquisition to model evaluation.
