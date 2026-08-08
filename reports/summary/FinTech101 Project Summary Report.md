# Project Summary Report (Option C): FinTech101 — Stock Price Prediction System

- **Subject:** COS30018 – Intelligent Systems
- **Unit Convenor:** Dr. Nguyen Phong Son
- **Group member:**
  - Nguyen Hoang Dung (105724554)
  - Le Ngoc Quynh Trang (105028463)
  - Nguyen Huu Hieu (104988566)

---

## Table of Contents

1. Introduction
2. Overall System Architecture
3. Implemented Data Processing Techniques
4. Experimented Machine Learning Techniques
5. Task C.7 Extension: Sentiment-Based Classification
6. Scenarios and Examples
7. Critical Analysis
8. Summary and Conclusion

---

## 1. Introduction

FinTech101 is a stock price prediction system built to investigate how different statistical and deep learning forecasting approaches perform on a real equity time series. The project began from two starting points supplied for Task C.1: a single-file baseline (`v0.1`) using an LSTM model on closing price alone, and a more modular reference project (`P1`) with separated data, model, and evaluation logic. Both were tested, compared, and used to motivate the redesign carried out across Tasks C.2–C.7.

The target instrument throughout the project is Commonwealth Bank of Australia (`CBA.AX`), chosen so that every task builds on the same dataset and every model comparison is fair. Over seven tasks, the project progressed from a single hardcoded model to a configurable multi-architecture forecasting framework (C.2–C.5), a statistically justified hybrid forecasting approach combining ARIMA with residual deep learning (C.6), and an independent extension that tests whether financial news sentiment improves next-day price direction prediction (C.7).

This report summarises the system as a whole. Detailed weekly implementation notes, code-level explanations, and screenshots for each task are maintained separately in the project's weekly reports and GitHub Wiki, referenced throughout this report rather than reproduced here.

---

## 2. Overall System Architecture

### 2.1 Design Principle

The project is organised as a single, reusable stock forecasting pipeline rather than as seven disconnected task implementations. Data processing, model construction, training, and evaluation are each isolated into their own module so that later tasks can reuse earlier components instead of duplicating logic. This decision was made directly in response to the baseline codebases reviewed in Task C.1: `v0.1` mixed data loading, model definition, training, and plotting in a single file, which made it difficult to change one part of the pipeline (for example, adding a new feature column) without risk of breaking another.

### 2.2 Core Forecasting Workflow (C.1–C.5)

The core workflow follows a fixed path: **data source → data processing → model construction → model training → model evaluation → results**.

- `data_processing.py` loads historical price data, cleans it, constructs sliding-window input sequences and forecasting targets, performs a chronological train/test split, and fits feature scalers strictly on the training partition to avoid data leakage.
- `model_factory.py` builds a compiled LSTM, GRU, or SimpleRNN model from a set of hyperparameters (cell type, depth, width, dropout, loss function), decoupling model architecture from the training logic that consumes it.
- `train.py` and `test.py` orchestrate fitting and evaluation respectively, so that a model can be evaluated repeatedly without being retrained.
- `base_sweep.py`, together with `run_c4_sweeps.py` and `run_c5_sweeps.py`, automates running many model configurations through this same workflow, so that Task C.4's hyperparameter comparisons and Task C.5's multivariate/multistep experiments produce directly comparable results.
- `config.py` centralises every dataset, model, and sweep parameter, so that a configuration change is made once and applied consistently everywhere it is used.

### 2.3 Extensions Beyond the Core Workflow (C.6–C.7)

Task C.6 and Task C.7 reuse specific components of the core workflow but do not follow its train-then-evaluate shape, because neither task fits that shape by design.

`run_c6.py` reuses `data_processing.py` for its univariate input windows and `model_factory.py` for its LSTM/GRU builder, but trains and evaluates in a single inline sequence: fit an ARIMA baseline, train a deep learning model on the ARIMA model's residuals, then combine both predictions. This differs from the core workflow's separated train/test scripts because the residual learner's training target (the ARIMA error) depends on the ARIMA model's output, so the two models cannot be trained independently.

The Task C.7 `c7_*.py` scripts replace `data_processing.py` entirely with a dedicated news-processing chain, because their input is financial news text rather than price history. This chain downloads GDELT news records, extracts sentiment features using two independent methods (GDELT's V2Tone and FinBERT), aligns them to Australian trading days, and merges them with market data into a labelled classification dataset. The final model is a Logistic Regression classifier predicting price direction, not a recurrent price forecaster, because C.7 reframes the problem as binary classification rather than regression.

Full architecture diagrams, component responsibility tables, and the six formal architectural decisions behind this design are documented in the **System Architecture** and **Experiment Pipeline** Wiki pages.

---

## 3. Implemented Data Processing Techniques

### 3.1 Multi-Feature, Configurable Data Loading (C.2)

The Task C.1 baseline (`v0.1`) used only the closing price as input and required the user to manually choose separate start/end dates for training and testing data. Task C.2 replaced this with `load_and_process_data()`, a single configurable entry point that:

- Accepts a single overall date range and derives train/test partitions from it automatically, rather than requiring four manually chosen dates.
- Supports multiple feature columns (`adjclose`, `volume`, `open`, `high`, `low`) instead of closing price alone.
- Handles missing values before sequence construction.
- Supports three train/test split strategies — chronological by date, chronological by ratio, and random — selected through a single `split_method` parameter.
- Caches downloaded data locally (`data/<TICKER>_cache.csv`) so repeated runs do not re-query the data source.
- Fits and stores `MinMaxScaler` objects per feature column, so that scalers used during training can be reloaded for consistent inverse-transformation during evaluation.

### 3.2 Leakage-Safe Ordering

A specific ordering is enforced to prevent information from the test period leaking into training: forecasting targets and sliding windows are constructed first, the chronological split is then performed by the *target* date rather than the input date, and scalers are fitted only on the resulting training partition before being applied to both partitions. This ordering was adopted because splitting by input date alone can allow a window whose *target* falls after the split boundary to still be included in training, which would let the model see outcomes from the test period indirectly.

### 3.3 Data Visualisation (C.3)

Task C.3 added two visualisation techniques to support exploratory analysis of the processed dataset: multi-day candlestick charts, which aggregate a configurable number of trading days per candle, and moving-window boxplots, which summarise the price distribution over a rolling window of trading days. Both were implemented as reusable functions in `visualization.py` and are used to sanity-check the dataset before it is used for training.

---

## 4. Experimented Machine Learning Techniques

### 4.1 Configurable Deep Learning Models (C.4)

Task C.4 replaced the single hardcoded LSTM architecture from `v0.1` with `model_factory.py`, which dynamically builds LSTM, GRU, or SimpleRNN networks from hyperparameters (cell type, number of layers, units per layer, dropout rate, loss function). Ten configurations were trained and evaluated under the same experimental conditions using `run_c4_sweeps.py`; the key results are summarised in Table 1.

**Table 1 — Task C.4 Hyperparameter Sweep Results** (selected columns; full results in `results/c4/c4_sweep_results.csv`)

| Configuration | Cell | Layers | Units | MAE ($) | RMSE ($) | Directional Acc. (%) |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: |
| GRU_BASE | GRU | 2 | 128 | **2.2256** | **2.5751** | 44.83 |
| LSTM_STACKED | LSTM | 3 | 128 | 2.6520 | 3.2555 | 44.40 |
| LSTM_SMALLBATCH | LSTM | 2 | 128 | 2.7512 | 3.4132 | 43.97 |
| LSTM_BASE | LSTM | 2 | 128 | 2.9312 | 3.4940 | 44.83 |
| LSTM_MSE | LSTM | 2 | 128 | 2.9251 | 3.4876 | 44.83 |
| LSTM_WIDE | LSTM | 2 | 256 | 3.4934 | 4.0006 | **45.69** |
| RNN_BASE | SimpleRNN | 2 | 128 | 3.7622 | 4.4339 | 46.12 |
| LSTM_SHALLOW | LSTM | 1 | 128 | 4.3150 | 4.6711 | 45.26 |

The GRU cell produced the lowest forecasting error among all ten configurations, ahead of every LSTM variant tested, despite GRU's simpler internal gating structure. Directional accuracy varied only narrowly across configurations (43.97%–46.12%), which suggests that hyperparameter tuning alone had limited effect on the model's ability to predict price *direction*, as distinct from minimising price *error*.

### 4.2 Multivariate and Multistep Forecasting (C.5)

Task C.5 extended the single-feature, single-step forecasting from C.4 into two more advanced problem formulations: predicting multiple future days at once (multistep), and predicting from multiple input features at once (multivariate). Table 2 reports the three GRU-based configurations evaluated with `run_c5_sweeps.py`.

**Table 2 — Task C.5 Multivariate/Multistep Results**

| Configuration | Features | Future Steps | MAE ($) | Directional Acc. (%) |
| :--- | :--- | :---: | ---: | ---: |
| gru_uni_multistep | Close only | 5 | **1.6143** | 49.43 |
| gru_multi_singlestep | All 6 features | 1 | 2.7635 | 44.83 |
| gru_multi_multistep | All 6 features | 5 | 3.9973 | 39.33 |

Forecasting error increased substantially as more features and more forecast steps were added simultaneously — the fully multivariate, multistep configuration produced roughly 2.5× the MAE of the univariate multistep configuration. This indicates that adding input features did not, by itself, improve predictive accuracy for this dataset, and that error compounds across a longer forecast horizon.

### 4.3 Statistical–Deep Learning Hybrid Forecasting (C.6)

Task C.6 investigated whether combining a statistical ARIMA model with a deep learning residual learner improves on either approach alone. An Augmented Dickey-Fuller test confirmed that the training Close series is non-stationary (p ≈ 0.819) while its first difference is stationary (p ≈ 5.19 × 10⁻¹³), statistically justifying a differencing order of *d = 1* for all ARIMA candidates. Three ARIMA orders — (1,1,1), (2,1,2), (5,1,0) — were fitted, and an LSTM and a GRU were each trained to predict the residual error of every ARIMA model, with the final prediction formed by adding the learned residual back to the ARIMA forecast.

**Table 3 — Task C.6 Results (selected models)**

| Model | MAE ($) | RMSE ($) | Directional Acc. (%) | Total Profit ($) |
| :--- | ---: | ---: | ---: | ---: |
| ARIMA(2,1,2) + LSTM Hybrid | **0.7679** | 0.9798 | 53.02 | 11.04 |
| ARIMA(2,1,2) Baseline | 0.7729 | 0.9852 | 51.29 | 10.35 |
| GRU Baseline (deep learning only) | 1.0660 | 1.3697 | **54.31** | **28.98** |
| LSTM Baseline (deep learning only) | 1.6174 | 2.0667 | 52.16 | 13.24 |

The best hybrid model, ARIMA(2,1,2) + LSTM, achieved the lowest forecasting error overall — a 0.65% MAE improvement over the best standalone ARIMA baseline — confirming that a deep learning residual learner can extract additional signal left in ARIMA's errors. However, the standalone GRU baseline achieved both the highest directional accuracy and the highest simulated trading profit, despite having a substantially higher MAE than every ARIMA-based model. This shows that minimising price error and maximising trading profitability are not the same objective, and a model selected purely on MAE would not have been the most profitable choice in this experiment.

---

## 5. Task C.7 Extension: Sentiment-Based Classification

### 5.1 Motivation and Reformulation

Tasks C.1–C.6 treat stock forecasting as a regression problem: predict the next price. Task C.7 instead asks whether financial news sentiment can improve a *classification* problem — predicting whether tomorrow's closing price will be higher or lower than today's — using historical market data as the baseline and testing whether adding sentiment features improves on it.

### 5.2 Data Collection and Feature Engineering

News data was collected from GDELT's Global Knowledge Graph, filtered for Commonwealth Bank-related coverage, and processed through a ten-stage pipeline (`run_c7.py`) that downloads and caches raw records, enriches them with English headlines, computes two independent sentiment representations, aligns both to Australian trading days, and merges them with market data into a labelled dataset:

- **V2Tone** — GDELT's built-in rule-based tone score, aggregated to a daily level.
- **FinBERT** — a transformer model trained specifically on financial text, run at the article level and aggregated to daily features.

A weekend/holiday alignment step ensures that news published outside a trading day is attributed to the next available trading day, rather than being discarded or misaligned.

### 5.3 Results

Six feature sets were evaluated using an identical Logistic Regression classifier and chronological train/validation/test split, isolating the sentiment representation as the only variable between experiments.

**Table 4 — Task C.7 Classification Results**

| Feature Set | Accuracy | Balanced Acc. | ROC AUC |
| :--- | ---: | ---: | ---: |
| Market only (baseline) | **0.554** | 0.515 | **0.584** |
| Market + V2Tone | 0.511 | 0.515 | 0.496 |
| Market + Full FinBERT | 0.502 | 0.501 | 0.531 |
| Market + Reduced FinBERT | 0.511 | **0.528** | 0.550 |
| Market + V2Tone + Reduced FinBERT | 0.528 | 0.523 | 0.544 |

The market-only baseline achieved the highest Accuracy and ROC AUC of any configuration tested. No sentiment-enhanced feature set surpassed it. FinBERT consistently outperformed V2Tone across nearly every metric, indicating that a transformer trained on financial text captures more useful sentiment signal than a general rule-based tone score, but the improvement was not large enough to close the gap to the market-only model.

### 5.4 Independent Research Component

Following the initial results, a feature audit was conducted to diagnose why the FinBERT feature set underperformed expectations. The audit identified substantial redundancy among the original FinBERT-derived variables — several features were mathematically dependent on each other or duplicated information already captured elsewhere. A reduced, non-redundant FinBERT feature set was designed and re-evaluated: it improved both Accuracy (0.502 → 0.511) and ROC AUC (0.531 → 0.550) relative to the full FinBERT set, while using fewer features. This demonstrates that feature engineering quality, not feature quantity, was the limiting factor in the original FinBERT representation.

---

## 6. Scenarios and Examples

**Scenario 1 — Comparing model architectures for price forecasting (C.4).** A user wants to know whether GRU or LSTM predicts CBA.AX prices more accurately under identical conditions. Running `run_c4_sweeps.py` trains and evaluates both architectures (plus SimpleRNN) across depth, width, and loss-function variants through the same pipeline, producing the directly comparable results in Table 1 — showing GRU_BASE as the most accurate configuration tested.

**Scenario 2 — Forecasting five days ahead instead of one (C.5).** A user wants a five-day-ahead forecast rather than a single-day forecast. Setting `future_steps=5` and running `run_c5_sweeps.py` (or `train.py --future_steps 5`) reuses the identical data pipeline and model builder, changing only the sliding-window target construction — no architecture code changes are required, illustrating the benefit of the shared pipeline design described in Section 2.

**Scenario 3 — Testing whether a statistical model improves a deep learning forecast (C.6).** A user wants to check whether combining ARIMA with a neural residual learner beats either approach alone. Running `run_c6.py` performs the full pipeline: stationarity testing, ARIMA fitting, residual learner training, and hybrid evaluation, producing Table 3 and confirming a measurable, if modest, improvement from the hybrid approach.

**Scenario 4 — Testing whether news sentiment improves next-day direction prediction (C.7).** A user wants to know whether adding financial news sentiment to a price-direction classifier helps. Running `run_c7.py` executes the full ten-stage news pipeline and produces Table 4, showing that for this dataset and time period, market data alone remained the strongest predictor.

---

## 7. Critical Analysis

**The core pipeline's reusability held up under real extension pressure.** The clearest evidence for the C.2–C.5 architectural decisions (Section 2.2) is that Task C.6 could reuse `data_processing.py` and `model_factory.py` for an entirely different forecasting strategy (ARIMA + residual learning) without modifying either module, and Task C.4's sweep results remained valid as a baseline for later comparisons because every task shared the same preprocessing and leakage-safe splitting logic.

**Minimising forecasting error and maximising trading profitability produced different winners.** In Task C.6, the ARIMA-based hybrid models had the lowest MAE, but the standalone GRU baseline had the highest directional accuracy and by far the highest simulated trading profit ($28.98 vs. $11.04 for the best hybrid). A project that optimised purely for MAE, as is common practice, would have overlooked the model that actually performed best under a trading-oriented evaluation. This is treated in Section 4.3 as a genuine finding, not a contradiction: MAE, directional accuracy, and trading profit measure different things, and no single metric was a good proxy for the other two throughout this project.

**Task C.7's negative result is a valid finding, not a failure.** None of the sentiment-enhanced feature sets outperformed the market-only baseline (Table 4). Rather than treating this as an implementation defect, the project used the feature audit (Section 5.4) to establish that the shortfall was attributable to feature redundancy rather than the sentiment signal itself being uninformative — the reduced FinBERT set measurably improved on the full FinBERT set once redundant variables were removed. This distinguishes "sentiment doesn't help this problem" from "the sentiment features were engineered poorly," which is a more defensible and more useful conclusion.

**Limitations.** All experiments were run on a single ticker (CBA.AX) over one historical period, so results may not generalise to other stocks, sectors, or market regimes. Task C.7's sentiment features were aggregated at a daily level, discarding intraday timing information about when news was published relative to market close. Logistic Regression was deliberately chosen for C.7 as an interpretable baseline; a more expressive classifier might capture nonlinear interactions between market and sentiment features that a linear model cannot.

---

## 8. Summary and Conclusion

FinTech101 progressed from a single-file, single-feature LSTM baseline to a modular forecasting framework capable of comparing recurrent architectures (C.4), multivariate and multistep forecasting (C.5), statistically justified hybrid forecasting (C.6), and sentiment-enhanced classification (C.7), all built on one leakage-safe, reusable data and evaluation pipeline.

Across the regression tasks, GRU consistently performed competitively against LSTM despite its simpler gating structure (C.4), additional input features did not by themselves reduce forecasting error (C.5), and combining ARIMA with a deep learning residual learner produced a small but real accuracy improvement over either technique alone (C.6) — though a standalone GRU model remained the most profitable under simulated trading. In the classification extension, historical market data proved to be a stronger predictor of next-day price direction than either of the two sentiment representations tested, though the independent feature audit showed that careful feature engineering materially improved the sentiment-based models even where it did not surpass the baseline.

Together, these results support a central conclusion: for CBA.AX over the period studied, model architecture and feature engineering quality mattered more than the sheer amount of information given to the model, whether that meant adding more price features (C.5) or more sentiment features (C.7). The reusable pipeline built across C.2–C.5 made it possible to test this conclusion consistently across seven increasingly different forecasting and classification problems without rebuilding the underlying data or evaluation logic each time.
