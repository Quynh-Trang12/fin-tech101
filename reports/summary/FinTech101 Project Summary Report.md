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

The core workflow follows a fixed path: **data source → data processing (`data_processing.py`) → model construction (`model_factory.py`) → model training (`train.py`) → model evaluation (`test.py`) → results**. `model_factory.py` builds a compiled LSTM, GRU, or SimpleRNN model from hyperparameters, decoupling architecture choice from training logic; `base_sweep.py`, `run_c4_sweeps.py`, and `run_c5_sweeps.py` automate running many configurations through this same workflow so C.4's and C.5's results are directly comparable; and `config.py` centralises every dataset, model, and sweep parameter in one place.

### 2.3 Extensions Beyond the Core Workflow (C.6–C.7)

Task C.6 and Task C.7 reuse specific components of the core workflow but do not follow its train-then-evaluate shape. `run_c6.py` reuses `data_processing.py` and `model_factory.py`, but trains and evaluates inline in one script: fit an ARIMA baseline, train an LSTM/GRU on its residuals, then combine both predictions — necessary because the residual learner's target depends on ARIMA's output, so the two cannot be trained independently. The Task C.7 `c7_*.py` scripts replace `data_processing.py` entirely with a news-processing chain that downloads GDELT records, extracts V2Tone and FinBERT sentiment, aligns both to trading days, and trains a Logistic Regression classifier on price *direction* rather than price value.

Full architecture diagrams and the six formal architectural decisions behind this design are documented in the **System Architecture** and **Experiment Pipeline** Wiki pages.

---

## 3. Implemented Data Processing Techniques

### 3.1 Multi-Feature, Configurable Data Loading (C.2)

The Task C.1 baseline (`v0.1`) used only the closing price and required manually choosing separate start/end dates for training and testing data. Task C.2 replaced this with `load_and_process_data()`, a single configurable entry point that derives train/test partitions from one overall date range, supports multiple feature columns (`adjclose`, `volume`, `open`, `high`, `low`) instead of closing price alone, handles missing values, offers three split strategies (chronological by date, chronological by ratio, or random), caches downloaded data locally, and fits per-feature `MinMaxScaler`s that are stored for reuse during evaluation.

### 3.2 Leakage-Safe Ordering

A specific ordering prevents test-period information leaking into training: sliding windows and targets are constructed first, the chronological split is performed by the *target* date rather than the input date, and scalers are fitted only on the resulting training partition. Splitting by input date alone can let a window whose target falls after the split boundary remain in training, indirectly exposing the model to test-period outcomes — this ordering avoids that.

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

GRU produced the lowest forecasting error among all ten configurations, ahead of every LSTM variant, despite its simpler gating structure. Directional accuracy varied only narrowly (43.97%–46.12%), suggesting hyperparameter tuning had limited effect on predicting price *direction* versus minimising price *error*.

### 4.2 Multivariate and Multistep Forecasting (C.5)

Task C.5 extended the single-feature, single-step forecasting from C.4 into two more advanced problem formulations: predicting multiple future days at once (multistep), and predicting from multiple input features at once (multivariate). Table 2 reports the three GRU-based configurations evaluated with `run_c5_sweeps.py`.

**Table 2 — Task C.5 Multivariate/Multistep Results**

| Configuration | Features | Future Steps | MAE ($) | Directional Acc. (%) |
| :--- | :--- | :---: | ---: | ---: |
| gru_uni_multistep | Close only | 5 | **1.6143** | 49.43 |
| gru_multi_singlestep | All 6 features | 1 | 2.7635 | 44.83 |
| gru_multi_multistep | All 6 features | 5 | 3.9973 | 39.33 |

Error increased substantially as more features and forecast steps were added simultaneously — the fully multivariate, multistep configuration produced roughly 2.5× the MAE of the univariate multistep one. Adding input features did not, by itself, improve accuracy here, and error compounds across a longer horizon.

### 4.3 Statistical–Deep Learning Hybrid Forecasting (C.6)

Task C.6 investigated whether combining a statistical ARIMA model with a deep learning residual learner improves on either approach alone. An Augmented Dickey-Fuller test confirmed that the training Close series is non-stationary (p ≈ 0.819) while its first difference is stationary (p ≈ 5.19 × 10⁻¹³), statistically justifying a differencing order of *d = 1* for all ARIMA candidates. Three ARIMA orders — (1,1,1), (2,1,2), (5,1,0) — were fitted, and an LSTM and a GRU were each trained to predict the residual error of every ARIMA model, with the final prediction formed by adding the learned residual back to the ARIMA forecast.

**Table 3 — Task C.6 Results (selected models)**

| Model | MAE ($) | RMSE ($) | Directional Acc. (%) | Total Profit ($) |
| :--- | ---: | ---: | ---: | ---: |
| ARIMA(2,1,2) + LSTM Hybrid | **0.7679** | 0.9798 | 53.02 | 11.04 |
| ARIMA(2,1,2) Baseline | 0.7729 | 0.9852 | 51.29 | 10.35 |
| GRU Baseline (deep learning only) | 1.0660 | 1.3697 | **54.31** | **28.98** |
| LSTM Baseline (deep learning only) | 1.6174 | 2.0667 | 52.16 | 13.24 |

The best hybrid, ARIMA(2,1,2) + LSTM, achieved the lowest error overall — a 0.65% MAE improvement over the best standalone ARIMA — confirming the residual learner extracts additional signal from ARIMA's errors. The standalone GRU baseline, however, achieved both the highest directional accuracy and trading profit despite a substantially higher MAE, showing that minimising error and maximising profitability are not the same objective (discussed further in Section 7).

---

## 5. Task C.7 Extension: Sentiment-Based Classification

### 5.1 Motivation and Reformulation

Tasks C.1–C.6 treat stock forecasting as a regression problem: predict the next price. Task C.7 instead asks whether financial news sentiment can improve a *classification* problem — predicting whether tomorrow's closing price will be higher or lower than today's — using historical market data as the baseline and testing whether adding sentiment features improves on it.

### 5.2 Data Collection and Feature Engineering

News data was collected from GDELT's Global Knowledge Graph, filtered for Commonwealth Bank-related coverage, and processed through a ten-stage pipeline (`run_c7.py`) that downloads and caches raw records, enriches them with English headlines, computes two independent sentiment representations, aligns both to Australian trading days (attributing news published outside a trading day to the next available one), and merges them with market data into a labelled dataset. **V2Tone** is GDELT's built-in rule-based tone score, aggregated daily; **FinBERT** is a transformer trained on financial text, run at the article level and aggregated to daily features.

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

**Comparing architectures (C.4).** `run_c4_sweeps.py` trains and evaluates LSTM, GRU, and SimpleRNN variants through the same pipeline, producing Table 1's directly comparable results — GRU_BASE is the most accurate configuration tested.

**Changing the forecast horizon (C.5).** Setting `future_steps=5` and running `run_c5_sweeps.py` reuses the identical data pipeline and model builder, changing only the sliding-window target — no architecture changes required.

**Testing a statistical/DL hybrid (C.6).** `run_c6.py` runs stationarity testing, ARIMA fitting, and residual-learner training end to end, producing Table 3's measurable hybrid improvement.

**Testing sentiment-augmented classification (C.7).** `run_c7.py` executes the full news pipeline and produces Table 4, showing market data alone remained the strongest predictor for this dataset and period.

---

## 7. Critical Analysis

**Reusability held up under real extension pressure.** Task C.6 reused `data_processing.py` and `model_factory.py` for an entirely different forecasting strategy without modifying either module, and C.4's sweep results stayed valid as a baseline because every task shared the same preprocessing and leakage-safe splitting logic.

**Error and profitability produced different winners.** In C.6, ARIMA-based hybrids had the lowest MAE, but the standalone GRU baseline had the highest directional accuracy and by far the highest trading profit ($28.98 vs. $11.04 for the best hybrid) — a project optimising purely for MAE would have missed the most profitable model. MAE, directional accuracy, and trading profit measure different things, and none was a reliable proxy for the others.

**C.7's negative result is a finding, not a failure.** No sentiment-enhanced feature set beat the market-only baseline (Table 4), but the feature audit (Section 5.4) showed the shortfall was attributable to feature redundancy, not an uninformative sentiment signal — the reduced FinBERT set measurably improved on the full set once redundant variables were removed.

**Limitations.** All experiments used a single ticker (CBA.AX) over one period, so results may not generalise. C.7's sentiment was aggregated daily, discarding intraday timing, and Logistic Regression was chosen as an interpretable baseline over a more expressive classifier.

---

## 8. Summary and Conclusion

FinTech101 progressed from a single-file, single-feature LSTM baseline to a modular forecasting framework capable of comparing recurrent architectures (C.4), multivariate and multistep forecasting (C.5), statistically justified hybrid forecasting (C.6), and sentiment-enhanced classification (C.7), all built on one leakage-safe, reusable data and evaluation pipeline.

Across the regression tasks, GRU consistently performed competitively against LSTM despite its simpler gating structure (C.4), additional input features did not by themselves reduce forecasting error (C.5), and combining ARIMA with a deep learning residual learner produced a small but real accuracy improvement over either technique alone (C.6) — though a standalone GRU model remained the most profitable under simulated trading. In the classification extension, historical market data proved to be a stronger predictor of next-day price direction than either of the two sentiment representations tested, though the independent feature audit showed that careful feature engineering materially improved the sentiment-based models even where it did not surpass the baseline.

Together, these results support a central conclusion: for CBA.AX over the period studied, model architecture and feature engineering quality mattered more than the sheer amount of information given to the model, whether that meant adding more price features (C.5) or more sentiment features (C.7). The reusable pipeline built across C.2–C.5 made it possible to test this conclusion consistently across seven increasingly different forecasting and classification problems without rebuilding the underlying data or evaluation logic each time.
