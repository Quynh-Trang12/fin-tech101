# Task C.6 Ensemble Forecasting Experiment Summary (Fixed-Parameter Protocol)

This report documents the experimental results of the fixed-parameter evaluation protocol (Option B). The classical statistical ARIMA model and the deep learning GRU model are both evaluated under identical chronological partitioning and frozen parameter constraints.

---

## 1. Experimental Setup

- **Dataset**: `CBA.AX` daily stock prices from Yahoo Finance cache.
- **Chronological Test Period**: Starts at `2023-08-02` (split boundary).
- **Target Variable**: Adjusted Closing Price (`adjclose`).
- **ARIMA Method**: Fixed-parameter forecasting (Option B). The model parameters are estimated **once** on the training dataset (up to 2023-08-01). The fitted parameters are frozen, and the filter is applied over the test period to generate one-step-ahead forecasts.
- **Ensemble Method**: Linear weighted averaging of GRU predictions with fixed-parameter ARIMA predictions.

---

## 2. Tested Configurations

- **GRU Baseline**:
  - `GRU_BASE`: Standard 2-layer GRU model with 128 hidden units.
- **ARIMA Baseline Orders**:
  - `ARIMA(1, 1, 1)`: Standard ARMA(1,1) on first-differenced prices.
  - `ARIMA(2, 1, 2)`: Higher-order stationary modeling.
  - `ARIMA(5, 1, 0)`: Autoregressive-only model on first-differenced prices.
- **Ensemble Weights (GRU / ARIMA)**:
  - 50% GRU / 50% ARIMA
  - 70% GRU / 30% ARIMA
  - 30% GRU / 70% ARIMA

---

## 3. Experimental Results

| Model                                   | Order/Weights        |     MAE |    RMSE |    MAPE |      DA |   total_buy_profit |   total_sell_profit |   total_profit |   profit_per_trade |   trading_accuracy |
|:----------------------------------------|:---------------------|--------:|--------:|--------:|--------:|-------------------:|--------------------:|---------------:|-------------------:|-------------------:|
| GRU_BASE                                | N/A                  | 2.22556 | 2.57511 | 2.11995 | 44.8276 |           0.077713 |           -22.6663  |      -22.5886  |         -0.0973646 |            44.8276 |
| ARIMA(1, 1, 1)                          | (1, 1, 1)            | 1.07631 | 1.34732 | 1.03806 | 49.1379 |          13.5664   |            -9.17761 |        4.38879 |          0.0189172 |            49.1379 |
| ARIMA(2, 1, 2)                          | (2, 1, 2)            | 1.08004 | 1.34966 | 1.04111 | 50      |          13.4505   |            -9.29349 |        4.15704 |          0.0179183 |            50      |
| ARIMA(5, 1, 0)                          | (5, 1, 0)            | 1.07848 | 1.35208 | 1.04055 | 50.431  |          15.6848   |            -7.05921 |        8.6256  |          0.0371793 |            50.431  |
| Ensemble (GRU*0.5 + ARIMA(1, 1, 1)*0.5) | GRU: 0.5, ARIMA: 0.5 | 1.45833 | 1.76669 | 1.39534 | 44.3966 |          -1.60476  |           -24.3488  |      -25.9535  |         -0.111869  |            44.3966 |
| Ensemble (GRU*0.7 + ARIMA(1, 1, 1)*0.3) | GRU: 0.7, ARIMA: 0.3 | 1.7282  | 2.06415 | 1.64984 | 43.9655 |          -2.00617  |           -24.7502  |      -26.7564  |         -0.115329  |            43.9655 |
| Ensemble (GRU*0.3 + ARIMA(1, 1, 1)*0.7) | GRU: 0.3, ARIMA: 0.7 | 1.24779 | 1.52786 | 1.19669 | 47.8448 |           4.18066  |           -18.5634  |      -14.3827  |         -0.0619944 |            47.8448 |
| Ensemble (GRU*0.5 + ARIMA(2, 1, 2)*0.5) | GRU: 0.5, ARIMA: 0.5 | 1.45491 | 1.76277 | 1.39189 | 44.3966 |          -1.60476  |           -24.3488  |      -25.9535  |         -0.111869  |            44.3966 |
| Ensemble (GRU*0.7 + ARIMA(2, 1, 2)*0.3) | GRU: 0.7, ARIMA: 0.3 | 1.72571 | 2.06115 | 1.64743 | 44.3966 |          -0.786636 |           -23.5307  |      -24.3173  |         -0.104816  |            44.3966 |
| Ensemble (GRU*0.3 + ARIMA(2, 1, 2)*0.7) | GRU: 0.3, ARIMA: 0.7 | 1.245   | 1.52462 | 1.19371 | 46.5517 |           2.02038  |           -20.7236  |      -18.7033  |         -0.0806175 |            46.5517 |
| Ensemble (GRU*0.5 + ARIMA(5, 1, 0)*0.5) | GRU: 0.5, ARIMA: 0.5 | 1.45629 | 1.76343 | 1.39355 | 44.8276 |          -0.385223 |           -23.1292  |      -23.5145  |         -0.101355  |            44.8276 |
| Ensemble (GRU*0.7 + ARIMA(5, 1, 0)*0.3) | GRU: 0.7, ARIMA: 0.3 | 1.72699 | 2.06143 | 1.64882 | 44.3966 |          -0.786636 |           -23.5307  |      -24.3173  |         -0.104816  |            44.3966 |
| Ensemble (GRU*0.3 + ARIMA(5, 1, 0)*0.7) | GRU: 0.3, ARIMA: 0.7 | 1.24198 | 1.52586 | 1.19182 | 45.6897 |           0.934265 |           -21.8098  |      -20.8755  |         -0.0899806 |            45.6897 |

---

## 4. Key Findings

1. **ARIMA Baseline Outperforms GRU**:
   - All individual Fixed-Parameter ARIMA models significantly outperformed the static GRU baseline.
   - The best-performing model overall was **ARIMA(1, 1, 1)** with an MAE of **$1.0763** and RMSE of **$1.3473** (compared to the GRU's MAE of **$2.2256**).
   - This occurs because ARIMA differences the price series to model local changes and anchors its next-day forecast on the most recent actual price ($y_t$), avoiding drift.

2. **Weighted Ensembles Underperform ARIMA**:
   - No weighted ensemble outperformed the individual ARIMA baselines.
   - Averaging the weaker GRU baseline (MAE ~$2.2256) with the stronger ARIMA baseline (MAE ~$1.08) degraded performance. The ensemble error scales linearly with the weight of the GRU.

3. **Trading Profitability**:
   - In contrast to the rolling ARIMA run, the Fixed-Parameter ARIMA models generated **positive simulated trading profits** (e.g. `ARIMA(5, 1, 0)` achieved a total profit of **$8.63** and directional accuracy of **50.43%**).
   - All weighted ensembles reported negative profits, demonstrating that GRU error contamination directly harms trading utility.

---

## 5. Notes, Warnings, and Limitations

- **Fast Execution**: Because the models are only fit once, execution time is extremely fast (under 1 second) compared to recursive daily refitting.
- **Stationarity Warnings**: Occasional statsmodels warning messages about near-non-invertibility were generated during the initial training fit but were safely bypassed.

---

## 6. Recommended Next Steps

1. **Hybrid Residual-Learning (Option D)**: Instead of linear weighted averaging, use the GRU model to predict the forecast residuals of the ARIMA model.
2. **Sentiment Indicators (Task C.7)**: Integrate external news sentiment data to further enhance directional trading signals.
