# Task C.6 Redo Experiment Summary Report

This report presents the consolidated findings of the statistically justified ARIMA + residual deep learning framework under Task C.6. Only univariate inputs (`adjclose`) and one-step-ahead forecasts were evaluated.

## 1. Stationarity Analysis (Augmented Dickey-Fuller Test)

The ADF test was executed strictly on the training history to determine the differencing parameter $d$. No lookahead leakage occurred.

| Series                  |   Test Statistic |     p-value |   Lags Used |   Observations |   Critical Value 1% |   Critical Value 5% |   Critical Value 10% | Is Stationary   |
|:------------------------|-----------------:|------------:|------------:|---------------:|--------------------:|--------------------:|---------------------:|:----------------|
| Original Close          |        -0.800857 | 0.81887     |          13 |            763 |            -3.43895 |            -2.86534 |             -2.56879 | False           |
| First-Differenced Close |        -8.25738  | 5.19309e-13 |          12 |            763 |            -3.43895 |            -2.86534 |             -2.56879 | True            |

**Justification:** The original training Close series was non-stationary ($p \approx 0.818870 \ge 0.05$), while the first-differenced series was highly stationary ($p \approx 5.193090e-13 < 0.05$). This statistically justifies $d=1$ in all ARIMA model orders.

## 1.1 Significant Autocorrelations (ACF) and Partial Autocorrelations (PACF)

The 95% approximate confidence threshold for significance is $\pm 0.070360$ ($1.96 / \sqrt{N}$ where $N = 776$ training observations). Significant lags are shown below:

|   Lag |        ACF | Is ACF Significant   |       PACF | Is PACF Significant   |
|------:|-----------:|:---------------------|-----------:|:----------------------|
|     1 | -0.080334  | True                 | -0.0804377 | True                  |
|     3 |  0.0780001 | True                 |  0.0791874 | True                  |
|     5 |  0.108813  | True                 |  0.100589  | True                  |
|     6 |  0.090628  | True                 |  0.104161  | True                  |
|    10 | -0.102537  | True                 | -0.103295  | True                  |
|    12 | -0.0885336 | True                 | -0.10403   | True                  |

This analysis shows the presence of short-term correlations in the differenced series, which justifies evaluating autoregressive (AR) and moving-average (MA) orders in the ARIMA candidates (such as ARIMA(1,1,1), ARIMA(2,1,2), and ARIMA(5,1,0)).

## 2. ARIMA Diagnostic Metrics

AIC, BIC, and Ljung-Box autocorrelation test results at lag 10 are presented below:

| Model        |     AIC |     BIC |   Ljung-Box p-value (lag 10) |
|:-------------|--------:|--------:|-----------------------------:|
| ARIMA(1,1,1) | 2448.74 | 2462.69 |                  0.895435    |
| ARIMA(2,1,2) | 2435.25 | 2458.5  |                  2.21691e-34 |
| ARIMA(5,1,0) | 2431.95 | 2459.84 |                  0.760093    |

## 2.1 ARIMA Candidate Ranking

| ARIMA Model   |     AIC |     BIC |   Ljung-Box p-value (lag 10) |      MAE |     RMSE |   MAPE (%) |   DA (%) |   Total Profit ($) |   AIC Rank |   BIC Rank |   MAE Rank |   RMSE Rank |
|:--------------|--------:|--------:|-----------------------------:|---------:|---------:|-----------:|---------:|-------------------:|-----------:|-----------:|-----------:|------------:|
| ARIMA(1,1,1)  | 2448.74 | 2462.69 |                  0.895435    | 0.776376 | 0.98289  |     0.746  |    50.43 |               4.7  |          3 |          3 |          2 |           1 |
| ARIMA(2,1,2)  | 2435.25 | 2458.5  |                  2.21691e-34 | 0.772908 | 0.985228 |     0.7436 |    51.29 |              10.35 |          2 |          1 |          1 |           2 |
| ARIMA(5,1,0)  | 2431.95 | 2459.84 |                  0.760093    | 0.780443 | 0.995072 |     0.7506 |    49.57 |               0.59 |          1 |          2 |          3 |           3 |## 3. Consolidated Prediction and Trading Evaluation

Evaluation results over the chronological test set (including deep learning baselines, standalone ARIMA configurations, and hybrid models):

| Model                      |      MAE |     RMSE |   MAPE (%) |   DA (%) |   Trading Accuracy (%) |   Total Profit ($) |   Profit/Trade ($) |
|:---------------------------|---------:|---------:|-----------:|---------:|-----------------------:|-------------------:|-------------------:|
| LSTM Baseline              | 1.61735  | 2.06665  |     1.6032 |    52.16 |                  52.16 |              13.24 |             0.0571 |
| GRU Baseline               | 1.06598  | 1.36969  |     1.0434 |    54.31 |                  54.31 |              28.98 |             0.1249 |
| ARIMA(1,1,1) Baseline      | 0.776376 | 0.98289  |     0.746  |    50.43 |                  50.43 |               4.7  |             0.0203 |
| ARIMA(1,1,1) + LSTM Hybrid | 0.769629 | 0.977275 |     0.7401 |    53.88 |                  53.88 |              28.75 |             0.1239 |
| ARIMA(1,1,1) + GRU Hybrid  | 0.76961  | 0.977262 |     0.7401 |    52.59 |                  52.59 |              26.66 |             0.1149 |
| ARIMA(2,1,2) Baseline      | 0.772908 | 0.985228 |     0.7436 |    51.29 |                  51.29 |              10.35 |             0.0446 |
| ARIMA(2,1,2) + LSTM Hybrid | 0.767905 | 0.979835 |     0.7394 |    53.02 |                  53.02 |              11.04 |             0.0476 |
| ARIMA(2,1,2) + GRU Hybrid  | 0.76809  | 0.980025 |     0.7395 |    53.45 |                  53.45 |              10.58 |             0.0456 |
| ARIMA(5,1,0) Baseline      | 0.780443 | 0.995072 |     0.7506 |    49.57 |                  49.57 |               0.59 |             0.0025 |
| ARIMA(5,1,0) + LSTM Hybrid | 0.777466 | 0.990948 |     0.7481 |    52.16 |                  52.16 |              12.56 |             0.0541 |
| ARIMA(5,1,0) + GRU Hybrid  | 0.777701 | 0.991076 |     0.7483 |    53.02 |                  53.02 |              17.13 |             0.0738 |

## 4. Key Experimental Findings

- **Lowest Forecasting Error:** `ARIMA(2,1,2) + LSTM Hybrid` achieved the lowest MAE of **$0.767905** (RMSE: **$0.979835**, MAPE: **0.7394%**).
- **Highest Directional Accuracy:** `GRU Baseline` correctly identified short-term trend directions **54.31%** of the time.
- **Most Profitable Strategy:** `GRU Baseline` achieved the highest total trading return of **$28.98**.

## 5. Performance Comparison: Hybrids vs. Standalone Baselines

- **Best Standalone Baseline Model:** `ARIMA(2,1,2) Baseline` with an MAE of **$0.772908**.
- **Best Residual Hybrid Model:** `ARIMA(2,1,2) + LSTM Hybrid` with an MAE of **$0.767905**.

The best hybrid model achieved an improvement of **0.65%** in MAE over the best standalone baseline model, demonstrating the effectiveness of the statistical + residual deep learning framework.
