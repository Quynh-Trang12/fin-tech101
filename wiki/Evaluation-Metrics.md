# Evaluation Metrics

## Purpose

FinTech101 evaluates two different kinds of prediction problem: price forecasting (Tasks C.1–C.6) and next-day price direction classification (Task C.7). Each uses its own metric set, because a regression error metric like MAE cannot judge a classifier and an accuracy score cannot judge how close a predicted price is to the true one. This page defines every metric reported in the project's evaluation outputs and explains what it is used for.

---

## Forecasting Metrics (C.1–C.6)

These metrics are computed in `test.py`'s `calculate_metrics()` and `get_trading_profits()`, on **unscaled** prices — predictions are inverse-transformed through the fitted `MinMaxScaler` before any metric is calculated, so reported errors are in dollars, not normalised units.

| Metric | Definition | What it measures |
| :----- | :---------- | :---------------- |
| **MAE** (Mean Absolute Error) | Average of `\|actual − predicted\|` across the test set | Average dollar error per prediction, in the same units as the stock price |
| **RMSE** (Root Mean Squared Error) | Square root of the average squared error | Dollar error that penalises large mistakes more heavily than MAE |
| **MAPE** (Mean Absolute Percentage Error) | Average of `\|actual − predicted\| / actual`, as a percentage | Error scaled relative to price level, so it stays comparable across different price ranges |
| **DA** (Directional Accuracy) | Percentage of test days where the predicted price movement (up or down from the previous actual price) matches the true movement | Whether the model gets the *direction* of the next move right, independent of how close the price value itself was |

### Trading Simulation Metrics

`get_trading_profits()` simulates a simple next-day trading strategy on top of the forecast: buy if the predicted price is above the current price, sell (short) if below, and realise the resulting profit or loss against the true price.

| Metric | Definition |
| :----- | :---------- |
| **Trading Accuracy** | Percentage of simulated trades that were profitable |
| **Total Profit** | Sum of profit/loss across all simulated trades over the test period |
| **Profit per Trade** | Total profit divided by the number of trades |

**Why report both forecasting and trading metrics.** A model with the lowest MAE is not automatically the most useful one. In the Task C.6 hybrid experiments, the ARIMA + LSTM hybrid achieved the lowest MAE of any model tested, but a standalone GRU baseline achieved both the highest directional accuracy and by far the highest total trading profit. Reporting only MAE would have hidden this difference; the trading simulation metrics exist specifically to surface it.

---

## Classification Metrics (C.7)

Task C.7 reframes the problem as binary classification — will tomorrow's closing price be higher or lower than today's — so it is evaluated with standard classification metrics instead of forecasting error. These are computed in `c7_baseline.py` using `scikit-learn`.

| Metric | Definition | Why it is used here |
| :----- | :---------- | :-------------------- |
| **Accuracy** | Percentage of test-day direction predictions that were correct | Simple overall correctness, but sensitive to class imbalance |
| **Balanced Accuracy** | Average of recall computed separately for each class | Corrects for Accuracy's sensitivity to a model that over-predicts one direction |
| **Precision** | Of the days predicted "price rises," the percentage that actually rose | How trustworthy a positive (rise) prediction is |
| **Recall** | Of the days that actually rose, the percentage the model correctly predicted | How many true rises the model catches |
| **F1-score** | Harmonic mean of Precision and Recall | Single score balancing both, useful when the two trade off against each other |
| **ROC AUC** | Area under the Receiver Operating Characteristic curve | The model's ability to rank a random "rise" day above a random "fall" day, independent of the classification threshold chosen |

**Why Balanced Accuracy and ROC AUC matter more than raw Accuracy here.** The Task C.7 market-only baseline strongly favoured predicting price increases, which inflates Recall and raw Accuracy without indicating real predictive skill. Balanced Accuracy and ROC AUC do not reward this bias, which is why the project's C.7 report treats them as the more informative metrics when comparing feature sets — see **Weekly Reports → Task C.7 Report** for the full comparison.

---

## Continue Exploring

- **Home** — Project background, scope, and objectives.
- **System Architecture** — High-level architecture and module interactions.
- **Experiment Pipeline** — End-to-end workflow from data acquisition to model evaluation.
- **Weekly Reports** — Archived reports for Tasks C.1–C.7 documenting weekly progress.
