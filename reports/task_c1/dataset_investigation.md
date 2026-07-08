# Dataset Investigation

This document provides a comprehensive exploratory investigation of the historical stock market dataset used throughout the FinTech101 stock prediction project, focusing on the target stock **Commonwealth Bank of Australia (`CBA.AX`)**.

---

## 1. Dataset Overview

* **Ticker**: `CBA.AX` (listed on the Australian Securities Exchange - ASX)
* **Date Range**: `2020-01-01` to `2024-07-02` (inclusive)
* **Number of Trading Days**: 1,138 trading days
* **Number of Features**: 5 features used as model inputs
* **Feature Names**: `adjclose`, `volume`, `open`, `high`, `low`
* **Target Variable**: `adjclose` (adjusted close price)
* **Lookback Window**: 50 trading days (sequence length $T = 50$)
* **Forecast Horizon**: 1 trading day ahead (forecast offset $= 1$, predicting $D_{t+1}$ based on inputs from $D_{t-49}$ to $D_t$)

---

## 2. Missing Values

* **Missing Values per Feature**:
  * `adjclose`: 0
  * `volume`: 0
  * `open`: 0
  * `high`: 0
  * `low`: 0
  * *Total Missing Values*: 0

### Handling of Missing Values in Preprocessing
Although the raw cached file `CBA.AX_cache.csv` contains zero missing values for all columns, the project's standardized preprocessing module [`src/data_processing.py`](file:///e:/fin-tech101/src/data_processing.py#L152) applies a **forward-fill (`ffill`)** operation to handle missing records defensively. Forward-filling replaces any missing values with the most recent valid observation chronologically, preserving the continuity of the time series without leaking future information.

---

## 3. Basic Statistics

The table below presents the basic descriptive statistics computed directly from the cleaned, unscaled historical daily stock records:

| Feature | Mean | Std. Dev. | Minimum | 25% (Q1) | 50% (Median) | 75% (Q3) | Maximum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **adjclose** ($) | 82.84 | 16.74 | 43.83 | 70.95 | 86.35 | 92.36 | 121.07 |
| **volume** | 2,763,692.00 | 1,672,452.00 | 207,240.00 | 1,772,995.00 | 2,319,145.00 | 3,132,848.00 | 17,021,970.00 |
| **open** ($) | 95.11 | 15.39 | 56.01 | 86.02 | 99.09 | 104.11 | 128.11 |
| **high** ($) | 95.79 | 15.32 | 57.80 | 86.75 | 99.80 | 104.96 | 128.68 |
| **low** ($) | 94.36 | 15.46 | 53.44 | 85.48 | 98.50 | 103.54 | 127.51 |

---

## 4. Distribution Observations

* **Feature Scale Similarities**:
  * The price-related features (`adjclose`, `open`, `high`, `low`) are on similar scales, with values ranging from approximately $\$43.83$ to $\$128.68$.
* **Range Discrepancies**:
  * The `volume` feature is on a completely different scale. It ranges from a minimum of $207,240$ to a maximum of $17,021,970$ (daily volume of shares traded). The values of `volume` are **4 to 5 orders of magnitude larger** than the price features.
* **Justification for Scaling**:
  * Because the magnitudes of the raw feature ranges are highly mismatched (e.g., volume in millions vs. prices in double or triple digits), fitting a neural network on raw unscaled values would cause the weight updates and loss gradients to be entirely dominated by the `volume` dimension. This would prevent the optimizer from learning meaningful relationships from the price channels. 
  * Therefore, **MinMax scaling** (standardizing each feature independently to a $[0, 1]$ range) is fully justified and mathematically necessary.

---

## 5. Time-Series Characteristics

* **Overall Trend**: 
  * The `CBA.AX` stock exhibits a strong, long-term upward trend over the four-and-a-half-year period, rising from an initial adjusted close price of $\$63.10$ on January 1, 2020, to a peak adjusted close price of $\$118.27$ on July 2, 2024 (an overall price increase of $+87.4\%$).
* **Volatility and Sub-Period Shifts**:
  * The dataset is characterized by a significant shift in volatility across the chronological splits:
    * **Training Period (2020-01-01 to 2023-01-24)**: Highly volatile, exhibiting an annualized volatility of **28.29%**. This period is dominated by the massive shock and rapid recovery associated with the COVID-19 pandemic.
    * **Validation Period (2023-01-25 to 2023-07-31)**: Moderate volatility (**17.19%**) and price consolidation.
    * **Test Period (2023-08-01 to 2024-07-02)**: Low volatility (**14.79%**), characterized by a steady, persistent upward trend.
* **Sudden Spikes and Drawdowns**:
  * A major market drawdown occurred at the beginning of the dataset during the COVID-19 crash, driving the adjusted close to its historical minimum of **$\$43.83$ on March 22, 2020** (a drop of over 30% from the pre-pandemic peak). This was followed by a sharp, high-volatility rebound.
* **Noisiness and Seasonality**:
  * The daily returns are highly noisy and show typical stock-market random walk properties in the short term.
  * No obvious, clean calendar seasonality exists in the raw price series, which is typical for liquid equity prices that adjust to continuous information arrival.

---

## 6. Dataset Size and Tensor Shapes

After sequence generation with a 50-day lookback window, a 1-day forecast offset, and a 1-day future step, the supervised samples are split chronologically (with validation as the tail 15% of the training split):

* **Total Supervised Samples**: 1,088 samples
  * **Training Samples**: 728 samples
  * **Validation Samples**: 128 samples
  * **Testing Samples**: 232 samples

### Tensor Shapes
* **X_train**: `(728, 50, 5)`
* **y_train**: `(728, 1)`
* **X_val**:   `(128, 50, 5)`
* **y_val**:   `(128, 1)`
* **X_test**:  `(232, 50, 5)`
* **y_test**:  `(232, 1)`

---

## 7. Implications for Model Design

The following considerations and hypotheses emerge directly from the observed data characteristics and guide the model design:

1. **Recurrent Neural Networks (RNN) Suitability**:
   * Stock price data is sequential and chronologically dependent. RNN-based architectures (LSTM, GRU, SimpleRNN) are highly suitable because they maintain hidden state representations that can capture temporal dependencies across the 50-day lookback window.
2. **Moderate Model Complexity & Overfitting Risks**:
   * With a total of only 728 training sequences and highly noisy inputs, the risk of **overfitting** is exceptionally high. 
   * Complex models with high parameter counts (e.g., deep stacked LSTMs with large hidden units) are highly susceptible to memorizing training noise (such as the high-volatility COVID-19 fluctuations), which will degrade their out-of-sample generalization.
3. **Deep vs. Shallow Networks**:
   * Extremely deep networks (e.g., 4+ layers) are not necessary and would likely overfit rapidly due to the small sample size (728 samples). A small-to-medium recurrent architecture (e.g., 1 or 2 layers, with 64 or 128 hidden units) serves as a much more reasonable starting point.
4. **Cell Type Performance Hypotheses**:
   * *Hypothesis A (SimpleRNN)*: Because the testing set is a relatively steady, linear upward trend (with low volatility of 14.79%) compared to the volatile training set, a lower-capacity model like `SimpleRNN` might perform better than LSTMs by acting as a structural regularizer, avoiding the memorization of complex training-set noise.
   * *Hypothesis B (LSTM/GRU)*: Gated architectures (LSTMs, GRUs) will prevent gradient vanishing over the 50-day lookback sequence but require strong dropout regularization to prevent noise overfitting.
5. **Feature Scaling Importance**:
   * Independent MinMax normalization is critical because it bounds the raw `volume` feature (which reaches up to $1.7 \times 10^7$) to the same $[0, 1]$ scale as the price features, ensuring stable and balanced gradient updates during backpropagation.
