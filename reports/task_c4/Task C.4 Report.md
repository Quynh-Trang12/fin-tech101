# Option C - Task C.4 Machine Learning 1 Report

## Project Details
- **Project:** FinTech101 Stock Price Prediction System
- **Subject:** COS30018 - Intelligent Systems
- **Task:** Option C - Task C.4: Machine Learning 1 - Recurrent Neural Networks & Hyperparameter Sweeps
- **Target Ticker:** Commonwealth Bank of Australia (`CBA.AX`)
- **Report Date:** 22 June 2026

---

## Introduction
This report documents the design, implementation, and empirical results of the deep learning forecasting models and hyperparameter sweep for Task C.4. We designed a dynamic model factory supporting Vanilla Recurrent Neural Networks (SimpleRNN), Gated Recurrent Units (GRU), and Long Short-Term Memory (LSTM) cells. We developed training and testing pipelines with deterministic seed constraints and resolved the Keras Mean Absolute Error (MAE) scaling bug. Finally, we conducted 8 distinct configuration sweeps on the real historical `CBA.AX` stock price dataset under a strict chronological time-series split, analyzing the impact of recurrent cell mechanics, depth, layer width, and loss formulations.

---

## 1. Pipeline Architecture & Implementation Details

### 1.1 Dynamic Model Factory (`src/model_factory.py`)
To build models programmatically, we implemented the `build_dl_model` function in [model_factory.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/model_factory.py). The factory:
* Resolves cell string keys (`"LSTM"`, `"GRU"`, `"SimpleRNN"`) to respective Keras layer classes.
* Assembles stacked recurrent layers. To support stack depth, all layers except the last must set `return_sequences=True`. The terminal recurrent layer sets `return_sequences=False` to produce a 2D state vector.
* Supports Keras `Bidirectional` wrapper layers.
* Appends `Dropout` regularization layers after each recurrent layer to prevent overfitting.
* Appends a final single-neuron `Dense(1, activation="linear")` layer for next-step price prediction.

### 1.2 Training Pipeline (`src/train.py`)
The training script [train.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/train.py) coordinates data preprocessing and model optimization. Key features:
* **Global Seed Determinism**: Sets `os.environ['PYTHONHASHSEED']`, `random.seed(314)`, `np.random.seed(314)`, and `tf.random.set_seed(314)` at import time and function execution to ensure reproducible model initializations.
* **Date Splitting Integration**: Loads the real daily dataset from [data_processing.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/data_processing.py) with a strict chronological split (`split_date="2023-08-02"`).
* **Weight Checkpoint Saving**: Trains models using TensorFlow callbacks/fit interfaces and writes weight matrices to `results/{model_name}.weights.h5`.

### 1.3 Testing Pipeline & Metric Evaluation (`src/test.py`)
The evaluation script [test.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/test.py) loads weight checkpoints and evaluates test data. It resolves the Keras MAE scaling bug by inverting predicted and actual prices *before* calculating errors, ensuring correct unscaled metrics:
1. **Mean Absolute Error (MAE)**:
   $$MAE = \frac{1}{N} \sum_{t=1}^{N} |y_t - \hat{y}_t|$$
2. **Root Mean Squared Error (RMSE)**:
   $$RMSE = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2}$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$MAPE = \frac{100\%}{N} \sum_{t=1}^{N} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$
4. **Directional Accuracy (DA)**:
   $$DA = \frac{1}{N-1} \sum_{t=2}^{N} \mathbb{I}\left(\text{sgn}(y_t - y_{t-1}) == \text{sgn}(\hat{y}_t - y_{t-1})\right)$$

It also runs a simulated trading strategy (buying when predicted price is higher than current price, selling when lower) to report **Trading Accuracy**, **Total Profit**, and **Profit per Trade**.

---

## 2. Hyperparameter Sweep Strategy

To isolate and test specific neural architectural assumptions, we defined 8 configurations grouped by the hypothesis each tests. Each configuration modifies exactly one structural parameter relative to the control baseline (`LSTM_BASE`).

The configuration keys are self-descriptive: the prefix identifies the recurrent cell type (`LSTM`, `GRU`, `RNN`) and the suffix describes the structural variant (`BASE`, `STACKED`, `SHALLOW`, `WIDE`, `NARROW`, `MSE`).

| Model Name | Cell Type | Layers | Units | Loss | Epochs | Batch Size | Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM_BASE** | LSTM | 2 | 128 | Huber | 20 | 64 | Standard baseline control benchmark |
| **GRU_BASE** | GRU | 2 | 128 | Huber | 20 | 64 | GRU parameter efficiency (no separate cell state) |
| **RNN_BASE** | SimpleRNN | 2 | 128 | Huber | 20 | 64 | Vanishing gradients on 50-day windows |
| **LSTM_STACKED** | LSTM | 3 | 128 | Huber | 20 | 64 | Representational hierarchy vs parameter overfitting |
| **LSTM_SHALLOW** | LSTM | 1 | 128 | Huber | 20 | 64 | Occam's Razor: simple regularization |
| **LSTM_WIDE** | LSTM | 2 | 256 | Huber | 20 | 64 | State capacity expansion vs overfitting noise |
| **LSTM_NARROW** | LSTM | 2 | 64 | Huber | 20 | 64 | Bottleneck compression as regularizer |
| **LSTM_MSE** | LSTM | 2 | 128 | MSE | 20 | 64 | Outlier sensitivity (quadratic) vs robust (Huber) fitting |

---

## 3. Empirical Results & Comparative Analysis

We executed the sweep using the modular experiment runner [run_sweeps.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/run_sweeps.py), which references parameters defined in [config.py](file:///s:/COS30018-Intelligent-System/fin-tech101/src/config.py). The results are tabulated below:

### 3.1 Sweep Metrics Table

| Model Name | Cell Type | Layers | Units | Loss | MAE ($) | RMSE ($) | MAPE (%) | DA (%) | Trading Acc (%) | Total Profit ($) | Profit/Trade ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM_BASE** | LSTM | 2 | 128 | huber | 3.0486 | 3.5555 | 2.87% | 45.02% | 45.02% | -$21.17 | -$0.09 |
| **GRU_BASE** | GRU | 2 | 128 | huber | 3.5731 | 3.8624 | 3.42% | 45.89% | 45.89% | -$24.90 | -$0.11 |
| **RNN_BASE** | SimpleRNN | 2 | 128 | huber | **2.0370** | **2.5901** | **1.89%** | 45.02% | 45.02% | -$21.12 | -$0.09 |
| **LSTM_STACKED** | LSTM | 3 | 128 | huber | 3.9252 | 4.6676 | 3.66% | **46.32%** | **46.32%** | **-$18.18** | **-$0.08** |
| **LSTM_SHALLOW** | LSTM | 1 | 128 | huber | 4.4890 | 4.8137 | 4.30% | 45.89% | 45.89% | -$24.90 | -$0.11 |
| **LSTM_WIDE** | LSTM | 2 | 256 | huber | 3.9878 | 4.4780 | 3.76% | 45.45% | 45.45% | -$25.54 | -$0.11 |
| **LSTM_NARROW** | LSTM | 2 | 64 | huber | 2.8820 | 3.4455 | 2.70% | 45.02% | 45.02% | -$25.56 | -$0.11 |
| **LSTM_MSE** | LSTM | 2 | 128 | mse | 3.0625 | 3.5699 | 2.88% | 45.02% | 45.02% | -$21.17 | -$0.09 |

### 3.2 Key Findings and Architectural Analysis

1. **SimpleRNN Outperforms Advanced Cells (Cell Type Comparison)**:
   Surprisingly, `RNN_BASE` (SimpleRNN) achieved the lowest error metrics overall (MAE: **$2.0370**, MAPE: **1.89%**), substantially outperforming `LSTM_BASE` ($3.0486) and `GRU_BASE` ($3.5731). This can be explained by **overfitting and signal complexity**. LSTMs and GRUs are designed to model long-term sequence dependencies, which makes them highly expressive. However, in noisy stock markets, high model capacity often leads to memorizing training noise, resulting in poor generalization on out-of-sample data. The SimpleRNN has lower capacity, which acts as a structural regularizer, forcing it to fit simpler, low-variance trends.
2. **Optimal Stack Depth (Occam's Razor)**:
   * `LSTM_SHALLOW` (1 layer) underperformed (MAE: $4.4890), indicating it lacked the representational depth to learn basic relationships.
   * `LSTM_BASE` (2 layers) improved performance (MAE: $3.0486), representing the optimal capacity sweet spot.
   * `LSTM_STACKED` (3 layers) degraded error performance (MAE: $3.9252), confirming that excess depth leads to overfitting on time-series noise.
3. **Narrow Width as a Regularizer**:
   Reducing LSTM layer width to 64 units (`LSTM_NARROW`) improved the MAE from $3.0486 to **$2.8820**. Conversely, widening the layers to 256 units (`LSTM_WIDE`) increased the MAE to $3.9878. This empirically validates the information bottleneck hypothesis: narrower layers restrict the model's ability to memorize transient noise, forcing it to focus on dominant price trends.
4. **Huber Loss vs MSE**:
   `LSTM_BASE` (Huber) slightly outperformed `LSTM_MSE` with an MAE of $3.0486 versus $3.0625. Huber Loss's linear penalty for large errors makes it less sensitive to stock price spikes and outliers, yielding more stable gradients and better generalization.
5. **Trading Realities and Directional Limits**:
   Across all sweeps, Directional Accuracy (DA) and Trading Accuracy hovered between **45% and 46%**. Because the model predicts the correct price direction less than half the time, the trading simulation incurred losses (~$18 to ~$25). This highlights the limitations of univariate/basic multivariate price history. In real-world applications, trading models require supplementary features (such as technical oscillators, volume flow, sentiment scores, and macroeconomic factors) to achieve directional edge.

---

## 4. Verification and Execution Evidence

### 4.1 Terminal Execution Screenshot
The screenshot below shows the PowerShell terminal execution of the hyperparameter sweep script:
![PowerShell Sweep Script Run](screenshots/c4_terminal.png)

### 4.2 Prediction Chart Comparison
The charts below show the actual versus predicted prices on the test set.
- The **LSTM_BASE** plot shows the typical smoothing effect of LSTM models.
- The **RNN_BASE** plot tracks the actual prices more closely, leading to its superior MAE score:

| LSTM_BASE | RNN_BASE |
| :---: | :---: |
| ![LSTM_BASE Prediction](file:///s:/COS30018-Intelligent-System/fin-tech101/results/LSTM_BASE_prediction.png) | ![RNN_BASE Prediction](file:///s:/COS30018-Intelligent-System/fin-tech101/results/RNN_BASE_prediction.png) |

---

## 5. Directory Structure & Professional Standards

The project maintains a modular layout that follows SOLID principles. Code dividers are implemented throughout all scripts to group functionalities logically.

```text
fin-tech101/
├── data/
│   └── CBA.AX_cache.csv             # Cleaned historical data cache (2020-01-01 to 2024-07-04)
├── results/
│   ├── LSTM_BASE_prediction.png     # LSTM_BASE Forecast Plot
│   ├── RNN_BASE_prediction.png      # RNN_BASE Forecast Plot
│   └── c4_sweep_results.csv         # Consolidated sweep results CSV
├── reports/
│   └── task_c4/
│       ├── Task C.4 Report.md       # This report
│       └── screenshots/
│           └── c4_terminal.png      # PowerShell sweep run screenshot
└── src/
    ├── config.py                    # Experiment parameters & sweep settings (Task C.4)
    ├── data_processing.py           # Preprocessing & chronological splits (Task C.2)
    ├── model_factory.py             # Dynamic recurrent model factory (Task C.4)
    ├── run_sweeps.py                # Sweeps experiment runner script (Task C.4)
    ├── train.py                     # Training pipeline (Task C.4)
    └── test.py                      # Unscaled testing pipeline (Task C.4)
```

---

## 6. Conclusion
Task C.4 has been successfully completed. We implemented a dynamic model factory and deterministic training/testing pipelines, resolving the reference codebase's MAE scaling bug. Running the 8 configurations on real `CBA.AX` stock prices under a chronological split showed that narrower, simpler model structures (such as `RNN_BASE` and `LSTM_NARROW`) generalize best to unseen prices by resisting market noise. These findings set a solid baseline for the multivariate and multi-step tasks in C.5.
