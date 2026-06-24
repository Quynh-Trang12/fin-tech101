# Option C - Task C.1 Setup and Verification Report

## Project Details

* **Project:** FinTech101 Stock Price Prediction System
* **Subject:** COS30018 – Intelligent Systems
* **Task:** Option C – Task C.1: Setup and Verification
* **Target Stock:** Commonwealth Bank of Australia (`CBA.AX`)
* **Report Date:** 23 June 2026

---

# 1. Introduction

Task C.1 required the setup and verification of two stock price prediction codebases: the provided baseline implementation (`v0.1`) and the external project (`P1`). Both codebases were downloaded, configured, executed, and analysed on a local machine using a shared Python virtual environment.

To support meaningful comparison, both systems were configured to operate on the same Commonwealth Bank of Australia (`CBA.AX`) stock dataset. The project repository was also established and organised to preserve the original downloaded code while providing runnable copies for experimentation and verification.

The objectives of this report are to document the environment setup process, verify that both codebases execute successfully, demonstrate understanding of the baseline implementation, and compare the prediction behaviour of the two systems using a common dataset.

---

# 2. Environment Setup and Repository Configuration

## 2.1 Virtual Environment Setup Rationale

A dedicated Python virtual environment (`.venv`) was created to isolate project dependencies from the host operating system and to ensure reproducible execution across future project tasks.

The environment was created and configured using the following commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.12 was selected deliberately rather than defaulting to Python 3.11 simply because TensorFlow and Keras projects often appear more stable on older Python versions. Although Python 3.11 remains a valid choice, its support window is shorter than Python 3.12. Since this project begins in mid-2026 and is expected to continue across multiple Option C tasks, Python 3.12 provides a longer maintenance window while still remaining compatible with TensorFlow 2.17.1 and the project dependency stack.

This decision reduces the chance that the project environment becomes outdated shortly after completion. Python 3.11 reaches end-of-life in October 2027, while Python 3.12 remains supported until October 2028. The below figure illustrates the Python release lifecycle used to justify the selection of Python 3.12 for this project.

![Python release lifecycle](screenshots/release-cycle.svg)

---

## 2.2 Dependency Specifications

A single `requirements.txt` file was created so that both `v0.1` and `P1` could share the same execution environment.

The primary dependencies are:

| Package            | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| TensorFlow 2.17.1  | Deep learning framework used to construct and train LSTM models |
| NumPy 1.26.4       | Numerical array operations                                      |
| Pandas 2.2.3       | Time-series processing and CSV handling                         |
| Scikit-learn 1.5.2 | Data normalisation and dataset preparation                      |
| Matplotlib 3.9.2   | Prediction visualisation                                        |
| yfinance 0.2.48    | Historical stock data retrieval                                 |
| yahoo-fin 0.8.9.1  | Additional market data utilities used by P1                     |

Using a common dependency specification simplifies environment management and ensures both codebases can be executed consistently throughout the project.

---

## 2.3 GitHub Repository and Wiki Setup

Task C.1 requires both the project repository and project Wiki to be established. A GitHub repository was therefore created to serve as the central location for source code, generated outputs, documentation, and future project deliverables.

The repository is organised as follows:

```text
fin-tech101/
├── baselines/
├── csv-results/
├── data/
├── docs/
├── references/
├── reports/
├── results/
├── src/
└── wiki/
```

Key directories used during Task C.1 include:

* `references/` – original downloaded versions of `v0.1` and P1.
* `baselines/` – runnable copies used for testing and compatibility fixes.
* `data/` – locally cached stock datasets.
* `results/` – generated prediction plots and model outputs.
* `reports/` – task reports submitted throughout the project.
* `wiki/` – project documentation and weekly reports.

A project Wiki was also created to satisfy the documentation requirements of Task C.1. The Wiki will be used throughout the project to maintain setup instructions, task reports, technical notes, and implementation progress.

---

# 3. Baseline Codebase Analysis and Testing

## 3.1 Understanding v0.1

The `v0.1` codebase is the initial stock price prediction program supplied for the project. The entire workflow is implemented within a single Python file and follows a conventional LSTM-based prediction pipeline.

### Pipeline Overview

1. Download historical stock data.
2. Extract closing prices.
3. Scale prices using `MinMaxScaler`.
4. Create rolling windows of historical observations.
5. Train a stacked LSTM network.
6. Generate predictions and visualisations.

### Key Characteristics

* Input feature: Closing price only.
* Historical input window: 60 trading days.
* Model architecture: Three stacked LSTM layers with 50 units each.
* Loss function: Mean Squared Error (MSE).
* Prediction target: Next trading-day closing price.

Because only the closing price is used as input, `v0.1` represents a univariate prediction model and serves as the project's baseline implementation.

---

## 3.2 Testing v0.1

The baseline was executed using the shared Python virtual environment created for Task C.1.

### Issues Encountered

During testing, the original `yfinance` download mechanism repeatedly failed when retrieving `CBA.AX` data:

```text
Failed to get ticker 'CBA.AX' reason: Expecting value: line 1 column 1 (char 0)

1 Failed download:
['CBA.AX']: YFTzMissingError('$%ticker%: possibly delisted; no timezone found')
```

To ensure reliable execution, a separate utility (`data_downloader.py`) was introduced to download real historical stock data directly from Yahoo Finance's Query2 API and store the dataset locally.

### Modifications Applied

* Added local CSV fallback support.
* Saved prediction plots to the project results directory.
* Added headless plotting support for non-interactive execution.

### Execution Command

```powershell
python baselines/v0.1/stock_prediction.py
```

![v0.1 Terminal Execution](screenshots/v01_terminal.png)

### Generated Output

Prediction plot:

```text
results/c1/v01_prediction.png
```

Testing results:

```text
Prediction: [[119.83187]]
Mean Squared Error: 16.091603202310264
```

![v0.1 Prediction Plot](../../results/c1/v01_prediction.png)

The script executed successfully and produced the expected prediction visualisation.

---

## 3.3 Understanding P1

P1 is a more advanced stock prediction project obtained from GitHub and evaluated alongside `v0.1`.

Unlike `v0.1`, which stores all logic in a single file, P1 separates configuration, model construction, training, and testing into multiple modules.

### Main Components

```text
baselines/p1/
├── parameters.py
├── stock_prediction.py
├── train.py
└── test.py
```

### Key Characteristics

* Input features:

  * Adjusted close price
  * Volume
  * Open price
  * High price
  * Low price

* Historical input window: configurable.
* Future prediction step: configurable.
* Model architecture: Two LSTM layers with 256 units each.
* Loss function: Huber loss.
* Model checkpointing support.
* Separate training and testing workflows.

Compared with `v0.1`, P1 provides a more modular and configurable foundation for experimentation.

---

## 3.4 Testing P1

The original P1 project was executed using the same virtual environment as `v0.1`.

### Modifications Applied

The following changes were made to support a fair comparison with `v0.1` and successful execution in the current environment:

* Stock ticker changed from `AMZN` to `CBA.AX`.
* Historical input window changed from 50 to 60 days.
* Future prediction step changed from 15 to 1.
* TensorFlow and Keras compatibility fixes applied.
* Local CSV fallback support extended.

These changes were intended to align the comparison setup with `v0.1` while preserving the original P1 architecture and workflow.

### Execution Commands

```powershell
python baselines/p1/train.py
python baselines/p1/test.py
```

![P1 Terminal Execution](screenshots/p1_terminal.png)

### Generated Outputs

Model weights:

```text
results/c1/
2026-06-23_CBA.AX-sh-1-sc-1-sbd-0-huber-adam-LSTM-seq-60-step-1-layers-2-units-256.weights.h5
```

Prediction plot:

```text
results/c1/p1_prediction.png
```

Testing results:

```text
Future price after 1 days is 119.57$
Huber loss: 0.001373790088109672
Mean Absolute Error: 46.94808660307785
```

![P1 Prediction Plot](../../results/c1/p1_prediction.png)

The training and testing pipelines completed successfully, confirming that P1 can be executed within the same environment and dataset configuration used for `v0.1`.

---

# 4. Comparative Evaluation

## 4.1 What Defines a Better Prediction?

Task C.1 requires the performance of `v0.1` and P1 to be compared. Before comparing the two models, it is necessary to define what constitutes a better prediction.

For this project, a better prediction is not simply a prediction that produces a visually appealing graph. Instead, a better prediction should satisfy the following criteria:

* Produces predictions that closely follow the actual stock price.
* Demonstrates lower prediction error on unseen data.
* Maintains consistent behaviour when evaluated on future observations.
* Can be reproduced using the same dataset and configuration.

A prediction model should therefore be evaluated using both visual inspection and quantitative measures rather than relying on a single graph or metric.

---

## 4.2 Comparison Setup

To support a meaningful comparison, both codebases were executed using the same stock dataset.

### Shared Configuration

* Stock ticker: `CBA.AX`
* Historical input window: 60 trading days
* Future prediction step: 1 day
* Dataset source: locally cached Yahoo Finance data

### Major Differences Remaining

| Aspect            | v0.1                       | P1                       |
| ----------------- | -------------------------- | ------------------------ |
| Input features    | Close price only           | Multiple stock features  |
| Model structure   | 3 × 50-unit LSTM layers    | 2 × 256-unit LSTM layers |
| Loss function     | MSE                        | Huber loss               |
| Project structure | Single-file implementation | Modular implementation   |
| Train-test split  | Chronological              | Randomly shuffled        |


Although the comparison setup was aligned where practical, both projects still retain their original modelling approaches.

---

## 4.3 Comparative Results and Analysis

### Prediction Results

| Codebase | Observed Characteristics                                                                                                                                                                                                                                                                                                                            | Prediction Plot                                              |
| :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
|   v0.1   | - Successfully predicts the overall market trend.<br>- Prediction curve responds slowly to rapid price movements.<br>- Noticeable lag appears when prices change direction.<br>- Testing results: <br> Prediction: [[119.83187]] <br> Mean Squared Error: 16.091603202310264                                                                        | ![v0.1 Prediction Plot](../../results/c1/v01_prediction.png) |
|    P1    | - Prediction curve follows the actual stock price more closely.<br>- Smaller visual deviation between predicted and actual values.<br>- Generates quantitative evaluation metrics during testing.<br>- Testing results: <br> Future price after 1 days is 119.57$ <br> Huber loss: 0.001373790088109672 <br> Mean Absolute Error: 46.94808660307785 | ![P1 Prediction Plot](../../results/c1/p1_prediction.png)    |

---

### Finding 1: The Prediction Plot and Reported MAE Do Not Match

The P1 prediction plot appears to remain close to the actual stock price throughout most of the testing period.

However, P1 reports:

```text
Mean Absolute Error: 46.94808660307785
```

A prediction error of approximately $46 would normally produce much larger deviations than those visible in the prediction plot.

This suggests that the reported MAE is not being calculated on the same scale as the displayed stock prices. As a result, the reported MAE cannot be used confidently when comparing P1 against `v0.1`.

---

### Finding 2: Multiple Factors Changed Between the Two Models

At first glance, P1 appears to outperform `v0.1`.

To understand why, the implementation of both projects was reviewed.

The comparison revealed that P1 differs from `v0.1` in several ways:

* More input features.
* Larger LSTM layers.
* Different loss function.
* Different train-test split strategy.

Because several factors changed at the same time, the observed performance difference cannot be linked to a single cause.

From the current results, it is not possible to determine how much of P1's improvement comes from the model architecture and how much comes from the other differences in the implementation.

---

### Finding 3: P1 Uses a Different Evaluation Strategy

One of the most significant differences between the two projects is how the dataset is divided into training and testing data.

`v0.1` uses a chronological split:

```text
Older stock prices  → Training set
Newer stock prices  → Testing set
```

The model is trained using past prices and evaluated using future prices.

P1 uses:

```python
SHUFFLE = True
SPLIT_BY_DATE = False
```

Before the dataset is split, all samples are randomly shuffled.

As a result, both the training set and testing set contain samples drawn from the entire 2020–2024 dataset.

For example:

```text
2020–2024 data
        ↓
Random shuffle
        ↓
Training set + Testing set
```

This means stock prices from 2024 may appear in the training set while other stock prices from 2024 appear in the testing set.

The model is therefore no longer evaluated exclusively on future stock prices that were completely unseen during training.

This makes P1's prediction results difficult to compare directly with `v0.1`, because the two models are being evaluated under different conditions.

---

### Overall Interpretation

Based on the prediction plots, P1 appears to produce stronger prediction results than `v0.1`.

However, the code review revealed three important findings:

1. The reported MAE appears inconsistent with the prediction plot.
2. Multiple differences exist between the two implementations, including input features, model architecture, loss functions, and data preparation.
3. P1 uses a substantially different train-test split strategy that introduces significant data leakage.

These findings make it difficult to determine how much of P1's stronger performance comes from the model itself and how much comes from differences in data preparation and evaluation.

The most valuable outcome of Task C.1 was therefore not identifying a definitive winner, but identifying the factors that must be understood before a fair comparison can be made.
