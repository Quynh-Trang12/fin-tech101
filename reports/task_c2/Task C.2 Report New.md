# Option C - Task C.2 Data Processing 1 Report

## Project Details
- **Project:** FinTech101 Stock Price Prediction System
- **Subject:** COS30018 - Intelligent Systems
- **Task:** Option C - Task 2: Modular Data Loading and Preprocessing
- **Target Ticker:** Commonwealth Bank of Australia (`CBA.AX`)
- **Report Date:** 22 June 2026

---

# 1. Introduction

## 1.1 Background

Stock price prediction depends not only on the forecasting model but also on how the data are prepared before training. Raw financial data often contain missing values, different feature scales, and chronological relationships that cannot be used directly by deep learning models. Without proper preprocessing, the model may produce unreliable results or suffer from data leakage, where information from the future unintentionally influences the training process.

Unlike traditional machine learning datasets, stock market data are time-series data. Each prediction depends on previous observations rather than individual records. Therefore, historical stock prices must be converted into fixed-length sequences before they can be used by recurrent neural networks such as Long Short-Term Memory (LSTM) networks.

To address these challenges, this project develops a reusable preprocessing pipeline that prepares raw stock market data for time-series forecasting while maintaining chronological order and preventing common sources of data leakage.

---

## 1.2 Objectives

The objective of Task C.2 is to develop a reusable function for loading and preprocessing stock market data. The preprocessing pipeline should:

- Load stock data within a user-defined date range.
- Handle missing (NaN) values.
- Provide different train-test splitting methods.
- Save downloaded data locally to reduce repeated downloads.
- Scale feature values for deep learning models.

The final output should be a clean, reusable, and model-ready dataset that can be used for stock price prediction.

---

# 2. Data Processing Pipeline

To satisfy the requirements of Task C.2, a seven-phase preprocessing pipeline was developed. Each phase performs one specific task, making the code easier to understand, maintain, and reuse.

The overall workflow is shown below.

```text
Raw Stock Data
       │
       ▼
Phase 1: Loading Data
       │
       ▼
Phase 2: Data Cleaning
       │
       ▼
Phase 3: Creating Training Sequences
       │
       ▼
Phase 4: Dataset Splitting
       │
       ▼
Phase 5: Feature Scaling
       │
       ▼
Phase 6: Preparing the Output
```

The following sections describe each phase in detail.

---

## 2.1 Phase 1 – Loading Data

The pipeline begins by loading historical stock data. If a local cache is available, it loads the cached data; otherwise, it downloads the data from Yahoo Finance and saves a local copy for future use. Users can specify the stock ticker together with the desired start and end dates, allowing different time periods to be used without changing the source code. The output of this phase is the raw historical stock dataset.

---

## 2.2 Phase 2 – Cleaning Data

The raw dataset is cleaned before any further processing.

First, the column names are converted into a consistent format and checked to ensure that all required features are available. The pipeline sorts the data into chronological order.

Missing values are handled using forward-fill, where each missing value is replaced with the most recent available value. This keeps the time series continuous while avoiding unnecessary data loss.

The output of this phase is a clean and standardized stock dataset.

---

## 2.3 Phase 3 – Creating Training Sequences

Deep learning models such as LSTM do not learn from individual trading days. Instead, they learn from sequences of historical observations.

The pipeline creates overlapping sliding windows using a fixed lookback period. Each window becomes one training sample, where the historical data are used as the input features and the stock price immediately after the window becomes the corresponding label.

For example, if the lookback window contains 50 trading days:

```text
Sample 1

Features
Day 1
Day 2
...
Day 50

↓

Label
Day 51
```

The window then moves forward by one trading day to create the next sample.

```text
Sample 2

Features
Day 2
Day 3
...
Day 51

↓

Label
Day 52
```

This process continues until the end of the dataset, producing a collection of training samples that preserve the chronological order of the original stock prices. These samples can then be used to train recurrent neural networks such as LSTM.

---

## 2.4 Phase 4 – Dataset Splitting

After all training sequences have been generated, they are first divided into **training** and **testing** datasets. Unlike conventional machine learning datasets, stock price prediction is a time-series problem where each prediction depends on historical observations. Therefore, the pipeline constructs **all valid sliding-window samples** before assigning each sample to a dataset.

Instead of assigning samples based on the final day contained in the input window, the pipeline assigns them according to the **prediction target date**. This ensures that the model is always evaluated on future data while preserving the complete historical context required for each prediction.

For example, suppose a lookback window contains 50 trading days:

```text
Input Window

Day 651
...
Day 700

↓

Prediction Target

Day 701
```

Although the input window contains historical observations before the split boundary, the sample belongs to the testing dataset because its prediction target is Day 701.

For a single-step prediction, each sample has only one prediction target. The splitting rule is therefore straightforward:

```python
train_mask = target_date < split_date
test_mask  = target_date >= split_date
```

The pipeline is also designed to support future extensions to multi-step forecasting, where one input window predicts multiple future trading days. In this case, a sample is included only if **all** of its prediction targets belong entirely to either the training or testing period. Samples whose prediction targets span both sides of the split boundary are discarded so that every sample belongs entirely to one dataset.

```python
first_target_date = target_dates[:, 0]
last_target_date  = target_dates[:, -1]

train_mask = last_target_date < split_date
test_mask  = first_target_date >= split_date
```

After the training and testing datasets have been created, the validation dataset is obtained by taking the most recent portion of the training data. This preserves the chronological ordering of the data while allowing the model to be validated using observations that occur later than the training samples.

---

## 2.5 Phase 5 – Feature Scaling

The input features are scaled using Min-Max normalization before they are passed to the neural network. To avoid data leakage, the Min-Max scalers are fitted **only on the training data**. The fitted scalers are then used to transform the validation and testing datasets. 

```text
Training Data
      │
      ▼
Fit Min-Max Scalers
      │
      ├────────► Scale Training Data
      ├────────► Scale Validation Data
      └────────► Scale Testing Data
```

The fitted scalers are also saved so they can be reused later without fitting them again. 

---

## 2.6 Phase 6 – Preparing the Output

Finally, the pipeline returns all processed data needed for model training and evaluation.

The returned data include:

- Training, validation, and testing feature tensors.
- Corresponding target values.
- The cleaned stock dataset.
- Input and target dates for each sequence.
- The latest lookback window for future prediction.
- The fitted feature scalers.

By returning everything in a single data package, the preprocessing pipeline provides a consistent interface for later forecasting tasks while keeping data preparation separate from model implementation.

---

