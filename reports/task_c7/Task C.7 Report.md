# Option C Weekly Report: Task C.7 − Sentiment-Based Stock Price Movement Prediction (Extension)

## Project Details
- **Project:** FinTech101 Stock Price Prediction System
- **Subject:** COS30018 - Intelligent Systems
- **Task:** Option C - Task C.7: Extension - Sentiment Based Stock Price Movement Prediction
- **Target Dataset:** GDELT ecosystem − **Global Document Graph (GDG)** dataset
- **Report Due:** Week 12


# 1. Introduction

## 1.1 Background

Previous tasks in this project focused on forecasting stock prices using historical market data. While historical prices contain valuable information about market behaviour, stock prices are also influenced by external factors such as company announcements, financial news, and investor sentiment. Incorporating these external information sources may improve a model's ability to predict future market movements.

Task C.7 extends the previous work by investigating whether financial news sentiment can improve **next-day stock direction prediction**. Rather than predicting the exact closing price, this task formulates the problem as a binary classification problem, where the objective is to predict whether the next trading day's closing price will be higher than the current day's closing price.

---

## 1.2 Research Objective

The primary objective of this task is to evaluate whether sentiment extracted from financial news provides additional predictive value beyond historical market information.

To answer this question, a series of controlled experiments were designed using progressively richer feature sets while keeping the classification model unchanged.

The experiments compare the following feature representations:

```
Market Features
        ↓
Market + V2Tone
        ↓
Market + FinBERT
        ↓
Market + Reduced FinBERT
```

Using the same preprocessing pipeline, train-validation-test split, and Logistic Regression classifier throughout the study ensures that differences in predictive performance can be attributed to the sentiment representations rather than changes in the machine learning model.

The research question investigated in this task is:

> **Can external news sentiment improve next-day stock direction prediction, and which sentiment representation contributes the most?**

---

## 1.3 Independent Research Overview

Beyond the baseline task requirements, this project investigates the use of **FinBERT**, a transformer model trained specifically on financial text, as an alternative to GDELT's built-in V2Tone sentiment representation.

Implementing this extension required substantially more than simply applying a pre-trained model. The original GDELT Global Knowledge Graph (GKG) dataset does not contain clean article headlines suitable for transformer-based natural language processing. An additional investigation of the GDELT ecosystem identified the **Global Document Graph (GDG)** dataset, which was used to recover English news headlines by joining the two datasets using article URLs.

Following the initial FinBERT experiments, a feature audit was conducted to investigate why the transformer-based representation did not initially outperform the baseline model. The audit identified significant feature redundancy, leading to the design and evaluation of a reduced FinBERT feature set. This investigation forms the independent research component of the project.

---

## 1.4 System Overview

Figure 1 illustrates the overall workflow developed for Task C.7.

```
Historical Market Data
        +
  Financial News
        │
        ▼
Sentiment Extraction
(V2Tone / FinBERT)
        │
        ▼
Feature Engineering
        │
        ▼
Logistic Regression
        │
        ▼
Next-Day Direction Prediction
```

To improve maintainability and reproducibility, the workflow was implemented as a modular pipeline consisting of dedicated scripts for data collection, headline enrichment, sentiment generation, feature engineering, dataset construction, preprocessing, model evaluation, and feature analysis. Intermediate datasets are cached throughout the pipeline, allowing each stage to be reproduced independently without repeating computationally expensive steps such as BigQuery queries or FinBERT inference.

# 2. Data Engineering Pipeline

A reproducible data engineering pipeline was developed to transform raw financial news into sentiment features suitable for stock direction prediction. Rather than relying on manually downloaded datasets, each stage of the workflow was implemented as an independent Python module, allowing intermediate results to be cached and reused throughout the project.

Figure 2 illustrates the complete data engineering pipeline.

```
                    Google BigQuery
                           │
                           ▼
                 c7_news_data.py
                           │
                           ▼
              Raw GKG Article Dataset
                 (61,221 articles)
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
  V2Tone Pipeline                 FinBERT Pipeline 
          |                      (Independent Research)
          │                                 │
 c7_news_features.py                c7_news_titles.py
          │                                 │
    Daily V2Tone                     English Headlines
          │                        (56,361 titles)
          │                                 │
          │                       c7_finbert_features.py
          │                                 │
          │                    Article-level Sentiment
          │                                 │
          │                        c7_finbert_daily.py
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                  Trading-Day Alignment
                   c7_news_alignment.py
                           ▼
                     c7_dataset.py
                           ▼
                  c7_preprocessing.py
                           ▼
                     c7_baseline.py
                           ▼
                Comparative Experiments
```

---

## 2.1 Historical Stock Data

Historical market data for **Commonwealth Bank of Australia (CBA.AX)** was obtained from Yahoo Finance using the data processing pipeline developed in previous tasks. The dataset spans **1 January 2020 to 2 July 2024** and contains the standard OHLCV variables (Open, High, Low, Close, Volume, and Adjusted Close).

Rather than rebuilding the stock preprocessing workflow, Task C.7 reuses the cleaned historical market dataset generated in Task C.2. This ensures consistency across all project tasks while allowing the new sentiment features to be evaluated independently.

---

## 2.2 Financial News Collection

Financial news articles were collected from the **GDELT Global Knowledge Graph (GKG)** using Google BigQuery. A dedicated collection script (`c7_news_data.py`) was developed to automatically retrieve articles mentioning Commonwealth Bank over the same period as the stock dataset.

To improve reproducibility and avoid repeated BigQuery queries, the downloaded articles were cached locally as a Parquet dataset together with metadata describing the download.

Duplicate articles were removed using the unique `DocumentIdentifier` (article URL).

The final collection contained:

| Item | Count |
|------|------:|
| Raw downloaded articles | 61,274 |
| Duplicate URLs removed | 53 |
| Final unique articles | **61,221** |

The resulting dataset was stored as:

- `data/c7/gdelt_cba_raw.parquet`
- `data/c7/gdelt_cba_metadata.json`

---

## 2.3 Headline Enrichment

Although the GKG dataset provides article metadata and V2Tone sentiment, it does not contain clean article headlines suitable for transformer-based sentiment analysis. To overcome this limitation, an additional investigation of the GDELT ecosystem was conducted. The **Global Document Graph (GDG)** dataset was found to contain article titles indexed by webpage URL. A second processing module (`c7_news_titles.py`) was developed to retrieve English headlines from GDG and merge them with the original GKG articles using

```
DocumentIdentifier (GKG)
        =
page_url (GDG)
```

Only English headlines were retained to match the language supported by FinBERT. This enrichment process successfully recovered **56,361 English headlines**, representing **92.06%** of all collected articles, while keeping the entire workflow within the GDELT ecosystem and avoiding external web scraping. The enriched dataset was stored as:

- `data/c7/gdelt_cba_enriched.parquet`

---

## 2.4 Time Alignment

Accurate temporal alignment is essential when combining financial news with stock prices. News publication timestamps obtained from GDELT are recorded in **UTC**, whereas Australian stock prices follow the **Australia/Sydney** timezone. Directly matching UTC calendar dates to trading dates would incorrectly assign some articles to the wrong market session. To prevent this issue, publication timestamps were converted using the following process:

```
UTC Timestamp
        ↓
Australia/Sydney
        ↓
Local Calendar Date
        ↓
Next Available Trading Day
```

News published on weekends or exchange holidays was automatically forwarded to the next available trading day using the stock trading calendar. The same alignment procedure was applied consistently to both the V2Tone and FinBERT pipelines.

This functionality was implemented in `c7_news_alignment.py`, ensuring that every sentiment feature corresponds to the earliest trading session that could reasonably react to the published news.

---

## 2.5 Reproducible Pipeline

One design objective of this project was reproducibility. Instead of constructing the dataset within a single script, each processing stage produces an intermediate artifact that can be inspected or reused by later stages. This modular design reduces unnecessary computation, particularly for expensive stages such as BigQuery retrieval and FinBERT inference. Table 1 summarizes the main intermediate datasets produced during the pipeline.

| Artifact | Generated By | Purpose |
|----------|--------------|---------|
| `gdelt_cba_raw.parquet` | `c7_news_data.py` | Raw GKG article cache |
| `gdelt_cba_enriched.parquet` | `c7_news_titles.py` | GKG articles enriched with GDG headlines |
| `gdelt_daily_v2tone.parquet` | `c7_news_features.py` | Daily V2Tone features |
| `gdelt_v2tone_aligned.parquet` | `c7_news_alignment.py` | Trading-day aligned V2Tone features |
| `gdelt_finbert_article.parquet` | `c7_finbert_features.py` | Article-level FinBERT predictions |
| `gdelt_daily_finbert.parquet` | `c7_finbert_daily.py` | Daily aggregated FinBERT features |
| `gdelt_finbert_aligned.parquet` | `c7_finbert_daily.py` | Trading-day aligned FinBERT features |
| `c7_dataset.parquet` | `c7_dataset.py` | Final classification dataset |

This modular pipeline made it possible to rerun or validate individual stages without repeating the entire workflow, improving both development efficiency and experimental reproducibility.

# 3. Sentiment Analysis

Following data collection and preprocessing, financial news articles were converted into numerical sentiment features suitable for machine learning. Two different sentiment representations were investigated: GDELT's built-in **V2Tone** sentiment and **FinBERT**, a pre-trained NLP model to analyze sentiment of financial text. Both representations were aggregated to daily features and aligned with stock trading days before being used for classification.

---

## 3.1 V2Tone Sentiment

The first sentiment representation used in this project was **V2Tone**, a sentiment measure provided directly by the GDELT Global Knowledge Graph (GKG). Unlike traditional sentiment analysis models that require processing raw text, V2Tone is generated automatically by GDELT during article indexing. Each article contains a V2Tone string describing multiple sentiment-related attributes, including:

- overall tone
- positive word proportion
- negative word proportion
- polarity
- article activity
- self-reference density

A dedicated processing module (`c7_news_features.py`) was developed to parse these article-level values and aggregate them into daily sentiment features. Before aggregation, publication timestamps were converted from UTC to the Australia/Sydney timezone to ensure consistency with the stock market trading calendar.

Since multiple news articles may be published on the same day, daily sentiment values were calculated using weighted averages, while article counts were accumulated through summation. The resulting daily features were then forwarded to the trading-day alignment pipeline described in Section 2.

---

## 3.2 FinBERT Sentiment

Although V2Tone provides a convenient rule-based sentiment representation, it does not analyze the contextual meaning of financial language. Therefore, the independent research component of this project investigated **ProsusAI/FinBERT**, a transformer model pretrained on financial news and financial reports. 

Unlike V2Tone, FinBERT requires natural language input. The English headlines recovered during the headline enrichment stage were therefore used as the input to the model. For each headline, FinBERT produced:

- Positive probability
- Neutral probability
- Negative probability
- Predicted sentiment label
- Prediction confidence

A total of **56,361 English headlines** were processed using FinBERT. The article-level predictions were stored as an intermediate dataset (`gdelt_finbert_article.parquet`) before being aggregated into daily sentiment features by `c7_finbert_daily.py`.

Daily aggregation followed the same methodology as the V2Tone pipeline. Article-level probabilities were averaged, sentiment label counts were accumulated, confidence statistics were computed, and all features were aligned to Australian trading days using the shared alignment pipeline.

This modular design ensured that expensive transformer inference only needed to be performed once, allowing later experiments to reuse the cached article-level predictions.

---

## 3.3 Comparison of Sentiment Representations

The two sentiment representations differ substantially in both methodology and computational cost.

| Representation | Characteristics | Advantages | Limitations |
|---------------|-----------------|------------|-------------|
| **V2Tone** | Rule-based sentiment provided directly by GDELT | Fast, immediately available, no additional inference required | Limited contextual understanding of financial language |
| **FinBERT** | Transformer model trained on financial text | Captures contextual meaning and finance-specific terminology | Computationally expensive and requires clean English headlines |

V2Tone serves as a lightweight baseline sentiment representation, while FinBERT represents a more advanced, domain-specific NLP approach. Evaluating both methods within the same classification framework allows the effect of increasingly sophisticated sentiment representations to be assessed under controlled experimental conditions.

# 4. Feature Engineering and Experimental Design

The final stage of the pipeline combines historical market information with daily sentiment features to construct the classification dataset. To evaluate the contribution of each sentiment representation fairly, all experiments share the same target definition, train-validation-test split, preprocessing pipeline, and classification model. Only the input feature sets differ between experiments.

---

## 4.1 Classification Target

Unlike previous tasks that predicted future stock prices, Task C.7 formulates stock prediction as a binary classification problem.

The target variable is defined as:

```
Tomorrow Close > Today Close
        ↓
        1
Otherwise
        ↓
        0
```

where:

- **1** indicates that the closing price increased on the next trading day.
- **0** indicates that the closing price decreased or remained unchanged.

The classification dataset was constructed using `c7_dataset.py`, which merges the historical stock dataset with the aligned sentiment features before generating the binary target variable.

---

## 4.2 Feature Engineering

The baseline feature set consists of historical market variables developed in previous tasks. These include adjusted closing price and additional market features used throughout the project.

To investigate the contribution of external information, sentiment features were progressively incorporated into the baseline dataset. This produced six experimental feature sets.

| Experiment | Feature Set | Purpose |
|------------|-------------|---------|
| Baseline | Market Features | Historical market information only |
| A | Market + V2Tone | Evaluate GDELT's built-in sentiment representation |
| B | Market + Full FinBERT | Evaluate a domain-specific transformer-based sentiment representation |
| C | Market + Reduced FinBERT | Evaluate a refined FinBERT feature representation |
| D | Market + V2Tone + Full FinBERT | Combine both sentiment representations |
| E | Market + V2Tone + Reduced FinBERT | Combine V2Tone with the refined FinBERT representation |

The preprocessing workflow was implemented in `c7_preprocessing.py`, which constructs the feature groups, performs chronological train-validation-test splitting, and prepares the datasets for model training.

---

## 4.3 Experimental Design

To ensure that differences in performance were caused only by the sentiment representations, every experiment followed the same experimental protocol.

The following components remained unchanged throughout the study:

- Historical stock dataset
- Train-validation-test split
- Classification target
- Feature scaling procedure
- Logistic Regression configuration
- Evaluation metrics

By controlling these factors, the experiments isolate the effect of adding external sentiment information while avoiding confounding variables introduced by changes to the machine learning model.

---

## 4.4 Classification Model

Logistic Regression was selected as the classifier for all experiments.

Although more complex classifiers are available, the objective of this task is not to compare machine learning algorithms. Instead, the objective is to compare different sentiment representations under identical modelling conditions.

Logistic Regression provides a simple and interpretable baseline that makes it easier to observe how the addition of V2Tone and FinBERT features influences predictive performance. This controlled design ensures that any observed improvements or degradations can be attributed to the engineered sentiment features rather than differences in model complexity.

Before training, all numerical features were standardised using statistics computed from the training set only. The fitted scaler was then applied to the validation and test sets, preventing information leakage while ensuring that features with different numerical ranges contributed fairly during optimisation.

---

## 4.5 Experimental Workflow

The complete experimental workflow is summarised below.

```
Historical Stock Data
           +
Aligned Sentiment Features
           │
           ▼
   c7_dataset.py
           │
           ▼
Feature Engineering
           │
           ▼
c7_preprocessing.py
           │
           ▼
Logistic Regression
(c7_baseline.py)
           │
           ▼
Performance Evaluation
```

This modular workflow separates dataset construction, preprocessing, and model evaluation into independent stages, making the experiments reproducible and allowing individual components to be modified or validated without affecting the rest of the pipeline.

# 5. Experimental Results

The six experimental feature sets described in Section 4 were evaluated using the same Logistic Regression classifier and chronological train-validation-test split. Performance was measured using Accuracy, Balanced Accuracy, Precision, Recall, F1-score, and ROC AUC. Confusion matrices were also analysed to better understand the prediction behaviour of each model.

---

## 5.1 Overall Performance Comparison

Table 2 summarizes the performance of all evaluated feature sets.

| Feature Set | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC |
|--------------|---------:|------------------:|----------:|--------:|---:|--------:|
| **Market** | **0.554** | **0.515** | **0.549** | **0.992** | **0.707** | **0.584** |
| Market + V2Tone | 0.511 | 0.515 | 0.543 | 0.664 | 0.597 | 0.496 |
| Market + Full FinBERT | 0.502 | 0.501 | 0.542 | 0.512 | 0.526 | 0.531 |
| Market + Reduced FinBERT | 0.511 | **0.528** | 0.551 | 0.600 | 0.574 | **0.550** |
| Market + V2Tone + Full FinBERT | 0.511 | 0.521 | 0.544 | 0.592 | 0.567 | 0.506 |
| Market + V2Tone + Reduced FinBERT | 0.528 | 0.523 | 0.551 | 0.744 | 0.633 | 0.544 |

Overall, the historical market features achieved the highest Accuracy and ROC AUC. Although several sentiment-enhanced models improved specific evaluation metrics, none surpassed the market-only baseline across the overall test set.

---

## 5.2 Baseline Performance

The market-only classifier achieved the strongest overall performance, with an Accuracy of **55.4%** and a ROC AUC of **0.584**.

Inspection of the confusion matrix showed that the model strongly favoured predicting the positive class (price increase), resulting in very high Recall but relatively poor discrimination between the two classes. Consequently, Balanced Accuracy and ROC AUC provide a more informative assessment than Accuracy alone.

The baseline establishes a reference against which the contribution of each sentiment representation can be evaluated.

---

## 5.3 V2Tone Results

Adding V2Tone features reduced overall predictive performance compared with the market-only baseline.

Although V2Tone provides a readily available sentiment representation within GDELT, the aggregated daily sentiment features did not provide additional predictive information for next-day stock direction. The decrease in ROC AUC further suggests that the rule-based sentiment representation was unable to improve the model's ranking ability.

---

## 5.4 FinBERT Results

Replacing V2Tone with FinBERT improved several evaluation metrics, including Balanced Accuracy and ROC AUC. This indicates that the transformer-based sentiment representation captured more informative financial sentiment than the rule-based V2Tone features.

However, the Full FinBERT feature set still did not outperform the market-only baseline, suggesting that richer sentiment representations alone are insufficient to improve prediction under the current experimental setting.

---

## 5.5 Reduced FinBERT Results

Following the initial FinBERT experiments, a reduced feature representation was evaluated.

Compared with the Full FinBERT feature set, the Reduced FinBERT representation improved both Accuracy and ROC AUC while using fewer features. When combined with V2Tone, the reduced feature set also produced the strongest performance among all sentiment-enhanced models.

These findings suggest that removing redundant sentiment variables produces a cleaner feature representation without increasing model complexity.

---

## 5.6 Result Summary

The experimental results highlight three key observations.

1. Historical market features remained the strongest predictor of next-day stock direction.

2. FinBERT consistently outperformed the rule-based V2Tone representation, indicating that a finance-specific transformer extracted more informative sentiment features.

3. Reducing redundancy within the FinBERT feature set produced measurable improvements over the original FinBERT representation, although the resulting models still did not surpass the market-only baseline.

Complete evaluation outputs, confusion matrices, and experiment summaries are included in the project repository under `csv-results/c7/` for reproducibility.

# 6. Independent Investigation

The initial experiments showed that replacing V2Tone with FinBERT did not outperform the market-only baseline, despite FinBERT being a transformer model trained specifically for financial text. Rather than accepting these results directly, an additional investigation was conducted to understand the behaviour of the engineered sentiment features and determine whether the feature representation itself could be improved.

---

## 6.1 Motivation

The Full FinBERT feature set contained ten engineered sentiment features, including probability estimates, confidence measures, article counts, and sentiment label proportions. Although these features provided a richer representation than V2Tone, the initial experiments showed only limited predictive performance. This suggested that increasing the number of features alone did not necessarily improve the classifier. To better understand this behavior, a diagnostic feature audit was performed using the dedicated analysis module `c7_feature_audit.py`.

---

## 6.2 Feature Audit

Several complementary analyzes were performed on the Full FinBERT feature set, including:

- Pearson correlation analysis
- Variance Inflation Factor (VIF)
- Logistic Regression coefficient analysis
- Permutation feature importance

The audit revealed substantial redundancy within the engineered features.

First, the three sentiment probabilities (positive, neutral, and negative) always sum to one for every article. After daily aggregation, this mathematical relationship is preserved, meaning that one probability can always be inferred from the other two.

Similarly, the three sentiment label shares also sum to one, providing another source of deterministic dependency.

In addition, several feature pairs exhibited extremely high correlations. For example, the average positive probability and positive article share measured nearly the same information, while the confidence statistics were also strongly correlated.

These findings indicated that the original feature representation contained unnecessary redundancy, leading to unstable model coefficients without providing additional predictive information.

---

## 6.3 Reduced FinBERT Representation

Based on the feature audit, a reduced FinBERT feature representation was designed to preserve the most informative aspects of the sentiment data while removing redundant variables.

The reduced representation contains four features:

| Feature | Purpose |
|----------|---------|
| `finbert_article_count` | Measures the amount of financial news published on each trading day. |
| `finbert_positive_probability_mean` | Represents the average positive sentiment strength. |
| `finbert_negative_probability_mean` | Represents the average negative sentiment strength. |
| `finbert_confidence_mean` | Represents the average confidence of the FinBERT predictions. |

Compared with the original ten-feature representation, the reduced feature set removes:

- neutral probability
- sentiment label shares
- maximum confidence
- binary news indicator

These variables were excluded because they either contained mathematically dependent information or duplicated information already represented by the retained features.

The resulting feature set remains fully interpretable while substantially reducing feature redundancy.

---

## 6.4 Re-evaluation

The reduced FinBERT feature set was evaluated using the same preprocessing pipeline, Logistic Regression classifier, and train-validation-test split as all previous experiments.

Compared with the original Full FinBERT representation, the reduced feature set produced measurable improvements across multiple evaluation metrics while using fewer features.

| Model | Accuracy | ROC AUC |
|--------|---------:|--------:|
| Market + Full FinBERT | 0.502 | 0.531 |
| Market + Reduced FinBERT | **0.511** | **0.550** |

When combined with V2Tone, the reduced representation also produced the strongest performance among all sentiment-enhanced models.

Although these improvements were not sufficient to outperform the market-only baseline, they demonstrate that careful feature engineering can improve the effectiveness of transformer-derived sentiment features.

---

## 6.5 Summary

The independent investigation extended the baseline implementation in three important ways.

1. An additional GDELT dataset (GDG) was investigated to recover English headlines suitable for transformer-based sentiment analysis.

2. A complete FinBERT inference pipeline was developed to generate article-level and daily financial sentiment features.

3. A systematic feature audit identified redundancy within the original FinBERT representation, leading to the design and evaluation of a reduced feature set that improved predictive performance while preserving interpretability.

Together, these investigations demonstrate that the contribution of domain-specific NLP models depends not only on the sentiment model itself, but also on how the resulting features are engineered and integrated into the downstream prediction pipeline.

# 7. Discussion

The experiments demonstrate that incorporating financial news sentiment is not guaranteed to improve stock direction prediction. Although both V2Tone and FinBERT provided additional external information, neither representation consistently outperformed the historical market features alone under the current experimental setting.

One possible explanation is that the historical market variables already capture a substantial amount of information reflected in publicly available news. Since financial markets react rapidly to new information, daily aggregated sentiment may contain limited additional predictive value for next-day price movements.

The comparison between V2Tone and FinBERT nevertheless provides several useful observations. Across nearly all experiments, FinBERT produced stronger performance than the rule-based V2Tone representation. As a transformer model trained specifically on financial text, FinBERT is better able to capture contextual meaning and domain-specific language, whereas V2Tone relies on predefined lexical rules. Although the improvement was modest, the results suggest that domain-specific sentiment representations are more suitable for financial prediction tasks than generic sentiment measures.

The feature audit also highlights the importance of feature engineering when working with transformer-generated outputs. Simply adding a larger number of sentiment features did not improve predictive performance. Instead, removing redundant and highly correlated variables produced a cleaner feature representation and improved several evaluation metrics while preserving interpretability. This finding reinforces that feature quality is often more important than feature quantity.

Several limitations should also be acknowledged. First, sentiment was aggregated at the daily level, which may lose valuable intraday information about when articles were published and how quickly markets reacted. Second, the study focused on a single Australian stock (CBA.AX), so the findings may not generalise to other companies or industries. Finally, Logistic Regression was intentionally selected as a simple and interpretable baseline to isolate the contribution of the engineered sentiment features. More sophisticated classifiers may capture nonlinear relationships between market data and sentiment, although this was beyond the scope of the current investigation.

Future work could explore higher-frequency news data, additional financial text sources, larger groups of stocks, or alternative classification models while retaining the reproducible data engineering pipeline developed in this project.

---

# 8. Conclusion

This project investigated whether external financial news sentiment can improve next-day stock direction prediction beyond historical market data.

A complete and reproducible pipeline was developed to collect financial news from GDELT, enrich articles with English headlines, generate both V2Tone and FinBERT sentiment features, align news with Australian trading days, and construct classification datasets for controlled experimentation.

The experiments showed that historical market features remained the strongest predictor of next-day price direction. While FinBERT consistently outperformed the rule-based V2Tone representation, neither sentiment approach surpassed the market-only baseline. An additional feature audit identified substantial redundancy within the original FinBERT feature set, leading to the development of a reduced representation that improved predictive performance while using fewer features.

Overall, the project demonstrates that integrating domain-specific NLP techniques into financial prediction requires more than simply applying a sentiment model. Careful data engineering, temporal alignment, feature design, and systematic evaluation are all essential for producing reliable and reproducible machine learning experiments.