# Option C - Task C.3 Data Processing 2 Report

## Project Details
- **Project:** FinTech101 Stock Price Prediction System
- **Subject:** COS30018 - Intelligent Systems
- **Task:** Option C - Task 3: Advanced Visualizations (Candlestick & Moving Boxplots)
- **Target Ticker:** Commonwealth Bank of Australia (`CBA.AX`)
- **Report Date:** 22 June 2026

---

# 1. Introduction

## 1.1 Background

Financial markets generate large volumes of time-series data that are often difficult to interpret directly from numerical tables. Although statistical summaries provide useful information, they do not clearly reveal market trends, price volatility, or short-term trading behaviour. Visualizing historical stock prices therefore plays an important role in exploratory data analysis by allowing important characteristics of the data to be identified before predictive models are developed.

Among the many visualization techniques used in financial analysis, **candlestick charts** and **boxplots** provide complementary perspectives of market behaviour. Candlestick charts summarize the opening, highest, lowest, and closing prices over a specified trading period, making them useful for observing price movements, trends, and trading activity. In contrast, boxplots summarize the statistical distribution of prices within consecutive trading windows, highlighting changes in variability, median price, and potential outliers over time.

Building upon the reusable preprocessing pipeline developed in Task C.2, this task implements a modular visualization module for the historical stock prices of **Commonwealth Bank of Australia (`CBA.AX`)**. The module generates configurable candlestick charts and windowed boxplots directly from the processed dataset, providing visual insight into the characteristics of the data that will later be used for machine learning and forecasting.

---

## 1.2 Objectives

The objective of Task C.3 is to develop a reusable visualization module for financial time-series data. The visualization pipeline should:

- Generate candlestick charts using historical stock price data.
- Support configurable *n*-trading-day candlestick aggregation.
- Generate boxplots showing the distribution of stock prices across consecutive trading-day windows.
- Produce publication-quality figures that can be reused throughout the project.
- Facilitate exploratory analysis of the historical `CBA.AX` dataset prior to the development of forecasting models.

Rather than focusing on predictive performance, this task aims to improve understanding of the dataset by presenting its temporal trends, price distributions, and periods of varying market volatility through appropriate financial visualizations.
---

# 2. Visualization Pipeline

To satisfy the requirements of Task C.3, a reusable visualization module was developed to generate financial charts directly from the processed stock dataset produced in Task C.2. Rather than implementing separate scripts for individual figures, the visualization module provides configurable functions that can be reused throughout the project to generate different visual representations of historical market data.

The overall visualization workflow is illustrated below.

```text
Processed Stock Dataset
          │
          ▼
  Visualization Module
          │
 ┌────────┴────────┐
 ▼                 ▼
Candlestick    Windowed Boxplot
          │
          ▼
     Saved Figures
```

The following sections describe each visualization method in detail.

---

## 2.1 Candlestick Visualization

Candlestick charts are one of the most widely used visualization techniques in technical analysis because they summarize the price movement over a trading period using four values:

- Opening price
- Highest price
- Lowest price
- Closing price

Each candlestick represents the price movement during a specified number of consecutive trading days. A green candlestick indicates that the closing price is higher than the opening price, while a red candlestick indicates that the closing price is lower than the opening price. Thin lines extending above and below the candlestick body represent the highest and lowest prices reached during the trading period.

Unlike calendar-based aggregation, the implementation groups **exactly _n_ consecutive trading days** into each candlestick. This avoids distortions caused by weekends and market holidays, ensuring that every candlestick represents the same number of trading sessions regardless of the calendar date.

For each aggregated trading window, the candlestick values are computed as follows.

| Price Component | Aggregation Rule |
| :-------------- | :--------------- |
| Open | First opening price |
| High | Maximum high price |
| Low | Minimum low price |
| Close | Final closing price |
| Volume | Sum of trading volumes |

Using configurable trading-day aggregation allows the same visualization function to produce either detailed daily charts or higher-level summaries that emphasize broader market trends while reducing short-term fluctuations.

---

## 2.2 Windowed Boxplot Visualization

While candlestick charts emphasize price movements over time, they provide limited information about the statistical distribution of prices within each period. To complement the candlestick visualization, the project also implements **windowed boxplots**.

Each boxplot summarizes the distribution of adjusted closing prices within a consecutive trading-day window. The box represents the interquartile range (IQR), the horizontal line inside the box indicates the median price, the whiskers illustrate the spread of the remaining observations, and any points beyond the whiskers are displayed as potential outliers.

Instead of analysing the entire dataset as a single distribution, the visualization divides the historical data into consecutive **10-trading-day windows**, allowing changes in price distribution and market volatility to be observed throughout the dataset.

This representation makes it easier to identify:

- Changes in the median stock price over time.
- Periods of high or low price variability.
- Windows containing unusually large price movements.
- Long-term changes in market behaviour.

Because the visualization is generated using configurable window sizes, the same implementation can be reused for different analysis periods without modifying the source code.

---

## 2.3 Generated Visualizations

The visualization module was executed using the processed **CBA.AX** historical dataset and generated three figures used throughout the remainder of this report:

1. **Daily Candlestick Chart (Test Period)** — provides a detailed view of daily price movements during the testing period.
2. **Five-Trading-Day Candlestick Chart** — aggregates the complete historical dataset into five-day trading intervals to emphasize long-term market trends.
3. **Ten-Trading-Day Windowed Boxplot** — summarizes the distribution of adjusted closing prices within consecutive trading-day windows across the entire dataset.

Together, these visualizations provide complementary perspectives of the historical stock data. The daily candlestick chart highlights short-term market behaviour, the aggregated candlestick chart emphasizes broader price trends, and the windowed boxplot illustrates how the statistical distribution of prices changes over time.
---

# 3. Visualization Results

The visualization module was applied to the historical **Commonwealth Bank of Australia (`CBA.AX`)** stock dataset covering the period from **1 January 2020 to 2 July 2024**. Three complementary visualizations were generated to examine the market from different perspectives: a detailed daily candlestick chart, an aggregated five-trading-day candlestick chart, and a ten-trading-day windowed boxplot. Together, these figures illustrate short-term price movements, long-term market trends, and changes in price distribution over time.

---

## 3.1 Daily Candlestick Chart (Test Period)

Figure 3.1 presents the daily candlestick chart for the testing period used in the later machine learning experiments.

<div align="center">

**Figure 3.1.** Daily candlestick chart of `CBA.AX` during the testing period.

![Daily Candlestick Chart](../../results/c3/CBA.AX_1day_candlestick_test_period.png)

</div>

Each candlestick represents a single trading day and summarizes the opening, highest, lowest, and closing prices together with the daily trading volume. Displaying only the testing period produces a clear visualization of short-term market behaviour without overcrowding the figure with several years of daily observations.

The chart shows that the stock price generally follows an upward trend throughout the testing period, rising from approximately **\$100** in August 2023 to above **\$125** by July 2024. Although several temporary pullbacks occur, particularly during March and April 2024, these corrections are relatively short-lived before the upward trend resumes. The accompanying volume bars also reveal occasional spikes in trading activity, indicating periods of increased market participation during larger price movements.

Overall, the daily candlestick chart provides a detailed view of short-term market dynamics while preserving individual trading-day information.

---

## 3.2 Five-Trading-Day Candlestick Chart

To better visualize the complete historical dataset, the candlestick aggregation interval was increased from one trading day to five consecutive trading days.

<div align="center">

**Figure 3.2.** Five-trading-day aggregated candlestick chart of `CBA.AX`.

![5-Day Candlestick Chart](../../results/c3/CBA.AX_5day_candlestick.png)

</div>

Aggregating five trading days into a single candlestick reduces short-term fluctuations and makes long-term market behaviour easier to interpret. The figure clearly highlights several major phases of the historical dataset.

At the beginning of 2020, the chart shows a sharp market decline associated with the COVID-19 pandemic, followed by a rapid recovery during the remainder of the year. Between 2021 and 2023, the stock exhibits alternating periods of growth and consolidation before entering a sustained upward trend throughout late 2023 and the first half of 2024.

Compared with the daily candlestick chart, the aggregated visualization removes much of the day-to-day market noise while preserving the overall price trend, making it more suitable for analysing long-term market behaviour.

---

## 3.3 Ten-Trading-Day Windowed Boxplot

While candlestick charts emphasize temporal price movement, they do not explicitly summarize the statistical distribution of prices. Figure 3.3 therefore presents the adjusted closing prices using consecutive ten-trading-day boxplots.

<div align="center">

**Figure 3.3.** Adjusted closing price distribution across consecutive ten-trading-day windows.

![10-Day Windowed Boxplot](../../results/c3/CBA.AX_10day_windowed_boxplot.png)

</div>

Each boxplot summarizes the distribution of adjusted closing prices within a ten-trading-day window. The median price is shown by the horizontal line inside each box, the box itself represents the interquartile range, the whiskers indicate the spread of the remaining observations, and individual points beyond the whiskers are displayed as potential outliers.

Several characteristics of the historical dataset can be observed from the figure. The earliest windows exhibit substantially wider spreads, reflecting the elevated market volatility during the COVID-19 period. As the market stabilizes, the boxplots generally become more compact, indicating lower short-term price variability. The median price also increases steadily throughout the dataset, illustrating the long-term appreciation of the stock over the four-and-a-half-year period.

Together, the three visualizations provide complementary perspectives of the historical market data. The daily candlestick chart emphasizes detailed trading behaviour, the five-trading-day candlestick chart highlights long-term price trends, and the windowed boxplots summarize changes in price distribution and market volatility throughout the dataset.

# 4. Discussion

The visualizations generated in Task C.3 provide complementary perspectives of the historical `CBA.AX` stock dataset. Rather than serving only as graphical outputs, they reveal important characteristics of the market that are less apparent from numerical summaries alone. These observations provide useful context for the machine learning models developed in the subsequent tasks.

---

## 4.1 Market Trend

Both candlestick charts indicate that the `CBA.AX` stock experienced a clear long-term upward trend between January 2020 and July 2024. Although the market experienced a significant decline during the COVID-19 pandemic, the subsequent recovery resulted in sustained price appreciation over the remainder of the observation period.

The five-trading-day candlestick chart presents this trend particularly clearly by reducing short-term fluctuations while preserving the overall direction of the market. In contrast, the daily candlestick chart provides a more detailed view of individual trading sessions, illustrating temporary pullbacks and periods of increased trading activity within the broader upward trend.

These observations suggest that the historical dataset contains both long-term trends and short-term fluctuations, making it an appropriate dataset for evaluating time-series forecasting models.

---

## 4.2 Market Volatility

The windowed boxplots provide additional insight into how market volatility changes throughout the dataset. The earliest trading windows exhibit noticeably wider interquartile ranges and longer whiskers, indicating greater variation in adjusted closing prices during the highly volatile market conditions associated with the COVID-19 pandemic.

As the dataset progresses, the boxplots generally become more compact, reflecting lower short-term price variability. At the same time, the median price continues to increase steadily across consecutive windows, reinforcing the long-term upward trend observed in the candlestick charts.

The combination of changing volatility and long-term price appreciation highlights the non-stationary nature of financial time-series data, where market behaviour evolves over time rather than remaining constant.

---

## 4.3 Complementary Visualizations

Although all three figures are generated from the same historical dataset, each visualization emphasizes different aspects of market behaviour.

The **daily candlestick chart** provides the highest level of detail, allowing individual trading sessions, price reversals, and changes in trading volume to be examined. The **five-trading-day candlestick chart** aggregates consecutive trading sessions to reduce market noise and reveal broader trends over the full observation period. Meanwhile, the **windowed boxplots** summarize the statistical distribution of prices within consecutive trading windows, making changes in median price, variability, and potential outliers easier to identify.

Using multiple visualization techniques therefore provides a more comprehensive understanding of the dataset than relying on a single chart. Together, these figures illustrate both the temporal behaviour and statistical characteristics of the historical stock prices, providing valuable exploratory analysis before the forecasting models introduced in the subsequent machine learning tasks.

# 5. Conclusion

Task C.3 has been successfully completed by developing a reusable visualization module for financial time-series data. The implementation supports configurable candlestick charts using *n*-trading-day aggregation and windowed boxplots that summarize the distribution of stock prices across consecutive trading periods. Both visualization methods are implemented as reusable functions, allowing different visualization settings to be generated without modifying the underlying code.

Three visualizations were produced using the historical **Commonwealth Bank of Australia (`CBA.AX`)** dataset. The daily candlestick chart provided a detailed view of short-term price movements during the testing period, while the five-trading-day candlestick chart highlighted the long-term market trend by reducing short-term fluctuations. The ten-trading-day windowed boxplots complemented these visualizations by illustrating changes in price distribution and market volatility throughout the dataset.

The exploratory analysis revealed several important characteristics of the historical data, including the sharp market decline during the COVID-19 pandemic, the subsequent recovery, a sustained long-term upward trend, and a gradual reduction in price volatility over time. These observations improve understanding of the dataset before predictive modelling and provide useful context for interpreting the forecasting results developed in the subsequent machine learning tasks.

Overall, Task C.3 establishes a reusable visualization framework that complements the preprocessing pipeline developed in Task C.2 and provides an effective foundation for the deep learning and forecasting experiments presented in the later stages of the project.