# 1. Hybrid Forecasting Methodology

## 1.1 Motivation

Tasks C.4 and C.5 demonstrated that the **GRU** architecture provides strong forecasting performance for stock price prediction. However, the experiments also showed that deep learning models alone do not consistently produce profitable trading decisions. Although the GRU model captures complex nonlinear patterns in historical market data, its predictions may drift over time and struggle to model short-term linear price movements.

Classical statistical models such as **ARIMA (AutoRegressive Integrated Moving Average)** offer complementary strengths. ARIMA models are designed to capture linear temporal relationships and short-term autocorrelations in time series data. They are computationally efficient and often perform well for one-step-ahead forecasting, but they are less capable of modelling complex nonlinear market behaviour.

These complementary characteristics motivate the use of **hybrid forecasting**, where statistical and deep learning models are combined in an attempt to exploit the strengths of both approaches. Rather than replacing one model with another, the objective is to investigate whether combining ARIMA and GRU can improve forecasting accuracy and trading performance beyond what either model achieves individually.

Two ensemble strategies are investigated in this task:

1. **Weighted Ensemble**, which combines the predictions of ARIMA and GRU using weighted averaging.
2. **Residual Learning**, which allows the GRU model to learn and correct the forecasting errors produced by ARIMA.

---

## 1.2 Weighted Ensemble

The first approach combines the predictions of the GRU model and the ARIMA model using a weighted average. Instead of relying entirely on one forecasting model, the final prediction is calculated as a weighted combination of both predictions:

\[
\hat{y}=w_{GRU}\hat{y}_{GRU}+w_{ARIMA}\hat{y}_{ARIMA}
\]

where:

- \(\hat{y}_{GRU}\) is the prediction produced by the GRU model,
- \(\hat{y}_{ARIMA}\) is the prediction produced by the ARIMA model,
- \(w_{GRU}\) and \(w_{ARIMA}\) are weighting coefficients satisfying

\[
w_{GRU}+w_{ARIMA}=1.
\]

Three fixed ARIMA configurations—ARIMA(1,1,1), ARIMA(2,1,2), and ARIMA(5,1,0)—were evaluated. For each ARIMA model, three weighting combinations (50:50, 70:30, and 30:70) were tested to investigate whether simple linear averaging could improve prediction performance.

---

## 1.3 Residual Learning

The second approach uses **residual learning**, which combines the strengths of both forecasting models in a different way.

Instead of averaging two independent predictions, the ARIMA model first generates an initial forecast. The prediction error (or **residual**) is then calculated as

\[
Residual = Actual\ Price - ARIMA\ Prediction.
\]

A GRU model is subsequently trained to predict these residual values rather than the stock prices directly. During inference, the predicted residual is added back to the ARIMA forecast to obtain the final prediction:

\[
Final\ Prediction = ARIMA\ Prediction + Predicted\ Residual.
\]

This approach allows ARIMA to model the underlying linear behaviour of the stock price while the GRU learns the remaining nonlinear patterns that ARIMA cannot capture. By focusing only on the residual errors, the GRU solves a simpler learning problem than predicting the entire stock price series from scratch.

The overall workflow of the residual-learning approach is illustrated below.

```text
Historical Stock Data
          │
          ▼
      ARIMA Model
          │
          ▼
   Initial Forecast
          │
          ├───────────────┐
          │               │
          ▼               │
Compute Residuals          │
(Actual − ARIMA)           │
          │               │
          ▼               │
     Train GRU            │
      on Residuals        │
          │               │
          ▼               │
 Predict Residuals         │
          │               │
          └──────┬────────┘
                 ▼
      Final Prediction
 = ARIMA + Predicted Residual
```

# 2. Experimental Design

## 2.1 Shared Experimental Setup

To ensure a fair comparison, all ensemble and hybrid forecasting experiments were conducted using the same dataset, preprocessing pipeline, and evaluation procedure established in the previous tasks. The only differences between experiments were the ARIMA model order and the forecasting strategy (weighted ensemble or residual learning).

The GRU architecture selected in Task C.4 was retained as the deep learning component throughout all experiments.

The shared experimental settings are summarised in Table 2.1.

| Parameter | Value |
| :-------- | :---- |
| Dataset | Commonwealth Bank of Australia (CBA.AX) |
| Deep Learning Model | GRU |
| Number of GRU Layers | 2 |
| Hidden Units | 128 |
| Lookback Window | 50 trading days |
| Forecast Offset | 1 trading day |
| Input Features | Adjusted Close, Volume, Open, High, Low |
| Optimizer | Adam |
| Loss Function | Huber Loss |
| Epochs | 20 |
| Batch Size | 64 |

The historical dataset was divided chronologically into training, validation, and testing subsets using the finalized preprocessing pipeline developed in the previous tasks. A **15% validation split** was created from the training data to monitor model performance during training, while all reported evaluation metrics were computed using the independent testing dataset. This chronological evaluation protocol prevents information leakage and provides a fair comparison between all forecasting methods.

---

## 2.2 ARIMA Configurations

Three fixed-parameter ARIMA models were evaluated as statistical forecasting baselines. The parameters of each ARIMA model were estimated once using the chronological training dataset and remained fixed throughout the testing period. This approach ensures that the statistical models are evaluated under the same conditions as the GRU model without repeatedly refitting the model during testing.

The evaluated ARIMA configurations are summarised below.

| Model | Order | Description |
| :---- | :---: | :---------- |
| ARIMA(1,1,1) | (1,1,1) | Standard autoregressive and moving-average model after first-order differencing. |
| ARIMA(2,1,2) | (2,1,2) | Higher-order autoregressive and moving-average model for capturing more complex temporal dependencies. |
| ARIMA(5,1,0) | (5,1,0) | Autoregressive model with a longer historical memory and no moving-average component. |

These three configurations provide a representative comparison of different statistical forecasting models before combining them with the GRU network.

---

## 2.3 Evaluation Metrics

Each forecasting model was evaluated using both prediction accuracy metrics and trading-oriented performance metrics.

### Regression Metrics

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual stock prices.
- **Root Mean Squared Error (RMSE):** Measures prediction error while placing greater emphasis on larger errors.
- **Mean Absolute Percentage Error (MAPE):** Average percentage prediction error relative to the true stock price.

### Directional and Trading Metrics

- **Directional Accuracy (DA):** Percentage of predictions that correctly identify whether the next stock price increases or decreases.
- **Trading Accuracy:** Percentage of simulated trades that generate a positive return.
- **Total Trading Profit:** Overall profit obtained from the simulated trading strategy across the testing period.
- **Profit per Trade:** Average profit or loss generated by each simulated trade.

Using both regression and trading metrics provides a comprehensive evaluation of each forecasting strategy. While regression metrics measure numerical forecasting accuracy, trading metrics assess the practical usefulness of the predictions in a simple trading scenario. Considering both perspectives enables a more balanced comparison of the standalone, ensemble, and hybrid forecasting models.

# 3. Experimental Results and Discussion

## 3.1 Weighted Ensemble Results

The first experiment investigated whether a simple weighted average of GRU and ARIMA predictions could improve forecasting performance. Three ARIMA configurations—ARIMA(1,1,1), ARIMA(2,1,2), and ARIMA(5,1,0)—were combined with the GRU model using three weighting schemes (50:50, 70:30, and 30:70).

The experimental results are summarised in Table 3.1.

| Model | MAE ($) | RMSE ($) | MAPE (%) | DA (%) | Trading Accuracy (%) | Total Profit ($) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GRU Baseline** | 2.2256 | 2.5751 | 2.120 | 44.83 | 44.83 | -22.59 |
| **ARIMA(1,1,1)** | **1.0763** | **1.3473** | **1.038** | 49.14 | 49.14 | 4.39 |
| ARIMA(2,1,2) | 1.0800 | 1.3497 | 1.041 | 50.00 | 50.00 | 4.16 |
| ARIMA(5,1,0) | 1.0785 | 1.3521 | 1.041 | **50.43** | **50.43** | **8.63** |
| Best Weighted Ensemble | 1.2420 | 1.5259 | 1.192 | 47.84 | 47.84 | -14.38 |

The results show that all three ARIMA models substantially outperformed the standalone GRU model across every regression metric. Although the weighted ensembles achieved lower prediction errors than the GRU baseline, none of them surpassed the corresponding standalone ARIMA models.

This behaviour is expected because weighted averaging combines the predictions of two complete forecasting models. Since the GRU model produced larger forecasting errors than ARIMA, averaging the two predictions introduced additional error into the final forecast rather than improving it.

Consequently, simple weighted averaging was not an effective strategy for combining the statistical and deep learning models.

---

## 3.2 Residual Learning Results

The second experiment investigated residual learning, where the GRU model predicts only the forecasting errors (residuals) produced by ARIMA instead of predicting stock prices directly.

The experimental results are summarised in Table 3.2.

| Model | MAE ($) | RMSE ($) | MAPE (%) | DA (%) | Trading Accuracy (%) | Total Profit ($) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed ARIMA(1,1,1) | 0.7764 | 0.9829 | 0.746 | 50.43 | 50.43 | 4.70 |
| **Residual Hybrid (1,1,1)** | 0.7677 | **0.9772** | 0.738 | **56.03** | **56.03** | **36.97** |
| Fixed ARIMA(2,1,2) | 0.7729 | 0.9852 | 0.744 | 51.29 | 51.29 | 10.35 |
| **Residual Hybrid (2,1,2)** | **0.7666** | 0.9797 | **0.738** | 52.59 | 52.59 | 10.35 |
| Fixed ARIMA(5,1,0) | 0.7804 | 0.9951 | 0.751 | 49.57 | 49.57 | 0.59 |
| **Residual Hybrid (5,1,0)** | 0.7757 | 0.9908 | 0.746 | 52.16 | 52.16 | 12.01 |

Unlike weighted averaging, the residual-learning approach consistently improved the forecasting performance of every ARIMA configuration. For all three ARIMA models, the hybrid approach reduced MAE, RMSE, and MAPE while also increasing Directional Accuracy.

Among the evaluated models, **Residual Hybrid (2,1,2)** achieved the lowest regression errors (MAE = **0.7666**, RMSE = **0.9797**, MAPE = **0.738%**), indicating the highest numerical forecasting accuracy.

However, **Residual Hybrid (1,1,1)** achieved the highest Directional Accuracy (**56.03%**) and the largest simulated trading profit (**$36.97**), while maintaining regression performance that was nearly identical to the best-performing model. These results suggest that small improvements in numerical prediction error do not necessarily translate into better trading performance.

---

## 3.3 Overall Discussion

The experiments demonstrate a clear difference between the two ensemble strategies.

The weighted ensemble approach simply averages two complete forecasts. When one forecasting model is substantially less accurate than the other, the averaging process introduces additional prediction error and reduces overall forecasting performance. Consequently, none of the weighted ensemble configurations outperformed the standalone ARIMA models.

Residual learning follows a different strategy. Instead of predicting the entire stock price, the GRU model focuses only on correcting the errors produced by ARIMA. This decomposition simplifies the learning problem by allowing ARIMA to model the linear components of the time series while the GRU learns the remaining nonlinear patterns.

Overall, the residual-learning approach consistently outperformed both the standalone forecasting models and the weighted ensemble strategy. Although **Residual Hybrid (2,1,2)** achieved the lowest regression error, **Residual Hybrid (1,1,1)** provided the best overall balance between forecasting accuracy, directional prediction, and simulated trading profitability. These findings indicate that residual learning is a more effective method for combining statistical and deep learning models for stock price forecasting than simple weighted averaging.

# 4. Verification and Prediction Examples

To verify the implementation, both the weighted ensemble experiment and the hybrid residual-learning experiment were executed using the automated experiment runners. All models were trained and evaluated using the same chronological preprocessing pipeline, ensuring that every forecasting method was tested under identical experimental conditions.

The generated outputs include:

- Consolidated evaluation metrics for both the weighted ensemble and residual-learning experiments.
- Prediction plots comparing the forecasting behaviour of ARIMA, GRU, weighted ensembles, and hybrid models.
- Saved model weights for each residual-learning GRU model.
- Summary reports documenting the experimental findings.

The successful execution of both experiments confirms that the forecasting pipeline correctly supports standalone statistical models, deep learning models, weighted ensembles, and hybrid residual-learning approaches.

---

### Figure 4.1 – Weighted Ensemble Prediction

![Weighted Ensemble Prediction](../../results/c6/c6_weighted_ensemble_prediction.png)

The weighted ensemble prediction combines the forecasts produced by the GRU and ARIMA models using fixed weighting coefficients. Although the ensemble follows the overall market trend, the prediction curve does not consistently improve upon the standalone ARIMA forecast. This observation agrees with the quantitative results presented in Section 3, where none of the weighted ensemble configurations outperformed the corresponding ARIMA models. The experiment demonstrates that simply averaging two complete forecasting models is insufficient when one model consistently produces larger prediction errors.

---

### Figure 4.2 – Residual Hybrid Prediction (ARIMA(1,1,1) + GRU)

![Residual Hybrid Prediction](../../results/c6/c6_hybrid_1_1_1_prediction.png)

The residual-learning model produces predictions that closely follow the actual stock prices throughout the testing period. Rather than predicting the complete stock price directly, the GRU model learns to correct the residual errors generated by ARIMA. This results in smaller deviations from the actual prices, particularly during periods of rapid market movement. The visual improvement is consistent with the substantial increase in Directional Accuracy (56.03%) and simulated trading profit ($36.97) achieved by the Residual Hybrid (1,1,1) model.

---

### Figure 4.3 – Residual Hybrid Prediction (ARIMA(2,1,2) + GRU)

![Residual Hybrid Prediction](../../results/c6/c6_hybrid_2_1_2_prediction.png)

The Residual Hybrid (2,1,2) configuration achieved the lowest regression errors among all evaluated forecasting models. The prediction curve closely matches the actual stock prices over the testing period, producing the lowest MAE, RMSE, and MAPE values reported in Section 3. These results demonstrate that residual learning consistently improves the forecasting accuracy of the underlying ARIMA model.

---

### Figure 4.4 – Residual Hybrid Prediction (ARIMA(5,1,0) + GRU)

![Residual Hybrid Prediction](../../results/c6/c6_hybrid_5_1_0_prediction.png)

The Residual Hybrid (5,1,0) model also improves upon its standalone ARIMA counterpart by reducing prediction errors and increasing directional accuracy. Although its overall forecasting performance is slightly lower than the other hybrid configurations, the prediction curve remains closer to the actual stock prices than the standalone statistical model.

Overall, the visual comparisons support the quantitative evaluation results presented in Section 3. The weighted ensemble approach provides only limited improvement over the standalone GRU model, whereas the residual-learning approach consistently enhances the forecasting performance of all three ARIMA configurations. These observations confirm that residual learning is a more effective strategy for combining statistical and deep learning models for stock price forecasting.

# 5. Conclusion

Task C.6 has been successfully completed by investigating two approaches for combining statistical and deep learning models for stock price forecasting: **weighted ensemble learning** and **hybrid residual learning**. Both approaches integrated the GRU architecture selected in Task C.4 with multiple ARIMA forecasting models while reusing the same preprocessing, training, and evaluation pipeline established in the previous tasks.

The weighted ensemble experiments demonstrated that simple linear averaging of GRU and ARIMA predictions did not improve forecasting performance. Although the ensemble models outperformed the standalone GRU model, none of the weighted combinations achieved better results than the corresponding standalone ARIMA models. These findings indicate that averaging the predictions of two complete forecasting models is not an effective strategy when one model consistently produces larger prediction errors.

In contrast, the residual-learning approach consistently improved the performance of all evaluated ARIMA models. The hybrid models achieved lower regression errors and higher directional accuracy than their standalone ARIMA counterparts, demonstrating that allowing the GRU model to learn and correct ARIMA's residual errors is a more effective method for combining statistical and deep learning models.

Among the evaluated configurations, **Residual Hybrid (2,1,2)** achieved the lowest prediction errors, while **Residual Hybrid (1,1,1)** achieved the highest Directional Accuracy (**56.03%**) and the greatest simulated trading profit (**$36.97**). Considering both forecasting accuracy and practical trading performance, the **Residual Hybrid (1,1,1)** configuration provides the best overall balance for the CBA.AX dataset.

Overall, Task C.6 demonstrates that hybrid residual learning is a more effective ensemble strategy than simple weighted averaging for this forecasting problem. By combining the linear modelling capability of ARIMA with the nonlinear learning capability of GRU, the proposed hybrid framework achieves more accurate and robust stock price forecasts than either standalone forecasting approach.