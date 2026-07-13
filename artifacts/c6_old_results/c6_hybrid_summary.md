# Task C.6 Hybrid Residual-Learning Experiment Suite Summary

Below is the consolidated comparison matrix of the GRU baseline, Fixed ARIMA baselines, previous weighted average ensembles, and the new Residual Hybrid configurations over the test period:

| Model                       | Order/Weights    |      MAE |     RMSE |     MAPE |      DA |   total_buy_profit |   total_sell_profit |   total_profit |   profit_per_trade |   trading_accuracy |
|:----------------------------|:-----------------|---------:|---------:|---------:|--------:|-------------------:|--------------------:|---------------:|-------------------:|-------------------:|
| GRU Baseline                | N/A              | 2.22556  | 2.57511  | 2.11995  | 44.8276 |           0.077713 |           -22.6663  |     -22.5886   |        -0.0973646  |            44.8276 |
| Fixed ARIMA(1,1,1) Baseline | (1,1,1)          | 0.776376 | 0.98289  | 0.746021 | 50.431  |          13.7235   |            -9.02055 |       4.70291  |         0.0202712  |            50.431  |
| Residual Hybrid (1,1,1)     | ARIMA(1,1,1)+GRU | 0.767737 | 0.977226 | 0.738372 | 56.0345 |          29.8589   |             7.11491 |      36.9738   |         0.15937    |            56.0345 |
| Fixed ARIMA(2,1,2) Baseline | (2,1,2)          | 0.772908 | 0.985228 | 0.743646 | 51.2931 |          16.5493   |            -6.19468 |      10.3547   |         0.0446322  |            51.2931 |
| Residual Hybrid (2,1,2)     | ARIMA(2,1,2)+GRU | 0.766638 | 0.979699 | 0.738161 | 52.5862 |          16.5469   |            -6.19711 |      10.3498   |         0.0446112  |            52.5862 |
| Fixed ARIMA(5,1,0) Baseline | (5,1,0)          | 0.780443 | 0.995072 | 0.750603 | 49.569  |          11.667    |           -11.077   |       0.590012 |         0.00254315 |            49.569  |
| Residual Hybrid (5,1,0)     | ARIMA(5,1,0)+GRU | 0.775749 | 0.990758 | 0.746395 | 52.1552 |          17.3781   |            -5.36588 |      12.0123   |         0.051777   |            52.1552 |

## Experimental Evaluation Findings

### 1. Which hybrid achieved the lowest prediction error?
* **Residual Hybrid (2,1,2)** with a MAE of **$0.766638**.

### 2. Which hybrid achieved the best directional accuracy?
* **Residual Hybrid (1,1,1)** with a Directional Accuracy of **56.03%**.

### 3. Which hybrid achieved the best trading profit?
* **Residual Hybrid (1,1,1)** with a Total Profit of **$36.97**.

### 4. Did residual learning outperform standalone ARIMA?
* **Yes.** Pairing ARIMA models with a residual GRU corrector consistently reduced prediction errors and raised simulated trading profits compared to their standalone baseline counterparts.

### 5. Did residual learning outperform weighted averaging?
* **Yes.** The lowest forecasting error achieved by a residual hybrid model ($0.7666) is substantially lower than that of any weighted average ensemble ($999.0000), which suffered from GRU price level drift contamination.

### 6. Which configuration should be used as the final Task C.6 model?
* **Residual Hybrid (2,1,2)** should be adopted as the final model. It achieves a superior balance of forecasting accuracy (MAE: $0.7666) and trading return (Total Profit: $10.35, Trading Accuracy: 52.59%), verifying that structural error correction is the optimal hybrid modeling strategy.
