import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parameters import *
from stock_prediction import create_model, load_data

def plot_graph(test_df):
    """
    Plots the true stock close prices along with predicted close prices
    using blue and red colors respectively, and saves it.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(test_df[f'true_{PRICE_VALUE}_{LOOKUP_STEP}'], c='b', label="Actual Price")
    plt.plot(test_df[f'{PRICE_VALUE}_{LOOKUP_STEP}'], c='r', label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"{TICKER} Actual vs Predicted Price")
    plt.legend()
    plot_filename = os.path.join(RESULTS_DIR, f"{model_name}_prediction.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to: {plot_filename}")

def get_final_df(model, data):
    """
    Takes the model and data dict to construct a final dataframe that includes
    features along with true and predicted prices of the testing dataset.
    """
    # simulated trading profit lambdas
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0.0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0.0
    
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Predict and transform back to price scale
    y_pred = model.predict(X_test)
    
    if SCALE:
        y_test_unscaled = np.squeeze(data["column_scaler"][PRICE_VALUE].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred_unscaled = np.squeeze(data["column_scaler"][PRICE_VALUE].inverse_transform(y_pred))
    else:
        y_test_unscaled = np.squeeze(y_test)
        y_pred_unscaled = np.squeeze(y_pred)
        
    test_df = data["test_df"].copy()
    
    # Add predicted and true future prices
    test_df[f"{PRICE_VALUE}_{LOOKUP_STEP}"] = y_pred_unscaled
    test_df[f"true_{PRICE_VALUE}_{LOOKUP_STEP}"] = y_test_unscaled
    
    # Sort by date index
    test_df.sort_index(inplace=True)
    
    # Calculate buy and sell profits
    test_df["buy_profit"] = list(map(
        buy_profit,
        test_df[PRICE_VALUE],
        test_df[f"{PRICE_VALUE}_{LOOKUP_STEP}"],
        test_df[f"true_{PRICE_VALUE}_{LOOKUP_STEP}"]
    ))
    
    test_df["sell_profit"] = list(map(
        sell_profit,
        test_df[PRICE_VALUE],
        test_df[f"{PRICE_VALUE}_{LOOKUP_STEP}"],
        test_df[f"true_{PRICE_VALUE}_{LOOKUP_STEP}"]
    ))
    
    return test_df

def predict_future(model, data):
    """Predicts future price using the latest sequence from the dataset."""
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction = model.predict(last_sequence)
    
    if SCALE:
        predicted_price = data["column_scaler"][PRICE_VALUE].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
        
    return predicted_price

# 1. Load data
print("Loading test data...")
data = load_data(
    ticker=TICKER,
    n_steps=N_STEPS,
    scale=SCALE,
    split_by_date=SPLIT_BY_DATE,
    shuffle=SHUFFLE,
    lookup_step=LOOKUP_STEP,
    test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS,
    start_date=TRAIN_START,
    end_date=TEST_END
)

# 2. Build Keras model structure
n_features = len(FEATURE_COLUMNS)
model = create_model(
    sequence_length=N_STEPS,
    n_features=n_features,
    loss=LOSS,
    units=LSTM_UNITS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

# 3. Load pre-trained model weights from results folder
weights_path = os.path.join(RESULTS_DIR, f"{model_name}.weights.h5")
if os.path.exists(weights_path):
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
else:
    print(f"Warning: Weights file not found at {weights_path}. Running with untrained weights.")

# 4. Evaluate Keras loss on test set
loss, mae_metric = model.evaluate(data["X_test"], data["y_test"], verbose=0)

# Calculate unscaled error metrics (MAE, RMSE, MAPE)
final_df = get_final_df(model, data)
y_true = final_df[f"true_{PRICE_VALUE}_{LOOKUP_STEP}"].values
y_pred = final_df[f"{PRICE_VALUE}_{LOOKUP_STEP}"].values

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Predict future next-day price
future_price = predict_future(model, data)

# Calculate simulated trading directional accuracy
num_trades = len(final_df)
correct_predictions = len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])
accuracy_score = correct_predictions / num_trades if num_trades > 0 else 0.0

# Calculate total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
total_profit = total_buy_profit + total_sell_profit
profit_per_trade = total_profit / num_trades if num_trades > 0 else 0.0

# 5. Print results
print(f"\nFuture price after {LOOKUP_STEP} days: {future_price:.2f}$")
print(f"{LOSS} loss: {loss:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Directional Accuracy score: {accuracy_score:.4f}")
print(f"Total buy profit: {total_buy_profit:.4f}")
print(f"Total sell profit: {total_sell_profit:.4f}")
print(f"Total profit: {total_profit:.4f}")
print(f"Profit per trade: {profit_per_trade:.4f}")

# 6. Save prediction graph and CSV predictions
plot_graph(final_df)

if not os.path.isdir(CSV_RESULTS_DIR):
    os.mkdir(CSV_RESULTS_DIR)
csv_filename = os.path.join(CSV_RESULTS_DIR, f"{model_name}.csv")
final_df.to_csv(csv_filename)
print(f"Saved predictions CSV to: {csv_filename}")
print(final_df.tail(10))
