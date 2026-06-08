import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from parameters import *
from stock_prediction import create_model, load_data

# Create output folders if they do not exist
if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

if not os.path.isdir(LOGS_DIR):
    os.mkdir(LOGS_DIR)

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

# 1. Load and process stock dataset
print(f"Loading and processing data for {TICKER}...")
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

# Save the raw data frame for reproducibility
raw_data_filename = os.path.join(DATA_DIR, f"{TICKER}_{date_now}.csv")
data["df"].to_csv(raw_data_filename)
print(f"Saved dataframe to local file: {raw_data_filename}")

# 2. Build the deep learning model architecture
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

# 3. Setup ModelCheckpoint and TensorBoard callbacks
weights_filename = os.path.join(RESULTS_DIR, f"{model_name}.weights.h5")
checkpointer = ModelCheckpoint(
    weights_filename,
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

tb_log_dir = os.path.join(LOGS_DIR, model_name)
tensorboard = TensorBoard(log_dir=tb_log_dir)

# 4. Train the model using ModelCheckpoint to save optimal weights
print(f"Training model for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
history = model.fit(
    data["X_train"],
    data["y_train"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(data["X_test"], data["y_test"]),
    callbacks=[checkpointer, tensorboard],
    verbose=1
)

print(f"Training completed. Optimal weights saved to: {weights_filename}")
