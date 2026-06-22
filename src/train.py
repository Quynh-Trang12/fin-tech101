# ==============================================================================
# File: train.py
# Purpose: Deep learning training pipeline with seed determinism.
# ==============================================================================

import os
import random
import argparse
import numpy as np
import tensorflow as tf

# Import local data processing and model factory modules
from data_processing import load_and_process_data
from model_factory import build_dl_model

# ==============================================================================
# SEED DETERMINISM SETUP
# ==============================================================================

def set_seed(seed=314):
    """Enforces random seed determinism across standard runtime environments."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configure TensorFlow single-thread execution settings for reproducibility if needed,
    # but standard TF seed is usually sufficient for classroom settings.
    print(f"[Seed Setup] Deterministic seed set to: {seed}")

# Set seed at import time
set_seed(314)

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_model(
    ticker="CBA.AX",
    start_date="2020-01-01",
    end_date="2024-07-02",
    split_date="2023-08-02",
    n_steps=50,
    lookup_step=1,
    scale=True,
    shuffle=True,
    feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
    units=128,
    cell_type="LSTM",
    n_layers=2,
    dropout=0.3,
    loss="huber",
    optimizer="adam",
    bidirectional=False,
    epochs=20,
    batch_size=64,
    model_name="lstm_model"
):
    """
    Orchestrates the entire training pipeline: loads/preprocesses data,
    creates the model from the factory, fits the model, and saves weights.
    
    Returns:
        dict: Training history and processed data dictionary for downstream use.
    """
    # --------------------------------------------------------------------------
    # Step 1: Enforce Determinism
    # --------------------------------------------------------------------------
    set_seed(314)
    
    # --------------------------------------------------------------------------
    # Step 2: Load and Segment Dataset
    # --------------------------------------------------------------------------
    print(f"\n[Train Pipeline] Loading dataset for {ticker} from {start_date} to {end_date}...")
    data = load_and_process_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        n_steps=n_steps,
        scale=scale,
        shuffle=shuffle,
        lookup_step=lookup_step,
        split_by_date=True,
        split_date=split_date,
        feature_columns=feature_columns,
        cache_dir="data"
    )
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    print(f"[Train Pipeline] Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"[Train Pipeline] Test shapes  - X: {data['X_test'].shape}, y: {data['y_test'].shape}")
    
    # --------------------------------------------------------------------------
    # Step 3: Instantiate Model Architecture
    # --------------------------------------------------------------------------
    n_features = len(feature_columns)
    print(f"[Train Pipeline] Building model architecture: {cell_type} | Units: {units} | Layers: {n_layers}")
    
    model = build_dl_model(
        sequence_length=n_steps,
        n_features=n_features,
        units=units,
        cell_type=cell_type,
        n_layers=n_layers,
        dropout=dropout,
        loss=loss,
        optimizer=optimizer,
        bidirectional=bidirectional
    )
    
    # --------------------------------------------------------------------------
    # Step 4: Model Training (Fitting)
    # --------------------------------------------------------------------------
    print(f"[Train Pipeline] Training model '{model_name}' for {epochs} epochs with batch size {batch_size}...")
    
    # Track performance using validation data to monitor overfitting
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data["X_test"], data["y_test"]),
        verbose=1
    )
    
    # --------------------------------------------------------------------------
    # Step 5: Save Model Weights and Metadata
    # --------------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    weights_path = os.path.join("results", f"{model_name}.weights.h5")
    model.save_weights(weights_path)
    print(f"[Train Pipeline] Successfully saved model weights to: {weights_path}")
    
    return {
        "history": history.history,
        "data": data,
        "model": model
    }

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep recurrent stock prediction models.")
    
    parser.add_argument("--ticker", type=str, default="CBA.AX", help="Stock ticker symbol.")
    parser.add_argument("--cell_type", type=str, default="LSTM", help="Recurrent cell type (LSTM, GRU, SimpleRNN).")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of recurrent layers.")
    parser.add_argument("--units", type=int, default=128, help="Number of recurrent hidden units.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--loss", type=str, default="huber", help="Loss function (huber, mse, mae).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--model_name", type=str, default="lstm_model", help="Name to save model weights as.")
    parser.add_argument("--bidirectional", action="store_true", help="Wrap recurrent layers in Bidirectional wrapper.")
    
    args = parser.parse_args()
    
    train_model(
        ticker=args.ticker,
        cell_type=args.cell_type,
        n_layers=args.n_layers,
        units=args.units,
        dropout=args.dropout,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        bidirectional=args.bidirectional
    )
