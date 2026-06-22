# ==============================================================================
# Purpose: 
# Dynamic construction of Deep Learning models (LSTM, GRU, SimpleRNN)
# for financial time-series prediction.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional

# ==============================================================================
# DYNAMIC MODEL BUILDER
# ==============================================================================

def build_dl_model(
    lookback_steps,
    n_features,
    units=128,
    cell_type="LSTM",
    n_layers=2,
    dropout=0.3,
    loss="huber",
    optimizer="adam",
    bidirectional=False,
    future_steps=1
):
    """
    Constructs and compiles a deep recurrent neural network using TensorFlow/Keras.
    
    Args:
        lookback_steps (int): Lookback window size (number of historical days).
        n_features (int): Number of feature columns in input sequences.
        units (int): Number of hidden units in recurrent layers.
        cell_type (str): Recurrent cell architecture ('LSTM', 'GRU', 'SimpleRNN').
        n_layers (int): Total number of recurrent layers.
        dropout (float): Dropout fraction applied after each recurrent layer.
        loss (str): Loss function identifier (e.g. 'huber', 'mse', 'mae').
        optimizer (str): Optimizer identifier (e.g. 'adam', 'rmsprop').
        bidirectional (bool): If True, wraps recurrent layers in Bidirectional wrappers.
        future_steps (int): Number of future steps/days to predict (default 1).
        
    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    # --------------------------------------------------------------------------
    # Step 1: Cell Type Resolve & Validation
    # --------------------------------------------------------------------------
    cell_type_upper = cell_type.upper()
    cell_map = {
        "LSTM": LSTM,
        "GRU": GRU,
        "SIMPLERNN": SimpleRNN,
        "RNN": SimpleRNN
    }
    
    if cell_type_upper not in cell_map:
        raise ValueError(
            f"Unsupported cell_type '{cell_type}'. "
            f"Available options are: {list(cell_map.keys())}"
        )
    cell_class = cell_map[cell_type_upper]
    
    # --------------------------------------------------------------------------
    # Step 2: Assemble Layer Architecture
    # --------------------------------------------------------------------------
    model = Sequential()
    
    for i in range(n_layers):
        # Determine sequence return behavior
        # Only the final recurrent layer should return a 2D state vector (return_sequences=False)
        # to interface properly with the Dense output layer.
        is_last = (i == n_layers - 1)
        return_seqs = not is_last
        
        # Configure arguments for this layer
        layer_kwargs = {
            "units": units,
            "return_sequences": return_seqs
        }
        
        # Set input shape for the initial recurrent layer
        if i == 0:
            layer_kwargs["input_shape"] = (lookback_steps, n_features)
            
        # Instantiate recurrent cell
        recurrent_layer = cell_class(**layer_kwargs)
        
        # Wrap in bidirectional layer if requested
        if bidirectional:
            if i == 0:
                # Keras Bidirectional wrapper needs input_shape passed via kwargs/args
                model.add(Bidirectional(recurrent_layer, input_shape=(lookback_steps, n_features)))
            else:
                model.add(Bidirectional(recurrent_layer))
        else:
            model.add(recurrent_layer)
            
        # Add regularization dropout
        if dropout > 0.0:
            model.add(Dropout(dropout))
            
    # --------------------------------------------------------------------------
    # Step 3: Compile Network with Targets
    # --------------------------------------------------------------------------
    # Output layer produces predictions for the specified future time steps
    model.add(Dense(future_steps, activation="linear"))
    
    # Compile model using unscaled MAE as a tracking metric during training
    model.compile(
        loss=loss,
        metrics=["mean_absolute_error"],
        optimizer=optimizer
    )
    
    return model
