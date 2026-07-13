# ==============================================================================
# Purpose:
# Preprocess the Task C.7 classification dataset by partitioning features into
# market, V2Tone news, and FinBERT news feature groups and performing
# chronological splits (train/val/test).
#
# Reuses the boundary configurations from Task C.2-C.6.
# ==============================================================================

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from config import (
    C7_DATA_DIR,
    SPLIT_DATE,
    VALIDATION_RATIO,
)

# ==============================================================================
# FEATURE GROUP DEFINITIONS
# ==============================================================================

MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "adjclose",
    "volume",
]

V2TONE_FEATURES = [
    "article_count",
    "valid_v2tone_count",
    "word_count_total",
    "positive_article_count",
    "negative_article_count",
    "neutral_article_count",
    "tone_min",
    "tone_max",
    "tone_mean",
    "positive_score_mean",
    "negative_score_mean",
    "polarity_mean",
    "activity_reference_density_mean",
    "self_group_reference_density_mean",
    "word_count_mean",
    "positive_article_share",
    "negative_article_share",
    "neutral_article_share",
    "has_news",
]

# We exclude raw predicted label counts (finbert_positive_article_count, etc.)
# because they are mathematically redundant when overall count and shares are included.
# This prevents perfect collinearity issues.
FINBERT_FEATURES = [
    "finbert_article_count",
    "finbert_positive_probability_mean",
    "finbert_neutral_probability_mean",
    "finbert_negative_probability_mean",
    "finbert_confidence_mean",
    "finbert_confidence_max",
    "finbert_positive_article_share",
    "finbert_neutral_article_share",
    "finbert_negative_article_share",
    "has_finbert_news",
]

REDUCED_FINBERT_FEATURES = [
    "finbert_article_count",
    "finbert_positive_probability_mean",
    "finbert_negative_probability_mean",
    "finbert_confidence_mean",
]

TARGET_COLUMN = "target"


def prepare_c7_classification_data(
    dataset_path: Path = C7_DATA_DIR / "c7_dataset.parquet",
    split_date: str = SPLIT_DATE,
    validation_ratio: float = VALIDATION_RATIO,
) -> Dict[str, Any]:
    """
    Load, validate, and chronologically split the Task C.7 dataset.
    
    Args:
        dataset_path: Path to c7_dataset.parquet.
        split_date: Date boundary separating train and test sets.
        validation_ratio: Portion of the training set used for validation.
        
    Returns:
        dict: A dictionary containing:
            - train_df: Training DataFrame.
            - val_df: Validation DataFrame.
            - test_df: Testing DataFrame.
            - feature_sets: Dict mapping feature set names to lists of columns.
            - target_column: Name of the target column ('target').
            - split_metadata: Metadata regarding the splitting process.
    """
    # --------------------------------------------------------------------------
    # 1. Load and Validate Dataset
    # --------------------------------------------------------------------------
    if not dataset_path.exists():
        raise FileNotFoundError(f"C.7 dataset not found at: {dataset_path}")
        
    df = pd.read_parquet(dataset_path)
    
    # Validation checks
    if "date" not in df.columns:
        raise ValueError("Dataset is missing the 'date' column.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing the target column '{TARGET_COLUMN}'.")
        
    # Check chronological uniqueness and sorting
    dates = df["date"]
    if not dates.is_unique:
        raise ValueError("Dataset contains duplicate dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dataset dates are not chronologically sorted.")
        
    # Check target contains only 0 and 1
    unique_targets = df[TARGET_COLUMN].unique()
    invalid_targets = [t for t in unique_targets if t not in [0, 1]]
    if invalid_targets:
        raise ValueError(f"Target column contains invalid classes: {invalid_targets}")
        
    # Check for missing values
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
        raise ValueError(
            f"Dataset contains {total_missing} missing values in columns: {cols_with_missing}."
        )
        
    # --------------------------------------------------------------------------
    # 2. Chronological Splitting (No Shuffling)
    # --------------------------------------------------------------------------
    # Parse splitting dates
    split_ts = pd.to_datetime(split_date)
    
    # Train full / test split
    train_full_df = df[df["date"] < split_ts].copy()
    test_df = df[df["date"] >= split_ts].copy()
    
    # Validate splits are not empty
    if train_full_df.empty:
        raise ValueError(f"Train set is empty. Check split_date: {split_date}")
    if test_df.empty:
        raise ValueError(f"Test set is empty. Check split_date: {split_date}")
        
    # Validation split from training set
    if validation_ratio > 0.0:
        val_size = int(len(train_full_df) * validation_ratio)
        train_size = len(train_full_df) - val_size
        
        train_df = train_full_df.iloc[:train_size].copy()
        val_df = train_full_df.iloc[train_size:].copy()
    else:
        train_df = train_full_df.copy()
        val_df = pd.DataFrame(columns=df.columns)
        
    # Verify split subsets
    if train_df.empty:
        raise ValueError("Training subset is empty after validation split.")
    if validation_ratio > 0.0 and val_df.empty:
        raise ValueError("Validation subset is empty after splitting.")
        
    # Compile metadata
    split_metadata = {
        "split_date": split_date,
        "validation_ratio": validation_ratio,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    
    # Define experiment feature sets
    feature_sets = {
        "market_only": MARKET_FEATURES.copy(),
        "market_plus_v2tone": MARKET_FEATURES + V2TONE_FEATURES,
        "market_plus_finbert": MARKET_FEATURES + FINBERT_FEATURES,
        "market_plus_v2tone_plus_finbert": MARKET_FEATURES + V2TONE_FEATURES + FINBERT_FEATURES,
        "market_plus_reduced_finbert": MARKET_FEATURES + REDUCED_FINBERT_FEATURES,
        "market_plus_v2tone_plus_reduced_finbert": MARKET_FEATURES + V2TONE_FEATURES + REDUCED_FINBERT_FEATURES,
    }
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_sets": feature_sets,
        "target_column": TARGET_COLUMN,
        "split_metadata": split_metadata,
    }


def main():
    print("=" * 80)
    print("STARTING C.7 CLASSIFICATION PREPROCESSING AUDIT (ENHANCED)")
    print("=" * 80)
    
    dataset_path = C7_DATA_DIR / "c7_dataset.parquet"
    data_dict = prepare_c7_classification_data(dataset_path)
    
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    test_df = data_dict["test_df"]
    meta = data_dict["split_metadata"]
    
    # 1. Dataset Shape Audit
    full_shape = (meta["total_rows"], len(train_df.columns))
    print(f"Full Dataset Shape:       {full_shape}")
    print(f"Train Dataset Shape:      {train_df.shape}")
    print(f"Validation Dataset Shape: {val_df.shape}")
    print(f"Test Dataset Shape:       {test_df.shape}\n")
    
    # 2. Date Boundaries Audit
    print("-" * 50)
    print("DATE BOUNDARIES AUDIT")
    print("-" * 50)
    print(f"Train Date Range:      {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Validation Date Range: {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Test Date Range:       {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}\n")
    
    # 3. Class Balance Audit
    print("-" * 50)
    print("CLASS BALANCE AUDIT")
    print("-" * 50)
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = df[TARGET_COLUMN].value_counts()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        total = len(df)
        print(f"{name} split:")
        print(f"  Class 0 (Down/Flat): {c0} ({c0/total*100:.2f}%)")
        print(f"  Class 1 (Rise):      {c1} ({c1/total*100:.2f}%)")
        
    # 4. Feature Groups Audit
    print("\n" + "-" * 50)
    print("FEATURE GROUPS AUDIT")
    print("-" * 50)
    print("Market-Only Features:")
    print(f"  {data_dict['feature_sets']['market_only']}")
    print("Market + V2Tone Features:")
    print(f"  {data_dict['feature_sets']['market_plus_v2tone']}")
    print("Market + FinBERT Features:")
    print(f"  {data_dict['feature_sets']['market_plus_finbert']}")
    print("Market + V2Tone + FinBERT Features:")
    print(f"  {data_dict['feature_sets']['market_plus_v2tone_plus_finbert']}")
    print("Market + Reduced FinBERT Features:")
    print(f"  {data_dict['feature_sets']['market_plus_reduced_finbert']}")
    print("Market + V2Tone + Reduced FinBERT Features:")
    print(f"  {data_dict['feature_sets']['market_plus_v2tone_plus_reduced_finbert']}")
    
    # 5. Overlap Checks
    print("\n" + "-" * 50)
    print("INTEGRITY & OVERLAP CHECKS")
    print("-" * 50)
    
    # Date Ranges overlap checks
    train_max = train_df["date"].max()
    val_min = val_df["date"].min()
    val_max = val_df["date"].max()
    test_min = test_df["date"].min()
    
    no_overlap_train_val = train_max < val_min
    no_overlap_val_test = val_max < test_min
    
    print(f"Train/Val Date Disjoint (no overlap): {no_overlap_train_val}")
    print(f"Val/Test Date Disjoint (no overlap):  {no_overlap_val_test}")
    
    # Check chronological ordering holds after split
    is_train_sorted = train_df["date"].is_monotonic_increasing
    is_val_sorted = val_df["date"].is_monotonic_increasing
    is_test_sorted = test_df["date"].is_monotonic_increasing
    
    print(f"Train subset chronologically sorted:  {is_train_sorted}")
    print(f"Val subset chronologically sorted:    {is_val_sorted}")
    print(f"Test subset chronologically sorted:   {is_test_sorted}")
    
    if not (no_overlap_train_val and no_overlap_val_test and is_train_sorted and is_val_sorted and is_test_sorted):
        raise ValueError("Data split integrity checks failed.")
    else:
        print("  - All data split integrity checks passed.")
        
    print("=" * 80)
    print("C.7 PREPROCESSING AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
