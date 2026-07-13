# ==============================================================================
# Purpose:
# Test a compact, non-redundant V2Tone feature representation to see if it
# improves Logistic Regression performance on the validation set.
#
# Prevents target leakages by keeping the test split completely unused.
# ==============================================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from c7_preprocessing import MARKET_FEATURES, TARGET_COLUMN, prepare_c7_classification_data

# Compact, non-redundant features to eliminate collinearity and constant columns
REDUCED_V2TONE_FEATURES = [
    "article_count",
    "tone_mean",
    "polarity_mean",
    "positive_article_share",
    "neutral_article_share",
    "activity_reference_density_mean",
    "self_group_reference_density_mean",
    "word_count_mean",
]


def evaluate_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    cm_save_path: Path,
) -> dict:
    """
    Fit StandardScaler and LogisticRegression on training set, then evaluate on validation.
    """
    # 1. Scale features: fit strictly on training, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 2. Train Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 3. Infer on validation set
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    # 4. Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_prob)
    
    # 5. Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title(f"Validation CM - {model_name}")
    
    cm_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Validation] Saved confusion matrix to: {cm_save_path.as_posix()}")
    
    return {
        "model": model_name,
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
    }


def main():
    print("=" * 80)
    print("STARTING C.7 REDUCED V2TONE VALIDATION SWEEP")
    print("=" * 80)
    
    # Define output folders
    csv_dir = Path("csv-results/c7")
    plots_dir = Path("results/c7")
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset splits (validation focus)
    data_dict = prepare_c7_classification_data()
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    
    # Print the exact feature lists
    print(f"Market Features (Count: {len(MARKET_FEATURES)}):")
    print(f"  {MARKET_FEATURES}")
    print(f"Reduced V2Tone Features (Count: {len(REDUCED_V2TONE_FEATURES)}):")
    print(f"  {REDUCED_V2TONE_FEATURES}\n")
    
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    
    # --------------------------------------------------------------------------
    # Run 1: Market Only features on Validation
    # --------------------------------------------------------------------------
    print("Evaluating Model 1: Market Only...")
    X_train_m = train_df[MARKET_FEATURES].values
    X_val_m = val_df[MARKET_FEATURES].values
    
    market_metrics = evaluate_on_validation(
        X_train=X_train_m,
        y_train=y_train,
        X_val=X_val_m,
        y_val=y_val,
        model_name="Market Only",
        cm_save_path=plots_dir / "logistic_market_validation_confusion_matrix.png",
    )
    
    # --------------------------------------------------------------------------
    # Run 2: Market + Reduced V2Tone features on Validation
    # --------------------------------------------------------------------------
    print("\nEvaluating Model 2: Market + Reduced V2Tone...")
    reduced_features = MARKET_FEATURES + REDUCED_V2TONE_FEATURES
    X_train_r = train_df[reduced_features].values
    X_val_r = val_df[reduced_features].values
    
    reduced_metrics = evaluate_on_validation(
        X_train=X_train_r,
        y_train=y_train,
        X_val=X_val_r,
        y_val=y_val,
        model_name="Market + Reduced V2Tone",
        cm_save_path=plots_dir / "logistic_reduced_v2tone_validation_confusion_matrix.png",
    )
    
    # --------------------------------------------------------------------------
    # Save & Display Comparison
    # --------------------------------------------------------------------------
    metrics_list = [market_metrics, reduced_metrics]
    metrics_df = pd.DataFrame(metrics_list)
    
    metrics_csv_path = csv_dir / "c7_reduced_v2tone_validation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[Validation] Saved validation comparison to: {metrics_csv_path.as_posix()}")
    
    # Print the validation comparison table
    print("\n" + "=" * 75)
    print("VALIDATION COMPARISON (MARKET ONLY vs MARKET + REDUCED V2TONE)")
    print("=" * 75)
    print(f"{'Metric':<22} {'Market Only':<15} {'Market + Red V2Tone':<22} {'Delta':<6}")
    print("-" * 75)
    
    metrics_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    for k in metrics_keys:
        val_m = float(market_metrics[k])
        val_r = float(reduced_metrics[k])
        diff = val_r - val_m
        metric_label = k.replace("_", " ").title()
        print(f"{metric_label:<22} {val_m:<15.6f} {val_r:<22.6f} {diff:<+.6f}")
        
    print("=" * 75)
    print("\nREDUCED V2TONE VALIDATION SWEEP COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
