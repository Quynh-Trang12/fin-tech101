# ==============================================================================
# Purpose:
# Evaluate the contribution of GDELT news sentiment (V2Tone) features to 
# stock market direction forecasting compared to the Market-Only baseline.
#
# Trains a Logistic Regression model using MARKET + V2TONE features.
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

from c7_preprocessing import MARKET_FEATURES, TARGET_COLUMN, V2TONE_FEATURES, prepare_c7_classification_data


def main():
    print("=" * 80)
    print("STARTING C.7 GDELT V2TONE FEATURE ADDITION EXPERIMENT")
    print("=" * 80)
    
    # Define output folders
    csv_dir = Path("csv-results/c7")
    plots_dir = Path("results/c7")
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load preprocessed classification splits
    # Reuses the exact splits and chronological partition logic from c7_preprocessing
    #
    # Controlled Experiment Design Explanation:
    # - We hold the model hyperparameters, StandardScaler strategy, split ratio, 
    #   and chronological partitions strictly constant.
    # - This isolates the only changing variable: the feature set itself. 
    # - Any change (delta) in evaluation metrics is therefore directly attributable 
    #   to the addition of GDELT V2Tone news sentiment features.
    data_dict = prepare_c7_classification_data()
    train_df = data_dict["train_df"]
    test_df = data_dict["test_df"]
    
    # Define experiment feature set (Market features + V2Tone features)
    features = MARKET_FEATURES + V2TONE_FEATURES
    
    X_train = train_df[features].values
    y_train = train_df[TARGET_COLUMN].values
    
    X_test = test_df[features].values
    y_test = test_df[TARGET_COLUMN].values
    
    # --------------------------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------------------------
    # Fit StandardScaler strictly on the training set only to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --------------------------------------------------------------------------
    # Model Training (Logistic Regression)
    # --------------------------------------------------------------------------
    print("\nTraining Logistic Regression with Market + V2Tone Features...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Infer predictions on the test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of rise for ROC AUC
    
    # --------------------------------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Plot and save confusion matrix using pure matplotlib
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Confusion Matrix - Logistic Regression (Market + V2Tone)")
    
    cm_path = plots_dir / "logistic_market_v2tone_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment] Saved confusion matrix to: {cm_path.as_posix()}")
    
    # Save metrics to CSV
    metrics_csv_path = csv_dir / "c7_v2tone_metrics.csv"
    metrics_df = pd.DataFrame([{
        "model": "Logistic Regression (Market + V2Tone)",
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
    }])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Experiment] Saved metrics to: {metrics_csv_path.as_posix()}")
    
    # Print C.7 V2Tone Experiment Summary
    print("\n" + "=" * 60)
    print("C.7 V2TONE EXPERIMENT SUMMARY")
    print("=" * 60)
    print("Feature Set:       Market + V2Tone")
    print("-" * 60)
    print(f"Accuracy:          {acc:.6f}")
    print(f"Balanced Accuracy: {bal_acc:.6f}")
    print(f"Precision:         {prec:.6f}")
    print(f"Recall:            {rec:.6f}")
    print(f"F1-score:          {f1:.6f}")
    print(f"ROC AUC:           {roc_auc:.6f}")
    print("=" * 60)
    
    # --------------------------------------------------------------------------
    # 3. Baseline Comparison
    # --------------------------------------------------------------------------
    baseline_csv_path = csv_dir / "c7_baseline_metrics.csv"
    if not baseline_csv_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found at: {baseline_csv_path.as_posix()}. Run c7_baseline.py first.")
        
    baseline_df = pd.read_csv(baseline_csv_path)
    lr_baseline_rows = baseline_df[baseline_df["model"] == "Logistic Regression"]
    if lr_baseline_rows.empty:
        raise ValueError("Could not find the 'Logistic Regression' baseline metrics.")
        
    lr_base = lr_baseline_rows.iloc[0]
    
    # Parse baseline values
    b_acc = float(lr_base["accuracy"])
    b_bal_acc = float(lr_base["balanced_accuracy"])
    b_prec = float(lr_base["precision"])
    b_rec = float(lr_base["recall"])
    b_f1 = float(lr_base["f1"])
    b_auc = float(lr_base["roc_auc"])
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (MARKET ONLY vs MARKET + V2TONE)")
    print("=" * 70)
    print(f"{'Metric':<22} {'Market Only':<15} {'Market + V2Tone':<18} {'Delta':<6}")
    print("-" * 70)
    print(f"{'Accuracy':<22} {b_acc:<15.6f} {acc:<18.6f} {acc - b_acc:<+.6f}")
    print(f"{'Balanced Accuracy':<22} {b_bal_acc:<15.6f} {bal_acc:<18.6f} {bal_acc - b_bal_acc:<+.6f}")
    print(f"{'Precision':<22} {b_prec:<15.6f} {prec:<18.6f} {prec - b_prec:<+.6f}")
    print(f"{'Recall':<22} {b_rec:<15.6f} {rec:<18.6f} {rec - b_rec:<+.6f}")
    print(f"{'F1-score':<22} {b_f1:<15.6f} {f1:<18.6f} {f1 - b_f1:<+.6f}")
    print(f"{'ROC AUC':<22} {b_auc:<15.6f} {roc_auc:<18.6f} {roc_auc - b_auc:<+.6f}")
    print("=" * 70)
    print("\nC.7 V2TONE SWEEP COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
