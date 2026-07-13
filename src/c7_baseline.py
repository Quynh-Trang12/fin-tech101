# ==============================================================================
# Purpose:
# Train and evaluate Logistic Regression classifiers for Task C.7 across six
# controlled feature sets to assess the contribution of full and reduced
# news sentiment features.
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

from c7_preprocessing import prepare_c7_classification_data


def evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_count: int,
    cm_save_path: Path,
) -> dict:
    """
    Fit StandardScaler and LogisticRegression strictly on training, infer on test,
    calculate evaluation metrics, and save confusion matrix.
    """
    # 1. Scale features (fit only on training, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 3. Predict on the test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    pred_pos = int(y_pred.sum())
    pred_neg = int(len(y_pred) - pred_pos)
    
    # 5. Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}")
    
    cm_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment] Saved confusion matrix for {model_name} to: {cm_save_path.as_posix()}")
    
    return {
        "model": model_name,
        "feature_count": feature_count,
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positives": pred_pos,
        "predicted_negatives": pred_neg,
    }


def main():
    print("=" * 80)
    print("STARTING TASK C.7 LOGISTIC REGRESSION COMPARATIVE EXPERIMENTS (ENHANCED)")
    print("=" * 80)
    
    # Define directories
    csv_dir = Path("csv-results/c7")
    plots_dir = Path("results/c7")
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load prepared chronological classification splits
    data_dict = prepare_c7_classification_data()
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    test_df = data_dict["test_df"]
    meta = data_dict["split_metadata"]
    
    y_train = train_df[data_dict["target_column"]].values
    y_val = val_df[data_dict["target_column"]].values
    y_test = test_df[data_dict["target_column"]].values
    
    # Log dataset split metadata
    print("Dataset Split Information:")
    print(f"  - Train split rows:      {meta['train_rows']} (Class 1 Share: {y_train.mean():.4f})")
    print(f"  - Validation split rows: {meta['val_rows']} (Class 1 Share: {y_val.mean():.4f})")
    print(f"  - Test split rows:       {meta['test_rows']} (Class 1 Share: {y_test.mean():.4f})")
    print("-" * 80)
    
    # 2. Run the six comparative experiments
    feature_sets = data_dict["feature_sets"]
    experiment_results = []
    
    # Experiment 1: Market only
    print("\n[Running] Experiment 1: Market only...")
    market_feats = feature_sets["market_only"]
    res_market = evaluate_model(
        X_train=train_df[market_feats].values,
        y_train=y_train,
        X_test=test_df[market_feats].values,
        y_test=y_test,
        model_name="Market Only",
        feature_count=len(market_feats),
        cm_save_path=plots_dir / "logistic_market_only_confusion_matrix.png",
    )
    experiment_results.append(res_market)
    
    # Experiment 2: Market + V2Tone
    print("\n[Running] Experiment 2: Market + V2Tone...")
    v2_feats = feature_sets["market_plus_v2tone"]
    res_v2 = evaluate_model(
        X_train=train_df[v2_feats].values,
        y_train=y_train,
        X_test=test_df[v2_feats].values,
        y_test=y_test,
        model_name="Market + V2Tone",
        feature_count=len(v2_feats),
        cm_save_path=plots_dir / "logistic_market_plus_v2tone_confusion_matrix.png",
    )
    experiment_results.append(res_v2)
    
    # Experiment 3: Market + Full FinBERT
    print("\n[Running] Experiment 3: Market + Full FinBERT...")
    fb_feats = feature_sets["market_plus_finbert"]
    res_fb = evaluate_model(
        X_train=train_df[fb_feats].values,
        y_train=y_train,
        X_test=test_df[fb_feats].values,
        y_test=y_test,
        model_name="Market + Full FinBERT",
        feature_count=len(fb_feats),
        cm_save_path=plots_dir / "logistic_market_plus_finbert_confusion_matrix.png",
    )
    experiment_results.append(res_fb)
    
    # Experiment 4: Market + V2Tone + Full FinBERT
    print("\n[Running] Experiment 4: Market + V2Tone + Full FinBERT...")
    combined_feats = feature_sets["market_plus_v2tone_plus_finbert"]
    res_combined = evaluate_model(
        X_train=train_df[combined_feats].values,
        y_train=y_train,
        X_test=test_df[combined_feats].values,
        y_test=y_test,
        model_name="Market + V2Tone + Full FinBERT",
        feature_count=len(combined_feats),
        cm_save_path=plots_dir / "logistic_market_plus_v2tone_plus_finbert_confusion_matrix.png",
    )
    experiment_results.append(res_combined)
    
    # Experiment 5: Market + Reduced FinBERT
    print("\n[Running] Experiment 5: Market + Reduced FinBERT...")
    reduced_fb_feats = feature_sets["market_plus_reduced_finbert"]
    res_reduced_fb = evaluate_model(
        X_train=train_df[reduced_fb_feats].values,
        y_train=y_train,
        X_test=test_df[reduced_fb_feats].values,
        y_test=y_test,
        model_name="Market + Reduced FinBERT",
        feature_count=len(reduced_fb_feats),
        cm_save_path=plots_dir / "logistic_market_plus_reduced_finbert_confusion_matrix.png",
    )
    experiment_results.append(res_reduced_fb)
    
    # Experiment 6: Market + V2Tone + Reduced FinBERT
    print("\n[Running] Experiment 6: Market + V2Tone + Reduced FinBERT...")
    combined_reduced_feats = feature_sets["market_plus_v2tone_plus_reduced_finbert"]
    res_combined_reduced = evaluate_model(
        X_train=train_df[combined_reduced_feats].values,
        y_train=y_train,
        X_test=test_df[combined_reduced_feats].values,
        y_test=y_test,
        model_name="Market + V2Tone + Reduced FinBERT",
        feature_count=len(combined_reduced_feats),
        cm_save_path=plots_dir / "logistic_market_plus_v2tone_plus_reduced_finbert_confusion_matrix.png",
    )
    experiment_results.append(res_combined_reduced)
    
    # --------------------------------------------------------------------------
    # 3. Save Results Comparison
    # --------------------------------------------------------------------------
    results_df = pd.DataFrame(experiment_results)
    comparison_csv_path = csv_dir / "c7_logistic_comparison.csv"
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"\n[Save] Saved consolidated metrics comparison to: {comparison_csv_path.as_posix()}")
    
    # --------------------------------------------------------------------------
    # 4. Print Comparison Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 135)
    print("C.7 COMPARATIVE LOGISTIC REGRESSION EXPERIMENT RESULTS")
    print("=" * 135)
    print(f"{'Model':<35} {'Feats':<5} {'Accuracy':<9} {'Bal Acc':<9} {'Precision':<10} {'Recall':<7} {'F1':<8} {'ROC AUC':<8} {'TN, FP, FN, TP':<16} {'Pred + / -':<12}")
    print("-" * 135)
    for res in experiment_results:
        cm_str = f"{res['tn']},{res['fp']},{res['fn']},{res['tp']}"
        pred_str = f"{res['predicted_positives']}/{res['predicted_negatives']}"
        print(f"{res['model']:<35} {res['feature_count']:<5} {res['accuracy']:<9.6f} {res['balanced_accuracy']:<9.6f} {res['precision']:<10.6f} {res['recall']:<7.6f} {res['f1']:<8.6f} {res['roc_auc']:<8.6f} {cm_str:<16} {pred_str:<12}")
    print("=" * 135)
    print("\nC.7 COMPARATIVE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 135)


if __name__ == "__main__":
    main()
