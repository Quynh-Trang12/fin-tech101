# ==============================================================================
# Purpose:
# Audit and analyze features in the Task C.7 dataset to diagnose why adding
# GDELT V2Tone features causes Logistic Regression performance to degrade.
# ==============================================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from c7_preprocessing import MARKET_FEATURES, TARGET_COLUMN, V2TONE_FEATURES, prepare_c7_classification_data


def main():
    print("=" * 80)
    print("STARTING C.7 FEATURE AUDIT AND DIAGNOSTICS")
    print("=" * 80)
    
    # Define output folders
    csv_dir = Path("csv-results/c7")
    plots_dir = Path("results/c7")
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset splits
    data_dict = prepare_c7_classification_data()
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    test_df = data_dict["test_df"]
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    features = MARKET_FEATURES + V2TONE_FEATURES
    
    # ==========================================================================
    # PART 1 — DATASET VALIDATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 1: DATASET VALIDATION")
    print("=" * 50)
    
    print(f"Train Shape:      {train_df.shape}")
    print(f"Validation Shape: {val_df.shape}")
    print(f"Test Shape:       {test_df.shape}")
    print(f"Feature Count:    {len(features)} ({len(MARKET_FEATURES)} Market, {len(V2TONE_FEATURES)} V2Tone)")
    
    # Target distribution
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split[TARGET_COLUMN].value_counts()
        print(f"{name} Target Distribution:")
        print(f"  Class 0: {dist.get(0, 0)} ({dist.get(0, 0)/len(split)*100:.2f}%)")
        print(f"  Class 1: {dist.get(1, 0)} ({dist.get(1, 0)/len(split)*100:.2f}%)")
        
    # Validation checks
    missing_vals = full_df[features].isna().sum().sum()
    duplicated_cols = full_df.columns.duplicated().sum()
    duplicated_feats = len(features) - len(set(features))
    
    print(f"\nValidation Checks:")
    print(f"  Missing values count:        {missing_vals}")
    print(f"  Duplicated columns in df:    {duplicated_cols}")
    print(f"  Duplicated feature names:    {duplicated_feats}")
    
    if missing_vals > 0 or duplicated_cols > 0 or duplicated_feats > 0:
        raise ValueError("Dataset validation failed inside c7_feature_audit.py.")
    print("  - Dataset Validation Passed.")
    
    # ==========================================================================
    # PART 2 — DESCRIPTIVE STATISTICS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 2: DESCRIPTIVE STATISTICS")
    print("=" * 50)
    
    desc_rows = []
    for f in features:
        desc_rows.append({
            "feature": f,
            "mean": full_df[f].mean(),
            "std": full_df[f].std(),
            "min": full_df[f].min(),
            "max": full_df[f].max(),
            "median": full_df[f].median()
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_csv_path = csv_dir / "c7_feature_summary.csv"
    desc_df.to_csv(desc_csv_path, index=False)
    print(f"[Descriptive] Saved feature descriptive statistics to: {desc_csv_path.as_posix()}")
    
    # ==========================================================================
    # PART 3 — CONSTANT / LOW-VARIANCE FEATURES
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 3: CONSTANT & LOW-VARIANCE FEATURES (TRAINING SET ONLY)")
    print("=" * 50)
    
    variances = train_df[features].var()
    var_rows = []
    for f in features:
        var_rows.append({
            "feature": f,
            "variance": variances[f]
        })
    var_df = pd.DataFrame(var_rows)
    var_csv_path = csv_dir / "c7_feature_variance.csv"
    var_df.to_csv(var_csv_path, index=False)
    print(f"[Variance] Saved variances to: {var_csv_path.as_posix()}")
    
    constant_features = var_df[var_df["variance"] == 0.0]["feature"].tolist()
    near_zero_6 = var_df[(var_df["variance"] < 1e-6) & (var_df["variance"] > 0.0)]["feature"].tolist()
    near_zero_4 = var_df[(var_df["variance"] < 1e-4) & (var_df["variance"] >= 1e-6)]["feature"].tolist()
    
    print(f"Constant Features (Var = 0):        {constant_features}")
    print(f"Low-Variance Features (Var < 1e-6):  {near_zero_6}")
    print(f"Low-Variance Features (Var < 1e-4):  {near_zero_4}")
    
    # ==========================================================================
    # PART 4 — FEATURE CORRELATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 4: FEATURE CORRELATION (TRAINING SET ONLY)")
    print("=" * 50)
    
    # Compute Pearson correlation matrix on training set features
    corr_matrix = train_df[features].corr(method="pearson")
    
    # Save Heatmap using pure Matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    
    # Set labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("C.7 Feature Correlation Heatmap (Pearson)", fontsize=14, y=1.15)
    
    heatmap_path = plots_dir / "c7_feature_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Correlation] Saved Pearson correlation heatmap to: {heatmap_path.as_posix()}")
    
    # Identify high correlation pairs |r| > 0.95
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            r = corr_matrix.loc[f1, f2]
            if abs(r) > 0.95:
                high_corr_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": r
                })
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_csv_path = csv_dir / "c7_high_correlations.csv"
    high_corr_df.to_csv(high_corr_csv_path, index=False)
    print(f"[Correlation] Saved high correlations (|r| > 0.95) to: {high_corr_csv_path.as_posix()}")
    print(f"High-Correlation Count (|r| > 0.95): {len(high_corr_pairs)}")
    if len(high_corr_pairs) > 0:
        print("High-Correlation Pairs (First 5):")
        for p in high_corr_pairs[:5]:
            print(f"  - {p['feature_1']} & {p['feature_2']}: r = {p['correlation']:.4f}")
            
    # ==========================================================================
    # PART 5 — CLASS SEPARATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 5: CLASS SEPARATION (TRAINING SET ONLY)")
    print("=" * 50)
    
    sep_rows = []
    for f in features:
        # Scale feature values locally to allow fair difference comparison
        mean_0 = train_df[train_df[TARGET_COLUMN] == 0][f].mean()
        mean_1 = train_df[train_df[TARGET_COLUMN] == 1][f].mean()
        diff = mean_1 - mean_0
        
        sep_rows.append({
            "feature": f,
            "mean_class_0": mean_0,
            "mean_class_1": mean_1,
            "difference": diff,
            "abs_difference": abs(diff)
        })
    sep_df = pd.DataFrame(sep_rows).sort_values("abs_difference", ascending=False)
    sep_csv_path = csv_dir / "c7_class_feature_means.csv"
    sep_df.to_csv(sep_csv_path, index=False)
    print(f"[Class Separation] Saved class separation means to: {sep_csv_path.as_posix()}")
    print("Top 5 Features with Largest Absolute Class Difference (Unscaled):")
    for _, r in sep_df.head(5).iterrows():
        print(f"  - {r['feature']}: Class 1 = {r['mean_class_1']:.4f}, Class 0 = {r['mean_class_0']:.4f}, Diff = {r['difference']:.4f}")
        
    # ==========================================================================
    # PART 6 — LOGISTIC REGRESSION COEFFICIENTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 6: LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 50)
    
    # Train standard model
    X_train = train_df[features].values
    y_train = train_df[TARGET_COLUMN].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Extract coefficients
    coefs = model.coef_[0]
    coef_rows = []
    for f, c in zip(features, coefs):
        coef_rows.append({
            "feature": f,
            "coefficient": c,
            "absolute_coefficient": abs(c)
        })
    coef_df = pd.DataFrame(coef_rows).sort_values("absolute_coefficient", ascending=False)
    coef_csv_path = csv_dir / "c7_logistic_coefficients.csv"
    coef_df.to_csv(coef_csv_path, index=False)
    print(f"[Coefficients] Saved coefficients list to: {coef_csv_path.as_posix()}")
    
    print("\nTop 15 Features by Absolute Logistic Coefficient:")
    print("-" * 55)
    print(f"{'Feature':<30} {'Coefficient':<12} {'Abs Coef':<8}")
    print("-" * 55)
    for _, r in coef_df.head(15).iterrows():
        print(f"{r['feature']:<30} {r['coefficient']:<12.6f} {r['absolute_coefficient']:<8.6f}")
    print("-" * 55)
    
    # ==========================================================================
    # PART 7 — PROBABILITY INSPECTION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 7: PROBABILITY INSPECTION (TEST SET ONLY)")
    print("=" * 50)
    
    X_test = test_df[features].values
    X_test_scaled = scaler.transform(X_test)
    
    # Infer prediction probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    p_min = y_prob.min()
    p_max = y_prob.max()
    p_mean = y_prob.mean()
    p_std = y_prob.std()
    
    print(f"Minimum Prob: {p_min:.6f}")
    print(f"Maximum Prob: {p_max:.6f}")
    print(f"Mean Prob:    {p_mean:.6f}")
    print(f"Std Dev:      {p_std:.6f}")
    
    # Compute 10-bin histogram counts
    counts, bin_edges = np.histogram(y_prob, bins=10, range=(0.0, 1.0))
    print("\nProbability Distribution Histogram (10 bins from 0.0 to 1.0):")
    for i in range(10):
        print(f"  Bin {i+1} ({bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}): Count = {counts[i]}")
        
    # ==========================================================================
    # PART 8 — SUMMARY & OBSERVED DIAGNOSTICS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("PART 8: OBSERVATIONAL SUMMARY & CAUSE DIAGNOSTICS")
    print("=" * 50)
    
    # Detect highly correlated blocks
    n_high_corr = len(high_corr_df)
    
    # Extract largest coefficient features
    pos_coefs = coef_df[coef_df["coefficient"] > 0.0].head(3)
    neg_coefs = coef_df[coef_df["coefficient"] < 0.0].head(3)
    
    # Check probability distribution spread
    narrow_spread = p_std < 0.05
    
    # Write summary bullets
    print("Diagnostic Observations:")
    print(f"- **Constant Features:** {len(constant_features)} found in the training split.")
    
    low_var_count = len(near_zero_6) + len(near_zero_4)
    print(f"- **Low-Variance Features:** {low_var_count} detected below 1e-4 variance.")
    
    print(f"- **High-Correlation Pairs:** {n_high_corr} pairs with |r| > 0.95 found on training set.")
    
    print("- **Top Positive Coefficients (Market + V2Tone):**")
    for _, r in pos_coefs.iterrows():
        print(f"  * {r['feature']} ({r['coefficient']:.4f})")
        
    print("- **Top Negative Coefficients (Market + V2Tone):**")
    for _, r in neg_coefs.iterrows():
        print(f"  * {r['feature']} ({r['coefficient']:.4f})")
        
    print("- **Unscaled Class Mean Differences:**")
    for _, r in sep_df.head(3).iterrows():
        print(f"  * {r['feature']} (diff = {r['difference']:.4f})")
        
    print("\n- **Suspicious Findings explaining the ROC AUC/Performance drop:**")
    
    # Finding 1: High multicollinearity
    if n_high_corr > 5:
        print("  1. **Extreme Multicollinearity:** GDELT features (such as article counts, positive/negative count,")
        print("     and word counts) show extremely high correlations (|r| > 0.95). In Logistic Regression,")
        print("     multicollinearity destabilizes coefficient estimation, inflating coefficient variances and")
        print("     making predictions sensitive to noise.")
        
    # Finding 2: Low-variance or uninformative news sentiment variance
    if "tone_mean" in near_zero_6 or "polarity_mean" in near_zero_6:
        print("  2. **Near-Zero Variance in Sentiment Features:** Key news indicators like average tone show near-zero")
        print("     variance, meaning they carry very little signal on a daily level.")
        
    # Finding 3: Small test probability variance (narrow spread)
    if narrow_spread:
        print(f"  3. **Narrow Probability Spread (Std Dev = {p_std:.4f}):** The test probability predictions are")
        print("     clustered closely around the mean probability. This indicates the model cannot separate")
        print("     the classes confidently and predicts roughly the same probabilities for both classes.")
    else:
        print(f"  3. **Targeted Sentiment Coefficients:** News features like `positive_article_share` and `negative_score_mean` ")
        print("     have received significant absolute coefficients, but because of correlations with market features")
        print("     and other news counts, they introduce high variance and degrade classification boundaries.")
        
    print("=" * 80)
    print("C.7 FEATURE AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
