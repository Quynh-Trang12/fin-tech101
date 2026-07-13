# ==============================================================================
# Purpose:
# Run article-level sentiment inference using FinBERT (ProsusAI/finbert)
# on GDELT English headlines for Task C.7.
#
# Supports interruption-safe execution through batch checkpointing, 
# GPU acceleration, and strict validation checks.
# ==============================================================================

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    GDELT_ENRICHED_CACHE_PATH,
    GDELT_FINBERT_ARTICLE_PATH,
    GDELT_FINBERT_CHECKPOINT_PATH,
)


def load_enriched_articles(path: Path) -> pd.DataFrame:
    """
    Load the enriched articles Parquet file and validate the required schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Enriched GDELT article cache not found at: {path.as_posix()}")
    
    df = pd.read_parquet(path)
    
    # Required input columns
    required = ["DocumentIdentifier", "published_at", "headline", "has_english_headline"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Enriched cache is missing required columns: {missing}")
        
    return df


def prepare_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare headlines for prediction:
    - Keep only has_english_headline == 1
    - Keep non-null and non-empty headlines after trimming
    - Preserve deterministic row ordering
    """
    data = df.copy()
    
    # 1. Keep English headlines
    data = data[data["has_english_headline"] == 1]
    
    # 2. Filter non-empty headlines
    data["headline"] = data["headline"].astype("string").str.strip()
    data = data[data["headline"].notna() & (data["headline"] != "")]
    
    # 3. Sort deterministically by publication timestamp and unique URL
    data = data.sort_values(by=["published_at", "DocumentIdentifier"], kind="stable").reset_index(drop=True)
    
    return data


def load_finbert_model(model_name: str, device: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load the FinBERT tokenizer and sequence classification model on the selected device.
    """
    print(f"[FinBERT] Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def resolve_label_mapping(model: AutoModelForSequenceClassification) -> Tuple[int, int, int]:
    """
    Map positive, neutral, and negative labels to model logit indices.
    """
    id2label = model.config.id2label
    mapping = {}
    for idx, label in id2label.items():
        mapping[label.lower()] = int(idx)
        
    pos_idx = mapping.get("positive")
    neu_idx = mapping.get("neutral")
    neg_idx = mapping.get("negative")
    
    if pos_idx is None or neu_idx is None or neg_idx is None:
        raise ValueError(f"Could not map all sentiment labels in config id2label: {id2label}")
        
    return pos_idx, neu_idx, neg_idx


def predict_batch(
    headlines: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str,
    max_length: int,
    pos_idx: int,
    neu_idx: int,
    neg_idx: int,
) -> Dict[str, List[Any]]:
    """
    Tokenize headlines and perform sentiment probability classification.
    """
    # Truncate and pad headlines
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Move inputs to target device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run model sequence classification with no gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
    # Extract prob vectors
    pos_probs = probs[:, pos_idx].tolist()
    neu_probs = probs[:, neu_idx].tolist()
    neg_probs = probs[:, neg_idx].tolist()
    
    labels = []
    confidences = []
    
    # Map index to model output labels
    id2label = model.config.id2label
    
    for i in range(len(headlines)):
        p = pos_probs[i]
        nu = neu_probs[i]
        ne = neg_probs[i]
        
        # Calculate maximum probability confidence
        max_prob = max(p, nu, ne)
        confidences.append(max_prob)
        
        # Fetch the predicted label name
        argmax_idx = int(np.argmax(probs[i]))
        labels.append(id2label[argmax_idx].lower())
        
    return {
        "finbert_positive_probability": pos_probs,
        "finbert_neutral_probability": neu_probs,
        "finbert_negative_probability": neg_probs,
        "finbert_predicted_label": labels,
        "finbert_confidence": confidences,
    }


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame | None:
    """Load existing checkpoint file if available."""
    if checkpoint_path.exists():
        try:
            df = pd.read_parquet(checkpoint_path)
            print(f"[Checkpoint] Loaded existing checkpoint with {len(df):,} predictions.")
            return df
        except Exception as error:
            print(f"[Warning] Failed to load checkpoint file: {error}. Starting fresh.")
    return None


def save_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Save the current predictions to a checkpoint parquet file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(checkpoint_path, index=False)


def validate_predictions(df: pd.DataFrame) -> None:
    """
    Validate that the generated predictions conform to all sanity checks.
    """
    print("[Validation] Running prediction sanity checks...")
    
    # 1. Unique DocumentIdentifier
    if not df["DocumentIdentifier"].is_unique:
        raise ValueError("Assertion failed: Output dataset contains duplicate DocumentIdentifier values.")
        
    # 2. Chronological sorting
    sort_keys = ["published_at", "DocumentIdentifier"]
    is_sorted = df.set_index(sort_keys).index.is_monotonic_increasing
    if not is_sorted:
        raise ValueError("Assertion failed: Output is not sorted chronologically.")
        
    # 3. Individual record validation
    for idx, row in df.iterrows():
        p_pos = row["finbert_positive_probability"]
        p_neu = row["finbert_neutral_probability"]
        p_neg = row["finbert_negative_probability"]
        pred_label = row["finbert_predicted_label"]
        conf = row["finbert_confidence"]
        headline_text = row["headline"]
        
        # Headline is not empty
        if pd.isna(headline_text) or str(headline_text).strip() == "":
            raise ValueError(f"Row {idx} fails: Headline text is empty.")
            
        # Probabilities in range [0, 1]
        if not (0.0 <= p_pos <= 1.0 and 0.0 <= p_neu <= 1.0 and 0.0 <= p_neg <= 1.0):
            raise ValueError(f"Row {idx} fails: Probabilities must be in [0, 1]. Got pos={p_pos}, neu={p_neu}, neg={p_neg}")
            
        # Sum to ~1.0
        prob_sum = p_pos + p_neu + p_neg
        if abs(prob_sum - 1.0) > 1e-4:
            raise ValueError(f"Row {idx} fails: Probabilities do not sum to 1. Got sum={prob_sum}")
            
        # Confidence matches maximum probability value
        expected_conf = max(p_pos, p_neu, p_neg)
        if abs(conf - expected_conf) > 1e-6:
            raise ValueError(f"Row {idx} fails: Confidence ({conf}) does not match max probability ({expected_conf})")
            
        # Predicted label matches maximum probability class label
        probs_dict = {"positive": p_pos, "neutral": p_neu, "negative": p_neg}
        expected_label = max(probs_dict, key=probs_dict.get)
        if pred_label != expected_label:
            raise ValueError(f"Row {idx} fails: Label ({pred_label}) does not match max probability label ({expected_label})")
            
    print("  - All sanity checks passed successfully.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run article-level FinBERT sentiment inference on GDELT headlines.")
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of headlines scored (useful for smoke tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FINBERT_BATCH_SIZE,
        help=f"Inference batch size. Default: {FINBERT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=FINBERT_MAX_LENGTH,
        help=f"Maximum sequence length. Default: {FINBERT_MAX_LENGTH}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (cpu or cuda). If not specified, CUDA will be automatically used if available.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume inference from the existing checkpoint if present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the final output file if it already exists.",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    # 1. Output file safety checks
    final_output_path = GDELT_FINBERT_ARTICLE_PATH
    if final_output_path.exists() and not args.force:
        print(f"[Error] Output file already exists: {final_output_path.as_posix()}")
        print("Use --force to overwrite. Exiting.")
        sys.exit(0)
        
    # 2. Select device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print("=" * 80)
    print("STARTING FINBERT SENTIMENT INFERENCE WORKFLOW")
    print("=" * 80)
    print(f"Target Device:      {device.upper()}")
    print(f"Model Name:         {FINBERT_MODEL_NAME}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Max Length:         {args.max_length}")
    print(f"Final Output Path:  {final_output_path.as_posix()}")
    print(f"Checkpoint Path:    {GDELT_FINBERT_CHECKPOINT_PATH.as_posix()}")
    print("=" * 80)
    
    # 3. Load and clean headlines
    raw_df = load_enriched_articles(GDELT_ENRICHED_CACHE_PATH)
    prepared_df = prepare_headlines(raw_df)
    
    total_headlines = len(prepared_df)
    print(f"[Data] Found {total_headlines:,} valid English headlines for scoring.")
    
    # Apply limit if requested
    if args.limit is not None:
        prepared_df = prepared_df.head(args.limit).copy()
        print(f"[Data] Limit applied. Scoring first {args.limit} headlines.")
        
    if prepared_df.empty:
        print("[Data] No headlines to process. Exiting.")
        sys.exit(0)
        
    # 4. Load Checkpoint or Resume
    checkpoint_df = None
    if args.resume:
        checkpoint_df = load_checkpoint(GDELT_FINBERT_CHECKPOINT_PATH)
        
    if checkpoint_df is not None:
        # Find which URL IDs are already processed
        processed_ids = set(checkpoint_df["DocumentIdentifier"].values)
        pending_df = prepared_df[~prepared_df["DocumentIdentifier"].isin(processed_ids)].copy()
        print(f"[Resume] Skip {len(processed_ids):,} already processed rows. {len(pending_df):,} rows remaining.")
    else:
        pending_df = prepared_df.copy()
        checkpoint_df = pd.DataFrame()
        
    if pending_df.empty:
        print("[Resume] All target rows have already been processed in checkpoint.")
    else:
        # 5. Load model and tokenizer
        tokenizer, model = load_finbert_model(FINBERT_MODEL_NAME, device)
        pos_idx, neu_idx, neg_idx = resolve_label_mapping(model)
        
        # Logit class labels
        id2label = model.config.id2label
        print(f"[FinBERT] Label indices mapping: positive={pos_idx}, neutral={neu_idx}, negative={neg_idx}")
        
        # 6. Run inference loop
        print(f"[Inference] Scoring {len(pending_df):,} headlines in batches of {args.batch_size}...")
        
        for start_idx in range(0, len(pending_df), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(pending_df))
            batch_df = pending_df.iloc[start_idx:end_idx].copy()
            
            # Predict
            batch_results = predict_batch(
                headlines=batch_df["headline"].tolist(),
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                pos_idx=pos_idx,
                neu_idx=neu_idx,
                neg_idx=neg_idx,
            )
            
            # Insert prediction columns
            for col_name, values in batch_results.items():
                batch_df[col_name] = values
                
            # Concat to checkpoint and save to disk
            checkpoint_df = pd.concat([checkpoint_df, batch_df], ignore_index=True)
            save_checkpoint(checkpoint_df, GDELT_FINBERT_CHECKPOINT_PATH)
            
            processed_count = len(checkpoint_df)
            print(f"Processed batch {start_idx // args.batch_size + 1}: scored {end_idx - start_idx} rows. Total: {processed_count}/{len(prepared_df)}")
            
    # 7. Finalize and Save
    print("\n[Finalize] Finalizing output dataset...")
    # Keep only the rows corresponding to our target run (if limit was applied, keep limit rows)
    # The checkpoint could contain more rows if limit was not applied or resume processed them.
    # To be safe, we filter checkpoint_df to matching URL IDs.
    target_ids = set(prepared_df["DocumentIdentifier"].values)
    final_df = checkpoint_df[checkpoint_df["DocumentIdentifier"].isin(target_ids)].copy()
    
    # Sort deterministically
    final_df = final_df.sort_values(by=["published_at", "DocumentIdentifier"], kind="stable").reset_index(drop=True)
    
    # Validate final output
    validate_predictions(final_df)
    
    # Save final dataset
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(final_output_path, index=False)
    print(f"\n[Finalize] Output successfully saved: {final_output_path.as_posix()} (Rows: {len(final_df):,})")
    
    # 8. Print audit stats and sample predictions
    print("\n" + "=" * 60)
    print("FINBERT SENTIMENT AUDIT STATISTICS")
    print("=" * 60)
    print(f"Total Rows Processed:   {len(final_df)}")
    print(f"Mean Confidence Score:  {final_df['finbert_confidence'].mean():.6f}")
    
    dist = final_df["finbert_predicted_label"].value_counts()
    print("Predicted Label Distribution:")
    for lbl in ["positive", "neutral", "negative"]:
        cnt = dist.get(lbl, 0)
        print(f"  * {lbl:<8}: {cnt:<6} ({cnt/len(final_df)*100:.2f}%)")
        
    print("\nSample Predictions (First 5):")
    for i, (_, row) in enumerate(final_df.head(5).iterrows()):
        print(f"Headline {i+1}: '{row['headline']}'")
        print(f"  - Positive Prob: {row['finbert_positive_probability']:.6f}")
        print(f"  - Neutral Prob:  {row['finbert_neutral_probability']:.6f}")
        print(f"  - Negative Prob: {row['finbert_negative_probability']:.6f}")
        print(f"  - Predicted:     {row['finbert_predicted_label'].upper()} (Confidence: {row['finbert_confidence']:.6f})")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[Error] Pipeline execution failed: {error}")
        sys.exit(1)
