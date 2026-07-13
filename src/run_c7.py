# ==============================================================================
# Purpose:
# Orchestrate and execute the complete Task C.7 workflow in chronological sequence,
# utilizing cached parquet datasets where possible.
# ==============================================================================

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list[str] = None) -> None:
    cmd = [sys.executable, str(Path("src") / script_name)]
    if args:
        cmd.extend(args)
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error: {script_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main() -> None:
    print("=" * 80)
    print("RUNNING THE COMPLETE C.7 NEWS SENTIMENT CLASSIFICATION WORKFLOW")
    print("=" * 80)

    # 1. GDELT News Data extraction/caching
    run_script("c7_news_data.py")

    # 2. GDELT News Title enrichment
    run_script("c7_news_titles.py")

    # 3. V2Tone daily calendar aggregation
    run_script("c7_news_features.py")

    # 4. V2Tone trading-day searchsorted alignment
    run_script("c7_news_alignment.py")

    # 5. FinBERT sentiment headline-level inference
    run_script("c7_finbert_features.py", ["--resume"])

    # 6. FinBERT daily aggregation and trading-day alignment
    run_script("c7_finbert_daily.py")

    # 7. Merge datasets to build final C7 classification matrix
    run_script("c7_dataset.py")

    # 8. Run validation tests and split data
    run_script("c7_preprocessing.py")

    # 9. Train and evaluate the six comparative Logistic Regression models
    run_script("c7_baseline.py")

    print("\n" + "=" * 80)
    print("C.7 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
