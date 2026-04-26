#!/usr/bin/env python
"""
Phase 5: Train/Test Split and Heuristic Baseline
Establish the performance bar that ML models must clear.
"""

from src.shared.phase_0_2_models import load_features, create_train_test_split, evaluate_heuristic, print_baseline_summary
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 5: Train/Test Split & Heuristic Baseline\n")

    # Step 1: Load features
    print("Step 1/3: Loading feature table...")
    features = load_features('data/processed/customer_features.parquet')
    print(f"  [OK] Loaded {len(features):,} customers with {len(features.columns)} columns")

    # Step 2: Create train/test split
    print("\nStep 2/3: Creating stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(features)
    print(f"  [OK] Train set: {len(X_train):,} customers")
    print(f"  [OK] Test set:  {len(X_test):,} customers")
    print(f"  [OK] Train churn rate: {y_train.mean():.1%}")
    print(f"  [OK] Test churn rate:  {y_test.mean():.1%}")

    # Step 3: Evaluate heuristic baseline
    print("\nStep 3/3: Evaluating heuristic baseline rule...")
    y_pred_heuristic, metrics = evaluate_heuristic(X_test, y_test)

    # Summary
    print_baseline_summary(metrics)

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train and test sets
    train_data = X_train.copy()
    train_data['churned'] = y_train.values
    train_data.to_csv(output_dir / "phase_5_train_set.csv", index=False)
    print(f"  [OK] phase_5_train_set.csv ({len(train_data):,} rows)")

    test_data = X_test.copy()
    test_data['churned'] = y_test.values
    test_data['heuristic_prediction'] = y_pred_heuristic
    test_data.to_csv(output_dir / "phase_5_test_set_with_heuristic.csv", index=False)
    print(f"  [OK] phase_5_test_set_with_heuristic.csv ({len(test_data):,} rows)")

    # Save baseline results summary
    results_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Score': list(metrics.values())
    })
    results_df.to_csv(output_dir / "phase_5_heuristic_baseline_metrics.csv", index=False)
    print(f"  [OK] phase_5_heuristic_baseline_metrics.csv")

    print("\n[SUCCESS] Phase 5 complete!")
    print("\nNext: Phase 6 - Logistic Regression Model\n")
