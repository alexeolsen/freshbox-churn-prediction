#!/usr/bin/env python
"""
Phase 4: Feature Engineering
Build ~25 features from raw data.
"""

from src.shared.phase_0_1_data_prep import load_raw_data, standardize_dates, standardize_flags, fix_churn_date_anomalies
from src.phase_3.phase_3_1_customer_base import build_customer_base
from src.phase_4.phase_4_1_features import assemble_feature_table
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 4: Feature Engineering\n")

    # Step 1: Load and prepare data
    print("Step 1/4: Loading and preparing raw data...")
    dfs = load_raw_data()
    dfs = standardize_dates(dfs)
    dfs = standardize_flags(dfs)
    dfs, _ = fix_churn_date_anomalies(dfs)
    print("  [OK] Data prepared")

    # Step 2: Build customer base with leakage prevention
    print("\nStep 2/4: Building customer-level base table...")
    dfs = build_customer_base(dfs)
    print("  [OK] Base table built")

    # Step 3: Engineer features
    print("\nStep 3/4: Engineering features...")
    feature_table = assemble_feature_table(
        dfs['customers'],
        dfs['activity_filtered'],
        dfs['tickets_filtered']
    )
    print("  [OK] Features engineered")

    # Step 4: Save and inspect
    print("\nStep 4/4: Saving and inspecting...")
    feature_table.to_parquet('data/processed/customer_features.parquet', index=False)
    print("  [OK] Features saved to data/processed/customer_features.parquet")

    print(f"\nFeature summary:")
    print(f"  Shape: {feature_table.shape}")
    print(f"  Churn rate: {feature_table['churned'].mean():.1%}")
    print(f"\nFirst few features:")
    print(feature_table.iloc[:5, :10].to_string())

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save engineered features
    feature_table.to_csv(output_dir / "phase_4_features_engineered.csv", index=False)
    print(f"  [OK] phase_4_features_engineered.csv ({len(feature_table):,} rows, {feature_table.shape[1]} columns)")

    print("\n[SUCCESS] Phase 4 complete!")
    print("\nNext: Phase 5 - Train/Test Split and Baseline Model\n")
