#!/usr/bin/env python
"""
Quick test script to load and validate the raw data.
Run this to check data quality before building features.
"""

from src.shared.phase_0_1_data_prep import load_raw_data, standardize_dates, standardize_flags, fix_churn_date_anomalies, validate_data

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    print("\n[*] Starting data validation...\n")

    # Step 1: Load raw CSVs
    print("Step 1/5: Loading raw CSV files...")
    dfs = load_raw_data()
    print(f"  [OK] Loaded 4 tables:")
    print(f"    - Customers: {len(dfs['customers']):,} rows")
    print(f"    - Weekly Activity: {len(dfs['weekly_activity']):,} rows")
    print(f"    - Support Tickets: {len(dfs['support_tickets']):,} rows")
    print(f"    - Calendar: {len(dfs['calendar']):,} rows")

    # Step 2: Standardize dates
    print("\nStep 2/5: Converting date columns to datetime...")
    dfs = standardize_dates(dfs)
    print(f"  [OK] Date columns converted")

    # Step 3: Standardize flags
    print("\nStep 3/5: Converting Y/N flags to boolean...")
    dfs = standardize_flags(dfs)
    print(f"  [OK] Flag columns converted")

    # Step 3.5: Fix churn date anomalies
    print("\nStep 3.5/5: Fixing churn date anomalies...")
    customers_before = dfs['customers'].copy()
    dfs, anomaly_count = fix_churn_date_anomalies(dfs)
    if anomaly_count > 0:
        print(f"  [OK] Fixed {anomaly_count} records with churn_date < signup_date")
    else:
        print(f"  [OK] No anomalies found")

    # Step 4: Validate
    print("\nStep 4/5: Running validation checks...")
    validate_data(dfs)

    print("[SUCCESS] Data loading and validation complete!")

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    dfs['customers'].to_csv(output_dir / "phase_2_customers_cleaned.csv", index=False)
    print(f"  [OK] phase_2_customers_cleaned.csv ({len(dfs['customers']):,} rows)")

    dfs['weekly_activity'].to_csv(output_dir / "phase_2_activity_cleaned.csv", index=False)
    print(f"  [OK] phase_2_activity_cleaned.csv ({len(dfs['weekly_activity']):,} rows)")

    dfs['support_tickets'].to_csv(output_dir / "phase_2_tickets_cleaned.csv", index=False)
    print(f"  [OK] phase_2_tickets_cleaned.csv ({len(dfs['support_tickets']):,} rows)")

    # Save removed data (churn date anomalies)
    if anomaly_count > 0:
        bad_mask = (customers_before['churned']) & (customers_before['churn_date'] < customers_before['signup_date'])
        anomalies = customers_before[bad_mask]
        anomalies.to_csv(output_dir / "phase_2_customers_churn_date_anomalies_REMOVED.csv", index=False)
        print(f"  [OK] phase_2_customers_churn_date_anomalies_REMOVED.csv ({len(anomalies)} rows)")

    print(f"\nYou can now proceed to Phase 3: Building customer base table.\n")
