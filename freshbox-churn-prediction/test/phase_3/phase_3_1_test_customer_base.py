#!/usr/bin/env python
"""
Phase 3: Build customer-level base table with data leakage prevention.
"""

from src.shared.phase_0_1_data_prep import load_raw_data, standardize_dates, standardize_flags, fix_churn_date_anomalies
from src.phase_3.phase_3_1_customer_base import build_customer_base, validate_no_leakage
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 3: Building customer base table (with leakage prevention)\n")

    # Step 1: Load and prepare data (from Phase 2)
    print("Step 1/3: Loading and preparing raw data...")
    dfs = load_raw_data()
    dfs = standardize_dates(dfs)
    dfs = standardize_flags(dfs)
    dfs, _ = fix_churn_date_anomalies(dfs)
    print("  [OK] Data prepared")

    # Capture data before leakage prevention (for CSV dumps)
    activity_before = dfs['weekly_activity'].copy()
    tickets_before = dfs['support_tickets'].copy()

    # Step 2: Build customer base with leakage prevention
    print("\nStep 2/3: Building customer-level base table...")
    dfs = build_customer_base(dfs)

    # Step 3: Validate no data leakage
    print("\nStep 3/3: Validating data leakage prevention...")
    validate_no_leakage(dfs)

    print("\n[SUCCESS] Phase 3 complete!")

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered (kept) data
    dfs['activity_filtered'].to_csv(output_dir / "phase_3_activity_post_churn_prevention.csv", index=False)
    print(f"  [OK] phase_3_activity_post_churn_prevention.csv ({len(dfs['activity_filtered']):,} rows)")

    dfs['tickets_filtered'].to_csv(output_dir / "phase_3_tickets_post_churn_prevention.csv", index=False)
    print(f"  [OK] phase_3_tickets_post_churn_prevention.csv ({len(dfs['tickets_filtered']):,} rows)")

    # Save removed data (post-churn records)
    customers = dfs['customers']
    churned_customers = customers[customers['churned']][['customer_id', 'churn_date']]

    # Post-churn activity removed
    activity_churned = activity_before.merge(churned_customers, on='customer_id', how='inner')
    activity_removed = activity_churned[
        activity_churned['week_commencing'] >= activity_churned['churn_date']
    ].drop('churn_date', axis=1)
    if len(activity_removed) > 0:
        activity_removed.to_csv(output_dir / "phase_3_activity_post_churn_REMOVED.csv", index=False)
        print(f"  [OK] phase_3_activity_post_churn_REMOVED.csv ({len(activity_removed)} rows REMOVED)")

    # Post-churn tickets removed
    tickets_churned = tickets_before.merge(churned_customers, on='customer_id', how='inner')
    tickets_removed = tickets_churned[
        tickets_churned['ticket_date'] >= tickets_churned['churn_date']
    ].drop('churn_date', axis=1)
    if len(tickets_removed) > 0:
        tickets_removed.to_csv(output_dir / "phase_3_tickets_post_churn_REMOVED.csv", index=False)
        print(f"  [OK] phase_3_tickets_post_churn_REMOVED.csv ({len(tickets_removed)} rows REMOVED)")

    print(f"\nNext: Phase 4 - Feature Engineering (will use activity_filtered and tickets_filtered)\n")
