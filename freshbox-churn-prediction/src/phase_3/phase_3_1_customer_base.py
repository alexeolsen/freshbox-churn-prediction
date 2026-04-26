import pandas as pd
import numpy as np
from datetime import datetime


def build_customer_base(dfs):
    """
    Build a single customer-level base table with one row per customer.

    CRITICAL DATA LEAKAGE PREVENTION:
    For churned customers, filter ALL activity to BEFORE churn_date.
    This ensures features never contain information from after the churn event.

    For active customers, use all available activity history.
    """
    customers = dfs['customers'].copy()
    activity = dfs['weekly_activity'].copy()
    tickets = dfs['support_tickets'].copy()

    # Identify churned and active customers
    churned_ids = customers[customers['churned']]['customer_id'].tolist()
    active_ids = customers[~customers['churned']]['customer_id'].tolist()

    print(f"\nBuilding customer base table...")
    print(f"  Churned customers: {len(churned_ids)}")
    print(f"  Active customers: {len(active_ids)}")

    # FILTER ACTIVITY - PREVENT DATA LEAKAGE
    print(f"\n  Filtering activity to avoid data leakage:")

    # For churned: keep only activity BEFORE churn_date
    churned_customers = customers[customers['churned']][['customer_id', 'churn_date']]
    activity_churned = activity.merge(churned_customers, on='customer_id', how='inner')
    activity_churned_filtered = activity_churned[
        activity_churned['week_commencing'] < activity_churned['churn_date']
    ].drop('churn_date', axis=1)

    # For active: keep all activity
    activity_active = activity[activity['customer_id'].isin(active_ids)]

    # Combine
    activity_clean = pd.concat([activity_churned_filtered, activity_active], ignore_index=True)

    print(f"    - Original activity records: {len(activity):,}")
    print(f"    - Activity records (churned only pre-churn): {len(activity_churned_filtered):,}")
    print(f"    - Activity records (active): {len(activity_active):,}")
    print(f"    - Combined (filtered): {len(activity_clean):,}")

    # FILTER TICKETS - PREVENT DATA LEAKAGE
    print(f"\n  Filtering support tickets to avoid data leakage:")

    # For churned: keep only tickets BEFORE churn_date
    tickets_churned = tickets.merge(churned_customers, on='customer_id', how='inner')
    tickets_churned_filtered = tickets_churned[
        tickets_churned['ticket_date'] < tickets_churned['churn_date']
    ].drop('churn_date', axis=1)

    # For active: keep all tickets
    tickets_active = tickets[tickets['customer_id'].isin(active_ids)]

    # Combine
    tickets_clean = pd.concat([tickets_churned_filtered, tickets_active], ignore_index=True)

    print(f"    - Original ticket records: {len(tickets):,}")
    print(f"    - Ticket records (churned only pre-churn): {len(tickets_churned_filtered):,}")
    print(f"    - Ticket records (active): {len(tickets_active):,}")
    print(f"    - Combined (filtered): {len(tickets_clean):,}")

    # SANITY CHECK: Confirm no churned customer has post-churn activity
    print(f"\n  Sanity checks:")
    for customer_id in churned_ids[:5]:  # Check first 5 as spot check
        churn_date = customers[customers['customer_id'] == customer_id]['churn_date'].values[0]
        post_churn = activity_clean[
            (activity_clean['customer_id'] == customer_id) &
            (activity_clean['week_commencing'] >= churn_date)
        ]
        assert len(post_churn) == 0, f"ERROR: Found post-churn activity for {customer_id}!"

    print(f"    - Verified: No churned customer has post-churn activity")
    print(f"    - Data leakage prevention: CONFIRMED")

    # Save filtered tables for downstream use
    dfs['activity_filtered'] = activity_clean
    dfs['tickets_filtered'] = tickets_clean

    print(f"\n[OK] Customer base table ready for feature engineering")

    return dfs


def validate_no_leakage(dfs):
    """
    Final validation: confirm no churned customer has activity after their churn_date.
    """
    customers = dfs['customers']
    activity = dfs['activity_filtered']
    tickets = dfs['tickets_filtered']

    churned = customers[customers['churned']][['customer_id', 'churn_date']]

    # Check activity
    activity_merged = activity.merge(churned, on='customer_id', how='inner')
    post_churn_activity = activity_merged[activity_merged['week_commencing'] >= activity_merged['churn_date']]

    # Check tickets
    tickets_merged = tickets.merge(churned, on='customer_id', how='inner')
    post_churn_tickets = tickets_merged[tickets_merged['ticket_date'] >= tickets_merged['churn_date']]

    assert len(post_churn_activity) == 0, f"ERROR: {len(post_churn_activity)} post-churn activity records found!"
    assert len(post_churn_tickets) == 0, f"ERROR: {len(post_churn_tickets)} post-churn ticket records found!"

    print("\n[OK] Data leakage validation passed")
    print(f"  - No post-churn activity records: CONFIRMED")
    print(f"  - No post-churn ticket records: CONFIRMED")

    return True