import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data():
    """
    Load all raw CSV files from data/raw/ directory.
    Returns a dictionary of DataFrames.
    """
    data_path = Path(__file__).parent.parent.parent / "data" / "raw"

    dfs = {
        'customers': pd.read_csv(data_path / 'freshbox_dim_customers.csv'),
        'weekly_activity': pd.read_csv(data_path / 'freshbox_fact_weekly_activity.csv'),
        'support_tickets': pd.read_csv(data_path / 'freshbox_fact_support_tickets.csv'),
        'calendar': pd.read_csv(data_path / 'freshbox_445_calendar.csv'),
    }

    return dfs


def standardize_dates(dfs):
    """
    Convert all date columns from strings to datetime format.
    This allows pandas to understand and work with dates properly.
    """
    # Customers: signup and churn dates
    dfs['customers']['signup_date'] = pd.to_datetime(dfs['customers']['signup_date'])
    dfs['customers']['churn_date'] = pd.to_datetime(dfs['customers']['churn_date'])

    # Weekly activity: week starting date
    dfs['weekly_activity']['week_commencing'] = pd.to_datetime(dfs['weekly_activity']['week_commencing'])

    # Support tickets: ticket creation and resolution dates
    dfs['support_tickets']['ticket_date'] = pd.to_datetime(dfs['support_tickets']['ticket_date'])
    dfs['support_tickets']['resolution_date'] = pd.to_datetime(dfs['support_tickets']['resolution_date'])

    # Calendar: week dates
    dfs['calendar']['week_commencing'] = pd.to_datetime(dfs['calendar']['week_commencing'])
    dfs['calendar']['week_ending'] = pd.to_datetime(dfs['calendar']['week_ending'])

    return dfs


def standardize_flags(dfs):
    """
    Convert Y/N string flags to boolean True/False.
    Makes it easier to filter and aggregate later.
    """
    # Customer flags
    bool_cols_customers = ['churned', 'referral_flag', 'discount_applied_flag', 'menu_customised_flag']
    for col in bool_cols_customers:
        if col in dfs['customers'].columns:
            dfs['customers'][col] = (dfs['customers'][col] == 'Y').astype(bool)

    return dfs


def fix_churn_date_anomalies(dfs):
    """
    Handle the 12 records where churn_date < signup_date (data quality issue).
    Set churn_date = signup_date for these anomalous records.
    This assumes they churned on the same day they signed up.
    """
    churned = dfs['customers'][dfs['customers']['churned']]
    bad_mask = churned['churn_date'] < churned['signup_date']
    bad_count = bad_mask.sum()

    if bad_count > 0:
        dfs['customers'].loc[dfs['customers']['churned'] & (dfs['customers']['churn_date'] < dfs['customers']['signup_date']), 'churn_date'] = \
            dfs['customers'].loc[dfs['customers']['churned'] & (dfs['customers']['churn_date'] < dfs['customers']['signup_date']), 'signup_date']

    return dfs, bad_count


def inspect_columns():
    """Quick inspection to see actual column names in each table."""
    dfs = load_raw_data()
    print("\nCOLUMN NAMES IN EACH TABLE:")
    for table, df in dfs.items():
        print(f"\n{table.upper()}:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")


def validate_data(dfs):
    """
    Run 5 critical data quality checks:
    1. No duplicate customers
    2. No duplicate tickets
    3. All activity/ticket customer IDs exist in customer table (referential integrity)
    4. Churn dates are after signup dates (logic check)
    5. Churn rate is in expected range (sanity check)
    """
    print("\n" + "="*70)
    print("DATA VALIDATION REPORT")
    print("="*70)

    # CHECK 1: Duplicate customers
    dup_customers = dfs['customers']['customer_id'].duplicated().sum()
    status_1 = "[PASS]" if dup_customers == 0 else "[FAIL]"
    print(f"\n{status_1} | No duplicate customers: {dup_customers} duplicates found")
    assert dup_customers == 0, "ERROR: Found duplicate customer IDs!"

    # CHECK 2: Duplicate tickets
    dup_tickets = dfs['support_tickets']['ticket_id'].duplicated().sum()
    status_2 = "[PASS]" if dup_tickets == 0 else "[FAIL]"
    print(f"{status_2} | No duplicate tickets: {dup_tickets} duplicates found")
    assert dup_tickets == 0, "ERROR: Found duplicate ticket IDs!"

    # CHECK 3: Referential integrity
    activity_customers = set(dfs['weekly_activity']['customer_id'].unique())
    ticket_customers = set(dfs['support_tickets']['customer_id'].unique())

    activity_valid = activity_customers.issubset(set(dfs['customers']['customer_id']))
    tickets_valid = ticket_customers.issubset(set(dfs['customers']['customer_id']))

    status_3a = "[PASS]" if activity_valid else "[FAIL]"
    status_3b = "[PASS]" if tickets_valid else "[FAIL]"
    print(f"{status_3a} | All weekly activity customer IDs exist in customer table")
    print(f"{status_3b} | All support ticket customer IDs exist in customer table")
    assert activity_valid and tickets_valid, "ERROR: Referential integrity violation!"

    # CHECK 4: Churn date logic (churn_date >= signup_date) - after fix
    churned = dfs['customers'][dfs['customers']['churned']]
    valid_churn_dates = (churned['churn_date'] >= churned['signup_date']).all()
    status_4 = "[PASS]" if valid_churn_dates else "[FAIL]"
    print(f"{status_4} | Churn dates >= signup dates (for churned customers)")
    assert valid_churn_dates, "ERROR: Found churn_date < signup_date after fix!"

    # CHECK 5: Churn rate sanity check (should be 65-70% per brief)
    churn_rate = dfs['customers']['churned'].mean()
    status_5 = "[PASS]" if 0.65 <= churn_rate <= 0.70 else "[WARNING]"
    print(f"{status_5} | Churn rate in expected range: {churn_rate:.1%} (expected 65-70%)")

    # SUMMARY STATISTICS
    print(f"\n" + "-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70)

    print(f"\nCUSTOMERS:")
    print(f"  • Total customers: {len(dfs['customers']):,}")
    print(f"  • Churned: {dfs['customers']['churned'].sum():,} ({dfs['customers']['churned'].mean():.1%})")
    print(f"  • Active: {(~dfs['customers']['churned']).sum():,} ({(~dfs['customers']['churned']).mean():.1%})")

    print(f"\nDATE RANGES:")
    print(f"  • Customer signup: {dfs['customers']['signup_date'].min().date()} to {dfs['customers']['signup_date'].max().date()}")
    print(f"  • Weekly activity: {dfs['weekly_activity']['week_commencing'].min().date()} to {dfs['weekly_activity']['week_commencing'].max().date()}")
    print(f"  • Support tickets: {dfs['support_tickets']['ticket_date'].min().date()} to {dfs['support_tickets']['ticket_date'].max().date()}")

    print(f"\nACTIVITY & SUPPORT:")
    print(f"  • Weekly activity records: {len(dfs['weekly_activity']):,}")
    print(f"  • Support tickets: {len(dfs['support_tickets']):,}")
    print(f"  • Customers with tickets: {dfs['support_tickets']['customer_id'].nunique():,}")

    print(f"\n" + "="*70)
    print("[SUCCESS] ALL VALIDATION CHECKS PASSED!")
    print("="*70 + "\n")

    return True
