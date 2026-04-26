import pandas as pd
import numpy as np

# Load and convert dates
df = pd.read_csv('data/raw/freshbox_dim_customers.csv')
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['churn_date'] = pd.to_datetime(df['churn_date'])
df['churned'] = (df['churned'] == 'Y').astype(bool)

# Find the problem
churned = df[df['churned']]
bad_dates = churned[churned['churn_date'] < churned['signup_date']]

print(f"Total customers: {len(df)}")
print(f"Total churned customers: {len(churned)}")
print(f"Customers with churn_date < signup_date: {len(bad_dates)}")
print(f"Percentage: {len(bad_dates) / len(churned) * 100:.1f}%")

print(f"\nFirst 10 problematic records:")
print(bad_dates[['customer_id', 'signup_date', 'churn_date']].head(10).to_string())

print(f"\nDate difference statistics (for bad records):")
bad_dates_copy = bad_dates.copy()
bad_dates_copy['days_diff'] = (bad_dates_copy['churn_date'] - bad_dates_copy['signup_date']).dt.days
print(f"  Min days diff: {bad_dates_copy['days_diff'].min()}")
print(f"  Max days diff: {bad_dates_copy['days_diff'].max()}")
print(f"  Mean days diff: {bad_dates_copy['days_diff'].mean():.1f}")

print(f"\nAnalysis:")
print(f"  These records likely represent a data quality issue.")
print(f"  For modeling, we should EXCLUDE them or handle them specially.")
print(f"  Option 1: Drop these {len(bad_dates)} records (affects {len(bad_dates)/len(df)*100:.1f}% of data)")
print(f"  Option 2: Mark them as data anomalies and investigate separately")
