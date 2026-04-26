import pandas as pd
import numpy as np
from datetime import datetime


def engineer_tenure_features(customers, activity):
    """
    TENURE & LIFECYCLE FEATURES
    How long has the customer been with us, and what's their activity level?
    """
    features = customers[['customer_id']].copy()

    # Get dataset max date for tenure calculation
    max_date = activity['week_commencing'].max()

    # For churned customers: tenure = weeks from signup to churn
    # For active customers: tenure = weeks from signup to dataset end
    features['tenure_weeks'] = 0.0  # Float to handle decimal weeks
    for idx, row in customers.iterrows():
        if row['churned']:
            tenure = (row['churn_date'] - row['signup_date']).days / 7
        else:
            tenure = (max_date - row['signup_date']).days / 7
        features.loc[features['customer_id'] == row['customer_id'], 'tenure_weeks'] = tenure

    # Order completion metrics
    activity_by_customer = activity.groupby('customer_id').agg({
        'order_status': lambda x: (x == 'Ordered').sum(),  # total orders
    }).rename(columns={'order_status': 'total_orders'})

    activity_by_customer['total_skips'] = activity.groupby('customer_id')['order_status'].apply(
        lambda x: (x == 'Skipped').sum()
    )

    # Order completion rate
    activity_by_customer['order_completion_rate'] = (
        activity_by_customer['total_orders'] /
        (activity_by_customer['total_orders'] + activity_by_customer['total_skips'])
    )
    activity_by_customer['order_completion_rate'] = activity_by_customer['order_completion_rate'].fillna(0)

    features = features.merge(activity_by_customer, on='customer_id', how='left').fillna(0)

    return features


def engineer_engagement_features(activity):
    """
    ENGAGEMENT INTENSITY FEATURES
    Quality of engagement: ratings, customisation, recipes tried.
    """
    engagement = activity.groupby('customer_id').agg({
        'order_value_gbp': 'mean',
        'avg_recipe_rating': 'mean',
        'recipes_rated': 'sum',
        'recipes_in_box': 'sum',
        'menu_customised_flag': lambda x: (x == True).sum(),
    }).rename(columns={
        'order_value_gbp': 'avg_order_value',
        'avg_recipe_rating': 'avg_recipe_rating',
        'recipes_rated': 'lifetime_recipes_rated',
        'recipes_in_box': 'lifetime_recipes_received',
    })

    # Rating engagement rate
    engagement['rating_engagement_rate'] = (
        engagement['lifetime_recipes_rated'] / engagement['lifetime_recipes_received']
    )
    engagement['rating_engagement_rate'] = engagement['rating_engagement_rate'].fillna(0)

    # Customisation rate
    engagement['menu_customisation_rate'] = (
        engagement['menu_customised_flag'] / activity.groupby('customer_id').size()
    )

    engagement = engagement.drop('menu_customised_flag', axis=1)

    return engagement


def engineer_recency_momentum_features(activity, customers):
    """
    RECENCY & MOMENTUM FEATURES (strongest predictors!)
    Is engagement trending up or down? How recent is recent activity?

    Churn is rarely about absolute behaviour. It's about deterioration.
    A customer whose ratings dropped from 4.2 to 2.8 is in much more trouble
    than one steady at 3.0.
    """
    momentum = activity.groupby('customer_id').apply(
        lambda df: pd.Series({
            'total_weeks': len(df),
            'earliest_week': df['week_commencing'].min(),
            'latest_week': df['week_commencing'].max(),
        })
    ).reset_index()

    features = customers[['customer_id']].copy()

    for customer_id in features['customer_id']:
        cust_activity = activity[activity['customer_id'] == customer_id].sort_values('week_commencing')

        if len(cust_activity) < 8:
            # Not enough weeks for early/late split
            features.loc[features['customer_id'] == customer_id, 'early_avg_rating'] = np.nan
            features.loc[features['customer_id'] == customer_id, 'late_avg_rating'] = np.nan
            features.loc[features['customer_id'] == customer_id, 'rating_trend'] = np.nan
            features.loc[features['customer_id'] == customer_id, 'early_order_freq'] = 0
            features.loc[features['customer_id'] == customer_id, 'late_order_freq'] = 0
            features.loc[features['customer_id'] == customer_id, 'frequency_trend'] = 0
            continue

        # Split into first 4 weeks vs last 4 weeks
        early = cust_activity.head(4)
        late = cust_activity.tail(4)

        # Ratings
        early_rating = early['avg_recipe_rating'].mean()
        late_rating = late['avg_recipe_rating'].mean()

        features.loc[features['customer_id'] == customer_id, 'early_avg_rating'] = early_rating
        features.loc[features['customer_id'] == customer_id, 'late_avg_rating'] = late_rating
        features.loc[features['customer_id'] == customer_id, 'rating_trend'] = late_rating - early_rating

        # Order frequency (orders per week)
        early_freq = (early['order_status'] == 'Ordered').sum() / len(early)
        late_freq = (late['order_status'] == 'Ordered').sum() / len(late)

        features.loc[features['customer_id'] == customer_id, 'early_order_freq'] = early_freq
        features.loc[features['customer_id'] == customer_id, 'late_order_freq'] = late_freq
        features.loc[features['customer_id'] == customer_id, 'frequency_trend'] = late_freq - early_freq

    return features


def engineer_economic_features(customers, activity):
    """
    ECONOMIC SIGNALS
    Pricing, discounts, and revenue indicators.
    """
    econ = customers[['customer_id', 'initial_discount_pct', 'weekly_price_gbp']].copy()

    # Discount dependency: % of weeks with discount applied
    discount_rate = activity.groupby('customer_id')['discount_applied_flag'].apply(
        lambda x: (x == True).sum() / len(x)
    )
    econ['discount_dependency_rate'] = econ['customer_id'].map(discount_rate).fillna(0)

    # Average discount when applied
    activity_with_discount = activity[activity['discount_applied_flag'] == True]
    avg_discount = activity_with_discount.groupby('customer_id')['discount_pct'].mean()
    econ['avg_discount_pct'] = econ['customer_id'].map(avg_discount).fillna(0)

    return econ


def engineer_friction_features(customers, tickets):
    """
    FRICTION SIGNALS
    Support interactions, complaints, and resolution times.
    """
    friction = customers[['customer_id']].copy()

    # Total tickets
    ticket_counts = tickets.groupby('customer_id').size()
    friction['total_tickets'] = friction['customer_id'].map(ticket_counts).fillna(0)

    # Delivery-related tickets (category = 'Delivery')
    delivery_tickets = tickets[tickets['category'] == 'Delivery'].groupby('customer_id').size()
    friction['delivery_complaints'] = friction['customer_id'].map(delivery_tickets).fillna(0)

    # Average resolution time (days)
    avg_resolution = tickets.groupby('customer_id')['resolution_days'].mean()
    friction['avg_resolution_days'] = friction['customer_id'].map(avg_resolution).fillna(0)

    # Tickets per tenure month
    tenure_months = customers.set_index('customer_id').loc[friction['customer_id'], 'signup_date']
    tenure_months = (datetime.now() - pd.to_datetime(tenure_months)).dt.days / 30.44

    friction['tickets_per_tenure_month'] = friction['total_tickets'] / tenure_months.values
    friction['tickets_per_tenure_month'] = friction['tickets_per_tenure_month'].replace([np.inf, -np.inf], 0).fillna(0)

    return friction


def engineer_demographic_features(customers):
    """
    DEMOGRAPHIC FEATURES
    Already in customers table, just select them.
    """
    demo = customers[[
        'customer_id',
        'acquisition_channel',
        'referral_flag',
        'age_band',
        'household_size',
        'region',
        'dietary_preference',
        'plan_type',
        'meals_per_week'
    ]].copy()

    return demo


def assemble_feature_table(customers, activity, tickets):
    """
    Assemble all features into single customer-level table.
    One-hot encode categoricals.
    Handle missing values.
    """
    print("\nAssembling feature table...")

    # Build feature groups
    print("  Building tenure & lifecycle features...")
    tenure_feats = engineer_tenure_features(customers, activity)

    print("  Building engagement intensity features...")
    engagement_feats = engineer_engagement_features(activity)

    print("  Building recency & momentum features...")
    momentum_feats = engineer_recency_momentum_features(activity, customers)

    print("  Building economic signal features...")
    econ_feats = engineer_economic_features(customers, activity)

    print("  Building friction signal features...")
    friction_feats = engineer_friction_features(customers, tickets)

    print("  Building demographic features...")
    demo_feats = engineer_demographic_features(customers)

    # Assemble
    features = tenure_feats
    features = features.merge(engagement_feats, on='customer_id', how='left')
    features = features.merge(momentum_feats, on='customer_id', how='left')
    features = features.merge(econ_feats, on='customer_id', how='left')
    features = features.merge(friction_feats, on='customer_id', how='left')
    features = features.merge(demo_feats, on='customer_id', how='left')

    # Add target variable
    features['churned'] = features['customer_id'].map(
        customers.set_index('customer_id')['churned']
    ).astype(int)

    # One-hot encode categoricals
    print("\n  One-hot encoding categorical features...")
    categorical_cols = ['acquisition_channel', 'age_band', 'region', 'dietary_preference', 'plan_type']
    for col in categorical_cols:
        if col in features.columns:
            dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
            features = pd.concat([features, dummies], axis=1)
            features = features.drop(col, axis=1)

    # Handle missing values
    print("  Handling missing values...")
    # Zero-fill engagement counts (no orders = 0)
    engagement_cols = ['total_orders', 'total_skips', 'lifetime_recipes_rated', 'lifetime_recipes_received']
    for col in engagement_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Median-fill ratings (when customer never rated)
    rating_cols = ['early_avg_rating', 'late_avg_rating', 'avg_recipe_rating', 'rating_trend']
    for col in rating_cols:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())

    # Drop reference flag (already one-hot encoded)
    if 'referral_flag' in features.columns:
        features = features.drop('referral_flag', axis=1)

    print(f"\n  Feature table assembled:")
    print(f"    - Customers: {len(features)}")
    print(f"    - Features: {len(features.columns) - 2}")  # -2 for customer_id and churned
    print(f"    - Missing values: {features.isna().sum().sum()}")

    return features
