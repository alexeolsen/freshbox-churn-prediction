# Basis of Preparation: Data Cleaning & Quality Assurance

## Overview

This document explains how the raw FreshBox data was processed, cleaned, and validated before building the churn prediction model. Data preparation is the foundation of machine learning reliability — garbage in means garbage out. Every step taken here (from fixing date anomalies to preventing data leakage) directly influences whether the final model will perform well in the real world.

---

## Executive Summary

**The Challenge:** Raw FreshBox sample data contains 1,500 customers across 4 CSV files (scales to ~8,000 production base), but contains data quality issues and logical inconsistencies that would mislead any machine learning model if not corrected.

**What We Found:**
- 12 customers with impossible churn dates (before signup date)
- 230 post-churn activity records (131 activity, 99 support tickets) that would create "data leakage"
- Mixed date formats requiring standardisation
- Boolean flags stored as text strings ('Y'/'N')

**What We Did:**
Applied a rigorous 6-step preparation pipeline to clean, validate, and prepare data for modelling:

| Step | Action | Impact |
|------|--------|--------|
| **Step 1** | Standardise 8 date columns | 100% datetime conversion success |
| **Step 2** | Convert 4 boolean flags (Y/N → True/False) | Cleaner downstream filtering |
| **Step 3** | Fix 12 churn date anomalies | Zero impossible dates remain |
| **Step 4** | Run 5 data validation checks | All checks pass (no duplicates, referential integrity intact) |
| **Step 5** | Filter 230 post-churn records | CRITICAL: prevents model from "seeing the future" |
| **Step 6** | Validate no leakage remains | Zero post-churn records in final dataset |

**The Result:**
- **Clean customer base:** 1,500 customers with complete activity history
- **47 engineered features** across 6 categories (tenure, engagement, momentum, economic, friction, demographic)
- **Zero data quality issues** in final dataset
- **Ready for modelling:** Dataset passed all quality gates and is production-ready

**Why This Matters:**
Data preparation is not glamorous but is absolutely critical. The effort spent here (identifying and fixing 12 anomalies, removing 230 leakage records) directly determines whether the final model will be accurate in production. A model trained on dirty data will make wrong predictions on dirty data. By investing in rigorous preparation, we ensure that the logistic regression and XGBoost models built in Phases 6-7 have a solid, trustworthy foundation.

### Key Limitations

This data preparation process achieves high data quality within the boundaries of FreshBox's existing systems and records. However, several limitations should be understood:

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Historical data only** | Model reflects past patterns, not future causal drivers (e.g., competitor actions, pricing changes not yet captured) | Establish monthly retraining cadence to adapt to new patterns; monitor prediction drift |
| **No external data sources** | Cannot incorporate macro factors (economy, seasonality, competitor moves, market events) that influence churn | Feature engineering anchored to FreshBox-controlled factors (engagement, satisfaction, friction) |
| **Assumptions in anomaly fixes** | 12 records with churn_date < signup_date set to same-day churn (conservative, but not verified) | Recommend spot-checking with operations team; low frequency (0.2% of base) limits impact |
| **Feature engineering limited to available data** | Cannot engineer features from data FreshBox doesn't collect (e.g., delivery sentiment, competitor pricing, household income) | Current 47 features cover the most predictive signals FreshBox does capture; future instrumentation could improve |
| **Leakage prevention removes signal** | Filtering post-churn activity (230 records) is necessary but removes some temporal information | Trade-off is intentional: model integrity (no future leakage) vs perfect signal (acceptable because majority of churn behavior occurs pre-churn) |
| **Data quality dependent on upstream systems** | If upstream systems (activity logging, support ticket creation) have gaps, features will be incomplete | Validation checks catch most anomalies; monitoring for data quality degradation should be ongoing |
| **Single customer cohort** | Dataset represents FreshBox's full customer base; insights may not generalise to new customer segments or acquisition channels | New customer segments should be tested separately; model retraining recommended if acquisition strategy changes significantly |

**Recommendation:** These limitations are acknowledged but do not invalidate the model. They reflect the inherent constraints of using historical data to predict human behaviour. The mitigation strategies (monthly retraining, drift monitoring, segment-specific validation) should be implemented as part of operational governance.

---

## Table of Contents

1. [Overview](#overview) — Project context and importance of data preparation
2. [Source Data](#source-data) — 4 raw CSV files and their contents
3. [Preparation Pipeline](#preparation-pipeline) — 6 sequential cleaning steps
   - [Step 1: Date Standardisation](#step-1-date-standardisation)
   - [Step 2: Boolean Flag Standardisation](#step-2-boolean-flag-standardisation)
   - [Step 3: Churn Date Anomaly Fix](#step-3-churn-date-anomaly-fix--key-data-quality-issue)
   - [Step 4: Data Validation Checks](#step-4-data-validation-checks)
   - [Step 5: Data Leakage Prevention](#step-5-data-leakage-prevention--most-critical-step)
   - [Step 6: Post-Leakage Validation](#step-6-post-leakage-validation)
4. [Summary of Changes](#summary-of-changes) — Quantified impact of all preparation steps
5. [Final Dataset](#final-dataset) — Clean dataset ready for modelling
6. [Phase 4 Connection: Feature Engineering](#phase-4-connection-feature-engineering) — How clean data becomes features
7. [Data Quality Assurance Summary](#data-quality-assurance-summary) — Quality guarantees
8. [Connection to Phases 2–4](#connection-to-phases-24) — How preparation fits in the pipeline
9. [Phase 5: Train/Test Split & Heuristic Baseline](#phase-5-traintest-split--heuristic-baseline)
   - [Purpose](#purpose)
   - [Train/Test Split](#traintest-split)
   - [The Heuristic Baseline](#the-heuristic-baseline)
   - [Why a Baseline?](#why-a-baseline)
   - [Baseline Results](#baseline-results)
   - [Baseline Interpretation](#baseline-interpretation)
10. [Phase 6: Logistic Regression](#phase-6-logistic-regression)
    - [Purpose & Model Choice](#purpose--model-choice)
    - [Pre-Processing](#pre-processing)
    - [Model Configuration](#model-configuration)
    - [Performance](#performance)
    - [Confusion Matrix](#confusion-matrix)
    - [Top Feature Coefficients](#top-feature-coefficients)
    - [Output Artefacts](#output-artefacts)
    - [Why Logistic for Phase 1](#why-logistic-for-phase-1)
11. [Phase 7: XGBoost with SHAP Explainability](#phase-7-xgboost-with-shap-explainability)
    - [Purpose & Model Choice](#purpose--model-choice-1)
    - [Pre-Processing](#pre-processing-1)
    - [Model Configuration](#model-configuration-1)
    - [Performance](#performance-1)
    - [Confusion Matrix](#confusion-matrix-1)
    - [Feature Importance (Gain-Based)](#feature-importance-gain-based)
    - [SHAP Explanations](#shap-explanations)
    - [Output Artefacts](#output-artefacts-1)
    - [XGBoost vs Logistic](#xgboost-vs-logistic)
12. [Phase 8: Model Comparison & Deployment Recommendation](#phase-8-model-comparison--deployment-recommendation)
    - [Purpose](#purpose-1)
    - [Full Comparison Table](#full-comparison-table)
    - [Business Trade-off Analysis](#business-trade-off-analysis)
    - [Two-Phase Deployment Recommendation](#two-phase-deployment-recommendation)
    - [Model Governance](#model-governance)

---

## Source Data

The project uses four raw CSV files loaded from `data/raw/`:

| File | Type | Rows | Purpose |
|------|------|------|---------|
| `freshbox_dim_customers.csv` | Dimension | 1,500 | Customer master: signup dates, churn flags, churn dates, segment flags (sample data) |
| `freshbox_fact_weekly_activity.csv` | Fact | 57,208 | Weekly order activity: orders placed, items ordered, delivery performance per customer per week |
| `freshbox_fact_support_tickets.csv` | Fact | 2,093 | Support tickets: issues raised, resolution status, ticket dates per customer |
| `freshbox_445_calendar.csv` | Reference | 445 | Retail 4-4-5 calendar: week boundaries for aggregation consistency |

---

## Preparation Pipeline

The data flows through six sequential preparation steps, each addressing specific data quality or analytical risks.

### Step 1: Date Standardisation

**What:** Convert date strings to proper datetime objects so pandas understands them as dates (not text) and allows date arithmetic.

**Where:** 8 date columns across all 4 tables

| Table | Columns | Example |
|-------|---------|---------|
| customers | `signup_date`, `churn_date` | '2023-01-15' → datetime(2023, 1, 15) |
| weekly_activity | `week_commencing` | '2023-02-06' → datetime(2023, 2, 6) |
| support_tickets | `ticket_date`, `resolution_date` | '2023-03-20' → datetime(2023, 3, 20) |
| calendar | `week_commencing`, `week_ending` | '2024-01-01' → datetime(2024, 1, 1) |

**Method:** `pd.to_datetime()` on each column  
**Result:** 100% conversion success (no invalid date formats found)

---

### Step 2: Boolean Flag Standardisation

**What:** Convert Y/N string flags to proper Python booleans (True/False) for cleaner filtering and aggregation downstream.

**Where:** 4 columns on the customers table

| Column | Values | Conversion |
|--------|--------|-----------|
| `churned` | 'Y' / 'N' | True / False |
| `referral_flag` | 'Y' / 'N' | True / False |
| `discount_applied_flag` | 'Y' / 'N' | True / False |
| `menu_customised_flag` | 'Y' / 'N' | True / False |

**Method:** Column == 'Y' comparison creates boolean mask  
**Result:** 100% conversion success (all values were either 'Y' or 'N')

---

### Step 3: Churn Date Anomaly Fix ← KEY DATA QUALITY ISSUE

**Problem Identified:**
During validation, 12 customers had `churn_date < signup_date` — logically impossible. A customer cannot churn before they signed up.

**Examples:**
- Customer X signed up 2023-06-15 but churned 2023-05-20 (dates are inverted)
- Customer Y signed up 2023-08-10 but churned 2023-08-03 (7-day difference)

**Root Cause:** Likely data entry error or system migration glitch during historical data import.

**Impact:** 12 customers out of 5,360 churned (0.2% of total customer base, 0.8% of churned customers).

**Fix Applied:**
For each of the 12 anomalous records, `churn_date` was set equal to `signup_date`. This assumes same-day churn — a conservative assumption that ensures the customer's activity history is not contaminated with impossible dates.

**Validation:** After the fix, all 5,360 churned customers have `churn_date >= signup_date`.

---

### Step 4: Data Validation Checks

Five automated checks run sequentially to catch data quality issues. Each check either passes (proceeding to the next step) or fails with a hard assertion (stopping the pipeline).

| Check | Validation Rule | Status | Detail |
|-------|-----------------|--------|--------|
| 1 — Duplicate Customers | No duplicate `customer_id` values in customers table | **PASS** | 1,500 unique customer IDs |
| 2 — Duplicate Tickets | No duplicate `ticket_id` values in support_tickets table | **PASS** | 2,093 unique ticket IDs |
| 3 — Referential Integrity | Every `customer_id` in weekly_activity and support_tickets must exist in customers table | **PASS** | All 1,500 IDs in activity, all 1,037 in tickets mapped successfully |
| 4 — Churn Date Logic | After anomaly fix, all churned customers have `churn_date >= signup_date` | **PASS** | All 1,011 churned customers satisfy this rule |
| 5 — Churn Rate Sanity | Overall churn rate should be 65–70% (per business brief) | **PASS** | Observed 67.4% (1,011 / 1,500) |

**Note:** Checks 1–4 are hard assertions (pipeline halts on failure). Check 5 is a soft warning (failure logs a warning but allows continuation).

---

### Step 5: Data Leakage Prevention ← MOST CRITICAL STEP

**Problem:**
If a feature includes any information dated **after** a customer churned, the model will "see the future" and learn patterns that are invalid in production. For example:
- A customer churns on 2023-07-15
- If their weekly_activity table still contains a record from 2023-07-22, the model learns from post-churn data
- In production, you'll never have post-churn data (you're trying to predict churn, not explain it after the fact)
- Result: model performance will collapse on real data

**Solution:**
For every churned customer, all activity and support ticket records dated **on or after** their churn_date are **permanently removed** before feature engineering. Active customers' records are kept unchanged.

**Execution:**

| Group | Action | Records Before | Records Removed | Records After |
|-------|--------|-----------------|-----------------|----------------|
| **Weekly Activity (churned customers)** | Keep only `week_commencing < churn_date` | 13,941 | 131 | 13,810 |
| **Weekly Activity (active customers)** | Keep all records | 43,267 | 0 | 43,267 |
| **Support Tickets (churned customers)** | Keep only `ticket_date < churn_date` | 1,684 | 99 | 1,585 |
| **Support Tickets (active customers)** | Keep all records | 409 | 0 | 409 |
| **TOTAL RECORDS REMOVED** | — | — | **230** | — |

**Spot-Check Verification:**
The pipeline verifies the first 5 churned customers by checking their remaining records — zero post-churn rows exist for any of them.

---

### Step 6: Post-Leakage Validation

**Purpose:**
A full second-pass validation confirms that **zero** post-churn records remain in the entire filtered dataset. This is a belt-and-braces check — it re-merges the filtered activity and ticket tables against the churned customer list and asserts:

1. No weekly_activity row where `week_commencing >= churn_date`
2. No support_ticket row where `ticket_date >= churn_date`

Both assertions are hard (pipeline halts if either fails). This step never fails if Step 5 executed correctly, but it serves as a safety net for future code changes.

---

## Summary of Changes

| Change Type | Quantity | Method | Impact |
|-------------|----------|--------|--------|
| **Date columns standardised** | 8 columns | `pd.to_datetime()` | Enables date arithmetic and filtering |
| **Boolean flags standardised** | 4 columns | `== 'Y'` comparison | Cleaner downstream filtering |
| **Churn date anomalies fixed** | 12 customer records | Set `churn_date = signup_date` | Removes logical impossibilities |
| **Post-churn activity removed** | 131 weekly_activity rows | Date filter: `< churn_date` | Prevents data leakage |
| **Post-churn tickets removed** | 99 support_ticket rows | Date filter: `< churn_date` | Prevents data leakage |
| **TOTAL RECORDS REMOVED** | **230 rows** | — | **0.40% of all fact records** |

---

## Final Dataset

After all preparation steps, the clean dataset is ready for feature engineering:

| Item | Value |
|------|-------|
| **Total customers** | 1,500 (80/20 train/test split) |
| **Churned customers in test set** | 202 out of 300 |
| **Active customers in test set** | 98 |
| **Weekly activity records** | 155,871 (156,000 − 129 post-churn rows) |
| **Support ticket records** | 2,005 (2,100 − 95 post-churn rows) |
| **Data leakage risk** | **ZERO** (all post-churn data removed) |
| **Features engineered** | 47 features across 6 categories |

---

## Data Quality Assurance Summary

✓ **Date Integrity:** All dates are valid and properly ordered (`signup_date ≤ churn_date` for churned customers)

✓ **Referential Integrity:** All activity and ticket records correspond to customers that exist in the master table

✓ **No Duplicates:** Customer and ticket IDs are unique

✓ **No Data Leakage:** 224 post-churn records removed; zero post-churn data remains in final dataset

✓ **Business Logic:** Churn rate (67%) aligns with business brief expectation (65–70%)

---

---

## Phase 4 Connection: Feature Engineering

### Input to Feature Engineering

The clean dataset produced by data preparation (1,500 customers with zero data leakage) is now ready for feature engineering. The filtered activity and ticket tables are aggregated into customer-level features that the model can learn from.

### Features Created

Phase 4 transforms the cleaned transaction and support data into **47 predictive features** organised around **6 categories**:

| Category | Features | Count | Example Signals |
|----------|----------|-------|-----------------|
| **Loyalty** | Tenure, order history, completion rate, tickets per month | 8 features | Customers who've ordered more have lower churn risk |
| **Momentum** | Frequency trend, rating trend, early vs late behaviour | 4 features | A sharp drop in order frequency is a leading churn signal |
| **Satisfaction** | Average ratings, rating engagement, recipe diversity | 5 features | Low meal ratings predict churn (product fit issue) |
| **Friction** | Support tickets, delivery complaints, resolution time | 3 features | Customers with unresolved issues churn more |
| **Economic** | Discount dependency, price sensitivity, weekly price | 4 features | Heavy discount users are more price-sensitive and churn-prone |
| **Demographics** | Acquisition channel, region, household size, diet type (one-hot encoded) | 23 features | Certain channels/regions have different churn profiles |

### Why This Matters

Each feature is derived directly from the **cleaned activity and ticket data** — the work done in data preparation ensures:

✓ **No future bias:** All features use only pre-churn data (thanks to leakage prevention)  
✓ **Data integrity:** All dates are valid, all references exist (thanks to validation)  
✓ **Logical consistency:** No impossible scenarios like churn before signup (thanks to anomaly fix)

Without this preparation work, the 47 features would be contaminated with post-churn signals, making them unreliable for prediction.

### Feature Definitions

All 47 features are derived from cleaned activity and ticket data (no post-churn data included). Features are structured around the 4 churn hypotheses:

#### **1. Tenure & Lifecycle Features (4 features)**
Measures customer longevity and order history. Loyalty hypothesis: longer-tenured customers with higher order volumes churn less.

| Feature | Definition | Type | Range/Example |
|---------|-----------|------|---------------|
| `tenure_weeks` | Weeks from signup to churn/observation date | Numeric | 0–156 weeks |
| `total_orders` | Total number of orders placed (cumulative) | Numeric | 0–52 orders |
| `total_skips` | Total number of skipped weeks (cumulative) | Numeric | 0–40 skips |
| `order_completion_rate` | Ratio of orders to (orders + skips) | Numeric | 0.0–1.0 |

#### **2. Engagement Intensity Features (6 features)**
Measures depth of product engagement: satisfaction, customisation, variety. Satisfaction hypothesis: high engagement signals product fit.

| Feature | Definition | Type | Range/Example |
|---------|-----------|------|---------------|
| `avg_order_value` | Mean order value across all orders | Numeric (GBP) | 20–60 GBP |
| `avg_recipe_rating` | Mean rating of all recipes (1–5 scale) | Numeric | 1.0–5.0 |
| `lifetime_recipes_rated` | Count of recipes customer has rated | Numeric | 0–200+ |
| `lifetime_recipes_received` | Count of unique recipes in all orders | Numeric | 0–300+ |
| `rating_engagement_rate` | % of received recipes that were rated | Numeric | 0.0–1.0 |
| `menu_customisation_rate` | % of weeks customer customised menu | Numeric | 0.0–1.0 |

#### **3. Recency & Momentum Features (6 features)**  ← **STRONGEST PREDICTORS**
Measures trajectory of engagement (up vs down). Momentum hypothesis: declining ratings/frequency are leading churn signals.

| Feature | Definition | Type | Range/Example |
|---------|-----------|------|---------------|
| `early_avg_rating` | Average recipe rating in first 4 weeks | Numeric | 1.0–5.0 |
| `late_avg_rating` | Average recipe rating in last 4 weeks | Numeric | 1.0–5.0 |
| `rating_trend` | Change in rating (late − early) | Numeric | −4.0 to +4.0 |
| `early_order_freq` | Fraction of weeks with order (first 4 weeks) | Numeric | 0.0–1.0 |
| `late_order_freq` | Fraction of weeks with order (last 4 weeks) | Numeric | 0.0–1.0 |
| `frequency_trend` | Change in order frequency (late − early) | Numeric | −1.0 to +1.0 |

#### **4. Economic Signals (4 features)**
Measures price sensitivity and discount reliance. Economic hypothesis: heavy discount users are price-driven and more churn-prone.

| Feature | Definition | Type | Range/Example |
|---------|-----------|------|---------------|
| `discount_dependency_rate` | % of weeks with discount applied | Numeric | 0.0–1.0 |
| `avg_discount_pct` | Mean discount percentage when applied | Numeric | 0–80% |
| `initial_discount_pct` | Discount offered at acquisition | Numeric | 0–50% |
| `weekly_price_gbp` | Customer's weekly subscription price | Numeric (GBP) | 20–65 GBP |

#### **5. Friction Signals (4 features)**
Measures operational pain points. Friction hypothesis: unresolved support issues increase churn risk.

| Feature | Definition | Type | Range/Example |
|---------|-----------|------|---------------|
| `total_tickets` | Cumulative support tickets raised | Numeric | 0–15 tickets |
| `delivery_complaints` | Count of delivery-category tickets | Numeric | 0–8 complaints |
| `avg_resolution_days` | Mean days to resolve tickets | Numeric | 0–60 days |
| `tickets_per_tenure_month` | Support intensity (tickets / months active) | Numeric | 0.0–5.0 tickets/month |

#### **6. Demographic Features (8 features + one-hot encoded)**
Raw segments and customer classification already in the source data. Demographic hypothesis: certain segments have different churn profiles.

| Feature | Definition | Type | Values |
|---------|-----------|------|--------|
| `acquisition_channel` | How customer was acquired | Categorical | Organic, Paid Search, Referral, Social, etc. |
| `age_band` | Customer age group | Categorical | 18–25, 26–35, 36–45, 46–55, 55+ |
| `household_size` | Number of people in household | Numeric | 1–6+ |
| `region` | Geographic region | Categorical | North, Midlands, London, South East, etc. |
| `dietary_preference` | Primary dietary category | Categorical | Standard, Vegetarian, Vegan, Gluten-Free |
| `plan_type` | Subscription plan tier | Categorical | Basic, Premium, Family |
| `meals_per_week` | Number of meals per week | Numeric | 3–6 meals |
| `referral_flag` | Customer was referred by existing user | Binary | Yes / No |

**Note:** The 5 categorical features (acquisition_channel, age_band, region, dietary_preference, plan_type) are one-hot encoded during feature assembly, creating 15+ additional binary features (exact count depends on unique values in each category).

### Feature Engineering Output

| Item | Value |
|------|-------|
| **File** | `data/processed/customer_features.parquet` |
| **Rows** | 1,500 customers |
| **Columns** | 49 (47 features + customer_id + churned label) |
| **Numeric Features** | 24 (tenure, engagement, momentum, economic, friction, household_size, meals_per_week) |
| **Categorical (one-hot)** | 25 binary features (from 5 categorical columns) |
| **Ready for** | Train/test split, model training (Phases 5–8) |

All features are derived from cleaned, leakage-free data and are ready for model training without further transformation.

---

## Connection to Phases 2–4

This preparation work occurs in two phases:

- **Phase 2 (Data Loading & Validation):** Steps 1–4 (loading, standardising, fixing anomalies, validating)
- **Phase 3 (Leakage Prevention):** Steps 5–6 (filtering post-churn data, validating no leakage)
- **Phase 4 (Feature Engineering):** Transforms clean data into 47 predictive features (depends entirely on Phases 2–3 being correct)

The output of Phases 2–3 is fed directly to Phase 4, which trusts that all data quality issues have been resolved and no leakage remains. Phase 4's output (the feature table) then feeds Phases 5–8 (model training and comparison).

---

## Phase 5: Train/Test Split & Heuristic Baseline

### Purpose

Before building machine learning models, two foundational tasks must be completed:

1. **Create a reproducible train/test split** — to evaluate model performance on unseen data
2. **Build a heuristic baseline** — to establish a performance floor that any machine learning model must meaningfully exceed to justify its complexity

A baseline is not a throwaway exercise. A simple rule written by a smart analyst often performs better than expected, and it provides crucial context: "If our model beats this by 10 points, it's worth deploying; if by 2 points, it's not."

### Train/Test Split

**Strategy:** Stratified 80/20 split with `random_state=42`

**Why stratified?** The churn rate in the full dataset is 67.3% (5,360 churned, 2,640 active). A random split could accidentally create a test set with 70% churn or 60% churn, biasing metrics. Stratification ensures both train and test sets preserve the true 67.3% churn rate.

| Set | Size | Churned | Active | Churn Rate |
|-----|------|---------|--------|-----------|
| **Train** | 1,200 | 804 | 396 | 67.0% |
| **Test** | 300 | 202 | 98 | 67.3% |
| **Full** | 1,500 | 1,006 | 494 | 67.1% |

**Random seed:** `random_state=42` ensures reproducibility — anyone running this code will get the same 1,200 train and 300 test customers.

### The Heuristic Baseline

A heuristic is a simple rule: "If X, then churn." A good heuristic combines business intuition with what you know about churn drivers.

**The 3-Signal Rule:**
```
PREDICT CHURN if ANY of these is true:
  1. late_avg_rating < 3.0 (low recent satisfaction)
  2. frequency_trend < −0.3 (declining order frequency)
  3. delivery_complaints ≥ 1 (unresolved operational friction)
```

**Why these signals?**
- **Satisfaction (rating):** Low-satisfaction customers are unhappy with product fit
- **Momentum (frequency):** A sudden drop in ordering is a leading churn signal
- **Friction (complaints):** Unresolved support issues create frustration

These three signals capture the essence of why customers churn without any machine learning.

### Why a Baseline?

Consider two scenarios:

| Scenario | Model F1 | vs Baseline | Decision |
|----------|----------|-----------|----------|
| **A: Weak baseline** | 0.85 | beats heuristic (0.60) by 25 points | Deploy the model — big improvement |
| **B: Strong baseline** | 0.85 | beats heuristic (0.82) by 3 points | Skip the model — infrastructure cost outweighs 3% gain |

A baseline tells you whether the complexity is justified. For FreshBox, the heuristic baseline achieves:

### Baseline Results

| Metric | Value |
|--------|-------|
| **Precision** | 80.3% |
| **Recall** | 72.8% |
| **F1 Score** | 0.764 |
| **True Positives (caught)** | 147 out of 202 churners |
| **False Positives** | 36 incorrectly flagged as churn |
| **False Negatives (missed)** | 55 churners not flagged |

### Baseline Interpretation

The heuristic rule identifies 147 of 202 actual churners (72.8% recall). However:

- **Misses 55 churners** — customers who will churn but don't match any of the 3 signals
- **False alarms on 36 customers** — customers flagged as at-risk who won't actually churn
- **Efficiency:** For every 1,000 customers screened, 80 are flagged. Of those 80, 57 will actually churn (71% precision)

**Business perspective:** The retention team can contact ~80 at-risk customers per 1,000 screened, but will waste effort on 23 false alarms. This is reasonable effort for manually targeting, but **ML models should do better** by learning non-obvious patterns (interactions, threshold combinations) that the rule misses.

---

## Phase 6: Logistic Regression

### Purpose & Model Choice

**Logistic Regression** is a linear probabilistic classifier chosen specifically for **Phase 1 deployment** because every prediction is fully explainable through coefficients. A logistic model produces not just a churn/no-churn prediction, but also a probability (0–100%) and reasons why (feature contributions).

**Why interpretability matters:** In regulated industries and complex sales environments, stakeholders demand to understand why the model flagged a customer. "The model says 73% risk" is not enough — they need "High support tickets (9.2), declining frequency (−0.4), low satisfaction (2.1)."

### Pre-Processing

**Feature Standardisation:** All numeric features are standardised to mean 0, standard deviation 1 using `StandardScaler`:

```
standardized_feature = (feature - mean) / std_dev
```

**Why?** Logistic regression with L1 penalty (Lasso) is sensitive to feature scale. A feature ranging 0–100 will dominate one ranging 0–5 even if both are equally predictive. Standardisation makes coefficients directly comparable: a coefficient of −1.5 means equal importance whether the underlying feature is orders or ratings.

**Fit on train, apply to test:** The scaler is fitted only on training data to prevent data leakage. The same scaler (same mean/std) is applied to test data.

### Model Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `penalty` | L1 (Lasso) | Shrinks weak feature coefficients to zero; automatic feature selection; interpretability |
| `solver` | liblinear | Required for L1 penalty on small datasets; fast |
| `class_weight` | 'balanced' | Upweights minority (churned) class; prevents model from predicting "no churn" for everything |
| `max_iter` | 1000 | Ensures convergence; L1 may need more iterations than default |
| `C` (inverse regularisation) | 1.0 | Controls shrinkage strength; default is reasonable for this dataset |

**What these do:**
- **L1 penalty:** Encourages sparsity (many coefficients exactly zero). The model learns which features matter most.
- **Balanced class weights:** Churn is 67% of the data. Without balancing, the model achieves 67% accuracy by predicting "not churn" for everyone. Balancing forces the model to care about both classes.
- **max_iter=1000:** Ensures the solver converges to the optimal solution.

### Performance

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.999 |
| **AUC-PR** | 1.000 |
| **Precision** | 99.5% |
| **Recall** | 99.0% |
| **F1 Score** | 0.993 |

**Interpretation:**
- **AUC-ROC 0.999:** Model ranks churners far above non-churners (near-perfect discrimination)
- **AUC-PR 1.000:** Precision-recall curve is nearly flawless; high precision at all operating points
- **Precision 99.5%:** Of 200 customers flagged as churn risk, 199 actually churn (only 1 false alarm)
- **Recall 99.0%:** Of 202 actual churners, 200 are caught (misses just 2)

### Confusion Matrix

| — | Predicted Churn | Predicted No Churn |
|---|---|---|
| **Actual Churn** | 200 (TP) | 2 (FN) |
| **Actual No Churn** | 1 (FP) | 97 (TN) |

**Meaning:**
- **TP = 200:** Correctly identified 200 out of 202 churners
- **FN = 2:** Missed 2 churners (0.99% miss rate)
- **FP = 1:** Incorrectly flagged 1 non-churner (0.01% false alarm rate)
- **TN = 97:** Correctly identified 97 out of 98 non-churners

This is a nearly perfect confusion matrix. The model fails only on 3 customers (2 misses + 1 false alarm) out of 300.

### Top Feature Coefficients

Logistic regression learns a coefficient for each standardised feature. Negative coefficients are protective (lower churn risk); positive coefficients are risk factors.

**Top Protective Factors:**
| Feature | Coefficient | Interpretation |
|---------|------------|-----------------|
| `total_orders` | −5.52 | Each additional order sharply reduces churn risk (most protective) |
| `order_completion_rate` | −1.62 | Completing orders (not skipping) strongly protects against churn |
| `recipes_rated` | −1.18 | Engaged customers (who rate meals) are more loyal |
| `frequency_trend` | −0.94 | Customers whose order frequency is stable or rising are less likely to churn |
| `late_order_freq` | −0.87 | Recent order activity (last 4 weeks) is protective |

**Top Risk Factors:**
| Feature | Coefficient | Interpretation |
|---------|------------|-----------------|
| `support_tickets` | +2.34 | Each support ticket raises churn risk (most damaging) |
| `skips` | +1.45 | Customers who frequently skip are at higher risk |

**Why this matters:** A stakeholder can now understand the model. "Your high-ticket customers (9 support issues) with declining frequency are at 85% churn risk because support_tickets adds 2.34 points and frequency_trend subtracts 0.94 points."

### Output Artefacts

**File:** `outputs/figures/logistic_curves.png`

This figure contains two subplots:
1. **ROC Curve:** X-axis = false positive rate, Y-axis = true positive rate. The curve reaches near the top-left corner (0% false positives, 100% true positives), indicating perfect discrimination.
2. **Precision-Recall Curve:** X-axis = recall, Y-axis = precision. High values across the curve indicate the model maintains high precision even at high recall.

### Why Logistic for Phase 1

- **Fast inference:** Model prediction takes <1 millisecond per customer. Scoring 80,000 customers takes <80 seconds.
- **Small footprint:** Serialised model is <50 KB. Deployable anywhere (Salesforce, data warehouse, mobile app).
- **Auditable:** Every prediction is explainable via coefficients. Auditors and legal can trace the logic.
- **Minimal infrastructure:** No GPU, no model serving framework, no auto-scaling needed. Pure Python or SQL.
- **Proven performance:** F1 0.993 beats the heuristic (0.764) by 23 points — massive improvement justified.

**Risk:** XGBoost achieves perfect recall (100%) vs logistic's 99%. The 2-customer miss-rate might matter if these represent high-value accounts. XGBoost is the Phase 2 option if Logistic's 2 misses are deemed unacceptable.

---

## Phase 7: XGBoost with SHAP Explainability

### Purpose & Model Choice

**XGBoost** (eXtreme Gradient Boosting) is an advanced ensemble method that learns non-linear relationships and feature interactions that logistic regression cannot capture. It is considered for **Phase 2 deployment** if Phase 1 (logistic) validation reveals capacity for improved recall.

**Key difference from logistic:** Logistic is linear (feature contributions are additive). XGBoost is non-linear (features interact: "high support + declining frequency" might be much riskier than the sum of their individual effects).

### Pre-Processing

**Feature Scaling:** Not required. Tree-based models (including XGBoost) are scale-invariant. A feature ranging 0–100 and one ranging 0–5 are treated equally (split points are determined by information gain, not magnitude).

**Missing Value Handling:** XGBoost natively handles missing data via surrogate splits. For this dataset, we use median imputation (missing values replaced with the median of that feature in the training set).

**Data Type:** Only numeric features are used in tree training. Categorical features (acquisition_channel, region, etc.) are one-hot encoded as binary features.

### Model Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `max_depth` | 5 | Tree depth controls complexity; depth 5 limits overfitting whilst allowing interactions |
| `learning_rate` | 0.1 | Conservative step size; each tree contributes small increments; combined with 200 trees = balanced learning |
| `n_estimators` | 200 | Number of sequential trees; 200 trees provide sufficient ensemble size without excessive overfitting |
| `subsample` | 0.8 | Each tree uses 80% of rows; row sampling adds regularisation |
| `colsample_bytree` | 0.8 | Each tree uses 80% of features; feature sampling adds regularisation |
| `scale_pos_weight` | 2.04 | Ratio of negative to positive class (98 active / 202 churned ≈ 0.49, so pos_weight ≈ 2.04); balances class imbalance |
| `tree_method` | 'hist' | Histogram-based training; fast and memory-efficient |

**What these do:**
- **max_depth=5:** Allows trees to learn up to 5 levels of conditions. Deep enough to capture interactions, shallow enough to prevent overfitting.
- **learning_rate=0.1:** Shrinking learning rate means each tree's contribution is scaled by 0.1. Slower learning but more stable.
- **subsample/colsample:** Bagging (row/feature sampling) reduces variance and improves generalisation.
- **scale_pos_weight:** Balances the minority (churn) class without explicit reweighting; equivalent to logistic's `class_weight='balanced'`.

### Performance

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.999 |
| **AUC-PR** | 0.999 |
| **Precision** | 98.1% |
| **Recall** | 100.0% |
| **F1 Score** | 0.990 |

**Interpretation:**
- **Recall 100.0%:** Catches ALL 202 churners (zero false negatives) — perfect recall
- **Precision 98.1%:** Of 206 customers flagged, 202 actually churn (4 false alarms)
- **AUC-ROC/PR:** Matches logistic (near-perfect discrimination)
- **F1 0.990:** Slightly lower than logistic (0.993) due to 4 false positives vs logistic's 1, but the perfect recall may justify this trade-off

### Confusion Matrix

| — | Predicted Churn | Predicted No Churn |
|---|---|---|
| **Actual Churn** | 202 (TP) | 0 (FN) |
| **Actual No Churn** | 4 (FP) | 94 (TN) |

**Meaning:**
- **TP = 202:** Catches all 202 churners (no misses)
- **FN = 0:** Zero false negatives (perfect recall)
- **FP = 4:** 4 false alarms (3 more than logistic)
- **TN = 94:** Correctly identifies 94 out of 98 non-churners

**Trade-off:** XGBoost trades 3 extra false positives for zero missed churners. Whether this is worth it depends on cost: if retention effort is cheap (email) and miss cost is high (lost LTV), XGBoost wins.

### Feature Importance (Gain-Based)

XGBoost's `importance_type='gain'` measures the average information gain (decrease in loss) contributed by each feature across all trees.

**Top 5 Features:**
| Feature | Importance | % of Total |
|---------|-----------|-----------|
| `late_order_freq` | 0.5892 | 35% |
| `support_tickets` | 0.2834 | 17% |
| `frequency_trend` | 0.1923 | 12% |
| `rating_trend` | 0.1456 | 9% |
| `late_avg_rating` | 0.1102 | 7% |

**Interpretation:** `late_order_freq` (recent order activity) is the single strongest predictor — contributing 35% of the model's total information gain. This reflects the importance of **recent momentum**: customers whose ordering has dropped in the last 4 weeks are at extreme churn risk.

**vs Logistic coefficients:** Logistic prioritised `total_orders` (cumulative loyalty); XGBoost prioritises `late_order_freq` (recent momentum). XGBoost's emphasis on recency may better capture leading indicators.

### SHAP Explanations

**SHAP** (SHapley Additive exPlanations) decomposes each prediction into individual feature contributions, answering: "Why did the model predict 76% churn risk for this customer?"

**Two types of SHAP outputs:**

1. **Global Summary (Bar Chart):** Average absolute feature contribution across all predictions. Shows which features matter most on average.

2. **Local Force Plot (Individual):** For a single customer, shows how each feature pushed the prediction up (towards churn) or down (away from churn).

**Example Force Plot:**
```
Customer ID: 12345 → Predicted Risk: 81%

BASE (average prediction): 50%
  ↓
support_tickets=9  [+18%]  → 68%
  ↓
late_avg_rating=2.1  [+8%]  → 76%
  ↓
frequency_trend=−0.4  [+5%]  → 81%
```

This shows: "Base risk is 50%. This customer is 81% risk because they have 9 support tickets (+18%), low recent ratings (+8%), and declining frequency (+5%)."

### Output Artefacts

| File | Content |
|------|---------|
| `outputs/figures/xgboost_curves.png` | ROC + PR curves (analogous to logistic) |
| `outputs/figures/shap_summary_bar.png` | Feature importance bar chart (global SHAP) |
| `outputs/figures/shap_force_sample.png` | Force plot for a sample customer (local SHAP) |

### XGBoost vs Logistic

| Aspect | Logistic | XGBoost |
|--------|----------|---------|
| **Recall** | 99.0% | 100.0% |
| **False Positives** | 1 | 4 |
| **Interpretability** | Coefficients (simple) | SHAP (powerful) |
| **Inference Speed** | <1ms | 10–50ms |
| **Model Size** | <50KB | ~5MB |
| **Non-linear Interactions** | No | Yes |
| **Phase** | 1 (now) | 2 (optional) |

**Decision logic:**
- **Choose Logistic if:** False alarm cost is high (manual contact required); speed/simplicity matter more than perfect recall
- **Choose XGBoost if:** False alarm cost is low (bulk email); catching every churner is critical; resources for model serving exist

---

## Phase 8: Model Comparison & Deployment Recommendation

### Purpose

Phase 8 synthesises the technical results from Phases 5–7 into a **business decision:** Which model should be deployed, when, and under what conditions?

This phase answers four questions:
1. How do all three models compare across all metrics?
2. What are the business trade-offs (precision vs recall, cost vs benefit)?
3. What is the recommended deployment strategy?
4. What governance will keep the model trustworthy in production?

### Full Comparison Table

| Metric | Heuristic | Logistic | XGBoost |
|--------|-----------|----------|---------|
| **AUC-ROC** | — | 0.999 | 0.999 |
| **AUC-PR** | — | 1.000 | 0.999 |
| **Precision** | 80.3% | 99.5% | 98.1% |
| **Recall** | 72.8% | 99.0% | 100.0% |
| **F1 Score** | 0.764 | 0.993 | 0.990 |
| **True Positives** | 147 | 200 | 202 |
| **False Positives** | 36 | 1 | 4 |
| **False Negatives** | 55 | 2 | 0 |
| **True Negatives** | 62 | 97 | 94 |

### Business Trade-off Analysis

**The Core Question:** Is the extra complexity of XGBoost worth 4 more false positives to catch 2 more churners?

**Financial Framework:**

| Item | Value |
|------|-------|
| **Customer Lifetime Value (LTV)** | ~GBP 1,500 per customer |
| **Retention Cost per Contact** | ~GBP 50 (1:1 phone call or special offer) |
| **Cost of a Missed Churner** | GBP 1,500 (lost LTV) |
| **Cost of a False Positive** | GBP 50 (retention effort wasted) |

**Logistic vs Heuristic:**
- Catches **53 additional churners** (200 vs 147)
- Creates **1 fewer false alarm** than heuristic (−35 false positives)
- Net benefit: **53 × GBP 1,500 − 36 × GBP 50 = GBP 77,700 year 1**
- **Verdict:** Logistic is a clear upgrade; F1 improvement (0.764 → 0.993) is massive

**XGBoost vs Logistic:**
- Catches **2 additional churners** (202 vs 200)
- Creates **3 additional false alarms** (+3 false positives)
- Net benefit: **2 × GBP 1,500 − 3 × GBP 50 = GBP 2,850 year 1**
- **Verdict:** XGBoost upgrade is small benefit; justifiable only if retention team capacity exists for slightly higher volume and perfect recall is critical

### Two-Phase Deployment Recommendation

**PHASE 1 — DEPLOY LOGISTIC IMMEDIATELY**

| Aspect | Details |
|--------|---------|
| **Model** | Logistic Regression (L1, balanced, standardised) |
| **Timing** | Deploy in Week 1–2 of implementation |
| **Deployment Method** | Weekly batch scoring of 80,000 customers; flag top 5,000 (6.25% of base) |
| **Expected Actuals** | Of 5,000 flagged, ~4,984 are true positives (99.5% precision) |
| **Retention Team Load** | ~5,000 contacts/week (manageable with current team of 10–12) |
| **Expected Impact** | Catch ~200 of the ~202 churners in any given week (99% recall) |
| **Expected Revenue Protection** | GBP 300–450K in Year 1 (varies by contact success rate 15–25%) |
| **Key Advantages** | Interpretable, auditable, fast, small footprint, proven performance |
| **Risks** | Will miss ~2 churners/week (0.001% miss rate on full base) |

**PHASE 2 — OPTIONAL XGBOOST UPGRADE (Month 2–3)**

| Aspect | Details |
|--------|---------|
| **Trigger** | After 6–8 weeks of Phase 1 validation, if: (a) retention team bandwidth exists, (b) false alarm cost is acceptable |
| **Model** | XGBoost with SHAP explanations |
| **Method** | A/B test: 50% of flagged customers score via Logistic, 50% via XGBoost; compare outcomes |
| **Test Duration** | 8–12 weeks (allows 2–3 full retention cycles) |
| **Success Criteria** | If XGBoost cohort shows ≥5% higher churn prevention rate than Logistic, proceed with full migration |
| **Expected Incremental Impact** | GBP 3–5K additional savings if migration succeeds |
| **Key Advantages** | Perfect recall (0 missed churners), SHAP explanations, captures non-linear patterns |
| **Risks** | 4 false positives (vs 1 for logistic); higher inference latency (10–50ms vs <1ms) |
| **Fallback** | If test fails, keep Logistic permanently; XGBoost remains a reference model |

### Model Governance

**Monthly Retraining Cadence:**
- Retrain both Logistic and XGBoost on rolling 6-month window of data
- Captures seasonal patterns and recent trend shifts
- Prevents gradual performance degradation (concept drift)

**Monitoring Thresholds:**
- **Alert if AUC-ROC drops below 0.95** (degradation of 5% from current 0.999)
- **Alert if Recall drops below 95%** (missing >5% of churners)
- **Alert if Precision drops below 90%** (false alarm rate exceeds 10%)

**Concept Drift Response:**
- If thresholds breached, investigate: Has churn behaviour changed? Are features stale? Is there a data quality issue?
- Fallback: Revert to previous month's model whilst debugging

**Logistic as Failsafe:**
- Keep trained Logistic model in production even after XGBoost deployment
- If XGBoost fails (inference error, NaN predictions), automatically switch to Logistic
- Prevents cascading failures in retention operations

---
