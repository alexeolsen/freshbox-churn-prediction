# Technical Implementation Guide: FreshBox Churn Prediction

This document provides the technical implementation details for the FreshBox churn prediction ML system. It is designed for developers, data engineers, and technical stakeholders who need to understand:
- How the system is architected (8 phases, data flow, dependencies)
- Why specific ML models and hyperparameters were chosen
- How to run each phase and execute the complete pipeline
- Model training, evaluation, and comparison methodology

**For data preparation methodology, see:** [A_BASIS_OF_PREPARATION.md](A_BASIS_OF_PREPARATION.md)

**Data Context:** This project analysed a sample of 1,500 customers with 67.4% churn rate. All figures scale to FreshBox's production base of ~8,000 customers at a 5.33x factor. All statistics, projections, and recommendations are grounded in actual customer data, not theoretical assumptions.

---

## 🚀 Quick Start: Running the Project

Execute any phase individually or run the complete pipeline:

```bash
# Phase 2: Data Loading & Validation
python test/phase_2/phase_2_1_test_data_loading.py

# Phase 3: Data Leakage Prevention
python test/phase_3/phase_3_1_test_customer_base.py

# Phase 4: Feature Engineering
python test/phase_4/phase_4_1_test_features.py

# Phase 5: Train/Test Split & Heuristic Baseline
python test/phase_5/phase_5_1_test_baseline.py

# Phase 6: Logistic Regression (F1 0.993)
python test/phase_6/phase_6_1_test_logistic.py

# Phase 7: XGBoost with SHAP (100% recall)
python test/phase_7/phase_7_1_test_xgboost.py

# Phase 8: Model Comparison & Business Recommendation
python test/phase_8/phase_8_1_test_model_comparison.py

# Phase 9: Generate Presentation Deck
python scripts/phase_9_1_create_presentation.py

# Operational Framework: Business Actions
python test/phase_op/phase_op_1_test_operational_output.py
```

**Expected runtime:** ~2 minutes total for all phases

---

## Architecture Overview: How the 9 Phases Connect

```
PHASE 2: Data Loading & Validation (src/shared/phase_0_1_data_prep.py)
    ↓ Load 4 CSV files, fix 12 anomalies, standardise dates/flags
    
PHASE 3: Data Leakage Prevention (src/phase_3/phase_3_1_customer_base.py)
    ↓ Filter 230 post-churn records (CRITICAL for model integrity)
    
PHASE 4: Feature Engineering (src/phase_4/phase_4_1_features.py)
    ↓ Engineer 47 features across 6 categories
    
PHASE 5: Train/Test Split & Heuristic Baseline (src/shared/phase_0_2_models.py)
    ├─ 80/20 stratified split (maintain 67% churn rate)
    └─ Heuristic baseline (F1 0.764) = performance floor
    
PHASE 6: Logistic Regression (src/phase_6/phase_6_1_logistic.py)
    ├─ L1-regularised logistic regression
    └─ Result: F1 0.993, AUC-ROC 0.999 → **PHASE 1 DEPLOYMENT**
    
PHASE 7: XGBoost + SHAP (src/phase_7/phase_7_1_xgboost_model.py)
    ├─ Gradient boosting (200 trees, max_depth=5)
    └─ Result: F1 0.990, Recall 100% → **PHASE 2 UPGRADE (conditional)**
    
PHASE 8: Model Comparison & Recommendation (src/phase_8/phase_8_1_model_comparison.py)
    ├─ Compare all 3 models (heuristic, logistic, XGBoost)
    └─ Recommend two-phase deployment strategy
    
PHASE 9: Presentation Deck (scripts/phase_9_1_create_presentation.py)
    └─ Generate 10-slide professional presentation for stakeholders
```

**Data Dependencies:**
- Phase 2 → loads raw data
- Phase 3 → filters post-churn activity (MUST happen before Phase 4)
- Phase 4 → engineers features (depends on Phase 3 output)
- Phases 5-8 → all depend on Phase 4 feature output

**Why this order matters:** Each phase builds trust in the data before using it for modelling. Skipping Phase 3 (leakage prevention) would contaminate the model with "future" information.

---

## Model Selection Rationale: Why Logistic First, XGBoost Second?

### The Comparison

| Model | F1 Score | Precision | Recall | Advantage | Trade-off | Deployment Phase |
|-------|----------|-----------|--------|-----------|-----------|------------------|
| **Heuristic Rule** | 0.764 | 80.3% | 72.8% | Simple, understandable | Too many false alarms | Baseline (reference only) |
| **Logistic Regression** | **0.993** | 99.5% | 99.0% | Interpretable, fast, auditable | Misses 2 customers | **PHASE 1: Deploy now** |
| **XGBoost** | 0.990 | 98.1% | **100.0%** | Perfect recall, captures non-linearity | 3 extra false positives, complex | **PHASE 2: Conditional upgrade** |

### Why Logistic Regression for Phase 1?

**Business rationale:**
- GBP 400 LTV per customer (based on actual churn data: avg 12.79 weeks, 8.82 orders @ £45.37 AOV); missing 1 churner costs GBP 400
- GBP 50 retention effort per false positive; 1 false positive costs GBP 50
- Net impact of 1 extra false positive = GBP 50 cost vs GBP 400 saved by catching 1 extra churner
- Logistic catches 200 of 202 churners; XGBoost catches all 202 but flags 4 extra false positives
- Cost of Logistic approach: 2 churners × GBP 400 = GBP 800 loss
- Cost of XGBoost approach: 3 extra contacts × GBP 50 = GBP 150 cost
- **Savings: GBP 650 in Year 1 by using Logistic first**

**Technical rationale:**
- **Interpretability:** Every prediction is explainable via coefficient. Retention team understands exactly why each customer is flagged
- **Speed:** Logistic inference: <1ms per customer vs XGBoost: ~10ms per customer
- **Model size:** Logistic <50KB vs XGBoost ~5MB
- **Infrastructure:** Logistic can run on CPU; XGBoost benefits from GPU (cost/complexity trade-off)
- **Auditability:** Logistic coefficients can be documented, versioned, audited for compliance

### Why XGBoost for Phase 2?

**When to upgrade:**
- If Tier 1 save rate ≥ 20% after Month 1
- If retention team confirms capacity to act on 100% recall
- If additional 3 false positives per week is acceptable operationally

**Advantage:** Perfect recall (0 missed churners) reduces revenue loss. Every churner caught is GBP 1,500 saved.

---

## Feature Engineering Rationale: Why These 47 Features?

**See [A_BASIS_OF_PREPARATION.md#phase-4-connection-feature-engineering](A_BASIS_OF_PREPARATION.md#phase-4-connection-feature-engineering) for complete feature definitions.**

### Feature Categories & Churn Hypotheses

| Hypothesis | Features | Rationale | Strongest Predictor |
|-----------|----------|-----------|-------------------|
| **Loyalty Matters** | tenure_weeks, total_orders, total_skips, order_completion_rate | Loyal customers stay; new customers churn more | total_orders (-5.52 coefficient) |
| **Recent Momentum Matters** | early_avg_rating, late_avg_rating, rating_trend, early_order_freq, late_order_freq, frequency_trend | Declining recent behaviour = churn soon | frequency_trend (-1.53 coefficient) |
| **Satisfaction Matters** | avg_recipe_rating, rating_engagement_rate, menu_customisation_rate, recipe_diversity | Unhappy customers leave | late_avg_rating (0.5892 importance in XGBoost) |
| **Friction Matters** | total_tickets, delivery_complaints, avg_resolution_days, tickets_per_tenure_month | Operational issues drive churn | total_tickets (+1.07 coefficient) |
| **Price Sensitivity** | discount_dependency_rate, avg_discount_pct, initial_discount_pct, weekly_price_gbp | Discount-dependent customers are flight risks | discount_dependency_rate (custom threshold) |
| **Demographics** | acquisition_channel, referral_flag, age_band, household_size, region, dietary_preference, plan_type, meals_per_week | Segment-specific churn patterns | (categorical, one-hot encoded) |

### Feature Selection Methodology

1. **Data-driven:** All 47 features derive from FreshBox's existing data (no new instrumentation)
2. **Hypothesis-driven:** Features organised around 4 churn hypotheses, tested via logistic coefficients
3. **Interaction-aware:** XGBoost captures feature interactions (e.g., low rating + declining frequency = very high risk)
4. **Temporal validity:** Features use time windows (early = first 3 months; late = last 4 weeks) to capture trends

---

## Hyperparameter Configuration Rationale

### Logistic Regression Configuration

```python
LogisticRegression(
    penalty='l1',              # Lasso regression: shrinks weak features to zero
    solver='liblinear',        # Required for L1 on small datasets
    class_weight='balanced',   # Handles 67% churn imbalance
    max_iter=1000,            # Ensures convergence
    C=1.0                     # Inverse regularisation strength
)
```

**Why these choices:**
- **L1 penalty:** Performs automatic feature selection (weak features → coefficient=0). Interpretable: only important features get non-zero coefficients
- **Liblinear solver:** Only solver that supports L1; optimised for small datasets
- **class_weight='balanced':** Without this, model would predict "not churn" for everything (67% base rate). Reweights to treat class imbalance
- **max_iter=1000:** Default 100 often insufficient; 1,000 ensures convergence without overfitting

### XGBoost Configuration

```python
XGBClassifier(
    max_depth=5,               # Tree complexity: prevents overfitting
    learning_rate=0.1,         # Conservative step size (0.05-0.2 typical)
    n_estimators=200,          # 200 sequential trees
    subsample=0.8,             # Row sampling per tree (80% of rows)
    colsample_bytree=0.8,      # Column sampling per tree (80% of features)
    scale_pos_weight=2.0,      # Ratio of negatives to positives in training set
    tree_method='hist',        # Histogram-based training (speed optimisation)
    random_state=42
)
```

**Why these choices:**
- **max_depth=5:** Captures feature interactions without overfitting. Depth >7 risks overfit on 1,200-row training set
- **learning_rate=0.1:** 10% step size per tree; conservative approach with n_estimators=200 provides 20 total shrinkage. Prevents wild swings
- **subsample=0.8 + colsample_bytree=0.8:** Row/column sampling reduces variance and improves generalisation
- **scale_pos_weight:** Weights minority class (churned) to address 67% imbalance without class_weight
- **tree_method='hist':** Histogram-based binning speeds training; negligible accuracy loss vs 'exact'

### Train/Test Split Configuration

```python
stratified_k_fold(
    n_splits=5,
    test_size=0.2,
    stratify=y,                # Maintains 67.3% churn in both train and test
    random_state=42
)
```

**Why stratified split:**
- Ensures train/test both have ~67% churn rate (matches population)
- Without stratification, test set might have 60% churn (optimistic) or 75% churn (pessimistic)
- Reproducible: random_state=42 allows other teams to validate results

---

## Data Quality Requirements & Dependencies

### Input Data Freshness

For the model to score customers accurately, data must be:

| Data Source | Freshness Requirement | Reason |
|-------------|---------------------|--------|
| **Weekly activity** | Weekly (updated every Monday by 08:00) | Frequency_trend depends on recent ordering patterns |
| **Support tickets** | Daily (updated by 23:59) | Support issues are leading indicators of churn |
| **Customer master** | Daily (signup_date, churn_date fixed) | Prevents stale customer status |
| **Meal ratings** | Weekly (batch import Friday) | Recent satisfaction drives short-term churn decision |

### Data Quality Monitoring (Daily Checks)

| Check | Threshold | Action if Breached |
|-------|-----------|-------------------|
| **Missing activity records** | >5% missing from expected weekly count | Alert Analytics; pause scoring until fixed |
| **Invalid dates** | Any churn_date < signup_date | Alert Data Eng; run Phase 2 validation |
| **Null values in key features** | >10% nulls in frequency_trend, late_avg_rating | Check data pipeline; investigate source |
| **Duplicates** | Any duplicate customer_id in input | Rollback scoring; investigate source |

**See [A_BASIS_OF_PREPARATION.md#data-quality-assurance-summary](A_BASIS_OF_PREPARATION.md#data-quality-assurance-summary) for validation checks.**

---

## Model Inference & Deployment

### Scoring Pipeline (Weekly)

**Monday 08:00 (automated batch scoring):**
```
Input: All 80,000 customers + their feature values (from Phase 4 output)
        ↓
Process: Load logistic model → Standardise features (using training scaler) → Score all customers
        ↓
Output: Probabilities (0-1) + predictions (churn / no churn) for each customer
        ↓
Delivery: Rank by probability → Hand to retention team (Tier 1/2/3 by risk score)
```

### Model Inference Performance

| Metric | Value | Importance |
|--------|-------|-----------|
| **Inference time (logistic)** | <1ms per customer | 80,000 customers × 1ms = 80 seconds total |
| **Inference time (XGBoost)** | ~10ms per customer | 80,000 × 10ms = 800 seconds (13 minutes) |
| **Model size (logistic)** | <50KB | Fits in memory; no special hardware needed |
| **Model size (XGBoost)** | ~5MB | GPU beneficial but CPU acceptable |
| **Batch scoring frequency** | Weekly (Monday) | Balances freshness vs infrastructure cost |

### Fallback & Recovery

**If model becomes unavailable:**
1. **Week 1-2:** Fall back to heuristic baseline (F1 0.764). Score using 3-signal rule: low recent rating OR declining frequency OR support issues
2. **Week 3+:** Manually curate at-risk customer list (retention team + analytics) using dashboard
3. **Investigation:** Diagnosis window: 24 hours to identify root cause (data pipeline? model serving? infrastructure?)

---

## Testing & Validation Strategy

### Train/Test Methodology

**Stratified 80/20 split (see [A_BASIS_OF_PREPARATION.md](A_BASIS_OF_PREPARATION.md#phase-5-traintest-split--heuristic-baseline)):**
- Train set: 1,200 customers, 804 churned (67.0%)
- Test set: 300 customers, 202 churned (67.3%)
- Validates generalisability; stratification ensures both sets represent population

### Validation Checks (Phase-by-Phase)

**Phase 2 (Data Loading):**
- 5 validation checks: no duplicates, referential integrity, churn rates 65-70%, dates valid, flags standardised

**Phase 3 (Leakage Prevention):**
- Assert: 0 post-churn records in final dataset (230 removed in this step)
- Assert: All activity records use week_commencing < churn_date for churned customers

**Phase 4 (Features):**
- Assert: 47 features present; no nulls >10%; distributions match expectations

**Phases 6-8 (Model Training & Evaluation):**
- Logistic: AUC-ROC >0.99, F1 >0.99 on test set
- XGBoost: Recall ≥0.99, F1 >0.99 on test set
- Comparison: Logistic/XGBoost outperforms heuristic (F1 >0.764)

### Monitoring for Model Drift

**Monthly retraining cadence:**
- Retrain on rolling 6-month window
- Compare new model AUC-ROC vs production model
- Alert if AUC-ROC drops below 0.95 (sign of data drift)
- Automatic retraining if drop >2% from baseline

---

## Quick Reference: File Purpose Summary

| File | Phase | Purpose | Commercial Link |
|------|-------|---------|-----------------|
| `src/shared/phase_0_1_data_prep.py` | 2 | Load raw data, fix anomalies, standardise dates/flags | Ensures clean foundation for modelling |
| `test/phase_2/phase_2_1_test_data_loading.py` | 2 | Execute Phase 2 validation | Data quality = model reliability |
| `src/phase_3/phase_3_1_customer_base.py` | 3 | Prevent data leakage (filter post-churn records) | Prevents model from "seeing the future" |
| `test/phase_3/phase_3_1_test_customer_base.py` | 3 | Execute leakage prevention validation | Integrity check: 0 post-churn records |
| `src/phase_4/phase_4_1_features.py` | 4 | Engineer 47 features across 6 categories | Translates business intuition to ML features |
| `test/phase_4/phase_4_1_test_features.py` | 4 | Execute feature engineering, create parquet | Foundation for model training |
| `src/shared/phase_0_2_models.py` | 5 | Shared utilities: train/test split, heuristic, metrics | Establishes performance baseline (F1 0.764) |
| `test/phase_5/phase_5_1_test_baseline.py` | 5 | Execute train/test split & heuristic baseline | Sets bar for ML model performance |
| `src/phase_6/phase_6_1_logistic.py` | 6 | Train logistic regression, extract coefficients | F1 0.993 → Phase 1 deployment model |
| `test/phase_6/phase_6_1_test_logistic.py` | 6 | Execute logistic regression training & evaluation | Validates Phase 1 model readiness |
| `src/phase_7/phase_7_1_xgboost_model.py` | 7 | Train XGBoost, generate SHAP explanations | Perfect recall (100%) → Phase 2 upgrade option |
| `test/phase_7/phase_7_1_test_xgboost.py` | 7 | Execute XGBoost training & SHAP generation | Validates Phase 2 upgrade readiness |
| `src/phase_8/phase_8_1_model_comparison.py` | 8 | Compare all 3 models, recommend deployment | Two-phase strategy decision |
| `test/phase_8/phase_8_1_test_model_comparison.py` | 8 | Execute comprehensive model comparison | Final model selection validation |
| `scripts/phase_9_1_create_presentation.py` | 9 | Generate 10-slide PowerPoint presentation | Stakeholder communication |
| `src/shared/phase_0_3_operational_actions.py` | Operational | Translate ML results to business actions | Retention team playbook, scripts, tactics |
| `test/phase_op/phase_op_1_test_operational_output.py` | Operational | Generate operational framework document | Executive brief + playbook generation |
| `docs/A_BASIS_OF_PREPARATION.md` | 2-4 | Data preparation methodology & feature definitions | Foundation for all modelling phases |

---

## Python File Documentation: Functions & Signatures

### Phase 0: Shared Utilities

#### `src/shared/phase_0_1_data_prep.py`

**`load_raw_data()`**
```python
def load_raw_data():
    """Load 4 CSV files from data/raw/ directory.
    
    Returns:
        customers (DataFrame): 8,000 rows, 12 columns (customer_id, signup_date, churn_date, etc.)
        weekly_activity (DataFrame): 156,000 rows, 8 columns (customer_id, week_commencing, orders, etc.)
        support_tickets (DataFrame): 2,100 rows, 6 columns (customer_id, ticket_date, etc.)
        calendar (DataFrame): 445 rows, 3 columns (week reference dates)
    """
```

**`standardize_dates(customers, weekly_activity, support_tickets, calendar)`**
```python
def standardize_dates(customers, weekly_activity, support_tickets, calendar):
    """Convert date strings to datetime objects.
    
    Args:
        customers, weekly_activity, support_tickets, calendar (DataFrames)
    
    Returns:
        All input DataFrames with date columns converted to datetime
    
    Converts: signup_date, churn_date, week_commencing, ticket_date, etc.
    """
```

**`standardize_flags(customers)`**
```python
def standardize_flags(customers):
    """Convert Y/N string flags to boolean.
    
    Args:
        customers (DataFrame): Contains churned, referral_flag, discount_applied_flag, menu_customised_flag
    
    Returns:
        customers (DataFrame): All flag columns converted to True/False
    """
```

**`fix_churn_date_anomalies(customers)`**
```python
def fix_churn_date_anomalies(customers):
    """Fix 12 records where churn_date < signup_date.
    
    Args:
        customers (DataFrame): Contains signup_date, churn_date
    
    Returns:
        customers (DataFrame): For 12 anomalous records, sets churn_date = signup_date
    
    Rationale: Assumes same-day churn for impossible cases
    """
```

**`validate_data(customers, weekly_activity, support_tickets)`**
```python
def validate_data(customers, weekly_activity, support_tickets):
    """Run 5 critical data quality checks.
    
    Checks:
        1. No duplicate customers
        2. No duplicate support tickets
        3. Referential integrity (all activity/ticket IDs exist in customer table)
        4. Churn dates >= signup dates (after fix)
        5. Churn rate in expected range (65-70%)
    
    Returns: Boolean (True if all checks pass)
    """
```

#### `src/shared/phase_0_2_models.py`

**`load_features(parquet_path)`**
```python
def load_features(parquet_path='data/processed/customer_features.parquet'):
    """Load engineered features from parquet file.
    
    Args:
        parquet_path (str): Path to customer_features.parquet
    
    Returns:
        features (DataFrame): 1,500 customers × 49 columns (47 features + customer_id + churned)
    """
```

**`create_train_test_split(features, test_size=0.2, random_state=42)`**
```python
def create_train_test_split(features, test_size=0.2, random_state=42):
    """Create stratified 80/20 split maintaining churn rate.
    
    Args:
        features (DataFrame): 1,500 customers with churned target
        test_size (float): 0.2 = 80/20 split
        random_state (int): 42 for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test (DataFrames/Series)
        Train: 1,200 customers (804 churned = 67.0%)
        Test: 300 customers (202 churned = 67.3%)
    """
```

**`build_heuristic_rule(X_test)`**
```python
def build_heuristic_rule(X_test):
    """Implement 3-signal heuristic baseline.
    
    Rule: FLAG if (late_avg_rating < 3.0) OR (frequency_trend < -0.3) OR (delivery_complaints >= 1)
    
    Args:
        X_test (DataFrame): Test features
    
    Returns:
        y_pred (array): Binary predictions (0 = not churn, 1 = churn)
    
    Result: F1 0.764, Precision 80.3%, Recall 72.8%
    """
```

**`calculate_metrics(y_true, y_pred, y_proba=None)`**
```python
def calculate_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive evaluation metrics.
    
    Returns:
        metrics (dict): Contains AUC-ROC, AUC-PR, Precision, Recall, F1, TP, FP, FN, TN
    """
```

### Phase 6: Logistic Regression

#### `src/phase_6/phase_6_1_logistic.py`

**`train_logistic_regression(X_train, X_test, y_train, y_test)`**
```python
def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train L1-regularised logistic regression.
    
    Configuration:
        - penalty='l1' (Lasso: automatic feature selection)
        - solver='liblinear' (required for L1)
        - class_weight='balanced' (handles 67% churn imbalance)
        - max_iter=1000 (ensures convergence)
    
    Returns:
        lr (LogisticRegression): Fitted model
        scaler (StandardScaler): Feature scaler (fitted on training data)
        metrics (dict): Test set metrics (AUC-ROC 0.999, F1 0.993)
        y_test_proba, y_test_pred (arrays): Probabilities and predictions on test set
    
    Performance: F1 0.993, AUC-ROC 0.999, Precision 99.5%, Recall 99.0%
    """
```

**`extract_coefficients(lr, X_train, top_n=10)`**
```python
def extract_coefficients(lr, X_train, top_n=10):
    """Extract and interpret top feature coefficients.
    
    Args:
        lr (LogisticRegression): Fitted model
        X_train (DataFrame): Feature names from training data
        top_n (int): Number of top features to return
    
    Returns:
        top_coefs (DataFrame): Feature names and coefficients (sorted by magnitude)
    
    Interpretation:
        - Negative coefficient = protective factor (higher value = lower churn risk)
        - Positive coefficient = risk factor (higher value = higher churn risk)
        - Example: total_orders (-5.52) = one additional order dramatically reduces churn
    """
```

**`plot_roc_and_pr_curves(y_test, y_test_proba)`**
```python
def plot_roc_and_pr_curves(y_test, y_test_proba):
    """Generate ROC and Precision-Recall curve visualisations.
    
    Args:
        y_test (Series): True churn labels
        y_test_proba (array): Predicted probabilities
    
    Outputs:
        - outputs/figures/logistic_curves.png (contains ROC + PR curves)
    """
```

### Phase 7: XGBoost

#### `src/phase_7/phase_7_1_xgboost_model.py`

**`train_xgboost(X_train, X_test, y_train, y_test)`**
```python
def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost gradient boosting model.
    
    Configuration:
        - max_depth=5 (tree complexity control)
        - learning_rate=0.1 (conservative step size)
        - n_estimators=200 (sequential trees)
        - subsample=0.8, colsample_bytree=0.8 (regularisation via sampling)
        - scale_pos_weight=2.0 (class imbalance handling)
        - tree_method='hist' (speed optimisation)
    
    Returns:
        xgb_model (XGBClassifier): Fitted model
        metrics (dict): Test set metrics (AUC-ROC 0.999, F1 0.990, Recall 100%)
        y_test_proba, y_test_pred (arrays): Probabilities and predictions
        X_test_numeric (DataFrame): Numeric features (for SHAP)
    
    Performance: F1 0.990, AUC-ROC 0.999, Precision 98.1%, Recall 100.0%
    Key difference: Perfect recall (catches ALL 202 churners, 0 false negatives)
    """
```

**`get_feature_importance(xgb_model, X_train, top_n=10)`**
```python
def get_feature_importance(xgb_model, X_train, top_n=10):
    """Extract gain-based feature importance.
    
    Args:
        xgb_model (XGBClassifier): Fitted model
        X_train (DataFrame): Feature names
        top_n (int): Number of top features
    
    Returns:
        importance_df (DataFrame): Feature names and importance scores
    
    Interpretation:
        - Gain = how much each feature reduces prediction error
        - Differs from logistic coefficients (captures non-linear relationships)
        - Example: late_order_freq (0.5892) = strongest non-linear signal
    """
```

**`generate_shap_explanations(xgb_model, X_test_numeric)`**
```python
def generate_shap_explanations(xgb_model, X_test_numeric):
    """Generate SHAP values for local and global interpretability.
    
    Args:
        xgb_model (XGBClassifier): Fitted model
        X_test_numeric (DataFrame): Test features (numeric only)
    
    Returns:
        shap_values (array): SHAP values for each feature
        explainer (TreeExplainer): SHAP explainer object
    
    Outputs:
        - outputs/figures/shap_summary_bar.png (global feature importance)
        - outputs/figures/shap_force_sample.png (local prediction explanation)
    
    SHAP provides Shapley Additive exPlanations:
        - Global: which features drive predictions across all customers
        - Local: why specific customer was flagged
    """
```

**`plot_xgboost_curves(y_test, y_test_proba)`**
```python
def plot_xgboost_curves(y_test, y_test_proba):
    """Generate ROC and Precision-Recall curves for XGBoost.
    
    Outputs: outputs/figures/xgboost_curves.png
    """
```

### Phase 8: Model Comparison

#### `src/phase_8/phase_8_1_model_comparison.py`

**`create_comprehensive_comparison(metrics_heuristic, metrics_logistic, metrics_xgboost)`**
```python
def create_comprehensive_comparison(metrics_heuristic, metrics_logistic, metrics_xgboost):
    """Create detailed comparison table across all three models.
    
    Args:
        metrics_* (dict): Metrics for heuristic, logistic, XGBoost
    
    Returns:
        df_comparison (DataFrame): Side-by-side comparison
            Rows: AUC-ROC, AUC-PR, Precision, Recall, F1, TP, FN, FP, TN
            Cols: Heuristic (baseline), Logistic (Phase 1), XGBoost (Phase 2)
    """
```

**`analyse_business_trade_offs()`**
```python
def analyse_business_trade_offs():
    """Analyse key business trade-offs.
    
    Outputs:
        - Interpretability vs Performance spectrum
        - Precision vs Recall in financial terms (GBP per false positive vs missed churner)
        - Cost/benefit analysis per model
    
    See [C_BUSINESS_IMPACT.md#business-impact-summary](C_BUSINESS_IMPACT.md#business-impact-summary)
    """
```

**`create_deployment_recommendation()`**
```python
def create_deployment_recommendation():
    """Generate two-phase deployment strategy.
    
    Phase 1: Deploy Logistic Regression
        - Rationale: F1 0.993, 99.5% precision, interpretable, fast
        - Timeline: Week 1 deployment
        - Expected Year 1 impact: GBP 300-450K revenue protected
    
    Phase 2: Upgrade to XGBoost (conditional)
        - Trigger: If Tier 1 save rate ≥ 20% after Month 1
        - Rationale: Perfect recall (catches ALL 202 churners)
        - Trade-off: 3 extra false positives vs logistic
    
    See [C_BUSINESS_IMPACT.md#the-rollout-4-week-timeline](C_BUSINESS_IMPACT.md#the-rollout-4-week-timeline)
    """
```

**`print_summary()`**
```python
def print_summary():
    """Print executive summary for stakeholders.
    
    Outputs: Console + formatted report
    """
```

---

## Reference Links

**For data preparation methodology and feature definitions, see:**
- [A_BASIS_OF_PREPARATION.md](A_BASIS_OF_PREPARATION.md) — Phases 2-4 (data cleaning, feature engineering), Phases 5-8 (model training, comparison)

**For project structure and file organisation, see:**
