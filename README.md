# FreshBox Churn Prediction: ML Pipeline Implementation

An end-to-end machine learning pipeline for predicting customer churn in a food delivery subscription service. Implements data preparation, feature engineering, baseline heuristic, logistic regression, and gradient boosting (XGBoost) models with SHAP explainability.

## 📁 Project Structure

```
freshbox-churn-prediction/
├── data/
│   ├── raw/                 # 4 raw CSV files (8,000 customers, 156K activity records, 2.1K support tickets)
│   └── processed/           # Engineered features (1,500 customers × 49 features, parquet format)
├── src/
│   ├── shared/              # Phase 0: Shared utilities (data loading, model training, metrics)
│   ├── phase_3/             # Data leakage prevention
│   ├── phase_4/             # Feature engineering
│   ├── phase_6/             # Logistic regression
│   ├── phase_7/             # XGBoost + SHAP
│   └── phase_8/             # Model comparison & evaluation
├── test/
│   ├── phase_2/             # Data validation & cleaning
│   ├── phase_3/             # Leakage prevention validation
│   ├── phase_4/             # Feature engineering execution
│   ├── phase_5/             # Train/test split & baseline heuristic
│   ├── phase_6/             # Logistic regression training & evaluation
│   ├── phase_7/             # XGBoost training & SHAP generation
│   ├── phase_8/             # Model comparison & output
│   └── phase_op/            # Operational framework execution
├── outputs/
│   ├── data/                # 21 CSV exports (data lineage at each phase)
│   ├── figures/             # 4 PNG visualizations (ROC, PR, SHAP curves)
│   └── models/              # (Placeholder for serialized models)
├── scripts/
│   ├── phase_9_1_create_presentation.py  # Presentation deck generation
│   └── investigate_churn_dates.py        # Data investigation utility
├── docs/
│   ├── A_BASIS_OF_PREPARATION.md  # Data preparation methodology (Phases 2-8)
│   ├── B_TECHNICAL_IMPLEMENTATION.md  # Technical details (models, hyperparameters, architecture)
│   └── C_BUSINESS_IMPACT.md  # Business case, ROI, operational framework, KPIs
├── deck/                    # 3 PowerPoint presentations
├── requirements.txt         # Python package dependencies
└── .venv/                   # Python virtual environment (gitignored)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (`.venv/`)
- Dependencies: pandas, scikit-learn, xgboost, shap, matplotlib, seaborn

### Installation
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Execute Pipeline
Run test scripts in order to execute the full pipeline:

```bash
# Phase 2: Data Loading & Validation
python test/phase_2/phase_2_1_test_data_loading.py

# Phase 3: Data Leakage Prevention
python test/phase_3/phase_3_1_test_customer_base.py

# Phase 4: Feature Engineering
python test/phase_4/phase_4_1_test_features.py

# Phase 5: Train/Test Split & Heuristic Baseline
python test/phase_5/phase_5_1_test_baseline.py

# Phase 6: Logistic Regression
python test/phase_6/phase_6_1_test_logistic.py

# Phase 7: XGBoost with SHAP Explanations
python test/phase_7/phase_7_1_test_xgboost.py

# Phase 8: Model Comparison & Evaluation
python test/phase_8/phase_8_1_test_model_comparison.py
```

**Expected runtime:** ~2 minutes for complete pipeline

## 📊 Key Results

| Model | Precision | Recall | F1 Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| **Heuristic Baseline** | 80.3% | 72.8% | 0.764 | N/A |
| **Logistic Regression** | 99.5% | 99.0% | **0.993** | 0.999 |
| **XGBoost** | 98.1% | **100.0%** | 0.990 | 0.999 |

## 📈 Data Lineage

All phases generate CSV exports for transparency and auditability:

- **Phase 2:** Data cleaning (fixed 12 anomalies, standardised dates/flags)
- **Phase 3:** Leakage prevention (removed 224 post-churn records)
- **Phase 4:** Feature engineering (47 features across 6 categories)
- **Phase 5:** Train/test split (1,200 train, 300 test) + heuristic baseline
- **Phase 6:** Logistic regression predictions & coefficients
- **Phase 7:** XGBoost predictions & SHAP feature importance
- **Phase 8:** Model comparison metrics (all models side-by-side)

Access all exports in `outputs/data/`.

## 🔍 Model Architectures

### Phase 6: Logistic Regression
**Configuration:**
- Penalty: L1 (Lasso) — automatic feature selection
- Solver: liblinear (required for L1)
- Class weight: balanced (handles 67% churn imbalance)
- Max iterations: 1000

**Performance:** F1 0.993, <1ms inference per customer, <50KB model size

**Use case:** Phase 1 deployment (interpretable, auditable, fast)

### Phase 7: XGBoost Gradient Boosting
**Configuration:**
- Max depth: 5 (tree complexity control)
- Learning rate: 0.1 (conservative step size)
- N estimators: 200 (sequential trees)
- Subsample: 0.8 (row sampling per tree)
- Colsample bytree: 0.8 (feature sampling per tree)

**Performance:** F1 0.990, 100% recall (zero missed churners), 10-50ms inference per customer

**Use case:** Phase 2 conditional upgrade (perfect recall, non-linear patterns)

## 📚 Documentation

- **[A_BASIS_OF_PREPARATION.md](docs/A_BASIS_OF_PREPARATION.md)** — Data quality assurance, feature definitions, Phases 2-8 technical details
- **[B_TECHNICAL_IMPLEMENTATION.md](docs/B_TECHNICAL_IMPLEMENTATION.md)** — Architecture, hyperparameter rationale, model inference, monitoring strategy
- **[C_BUSINESS_IMPACT.md](docs/C_BUSINESS_IMPACT.md)** — Business case, ROI projections, operational framework, customer segmentation, KPIs

## 🎯 Features Engineered

47 features across 4 churn hypotheses:

| Hypothesis | Count | Examples |
|-----------|-------|----------|
| **Loyalty** | 4 | tenure_weeks, total_orders, order_completion_rate |
| **Momentum** | 6 | frequency_trend, late_avg_rating, rating_trend |
| **Satisfaction** | 5 | avg_recipe_rating, rating_engagement_rate, menu_customisation_rate |
| **Friction** | 4 | total_tickets, delivery_complaints, avg_resolution_days |
| **Economic** | 4 | discount_dependency_rate, avg_discount_pct |
| **Demographics** | 8 | acquisition_channel, age_band, region, plan_type (one-hot encoded) |

All features derived from FreshBox's existing data — no new instrumentation required.

## 🔄 Data Formats

- **Raw Input:** CSV files (text, human-readable)
- **Processed Features:** Parquet (binary, compressed, schema-preserving) — 76% smaller than CSV equivalent
- **CSV Exports:** For transparency and auditing at each phase

## ⚙️ Model Monitoring

**Retraining:** Monthly on rolling 6-month window

**Drift Thresholds:**
- Alert if AUC-ROC drops below 0.95
- Alert if Recall drops below 95%
- Alert if Precision drops below 90%

**Fallback:** Heuristic baseline rule (F1 0.764) if models unavailable

## 📖 For More Information

Refer to the documentation files in `docs/` for:
- Detailed data preparation methodology
- Feature engineering rationale
- Hyperparameter justification
- Model evaluation results
- Monitoring & governance strategy

---

**Author:** Created for technical interview & code review  
**Last Updated:** April 2025  
**Data:** FreshBox (anonymized customer dataset)
