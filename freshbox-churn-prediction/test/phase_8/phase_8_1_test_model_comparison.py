#!/usr/bin/env python
"""
Phase 8: Model Comparison & Business Recommendation
Compare heuristic, logistic, and XGBoost models.
Make final recommendation for FreshBox deployment.
"""

from src.shared.phase_0_2_models import load_features, create_train_test_split, evaluate_heuristic
from src.phase_6.phase_6_1_logistic import train_logistic_regression
from src.phase_7.phase_7_1_xgboost_model import train_xgboost
from src.phase_8.phase_8_1_model_comparison import create_comprehensive_comparison, analyse_business_trade_offs, create_deployment_recommendation, print_summary
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 8: Model Comparison & Business Recommendation\n")

    # Step 1: Load and split
    print("Step 1/4: Loading features and creating train/test split...")
    features = load_features('data/processed/customer_features.parquet')
    X_train, X_test, y_train, y_test = create_train_test_split(features)
    print(f"  [OK] Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 2: Evaluate all three models
    print("\nStep 2/4: Evaluating all models...")

    print("\n  [1] Heuristic baseline...")
    y_pred_heuristic, metrics_heuristic = evaluate_heuristic(X_test, y_test)

    print("\n  [2] Logistic regression...")
    lr, scaler, metrics_logistic, y_test_proba_lr, y_test_pred_lr = train_logistic_regression(X_train, X_test, y_train, y_test)

    print("\n  [3] XGBoost...")
    xgb_model, metrics_xgboost, y_test_proba_xgb, y_test_pred_xgb, _ = train_xgboost(X_train, X_test, y_train, y_test)

    # Step 3: Comprehensive comparison
    print("\nStep 3/4: Creating comprehensive comparison...")
    df_comparison = create_comprehensive_comparison(metrics_heuristic, metrics_logistic, metrics_xgboost)

    # Step 4: Trade-offs and recommendation
    print("\nStep 4/4: Analysing trade-offs and generating recommendation...")
    analyse_business_trade_offs()

    create_deployment_recommendation()

    # Executive summary
    print_summary()

    print("\n[SUCCESS] Phase 8 complete!")

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive comparison
    df_comparison.to_csv(output_dir / "phase_8_all_models_comparison.csv", index=False)
    print(f"  [OK] phase_8_all_models_comparison.csv")

    # Save all predictions side-by-side
    all_predictions = X_test.copy()
    all_predictions['actual_churn'] = y_test.values
    all_predictions['heuristic_prediction'] = y_pred_heuristic
    # Handle both 1D and 2D probability arrays
    if y_test_proba_lr.ndim == 2:
        all_predictions['logistic_probability'] = y_test_proba_lr[:, 1]
    else:
        all_predictions['logistic_probability'] = y_test_proba_lr
    all_predictions['logistic_prediction'] = y_test_pred_lr
    if y_test_proba_xgb.ndim == 2:
        all_predictions['xgboost_probability'] = y_test_proba_xgb[:, 1]
    else:
        all_predictions['xgboost_probability'] = y_test_proba_xgb
    all_predictions['xgboost_prediction'] = y_test_pred_xgb
    all_predictions.to_csv(output_dir / "phase_8_all_models_predictions.csv", index=False)
    print(f"  [OK] phase_8_all_models_predictions.csv ({len(all_predictions):,} rows)")

    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'Model': ['Heuristic', 'Logistic', 'XGBoost'],
        'AUC-ROC': [0.5, metrics_logistic['auc_roc'], metrics_xgboost['auc_roc']],
        'AUC-PR': [0.5, metrics_logistic['auc_pr'], metrics_xgboost['auc_pr']],
        'Precision': [metrics_heuristic['precision'], metrics_logistic['precision'], metrics_xgboost['precision']],
        'Recall': [metrics_heuristic['recall'], metrics_logistic['recall'], metrics_xgboost['recall']],
        'F1 Score': [metrics_heuristic['f1'], metrics_logistic['f1'], metrics_xgboost['f1']],
        'TP': [metrics_heuristic['tp'], metrics_logistic['tp'], metrics_xgboost['tp']],
        'FP': [metrics_heuristic['fp'], metrics_logistic['fp'], metrics_xgboost['fp']],
        'FN': [metrics_heuristic['fn'], metrics_logistic['fn'], metrics_xgboost['fn']],
        'TN': [metrics_heuristic['tn'], metrics_logistic['tn'], metrics_xgboost['tn']]
    })
    metrics_summary.to_csv(output_dir / "phase_8_model_metrics_summary.csv", index=False)
    print(f"  [OK] phase_8_model_metrics_summary.csv")

    print("\nNext: Phase 9 - Create presentation deck for stakeholders\n")
