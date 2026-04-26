#!/usr/bin/env python
"""
Phase 7: XGBoost with SHAP Explainability
Train gradient boosting model and generate local + global explanations.
Compare non-linear model to logistic regression baseline.
"""

from src.shared.phase_0_2_models import load_features, create_train_test_split, evaluate_heuristic
from src.phase_6.phase_6_1_logistic import train_logistic_regression
from src.phase_7.phase_7_1_xgboost_model import train_xgboost, get_feature_importance, generate_shap_explanations, plot_xgboost_curves
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 7: XGBoost with SHAP Explainability\n")

    # Step 1: Load and split
    print("Step 1/5: Loading features and creating train/test split...")
    features = load_features('data/processed/customer_features.parquet')
    X_train, X_test, y_train, y_test = create_train_test_split(features)
    print(f"  [OK] Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 2: Train logistic regression (for comparison)
    print("\nStep 2/5: Training logistic regression (for comparison)...")
    lr, scaler, metrics_lr, y_test_proba_lr, y_test_pred_lr = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # Step 3: Train XGBoost
    print("\nStep 3/5: Training XGBoost model...")
    xgb_model, metrics_xgb, y_test_proba_xgb, y_test_pred_xgb, X_test_numeric = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    # Step 4: Feature importance
    print("\nStep 4/5: Extracting feature importance and SHAP explanations...")
    top_features_xgb = get_feature_importance(xgb_model, X_train, top_n=10)
    shap_values, explainer = generate_shap_explanations(xgb_model, X_test_numeric)
    plot_xgboost_curves(y_test, y_test_proba_xgb)

    # Step 5: Model comparison
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION vs XGBOOST")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Logistic':<15} {'XGBoost':<15} {'Improvement':<15}")
    print("-" * 70)

    auc_roc_improvement = metrics_xgb['auc_roc'] - metrics_lr['auc_roc']
    print(f"{'AUC-ROC':<20} {metrics_lr['auc_roc']:<15.3f} {metrics_xgb['auc_roc']:<15.3f} {auc_roc_improvement:+.3f}")

    auc_pr_improvement = metrics_xgb['auc_pr'] - metrics_lr['auc_pr']
    print(f"{'AUC-PR':<20} {metrics_lr['auc_pr']:<15.3f} {metrics_xgb['auc_pr']:<15.3f} {auc_pr_improvement:+.3f}")

    precision_improvement = metrics_xgb['precision'] - metrics_lr['precision']
    print(f"{'Precision':<20} {metrics_lr['precision']:<15.3f} {metrics_xgb['precision']:<15.3f} {precision_improvement:+.3f}")

    recall_improvement = metrics_xgb['recall'] - metrics_lr['recall']
    print(f"{'Recall':<20} {metrics_lr['recall']:<15.3f} {metrics_xgb['recall']:<15.3f} {recall_improvement:+.3f}")

    f1_improvement = metrics_xgb['f1'] - metrics_lr['f1']
    print(f"{'F1 Score':<20} {metrics_lr['f1']:<15.3f} {metrics_xgb['f1']:<15.3f} {f1_improvement:+.3f}")

    # Confusion matrices
    print(f"\n" + "-" * 70)
    print("CONFUSION MATRIX COMPARISON")
    print("-" * 70)

    print(f"\nLogistic Regression:")
    print(f"  True Negatives:  {int(metrics_lr['tn']):4d}")
    print(f"  False Positives: {int(metrics_lr['fp']):4d}")
    print(f"  False Negatives: {int(metrics_lr['fn']):4d}")
    print(f"  True Positives:  {int(metrics_lr['tp']):4d}")

    print(f"\nXGBoost:")
    print(f"  True Negatives:  {int(metrics_xgb['tn']):4d}")
    print(f"  False Positives: {int(metrics_xgb['fp']):4d}")
    print(f"  False Negatives: {int(metrics_xgb['fn']):4d}")
    print(f"  True Positives:  {int(metrics_xgb['tp']):4d}")

    # Summary
    print("\n[SUCCESS] Phase 7 complete!")
    print("\nKey Findings:")

    if metrics_xgb['f1'] > metrics_lr['f1']:
        improvement_pct = ((metrics_xgb['f1'] - metrics_lr['f1']) / metrics_lr['f1']) * 100
        print(f"  [OK] XGBoost outperforms logistic regression (F1 +{improvement_pct:.1f}%)")
    else:
        print(f"  [!] Logistic regression comparable to XGBoost")

    if metrics_xgb['recall'] >= 0.98:
        print(f"  [OK] XGBoost catches {metrics_xgb['recall']*100:.1f}% of churners (excellent recall)")
    elif metrics_xgb['recall'] >= 0.90:
        print(f"  [OK] XGBoost catches {metrics_xgb['recall']*100:.1f}% of churners (strong recall)")

    print(f"\n  SHAP explanations enable local interpretation of individual predictions")
    print(f"  Feature importance highlights non-linear relationships missed by linear model")

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save test predictions
    test_predictions = X_test.copy()
    test_predictions['actual_churn'] = y_test.values
    # Handle both 1D and 2D probability arrays
    if y_test_proba_xgb.ndim == 2:
        test_predictions['xgboost_probability'] = y_test_proba_xgb[:, 1]
    else:
        test_predictions['xgboost_probability'] = y_test_proba_xgb
    test_predictions['xgboost_prediction'] = y_test_pred_xgb
    test_predictions.to_csv(output_dir / "phase_7_xgboost_predictions.csv", index=False)
    print(f"  [OK] phase_7_xgboost_predictions.csv ({len(test_predictions):,} rows)")

    # Save feature importance
    # Get feature names - only use as many as there are importances
    feature_names = X_train.columns[:len(xgb_model.feature_importances_)]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(output_dir / "phase_7_xgboost_feature_importance.csv", index=False)
    print(f"  [OK] phase_7_xgboost_feature_importance.csv ({len(importance_df)} features)")

    # Save metrics comparison
    metrics_comp = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR'],
        'Logistic': [metrics_lr['precision'], metrics_lr['recall'], metrics_lr['f1'], metrics_lr['auc_roc'], metrics_lr['auc_pr']],
        'XGBoost': [metrics_xgb['precision'], metrics_xgb['recall'], metrics_xgb['f1'], metrics_xgb['auc_roc'], metrics_xgb['auc_pr']]
    })
    metrics_comp.to_csv(output_dir / "phase_7_logistic_vs_xgboost.csv", index=False)
    print(f"  [OK] phase_7_logistic_vs_xgboost.csv")

    print("\nNext: Phase 8 - Model Comparison & Business Recommendation\n")
