#!/usr/bin/env python
"""
Phase 6: Logistic Regression
Train interpretable baseline ML model and compare to heuristic.
"""

from src.shared.phase_0_2_models import load_features, create_train_test_split, evaluate_heuristic
from src.phase_6.phase_6_1_logistic import train_logistic_regression, extract_coefficients, plot_roc_and_pr_curves
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("\n[*] PHASE 6: Logistic Regression\n")

    # Step 1: Load and split
    print("Step 1/4: Loading features and creating train/test split...")
    features = load_features('data/processed/customer_features.parquet')
    X_train, X_test, y_train, y_test = create_train_test_split(features)
    print(f"  [OK] Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 2: Heuristic baseline for comparison
    print("\nStep 2/4: Getting heuristic baseline scores...")
    y_pred_heuristic, metrics_heuristic = evaluate_heuristic(X_test, y_test)

    # Step 3: Train logistic regression
    print("\nStep 3/4: Training logistic regression...")
    lr, scaler, metrics_lr, y_test_proba, y_test_pred = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # Step 4: Extract coefficients and plot curves
    print("\nStep 4/4: Extracting feature importance and plotting curves...")
    top_coefs = extract_coefficients(lr, X_train, top_n=10)
    plot_roc_and_pr_curves(y_test, y_test_proba)

    # Comparison
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION vs HEURISTIC BASELINE")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Heuristic':<15} {'Logistic':<15} {'Improvement':<15}")
    print("-" * 70)

    auc_roc_lr = metrics_lr['auc_roc']
    auc_roc_heuristic = 0.5  # Heuristic doesn't have probabilities
    print(f"{'AUC-ROC':<20} {'N/A':<15} {auc_roc_lr:<15.3f} {'+':<15}")

    precision_improvement = metrics_lr['precision'] - metrics_heuristic['precision']
    print(f"{'Precision':<20} {metrics_heuristic['precision']:<15.3f} {metrics_lr['precision']:<15.3f} {precision_improvement:+.3f}")

    recall_improvement = metrics_lr['recall'] - metrics_heuristic['recall']
    print(f"{'Recall':<20} {metrics_heuristic['recall']:<15.3f} {metrics_lr['recall']:<15.3f} {recall_improvement:+.3f}")

    f1_improvement = metrics_lr['f1'] - metrics_heuristic['f1']
    print(f"{'F1 Score':<20} {metrics_heuristic['f1']:<15.3f} {metrics_lr['f1']:<15.3f} {f1_improvement:+.3f}")

    print("\n[SUCCESS] Phase 6 complete!")
    print("\nKey Finding:")
    if auc_roc_lr > 0.75:
        print(f"  [OK] Logistic regression achieves strong discrimination (AUC-ROC = {auc_roc_lr:.3f})")
    else:
        print(f"  [!] Modest improvement from logistic regression (AUC-ROC = {auc_roc_lr:.3f})")

    # ====== CSV DUMPS ======
    print("\n[*] Saving data exports to outputs/data/...\n")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save test predictions
    test_predictions = X_test.copy()
    test_predictions['actual_churn'] = y_test.values
    # Handle both 1D and 2D probability arrays
    if y_test_proba.ndim == 2:
        test_predictions['logistic_probability'] = y_test_proba[:, 1]
    else:
        test_predictions['logistic_probability'] = y_test_proba
    test_predictions['logistic_prediction'] = y_test_pred
    test_predictions.to_csv(output_dir / "phase_6_logistic_predictions.csv", index=False)
    print(f"  [OK] phase_6_logistic_predictions.csv ({len(test_predictions):,} rows)")

    # Save feature coefficients
    # Get feature names - only use as many as there are coefficients
    feature_names = X_train.columns[:len(lr.coef_[0])]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    coef_df.to_csv(output_dir / "phase_6_logistic_coefficients.csv", index=False)
    print(f"  [OK] phase_6_logistic_coefficients.csv ({len(coef_df)} features)")

    # Save metrics comparison
    metrics_comp = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
        'Heuristic': [metrics_heuristic['precision'], metrics_heuristic['recall'], metrics_heuristic['f1'], 0.5],
        'Logistic': [metrics_lr['precision'], metrics_lr['recall'], metrics_lr['f1'], metrics_lr['auc_roc']]
    })
    metrics_comp.to_csv(output_dir / "phase_6_logistic_vs_heuristic.csv", index=False)
    print(f"  [OK] phase_6_logistic_vs_heuristic.csv")

    print("\nNext: Phase 7 - XGBoost with SHAP Explainability\n")
