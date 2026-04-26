import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model for churn prediction.

    XGBoost captures non-linear relationships and feature interactions that
    linear models miss. Hyperparameters tuned for recall (catch more churners)
    while maintaining precision.
    """
    print("\nTraining XGBoost...")
    print("-" * 70)

    # Step 1: Prepare features (no standardization needed for tree models)
    print("  Step 1: Preparing features for XGBoost...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numeric = X_train[numeric_cols].astype(float).fillna(X_train[numeric_cols].median())
    X_test_numeric = X_test[numeric_cols].astype(float).fillna(X_train[numeric_cols].median())
    print(f"    [OK] Using {len(numeric_cols)} numeric features")

    # Step 2: Train XGBoost
    print("  Step 2: Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )
    xgb_model.fit(
        X_train_numeric, y_train,
        eval_set=[(X_test_numeric, y_test)],
        verbose=False
    )
    print(f"    [OK] Model trained with {xgb_model.n_estimators} trees")

    # Step 3: Generate predictions
    print("  Step 3: Generating predictions...")
    y_train_pred = xgb_model.predict(X_train_numeric)
    y_test_pred = xgb_model.predict(X_test_numeric)
    y_train_proba = xgb_model.predict_proba(X_train_numeric)[:, 1]
    y_test_proba = xgb_model.predict_proba(X_test_numeric)[:, 1]
    print(f"    [OK] Predictions generated")

    # Step 4: Calculate metrics
    print("  Step 4: Calculating metrics...")
    metrics = {
        'model': 'XGBoost',
        'auc_roc': roc_auc_score(y_test, y_test_proba),
        'auc_pr': average_precision_score(y_test, y_test_proba),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp

    print(f"    [OK] Metrics calculated")

    # Print performance
    print(f"\nXGBoost Performance (Test Set):")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.3f}")
    print(f"  AUC-PR:   {metrics['auc_pr']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

    return xgb_model, metrics, y_test_proba, y_test_pred, X_test_numeric


def get_feature_importance(xgb_model, X_train, top_n=10):
    """
    Extract XGBoost feature importance (gain-based).
    Gain measures the average improvement in accuracy brought by each feature.
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE: XGBoost Gain-Based Scores")
    print("=" * 70)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    top_features = feature_importance_df.head(top_n)

    print(f"\n{'Feature':<35} {'Importance Score':>18}")
    print("-" * 60)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:<35} {row['importance']:>18.4f}")

    print(f"\nInterpretation:")
    print(f"  Importance measures how much each feature reduces prediction error")
    print(f"  Higher scores indicate features that XGBoost relies on most")
    print(f"  Differs from logistic coefficients: captures non-linear relationships")

    return top_features


def generate_shap_explanations(xgb_model, X_test, save_path='outputs/figures/'):
    """
    Generate SHAP (SHapley Additive exPlanations) visualisations.

    SHAP values show the contribution of each feature to individual predictions.
    Provides both global (feature importance) and local (individual prediction) explanations.
    """
    print("\n" + "=" * 70)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 70)

    print("\n  Creating SHAP explainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    print("    [OK] SHAP values calculated")

    # Create visualisations
    print("  Creating SHAP summary plot...")

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{save_path}shap_summary_bar.png', dpi=150, bbox_inches='tight')
    print(f"    [OK] SHAP summary plot saved to {save_path}shap_summary_bar.png")
    plt.close()

    print("  Creating SHAP dependence plots...")

    # Create force plot for first prediction
    fig, ax = plt.subplots(figsize=(14, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{save_path}shap_force_sample.png', dpi=150, bbox_inches='tight')
    print(f"    [OK] SHAP force plot saved to {save_path}shap_force_sample.png")
    plt.close()

    print(f"\n[OK] SHAP explanations generated successfully")

    return shap_values, explainer


def plot_xgboost_curves(y_test, y_test_proba, save_path='outputs/figures/'):
    """
    Plot ROC and Precision-Recall curves for XGBoost.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC plot
    axes[0].plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - XGBoost')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR plot
    axes[1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve - XGBoost')
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}xgboost_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Curves saved to {save_path}xgboost_curves.png")
    plt.close()
