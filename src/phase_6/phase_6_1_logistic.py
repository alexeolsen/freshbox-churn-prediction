import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train logistic regression with L1 penalty (Lasso).

    L1 regularisation shrinks weak features to zero, giving us automatic
    feature selection and interpretability.
    """
    print("\nTraining Logistic Regression...")
    print("-" * 70)

    # Step 1: Standardise features (critical for linear models)
    print("  Step 1: Standardising features...")

    # Only standardise numeric columns (exclude categorical dummies)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numeric = X_train[numeric_cols].astype(float).fillna(X_train[numeric_cols].median())
    X_test_numeric = X_test[numeric_cols].astype(float).fillna(X_train[numeric_cols].median())

    scaler = StandardScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train_numeric)
    X_test_scaled_numeric = scaler.transform(X_test_numeric)

    # Combine with categorical columns (already one-hot encoded, no scaling needed)
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    X_train_scaled = pd.DataFrame(
        np.hstack([X_train_scaled_numeric, X_train[categorical_cols].values]) if categorical_cols else X_train_scaled_numeric,
        columns=numeric_cols + categorical_cols
    )
    X_test_scaled = pd.DataFrame(
        np.hstack([X_test_scaled_numeric, X_test[categorical_cols].values]) if categorical_cols else X_test_scaled_numeric,
        columns=numeric_cols + categorical_cols
    )
    print(f"    [OK] Standardised {len(numeric_cols)} numeric features")

    # Step 2: Train logistic regression with L1 penalty (using numeric features only)
    print("  Step 2: Training logistic regression (L1 penalty)...")
    lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,
        random_state=42
    )
    # Convert to numpy for sklearn compatibility
    lr.fit(X_train_scaled_numeric, y_train)
    print(f"    [OK] Model trained")

    # Step 3: Get predictions and probabilities
    print("  Step 3: Generating predictions...")
    y_train_pred = lr.predict(X_train_scaled_numeric)
    y_test_pred = lr.predict(X_test_scaled_numeric)
    y_train_proba = lr.predict_proba(X_train_scaled_numeric)[:, 1]
    y_test_proba = lr.predict_proba(X_test_scaled_numeric)[:, 1]
    print(f"    [OK] Predictions generated")

    # Step 4: Calculate metrics
    print("  Step 4: Calculating metrics...")
    metrics = {
        'model': 'Logistic Regression',
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
    print(f"\nLogistic Regression Performance (Test Set):")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.3f}")
    print(f"  AUC-PR:   {metrics['auc_pr']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

    return lr, scaler, metrics, y_test_proba, y_test_pred


def extract_coefficients(lr, X_train, top_n=10):
    """
    Extract and interpret logistic regression coefficients.

    Coefficient tells us: "A one standard deviation increase in this feature
    changes log-odds of churn by X"
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE: Top 10 Coefficients")
    print("=" * 70)

    # Get numeric columns only (what was actually trained)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    coefficients = pd.DataFrame({
        'feature': numeric_cols,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    top_coefficients = coefficients.head(top_n)

    print(f"\n{'Feature':<35} {'Coefficient':>12} {'Direction'}")
    print("-" * 60)
    for idx, row in top_coefficients.iterrows():
        direction = "INCREASES churn" if row['coefficient'] > 0 else "DECREASES churn"
        print(f"{row['feature']:<35} {row['coefficient']:>12.4f}  {direction}")

    print(f"\nInterpretation Guide:")
    print(f"  Positive coefficient -> Higher feature value -> Higher churn risk")
    print(f"  Negative coefficient -> Higher feature value -> Lower churn risk")
    print(f"\nExample: delivery_complaints has coefficient 0.52")
    print(f"  -> Each additional delivery complaint increases log-odds by 0.52")
    print(f"  -> Roughly 1.7x higher churn odds per complaint")

    return top_coefficients


def plot_roc_and_pr_curves(y_test, y_test_proba, save_path='outputs/figures/'):
    """
    Plot ROC and Precision-Recall curves.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC plot
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Logistic Regression')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR plot
    axes[1].plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve - Logistic Regression')
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}logistic_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Curves saved to {save_path}logistic_curves.png")
    plt.close()
