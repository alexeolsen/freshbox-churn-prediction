import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score


def load_features(path):
    """Load engineered feature table from parquet."""
    return pd.read_parquet(path)


def create_train_test_split(features, test_size=0.2, random_state=42):
    """
    Create stratified train/test split.
    Stratified ensures both sets have same churn rate as overall data.
    """
    X = features.drop(['customer_id', 'churned'], axis=1)
    y = features['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratified split
    )

    return X_train, X_test, y_train, y_test


def build_heuristic_rule(X_test, y_test):
    """
    Build the obvious rule a smart retention manager would use.

    Churn Risk = Low recent ratings OR declining frequency OR complaints

    This is the baseline all ML models must beat.
    """
    predictions = (
        (X_test['late_avg_rating'] < 3.0) |  # Low recent satisfaction
        (X_test['frequency_trend'] < -0.3) |  # Declining order frequency
        (X_test['delivery_complaints'] >= 1)   # Had delivery problems
    ).astype(int)

    return predictions


def calculate_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics.
    """
    metrics = {
        'model': model_name,
        'auc_roc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else np.nan,
        'auc_pr': average_precision_score(y_true, y_pred_proba) if y_pred_proba is not None else np.nan,
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Additional metrics at top 20% threshold
    # (used for practical decision-making: "focus on top 20% of scored customers")
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp

    return metrics


def evaluate_heuristic(X_test, y_test):
    """
    Score the heuristic rule and calculate metrics.
    """
    print("\nEvaluating Heuristic Rule...")
    print("-" * 70)

    y_pred = build_heuristic_rule(X_test, y_test)
    metrics = calculate_metrics(y_test, y_pred, model_name="Heuristic Rule")

    print(f"\nHeuristic Rule Performance:")
    print(f"  Precision: {metrics['precision']:.3f}  (of flagged customers, how many actually churn?)")
    print(f"  Recall:    {metrics['recall']:.3f}  (of actual churners, how many do we catch?)")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

    # Confusion matrix interpretation
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {int(metrics['tn']):4d}  (correctly identified as staying)")
    print(f"  False Positives: {int(metrics['fp']):4d}  (incorrectly flagged as churning)")
    print(f"  False Negatives: {int(metrics['fn']):4d}  (missed churners)")
    print(f"  True Positives:  {int(metrics['tp']):4d}  (correctly identified as churning)")

    return y_pred, metrics


def print_baseline_summary(metrics):
    """
    Print interpretation of baseline performance.
    """
    print("\n" + "=" * 70)
    print("HEURISTIC BASELINE SUMMARY")
    print("=" * 70)

    print(f"\nInterpretation:")
    print(f"  - Of every 100 customers flagged as at-risk:")
    print(f"    {metrics['precision']*100:.0f} will actually churn")
    print(f"\n  - Of all customers who will churn:")
    print(f"    {metrics['recall']*100:.0f}% are caught by this rule")

    print(f"\nWhat this means:")
    if metrics['precision'] > 0.7:
        print(f"  [OK] Rule is fairly precise (low false alarm rate)")
    else:
        print(f"  [!] Rule flags many false positives (wastes retention budget)")

    if metrics['recall'] > 0.6:
        print(f"  [OK] Rule catches most of the churners")
    else:
        print(f"  [!] Rule misses many churners (incomplete coverage)")

    print(f"\nML models must beat these numbers to justify added complexity.")
    print("=" * 70 + "\n")
