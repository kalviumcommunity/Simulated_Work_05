import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model object
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        threshold (float): Classification threshold for probabilities
        
    Returns:
        Dict[str, Any]: Evaluation metrics and results
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_threshold = (y_proba >= threshold).astype(int)
    else:
        y_proba = None
        y_pred_threshold = y_pred
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Calculate metrics with threshold
    accuracy_thresh = accuracy_score(y_test, y_pred_threshold)
    precision_thresh = precision_score(y_test, y_pred_threshold, average='binary')
    recall_thresh = recall_score(y_test, y_pred_threshold, average='binary')
    f1_thresh = f1_score(y_test, y_pred_threshold, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC if probabilities are available
    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Compile results
    evaluation_results = {
        'basic_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'threshold_metrics': {
            'threshold': threshold,
            'accuracy': float(accuracy_thresh),
            'precision': float(precision_thresh),
            'recall': float(recall_thresh),
            'f1_score': float(f1_thresh)
        },
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'classification_report': class_report,
        'test_samples': len(y_test),
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist() if y_proba is not None else None
    }
    
    return evaluation_results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print evaluation results in a formatted way.
    
    Args:
        results (Dict): Evaluation results dictionary
    """
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    # Basic metrics
    basic = results['basic_metrics']
    print(f"Accuracy:  {basic['accuracy']:.4f}")
    print(f"Precision: {basic['precision']:.4f}")
    print(f"Recall:    {basic['recall']:.4f}")
    print(f"F1-Score:  {basic['f1_score']:.4f}")
    
    # Threshold metrics
    thresh = results['threshold_metrics']
    print(f"\nThreshold ({thresh['threshold']:.2f}) Metrics:")
    print(f"Accuracy:  {thresh['accuracy']:.4f}")
    print(f"Precision: {thresh['precision']:.4f}")
    print(f"Recall:    {thresh['recall']:.4f}")
    print(f"F1-Score:  {thresh['f1_score']:.4f}")
    
    # ROC AUC
    if results['roc_auc'] is not None:
        print(f"ROC AUC:   {results['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"True Negative:  {cm[0][0]}")
    print(f"False Positive: {cm[0][1]}")
    print(f"False Negative: {cm[1][0]}")
    print(f"True Positive:  {cm[1][1]}")
    
    print("=" * 50)


def calculate_business_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate business-relevant metrics for spam detection.
    
    Args:
        results (Dict): Evaluation results
        
    Returns:
        Dict[str, Any]: Business metrics
    """
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    total = tn + fp + fn + tp
    
    # Business metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Cost assumptions (can be customized)
    cost_fp = 1.0  # Cost of marking legitimate email as spam
    cost_fn = 10.0  # Cost of missing spam email
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    
    business_metrics = {
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'total_cost': float(total_cost),
        'cost_per_email': float(total_cost / total) if total > 0 else 0,
        'spam_caught_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'legitimate_preserved_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0
    }
    
    return business_metrics


def compare_models(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple models and return the best one.
    
    Args:
        model_results (Dict): Dictionary of model results
        
    Returns:
        Dict[str, Any]: Comparison results and best model
    """
    comparison = {}
    
    for model_name, results in model_results.items():
        basic = results['basic_metrics']
        comparison[model_name] = {
            'accuracy': basic['accuracy'],
            'precision': basic['precision'],
            'recall': basic['recall'],
            'f1_score': basic['f1_score'],
            'roc_auc': results.get('roc_auc', 0)
        }
    
    # Find best model based on F1-score
    best_model = max(comparison.keys(), key=lambda x: comparison[x]['f1_score'])
    
    return {
        'comparison': comparison,
        'best_model': best_model,
        'best_metrics': comparison[best_model]
    }
