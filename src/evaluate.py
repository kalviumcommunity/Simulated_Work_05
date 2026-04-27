"""
Evaluation module for spam email detection.
Computes comprehensive metrics on held-out data.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute basic classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        average (str): Averaging method for multi-class
        
    Returns:
        Dict[str, float]: Basic metrics
    """
    logger.info("Computing basic metrics...")
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average)),
        'recall': float(recall_score(y_true, y_pred, average=average)),
        'f1_score': float(f1_score(y_true, y_pred, average=average))
    }
    
    logger.info(f"Basic metrics computed: Accuracy={metrics['accuracy']:.4f}")
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute confusion matrix and related metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        normalize (Optional[str]): Normalization method ('true', 'pred', 'all')
        
    Returns:
        Dict[str, Any]: Confusion matrix and derived metrics
    """
    logger.info("Computing confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    # Extract values for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        confusion_metrics = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'true_negative_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        }
    else:
        confusion_metrics = {
            'confusion_matrix': cm.tolist()
        }
    
    logger.info(f"Confusion matrix computed with shape {cm.shape}")
    
    return confusion_metrics


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics that require probability predictions.
    
    Args:
        y_true (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities for positive class
        
    Returns:
        Dict[str, float]: Probabilistic metrics
    """
    logger.info("Computing probabilistic metrics...")
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'average_precision': float(average_precision_score(y_true, y_proba))
    }
    
    # Calculate precision-recall curve points
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    metrics['pr_curve'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }
    
    metrics['roc_curve'] = {
        'false_positive_rate': fpr.tolist(),
        'true_positive_rate': tpr.tolist()
    }
    
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
    
    return metrics


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
    output_dict: bool = True
) -> Dict[str, Any]:
    """
    Compute detailed classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        target_names (Optional[list]): Names for target classes
        output_dict (bool): Whether to return dict instead of string
        
    Returns:
        Dict[str, Any]: Classification report
    """
    logger.info("Computing classification report...")
    
    if target_names is None:
        target_names = ['Not Spam', 'Spam']
    
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=output_dict)
    
    logger.info("Classification report computed")
    
    return report


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0
) -> Dict[str, float]:
    """
    Compute business-relevant metrics for spam detection.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        cost_fp (float): Cost of false positive (legitimate email marked as spam)
        cost_fn (float): Cost of false negative (spam email not caught)
        
    Returns:
        Dict[str, float]: Business metrics
    """
    logger.info("Computing business metrics...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        total_samples = len(y_true)
        
        business_metrics = {
            'total_cost': float(total_cost),
            'cost_per_email': float(total_cost / total_samples),
            'cost_fp': float(fp * cost_fp),
            'cost_fn': float(fn * cost_fn),
            'spam_caught_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'legitimate_preserved_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'total_fp': int(fp),
            'total_fn': int(fn),
            'total_tp': int(tp),
            'total_tn': int(tn)
        }
    else:
        business_metrics = {'error': 'Binary classification required for business metrics'}
    
    logger.info(f"Business metrics computed. Total cost: {business_metrics.get('total_cost', 'N/A')}")
    
    return business_metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    target_names: Optional[list] = None,
    compute_business: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (Optional[np.ndarray]): Predicted probabilities
        target_names (Optional[list]): Names for target classes
        compute_business (bool): Whether to compute business metrics
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation results
    """
    logger.info("Starting comprehensive model evaluation...")
    
    evaluation_results = {
        'basic_metrics': compute_basic_metrics(y_true, y_pred),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred),
        'classification_report': compute_classification_report(y_true, y_pred, target_names)
    }
    
    # Add probabilistic metrics if probabilities are available
    if y_proba is not None:
        evaluation_results['probabilistic_metrics'] = compute_probabilistic_metrics(y_true, y_proba)
    
    # Add business metrics if requested
    if compute_business:
        evaluation_results['business_metrics'] = compute_business_metrics(y_true, y_pred)
    
    logger.info("Comprehensive evaluation completed")
    
    return evaluation_results


def print_evaluation_summary(evaluation_results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        evaluation_results (Dict[str, Any]): Evaluation results dictionary
    """
    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    # Basic metrics
    basic = evaluation_results['basic_metrics']
    print(f"Accuracy:    {basic['accuracy']:.4f}")
    print(f"Precision:   {basic['precision']:.4f}")
    print(f"Recall:      {basic['recall']:.4f}")
    print(f"F1-Score:    {basic['f1_score']:.4f}")
    
    # Probabilistic metrics
    if 'probabilistic_metrics' in evaluation_results:
        prob = evaluation_results['probabilistic_metrics']
        print(f"ROC AUC:     {prob['roc_auc']:.4f}")
        print(f"Avg Precision: {prob['average_precision']:.4f}")
    
    # Confusion matrix
    cm = evaluation_results['confusion_matrix']
    if 'true_positives' in cm:
        print(f"\nConfusion Matrix:")
        print(f"True Negative:  {cm['true_negatives']}")
        print(f"False Positive: {cm['false_positives']}")
        print(f"False Negative: {cm['false_negatives']}")
        print(f"True Positive:  {cm['true_positives']}")
    
    # Business metrics
    if 'business_metrics' in evaluation_results:
        biz = evaluation_results['business_metrics']
        print(f"\nBusiness Metrics:")
        print(f"Total Cost:       ${biz['total_cost']:.2f}")
        print(f"Cost per Email:   ${biz['cost_per_email']:.4f}")
        print(f"Spam Caught Rate: {biz['spam_caught_rate']:.4f}")
        print(f"Legitimate Preserved: {biz['legitimate_preserved_rate']:.4f}")
    
    print("=" * 60)


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    metric: str = 'f1_score'
) -> Dict[str, Any]:
    """
    Compare multiple models and rank them by specified metric.
    
    Args:
        model_results (Dict[str, Dict]): Results from multiple models
        metric (str): Metric to use for comparison
        
    Returns:
        Dict[str, Any]: Comparison results and rankings
    """
    logger.info(f"Comparing models by {metric}...")
    
    comparison = {}
    
    for model_name, results in model_results.items():
        basic_metrics = results.get('basic_metrics', {})
        if metric in basic_metrics:
            comparison[model_name] = basic_metrics[metric]
        else:
            logger.warning(f"Metric {metric} not found for model {model_name}")
    
    # Sort models by metric
    sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
    
    comparison_results = {
        'metric': metric,
        'rankings': dict(sorted_models),
        'best_model': sorted_models[0][0] if sorted_models else None,
        'best_score': sorted_models[0][1] if sorted_models else None
    }
    
    logger.info(f"Best model: {comparison_results['best_model']} with {metric}: {comparison_results['best_score']:.4f}")
    
    return comparison_results


def main():
    """
    Main function to demonstrate evaluation functionality.
    """
    logger.info("Starting evaluation demonstration...")
    
    # This would typically be called with actual model predictions
    # Just demonstrating the evaluation structure
    logger.info("Evaluation module ready for use!")
    
    return True


if __name__ == "__main__":
    main()
