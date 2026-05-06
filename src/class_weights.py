"""
Class Weights Implementation for Imbalanced Data

This module implements class weighting techniques to handle imbalanced datasets
by modifying the loss function to penalize minority-class errors more heavily.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def compute_balanced_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights using scikit-learn's formula.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class labels to their weights
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    
    logger.info(f"Computed balanced weights: {weight_dict}")
    return weight_dict


def train_weighted_model(X_train, y_train, model_type: str = "logistic", 
                        class_weight: str = "balanced", **model_params) -> Any:
    """
    Train a model with class weights.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('logistic', 'random_forest', 'decision_tree', 'svm')
        class_weight: Class weighting strategy ('balanced' or dict)
        **model_params: Additional model parameters
        
    Returns:
        Trained model
    """
    if model_type == "logistic":
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            **model_params
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            class_weight=class_weight,
            n_estimators=100,
            random_state=42,
            **model_params
        )
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=42,
            **model_params
        )
    elif model_type == "svm":
        model = SVC(
            class_weight=class_weight,
            probability=True,  # Required for probability predictions
            random_state=42,
            **model_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Training {model_type} with class_weight={class_weight}")
    model.fit(X_train, y_train)
    return model


def compare_weighted_models(X_train, X_test, y_train, y_test, 
                           model_type: str = "logistic") -> Tuple[Dict, Dict]:
    """
    Compare weighted vs unweighted models.
    
    Args:
        X_train, X_test, y_train, y_test: Split data
        model_type: Type of model to train
        
    Returns:
        Tuple of (unweighted_results, weighted_results)
    """
    # Train unweighted model
    model_unweighted = train_weighted_model(
        X_train, y_train, model_type=model_type, class_weight=None
    )
    
    # Train weighted model
    model_weighted = train_weighted_model(
        X_train, y_train, model_type=model_type, class_weight="balanced"
    )
    
    # Get predictions
    y_pred_unweighted = model_unweighted.predict(X_test)
    y_pred_weighted = model_weighted.predict(X_test)
    
    # Get probabilities for threshold analysis
    y_proba_unweighted = model_unweighted.predict_proba(X_test)[:, 1]
    y_proba_weighted = model_weighted.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    unweighted_results = {
        'model': model_unweighted,
        'predictions': y_pred_unweighted,
        'probabilities': y_proba_unweighted,
        'classification_report': classification_report(y_test, y_pred_unweighted, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_unweighted)
    }
    
    weighted_results = {
        'model': model_weighted,
        'predictions': y_pred_weighted,
        'probabilities': y_proba_weighted,
        'classification_report': classification_report(y_test, y_pred_weighted, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_weighted)
    }
    
    return unweighted_results, weighted_results


def tune_threshold(y_true, y_proba, thresholds: np.ndarray = None) -> Dict[float, Dict]:
    """
    Tune classification threshold for optimal precision-recall balance.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        thresholds: Array of thresholds to test
        
    Returns:
        Dictionary mapping thresholds to metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = {}
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
    
    return results


def cross_validate_with_weights(X, y, model_type: str = "logistic", 
                               class_weight: str = "balanced", 
                               cv_folds: int = 5) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation with class weights.
    
    Args:
        X: Features
        y: Target
        model_type: Type of model
        class_weight: Class weighting strategy
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary of cross-validation scores
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    model = train_weighted_model(X, y, model_type=model_type, class_weight=class_weight)
    
    scoring_metrics = ['recall', 'precision', 'f1', 'roc_auc']
    cv_scores = {}
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_scores[metric] = scores
            logger.info(f"CV {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
        except ValueError as e:
            logger.warning(f"Could not compute {metric}: {e}")
            cv_scores[metric] = np.array([0.0])
    
    return cv_scores


def grid_search_class_weights(X_train, y_train, model_type: str = "logistic",
                              weight_ratios: list = None, cv_folds: int = 5) -> Dict:
    """
    Grid search over different class weight configurations.
    
    Args:
        X_train, y_train: Training data
        model_type: Type of model
        weight_ratios: List of weight ratios to test
        cv_folds: Number of CV folds
        
    Returns:
        Grid search results
    """
    if weight_ratios is None:
        weight_ratios = [
            {0: 1, 1: 5},
            {0: 1, 1: 10},
            {0: 1, 1: 15},
            {0: 1, 1: 20},
            "balanced"
        ]
    
    param_grid = {"class_weight": weight_ratios}
    
    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type for grid search: {model_type}")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=skf,
        scoring="recall",  # Optimize for recall
        n_jobs=-1
    )
    
    logger.info("Starting grid search for class weights...")
    grid.fit(X_train, y_train)
    
    results = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'cv_results': grid.cv_results_,
        'best_estimator': grid.best_estimator_
    }
    
    logger.info(f"Best class_weight: {grid.best_params_}")
    logger.info(f"Best CV Recall: {grid.best_score_:.3f}")
    
    return results


def plot_comparison_results(unweighted_results: Dict, weighted_results: Dict, 
                           y_test: np.ndarray, save_path: str = None):
    """
    Plot comparison between weighted and unweighted models.
    
    Args:
        unweighted_results: Results from unweighted model
        weighted_results: Results from weighted model
        y_test: True labels
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion matrices
    sns.heatmap(unweighted_results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Unweighted Model - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    sns.heatmap(weighted_results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Weighted Model - Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Precision-Recall curves
    precision_u, recall_u, _ = precision_recall_curve(
        y_test, unweighted_results['probabilities']
    )
    precision_w, recall_w, _ = precision_recall_curve(
        y_test, weighted_results['probabilities']
    )
    
    axes[1, 0].plot(recall_u, precision_u, label='Unweighted', linewidth=2)
    axes[1, 0].plot(recall_w, precision_w, label='Weighted', linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # ROC curves
    fpr_u, tpr_u, _ = roc_curve(y_test, unweighted_results['probabilities'])
    fpr_w, tpr_w, _ = roc_curve(y_test, weighted_results['probabilities'])
    
    auc_u = auc(fpr_u, tpr_u)
    auc_w = auc(fpr_w, tpr_w)
    
    axes[1, 1].plot(fpr_u, tpr_u, label=f'Unweighted (AUC = {auc_u:.3f})', linewidth=2)
    axes[1, 1].plot(fpr_w, tpr_w, label=f'Weighted (AUC = {auc_w:.3f})', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def print_comparison_summary(unweighted_results: Dict, weighted_results: Dict):
    """
    Print a summary comparison between weighted and unweighted models.
    
    Args:
        unweighted_results: Results from unweighted model
        weighted_results: Results from weighted model
    """
    print("\n" + "="*60)
    print("CLASS WEIGHTS COMPARISON SUMMARY")
    print("="*60)
    
    # Extract metrics for minority class (assuming binary classification)
    minority_class = 1
    
    unweighted_metrics = unweighted_results['classification_report'][str(minority_class)]
    weighted_metrics = weighted_results['classification_report'][str(minority_class)]
    
    print(f"\nMinority Class (Class {minority_class}) Performance:")
    print("-" * 50)
    
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        unweighted_val = unweighted_metrics[metric]
        weighted_val = weighted_metrics[metric]
        change = weighted_val - unweighted_val
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        
        print(f"{metric.capitalize():12}: {unweighted_val:.3f} → {weighted_val:.3f} {arrow} ({change:+.3f})")
    
    print(f"\nOverall Accuracy:")
    print(f"Unweighted: {unweighted_results['classification_report']['accuracy']:.3f}")
    print(f"Weighted:   {weighted_results['classification_report']['accuracy']:.3f}")
    
    print(f"\nConfusion Matrix Comparison:")
    print("Unweighted:")
    print(unweighted_results['confusion_matrix'])
    print("Weighted:")
    print(weighted_results['confusion_matrix'])
    
    print("\n" + "="*60)


def analyze_class_distribution(y: np.ndarray) -> Dict:
    """
    Analyze class distribution and imbalance.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary with distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    distribution = {}
    for cls, count in zip(unique, counts):
        distribution[cls] = {
            'count': int(count),
            'percentage': float(count / total_samples * 100)
        }
    
    # Calculate imbalance ratio
    if len(unique) == 2:
        majority_count = max(counts)
        minority_count = min(counts)
        imbalance_ratio = majority_count / minority_count
    else:
        imbalance_ratio = max(counts) / min(counts)
    
    analysis = {
        'distribution': distribution,
        'total_samples': total_samples,
        'num_classes': len(unique),
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio < 1.5
    }
    
    logger.info(f"Class distribution: {distribution}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    return analysis
