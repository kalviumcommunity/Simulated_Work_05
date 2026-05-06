"""
Class Imbalance Analysis and Diagnosis Module

This module provides comprehensive tools for understanding and diagnosing class imbalance
in classification problems, including severity assessment, baseline comparisons, and
proper evaluation methodologies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                          precision_score, recall_score, f1_score, roc_auc_score,
                          average_precision_score, precision_recall_curve)
from typing import Dict, Tuple, Any, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ImbalanceAnalyzer:
    """
    Comprehensive class imbalance analysis and diagnosis tool.
    """
    
    def __init__(self, y: np.ndarray):
        """
        Initialize analyzer with target labels.
        
        Args:
            y: Target labels
        """
        self.y = np.array(y)
        self.classes_ = np.unique(self.y)
        self.n_classes_ = len(self.classes_)
        self.analysis_ = None
        
    def analyze_distribution(self) -> Dict:
        """
        Analyze class distribution and severity of imbalance.
        
        Returns:
            Dictionary with distribution analysis
        """
        unique, counts = np.unique(self.y, return_counts=True)
        total_samples = len(self.y)
        
        distribution = {}
        for cls, count in zip(unique, counts):
            distribution[cls] = {
                'count': int(count),
                'percentage': float(count / total_samples * 100),
                'ratio_to_majority': None,
                'ratio_to_minority': None
            }
        
        # Calculate ratios
        max_count = max(counts)
        min_count = min(counts)
        
        for cls in distribution:
            distribution[cls]['ratio_to_majority'] = float(distribution[cls]['count'] / max_count)
            distribution[cls]['ratio_to_minority'] = float(distribution[cls]['count'] / min_count)
        
        # Determine severity
        minority_percentage = min([dist['percentage'] for dist in distribution.values()])
        
        if minority_percentage >= 20:
            severity = "mild"
        elif minority_percentage >= 10:
            severity = "moderate"
        elif minority_percentage >= 5:
            severity = "severe"
        elif minority_percentage >= 1:
            severity = "extreme"
        else:
            severity = "critical"
        
        # Calculate imbalance ratio
        if len(unique) == 2:
            majority_count = max(counts)
            minority_count = min(counts)
            imbalance_ratio = majority_count / minority_count
        else:
            imbalance_ratio = max(counts) / min(counts)
        
        self.analysis_ = {
            'distribution': distribution,
            'total_samples': total_samples,
            'num_classes': self.n_classes_,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'minority_percentage': minority_percentage,
            'is_balanced': imbalance_ratio < 1.5,
            'recommendations': self._get_recommendations(severity, minority_percentage)
        }
        
        return self.analysis_
    
    def _get_recommendations(self, severity: str, minority_percentage: float) -> List[str]:
        """
        Get recommendations based on imbalance severity.
        
        Args:
            severity: Severity level
            minority_percentage: Percentage of minority class
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if severity == "mild":
            recommendations.extend([
                "Standard evaluation metrics (accuracy) still informative",
                "Consider precision/recall for business context",
                "Stratified splits recommended but not critical"
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Accuracy becomes partially misleading",
                "Use precision, recall, and F1 as primary metrics",
                "Stratified splits essential",
                "Consider threshold tuning"
            ])
        elif severity == "severe":
            recommendations.extend([
                "Accuracy is actively misleading",
                "F1-score or PR-AUC recommended as primary metric",
                "Class weights or resampling likely needed",
                "Strict stratification required"
            ])
        elif severity == "extreme":
            recommendations.extend([
                "Accuracy is completely useless",
                "PR-AUC preferred over ROC-AUC",
                "Special handling essential (class weights + resampling)",
                "Consider anomaly detection approaches"
            ])
        else:  # critical
            recommendations.extend([
                "Standard classification likely inappropriate",
                "Consider one-class classification or anomaly detection",
                "Data augmentation for minority class may be necessary",
                "Custom evaluation metrics based on business cost"
            ])
        
        return recommendations
    
    def plot_distribution(self, save_path: str = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot class distribution visualization.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.analysis_ is None:
            self.analyze_distribution()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        classes = list(self.analysis_['distribution'].keys())
        counts = [self.analysis_['distribution'][cls]['count'] for cls in classes]
        percentages = [self.analysis_['distribution'][cls]['percentage'] for cls in classes]
        
        bars = ax1.bar(classes, counts, color=['steelblue', 'coral', 'lightgreen', 'orange'][:len(classes)])
        ax1.set_title('Class Distribution - Counts', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(percentages, labels=[f'Class {cls}\n({pct:.1f}%)' for cls, pct in zip(classes, percentages)],
                colors=['steelblue', 'coral', 'lightgreen', 'orange'][:len(classes)],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution - Percentages', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Class Imbalance Analysis\nSeverity: {self.analysis_["severity"].upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print comprehensive analysis summary."""
        if self.analysis_ is None:
            self.analyze_distribution()
        
        print("\n" + "="*80)
        print("CLASS IMBALANCE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"  Total samples: {self.analysis_['total_samples']:,}")
        print(f"  Number of classes: {self.analysis_['num_classes']}")
        print(f"  Imbalance ratio: {self.analysis_['imbalance_ratio']:.2f}")
        print(f"  Severity: {self.analysis_['severity'].upper()}")
        print(f"  Is balanced: {self.analysis_['is_balanced']}")
        
        print(f"\nClass Distribution:")
        for cls, dist in self.analysis_['distribution'].items():
            print(f"  Class {cls}: {dist['count']:,} samples ({dist['percentage']:.2f}%)")
            print(f"    Ratio to majority: {dist['ratio_to_majority']:.3f}")
            print(f"    Ratio to minority: {dist['ratio_to_minority']:.3f}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(self.analysis_['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def compute_baseline_metrics(X_train, X_test, y_train, y_test, 
                        strategy: str = "most_frequent") -> Dict:
    """
    Compute majority-class baseline performance.
    
    Args:
        X_train, X_test, y_train, y_test: Split data
        strategy: Baseline strategy
        
    Returns:
        Dictionary with baseline metrics
    """
    baseline = DummyClassifier(strategy=strategy)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    
    # Get baseline probabilities if available
    if hasattr(baseline, 'predict_proba'):
        baseline_proba = baseline.predict_proba(X_test)[:, 1]
    else:
        baseline_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, baseline_pred)
    report = classification_report(y_test, baseline_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, baseline_pred)
    
    baseline_results = {
        'model': baseline,
        'predictions': baseline_pred,
        'probabilities': baseline_proba,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'strategy': strategy
    }
    
    # Extract minority class metrics (assuming binary classification)
    if len(np.unique(y_test)) == 2:
        minority_class = 1
        baseline_results['minority_recall'] = report[str(minority_class)]['recall']
        baseline_results['minority_precision'] = report[str(minority_class)]['precision']
        baseline_results['minority_f1'] = report[str(minority_class)]['f1-score']
    
    return baseline_results


def evaluate_with_stratification(X, y, model, cv_folds: int = 5, 
                            scoring_metrics: List[str] = None) -> Dict:
    """
    Evaluate model with proper stratified cross-validation.
    
    Args:
        X, y: Features and target
        model: Scikit-learn model
        cv_folds: Number of CV folds
        scoring_metrics: List of metrics to compute
        
    Returns:
        Dictionary with CV results
    """
    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Use stratified K-fold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {}
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'cv_folds': cv_folds
            }
            logger.info(f"CV {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
        except ValueError as e:
            logger.warning(f"Could not compute {metric}: {e}")
            cv_results[metric] = {'scores': np.array([0.0]), 'mean': 0.0, 'std': 0.0}
    
    return cv_results


def analyze_threshold_effects(y_true, y_proba, thresholds: np.ndarray = None) -> Dict:
    """
    Analyze effects of different classification thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to test
        
    Returns:
        Dictionary with threshold analysis
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'predictions': y_pred
        }
    
    # Find optimal thresholds for different metrics
    best_f1_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
    best_recall_threshold = max(results.keys(), key=lambda t: results[t]['recall'])
    best_precision_threshold = max(results.keys(), key=lambda t: results[t]['precision'])
    
    threshold_analysis = {
        'results': results,
        'best_f1_threshold': best_f1_threshold,
        'best_f1_score': results[best_f1_threshold]['f1'],
        'best_recall_threshold': best_recall_threshold,
        'best_recall_score': results[best_recall_threshold]['recall'],
        'best_precision_threshold': best_precision_threshold,
        'best_precision_score': results[best_precision_threshold]['precision'],
        'thresholds': thresholds
    }
    
    return threshold_analysis


def compare_evaluation_metrics(y_true, y_pred, y_proba=None) -> Dict:
    """
    Compare different evaluation metrics for the same predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Rates
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Probability-based metrics (if available)
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    # Classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics


def plot_metric_comparison(metrics_dict: Dict[str, Dict], save_path: str = None):
    """
    Plot comparison of different evaluation metrics.
    
    Args:
        metrics_dict: Dictionary of metrics for different models
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(metrics_dict.keys())
    
    # Accuracy comparison
    accuracies = [metrics_dict[name]['accuracy'] for name in model_names]
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall comparison
    precisions = [metrics_dict[name]['precision'] for name in model_names]
    recalls = [metrics_dict[name]['recall'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, precisions, width, label='Precision', color='lightcoral')
    axes[0, 1].bar(x + width/2, recalls, width, label='Recall', color='lightgreen')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 comparison
    f1_scores = [metrics_dict[name]['f1'] for name in model_names]
    axes[1, 0].bar(model_names, f1_scores, color='gold')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC vs PR-AUC (if available)
    if all('roc_auc' in metrics_dict[name] for name in model_names):
        roc_aucs = [metrics_dict[name]['roc_auc'] for name in model_names]
        pr_aucs = [metrics_dict[name]['pr_auc'] for name in model_names]
        
        axes[1, 1].bar(x - width/2, roc_aucs, width, label='ROC-AUC', color='plum')
        axes[1, 1].bar(x + width/2, pr_aucs, width, label='PR-AUC', color='orange')
        axes[1, 1].set_title('ROC-AUC vs PR-AUC')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC-AUC/PR-AUC\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC-AUC vs PR-AUC')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()


def create_stratified_splits(X, y, test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple:
    """
    Create properly stratified train/test splits.
    
    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y  # Critical for imbalanced data
    )


def check_evaluation_warnings(metrics: Dict) -> List[str]:
    """
    Check for common evaluation mistakes in imbalanced problems.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        List of warnings
    """
    warnings_list = []
    
    # Check if only accuracy is reported
    if len(metrics) == 1 and 'accuracy' in metrics:
        warnings_list.append("⚠️  Only accuracy reported - precision/recall/F1 missing")
    
    # Check for high accuracy but low recall
    if 'accuracy' in metrics and 'recall' in metrics:
        if metrics['accuracy'] > 0.9 and metrics['recall'] < 0.2:
            warnings_list.append("⚠️  High accuracy but very low recall - classic imbalance sign")
    
    # Check for zero minority predictions
    if 'true_positives' in metrics and 'false_positives' in metrics:
        if metrics['true_positives'] == 0 and metrics['false_positives'] == 0:
            warnings_list.append("⚠️  No positive predictions made - model predicts majority class only")
    
    # Check for confusion matrix absence
    if 'confusion_matrix' not in metrics:
        warnings_list.append("⚠️  Confusion matrix missing - essential for imbalanced evaluation")
    
    return warnings_list


def print_evaluation_recommendations(metrics: Dict, context: str = "general"):
    """
    Print evaluation recommendations based on metrics.
    
    Args:
        metrics: Metrics dictionary
        context: Business context (fraud, medical, spam, etc.)
    """
    print("\n" + "="*60)
    print("EVALUATION RECOMMENDATIONS")
    print("="*60)
    
    # Check for warnings
    warnings_list = check_evaluation_warnings(metrics)
    if warnings_list:
        print("⚠️  WARNINGS DETECTED:")
        for warning in warnings_list:
            print(f"  {warning}")
        print()
    
    # Context-specific recommendations
    if context == "fraud":
        print("📊 FRAUD DETECTION CONTEXT:")
        print("  • Prioritize recall - missing fraud is costly")
        print("  • Accept lower precision for higher recall")
        print("  • Monitor false positive rate for customer friction")
        print("  • Consider cost-based evaluation")
    
    elif context == "medical":
        print("🏥 MEDICAL DIAGNOSIS CONTEXT:")
        print("  • Maximize recall - missed diagnoses are critical")
        print("  • False positives acceptable (additional testing)")
        print("  • Use sensitivity (recall) as primary metric")
        print("  • Consider ROC-AUC for threshold selection")
    
    elif context == "spam":
        print("📧 SPAM DETECTION CONTEXT:")
        print("  • Balance precision and recall")
        print("  • False positives (legitimate → spam) are annoying")
        print("  • False negatives (spam → inbox) are inconvenient")
        print("  • F1-score is appropriate primary metric")
    
    elif context == "churn":
        print("👥 CUSTOMER CHURN CONTEXT:")
        print("  • Prioritize recall - lost customers are costly")
        print("  • False positives cost discounts but retain customers")
        print("  • Consider customer lifetime value in cost analysis")
        print("  • Monitor precision for ROI of retention efforts")
    
    else:  # general
        print("📈 GENERAL CONTEXT:")
        print("  • Report precision, recall, and F1 alongside accuracy")
        print("  • Use confusion matrix for detailed analysis")
        print("  • Consider business cost of each error type")
        print("  • Choose primary metric based on cost structure")
    
    print("\n" + "="*60)
