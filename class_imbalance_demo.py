"""
Class Imbalance Understanding Demonstration

This script demonstrates the comprehensive analysis of class imbalance problems,
including severity assessment, baseline comparisons, and proper evaluation methodologies.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import fit_preprocessor, transform_features
from src.class_imbalance import (
    ImbalanceAnalyzer, compute_baseline_metrics, evaluate_with_stratification,
    analyze_threshold_effects, compare_evaluation_metrics, plot_metric_comparison,
    create_stratified_splits, print_evaluation_recommendations
)
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_imbalanced_datasets():
    """
    Create synthetic datasets with different imbalance levels for demonstration.
    """
    from sklearn.datasets import make_classification
    
    datasets = {}
    
    # Mild imbalance (60-40)
    X_mild, y_mild = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.6, 0.4], flip_y=0.01, random_state=42
    )
    datasets['mild'] = (X_mild, y_mild)
    
    # Moderate imbalance (80-20)
    X_moderate, y_moderate = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.8, 0.2], flip_y=0.01, random_state=42
    )
    datasets['moderate'] = (X_moderate, y_moderate)
    
    # Severe imbalance (95-5)
    X_severe, y_severe = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0.01, random_state=42
    )
    datasets['severe'] = (X_severe, y_severe)
    
    # Extreme imbalance (99-1)
    X_extreme, y_extreme = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0.01, random_state=42
    )
    datasets['extreme'] = (X_extreme, y_extreme)
    
    return datasets


def demonstrate_imbalance_severity():
    """
    Demonstrate different levels of class imbalance.
    """
    print("="*80)
    print("DEMONSTRATING CLASS IMBALANCE SEVERITY LEVELS")
    print("="*80)
    
    datasets = create_imbalanced_datasets()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (severity, (X, y)) in enumerate(datasets.items()):
        analyzer = ImbalanceAnalyzer(y)
        analysis = analyzer.analyze_distribution()
        
        # Plot distribution
        classes = list(analysis['distribution'].keys())
        counts = [analysis['distribution'][cls]['count'] for cls in classes]
        percentages = [analysis['distribution'][cls]['percentage'] for cls in classes]
        
        bars = axes[i].bar(classes, counts, color=['steelblue', 'coral'])
        axes[i].set_title(f'{severity.upper()} Imbalance\n({percentages[0]:.1f}% / {percentages[1]:.1f}%)')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Print analysis
        print(f"\n{severity.upper()} IMBALANCE:")
        print(f"  Imbalance Ratio: {analysis['imbalance_ratio']:.2f}")
        print(f"  Severity: {analysis['severity']}")
        print(f"  Minority Percentage: {analysis['minority_percentage']:.2f}%")
    
    plt.tight_layout()
    plt.savefig("plots/imbalance_severity_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return datasets


def demonstrate_accuracy_failure():
    """
    Demonstrate how accuracy can be misleading in imbalanced problems.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING ACCURACY FAILURE IN IMBALANCED DATA")
    print("="*80)
    
    # Create severely imbalanced dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0.01, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = create_stratified_splits(X, y, test_size=0.2)
    
    # Train a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = compare_evaluation_metrics(y_test, y_pred, y_proba)
    
    print(f"\nDataset: 99% majority, 1% minority")
    print(f"Test set: {len(y_test)} samples")
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives: {metrics['true_positives']}")
    
    # Show what accuracy "sees" vs reality
    print(f"\nWhat Accuracy Reports vs Reality:")
    print(f"  Accuracy says: {metrics['accuracy']:.1%} - 'Excellent performance!'")
    print(f"  Reality: Model catches only {metrics['recall']:.1%} of minority cases")
    print(f"  Reality: {metrics['false_negatives']} minority cases missed completely")
    
    # Baseline comparison
    baseline_results = compute_baseline_metrics(X_train, X_test, y_train, y_test)
    
    print(f"\nBaseline (predict majority class always):")
    print(f"  Accuracy: {baseline_results['accuracy']:.3f} ({baseline_results['accuracy']:.1%})")
    print(f"  Minority Recall: {baseline_results['minority_recall']:.3f}")
    print(f"  Minority F1: {baseline_results['minority_f1']:.3f}")
    
    improvement = metrics['recall'] - baseline_results['minority_recall']
    print(f"\nImprovement over baseline:")
    print(f"  Recall improvement: {improvement:+.3f} ({improvement:+.1%})")
    
    return metrics, baseline_results


def demonstrate_stratified_importance():
    """
    Demonstrate importance of stratified splits.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING IMPORTANCE OF STRATIFIED SPLITS")
    print("="*80)
    
    # Create imbalanced dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0.01, random_state=42
    )
    
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(random_state=42)
    
    # Regular K-fold (non-stratified)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    regular_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
    
    # Stratified K-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    print(f"Dataset: 90% majority, 10% minority")
    print(f"CV F1-Scores (5-fold):")
    
    print(f"\nRegular K-Fold:")
    print(f"  Scores: {[f'{s:.3f}' for s in regular_scores]}")
    print(f"  Mean: {regular_scores.mean():.3f}")
    print(f"  Std: {regular_scores.std():.3f}")
    print(f"  Range: {regular_scores.max() - regular_scores.min():.3f}")
    
    print(f"\nStratified K-Fold:")
    print(f"  Scores: {[f'{s:.3f}' for s in stratified_scores]}")
    print(f"  Mean: {stratified_scores.mean():.3f}")
    print(f"  Std: {stratified_scores.std():.3f}")
    print(f"  Range: {stratified_scores.max() - stratified_scores.min():.3f}")
    
    # Show class distribution in each fold
    print(f"\nClass distribution in folds:")
    
    fold_distributions = {'regular': [], 'stratified': []}
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        fold_y_test = y[test_idx]
        unique, counts = np.unique(fold_y_test, return_counts=True)
        fold_distributions['regular'].append(dict(zip(unique, counts)))
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_y_test = y[test_idx]
        unique, counts = np.unique(fold_y_test, return_counts=True)
        fold_distributions['stratified'].append(dict(zip(unique, counts)))
    
    print(f"\nRegular K-Fold test set distributions:")
    for i, dist in enumerate(fold_distributions['regular']):
        minority_pct = dist.get(1, 0) / sum(dist.values()) * 100
        print(f"  Fold {i+1}: {dist} -> {minority_pct:.1f}% minority")
    
    print(f"\nStratified K-Fold test set distributions:")
    for i, dist in enumerate(fold_distributions['stratified']):
        minority_pct = dist.get(1, 0) / sum(dist.values()) * 100
        print(f"  Fold {i+1}: {dist} -> {minority_pct:.1f}% minority")
    
    return regular_scores, stratified_scores


def demonstrate_threshold_analysis():
    """
    Demonstrate threshold effects on imbalanced classification.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING THRESHOLD EFFECTS ON IMBALANCED CLASSIFICATION")
    print("="*80)
    
    # Create imbalanced dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0.01, random_state=42
    )
    
    # Split and train
    X_train, X_test, y_train, y_test = create_stratified_splits(X, y, test_size=0.2)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Analyze thresholds
    threshold_analysis = analyze_threshold_effects(y_test, y_proba)
    
    print(f"Threshold Analysis Results:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)
    
    for threshold in np.arange(0.1, 0.8, 0.1):
        results = threshold_analysis['results'][threshold]
        print(f"{threshold:>10.2f} {results['precision']:>10.3f} {results['recall']:>8.3f} "
              f"{results['f1']:>8.3f} {results['tp']:>4} {results['fp']:>4} {results['fn']:>4}")
    
    print(f"\nOptimal Thresholds:")
    print(f"  Best F1 threshold: {threshold_analysis['best_f1_threshold']:.2f} (F1 = {threshold_analysis['best_f1_score']:.3f})")
    print(f"  Best Recall threshold: {threshold_analysis['best_recall_threshold']:.2f} (Recall = {threshold_analysis['best_recall_score']:.3f})")
    print(f"  Best Precision threshold: {threshold_analysis['best_precision_threshold']:.2f} (Precision = {threshold_analysis['best_precision_score']:.3f})")
    
    # Plot threshold curves
    plt.figure(figsize=(12, 8))
    
    thresholds = threshold_analysis['thresholds']
    precisions = [threshold_analysis['results'][t]['precision'] for t in thresholds]
    recalls = [threshold_analysis['results'][t]['recall'] for t in thresholds]
    f1_scores = [threshold_analysis['results'][t]['f1'] for t in thresholds]
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default (0.5)')
    plt.axvline(x=threshold_analysis['best_f1_threshold'], color='orange', linestyle='--', 
                alpha=0.7, label=f'Best F1 ({threshold_analysis[\"best_f1_threshold\"]:.2f})')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall-F1 vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot confusion matrix components
    tps = [threshold_analysis['results'][t]['tp'] for t in thresholds]
    fps = [threshold_analysis['results'][t]['fp'] for t in thresholds]
    fns = [threshold_analysis['results'][t]['fn'] for t in thresholds]
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, tps, 'g-', linewidth=2, label='True Positives')
    plt.plot(thresholds, fps, 'r-', linewidth=2, label='False Positives')
    plt.plot(thresholds, fns, 'orange', linewidth=2, label='False Negatives')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Components vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    plt.subplot(2, 2, 3)
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Score distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_proba[y_test == 0], bins=30, alpha=0.7, label='Negative Class', color='blue')
    plt.hist(y_proba[y_test == 1], bins=30, alpha=0.7, label='Positive Class', color='red')
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default Threshold')
    plt.axvline(x=threshold_analysis['best_f1_threshold'], color='orange', linestyle='--', 
                alpha=0.7, label='Best F1 Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Score Distribution by Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("plots/threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return threshold_analysis


def demonstrate_business_contexts():
    """
    Demonstrate how business context affects metric selection.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING BUSINESS CONTEXT IMPACT ON METRIC SELECTION")
    print("="*80)
    
    # Create a moderately imbalanced dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=1, weights=[0.85, 0.15], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = create_stratified_splits(X, y, test_size=0.2)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compare_evaluation_metrics(y_test, y_pred, y_proba)
    
    print(f"Model Performance Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Different business contexts
    contexts = ['fraud', 'medical', 'spam', 'churn']
    
    for context in contexts:
        print(f"\n{'='*40}")
        print(f"{context.upper()} CONTEXT RECOMMENDATIONS")
        print('='*40)
        print_evaluation_recommendations(metrics, context)
    
    return metrics


def main():
    """
    Main demonstration function.
    """
    print("="*80)
    print("CLASS IMBALANCE UNDERSTANDING DEMONSTRATION")
    print("="*80)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Step 1: Demonstrate imbalance severity
    print("\n1. CLASS IMBALANCE SEVERITY ANALYSIS")
    datasets = demonstrate_imbalance_severity()
    
    # Step 2: Demonstrate accuracy failure
    print("\n2. ACCURACY FAILURE DEMONSTRATION")
    accuracy_metrics, baseline_results = demonstrate_accuracy_failure()
    
    # Step 3: Demonstrate stratified importance
    print("\n3. STRATIFIED SPLITS IMPORTANCE")
    regular_scores, stratified_scores = demonstrate_stratified_importance()
    
    # Step 4: Demonstrate threshold analysis
    print("\n4. THRESHOLD ANALYSIS DEMONSTRATION")
    threshold_results = demonstrate_threshold_analysis()
    
    # Step 5: Demonstrate business contexts
    print("\n5. BUSINESS CONTEXT IMPACT")
    context_metrics = demonstrate_business_contexts()
    
    # Step 6: Summary and recommendations
    print("\n" + "="*80)
    print("KEY TAKEAWAYS AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n🔍 DIAGNOSIS BEFORE TREATMENT:")
    print("  • Always check class distribution first")
    print("  • Compute majority-class baseline")
    print("  • Use stratified splits for imbalanced data")
    print("  • Examine confusion matrix, not just accuracy")
    
    print("\n📊 METRIC SELECTION GUIDELINES:")
    print("  • Mild imbalance (60-40): Accuracy still useful")
    print("  • Moderate imbalance (80-20): Use precision/recall/F1")
    print("  • Severe imbalance (95-5): F1 or PR-AUC preferred")
    print("  • Extreme imbalance (99-1): PR-AUC over ROC-AUC")
    
    print("\n⚠️  WARNING SIGNS:")
    print("  • High accuracy + low recall = imbalance blindness")
    print("  • No confusion matrix = incomplete evaluation")
    print("  • Only accuracy reported = suspicious")
    print("  • No baseline comparison = missing context")
    
    print("\n✅ PROFESSIONAL WORKFLOW:")
    print("  1. Analyze class distribution and severity")
    print("  2. Compute majority-class baseline")
    print("  3. Use stratified train/test splits and CV")
    print("  4. Evaluate with precision, recall, F1, confusion matrix")
    print("  5. Consider business cost structure")
    print("  6. Choose primary metric based on context")
    print("  7. Only then apply mitigation strategies")
    
    # Save summary results
    summary_results = {
        'accuracy_demo': {
            'model_metrics': accuracy_metrics,
            'baseline_metrics': baseline_results
        },
        'stratified_comparison': {
            'regular_cv': regular_scores.tolist(),
            'stratified_cv': stratified_scores.tolist()
        },
        'threshold_analysis': {
            'best_f1_threshold': threshold_results['best_f1_threshold'],
            'best_f1_score': threshold_results['best_f1_score']
        },
        'business_context_metrics': context_metrics
    }
    
    import json
    with open("class_imbalance_demo_results.json", "w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\n📁 Files created:")
    print(f"  • plots/imbalance_severity_comparison.png")
    print(f"  • plots/threshold_analysis.png")
    print(f"  • class_imbalance_demo_results.json")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return summary_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Class imbalance understanding demonstration completed successfully!")
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise
