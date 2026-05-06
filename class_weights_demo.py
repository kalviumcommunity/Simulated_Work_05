"""
Class Weights Demonstration

This script demonstrates the application of class weights to handle imbalanced data,
including comparisons between weighted and unweighted models, threshold tuning,
and cross-validation analysis.
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
from src.class_weights import (
    compute_balanced_weights, train_weighted_model, compare_weighted_models,
    tune_threshold, cross_validate_with_weights, grid_search_class_weights,
    plot_comparison_results, print_comparison_summary, analyze_class_distribution
)
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main demonstration of class weights functionality.
    """
    print("="*80)
    print("CLASS WEIGHTS DEMONSTRATION")
    print("="*80)
    
    # Step 1: Load and analyze data
    print("\n1. Loading and analyzing data...")
    X, y = load_data(synthetic=True)
    X_clean, y_clean = clean_data(X, y)
    
    # Analyze class distribution
    distribution_analysis = analyze_class_distribution(y_clean.values)
    print(f"Dataset contains {distribution_analysis['total_samples']} samples")
    print(f"Number of classes: {distribution_analysis['num_classes']}")
    print(f"Imbalance ratio: {distribution_analysis['imbalance_ratio']:.2f}")
    print(f"Is balanced: {distribution_analysis['is_balanced']}")
    
    # Step 2: Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    preprocessor, X_train_transformed = fit_preprocessor(X_train.values, y_train.values)
    X_test_transformed = transform_features(preprocessor, X_test.values)
    
    # Step 4: Compute balanced weights
    print("\n4. Computing balanced class weights...")
    balanced_weights = compute_balanced_weights(y_train.values)
    print(f"Balanced weights: {balanced_weights}")
    
    # Step 5: Compare weighted vs unweighted models
    print("\n5. Comparing weighted vs unweighted models...")
    unweighted_results, weighted_results = compare_weighted_models(
        X_train_transformed, X_test_transformed, 
        y_train.values, y_test.values,
        model_type="logistic"
    )
    
    # Print comparison summary
    print_comparison_summary(unweighted_results, weighted_results)
    
    # Step 6: Threshold tuning
    print("\n6. Tuning classification thresholds...")
    threshold_results = tune_threshold(
        y_test.values, weighted_results['probabilities']
    )
    
    print("\nThreshold Analysis:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for threshold, metrics in threshold_results.items():
        print(f"{threshold:>10.2f} {metrics['precision']:>10.3f} {metrics['recall']:>8.3f} {metrics['f1']:>8.3f}")
    
    # Find best F1 threshold
    best_threshold = max(threshold_results.keys(), 
                        key=lambda t: threshold_results[t]['f1'])
    best_f1 = threshold_results[best_threshold]['f1']
    print(f"\nBest threshold for F1: {best_threshold:.2f} (F1 = {best_f1:.3f})")
    
    # Step 7: Cross-validation with class weights
    print("\n7. Cross-validation with class weights...")
    cv_scores = cross_validate_with_weights(
        X_train_transformed, y_train.values,
        model_type="logistic",
        class_weight="balanced",
        cv_folds=5
    )
    
    print("\nCross-validation Results:")
    for metric, scores in cv_scores.items():
        print(f"{metric:>12}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Step 8: Grid search for optimal weights
    print("\n8. Grid search for optimal class weights...")
    grid_results = grid_search_class_weights(
        X_train_transformed, y_train.values,
        model_type="logistic",
        cv_folds=5
    )
    
    print(f"Best class weight configuration: {grid_results['best_params']}")
    print(f"Best CV recall score: {grid_results['best_score']:.3f}")
    
    # Step 9: Train model with best weights
    print("\n9. Training model with best weights...")
    best_model = train_weighted_model(
        X_train_transformed, y_train.values,
        model_type="logistic",
        class_weight=grid_results['best_params']['class_weight']
    )
    
    # Evaluate best model
    y_pred_best = best_model.predict(X_test_transformed)
    y_proba_best = best_model.predict_proba(X_test_transformed)[:, 1]
    
    print("\nBest Model Performance:")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test.values, y_pred_best))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test.values, y_pred_best))
    
    # Step 10: Visualization
    print("\n10. Generating visualizations...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Plot comparison results
    plot_comparison_results(
        unweighted_results, weighted_results,
        y_test.values,
        save_path="plots/class_weights_comparison.png"
    )
    
    # Plot threshold analysis
    plt.figure(figsize=(10, 6))
    thresholds = list(threshold_results.keys())
    precisions = [threshold_results[t]['precision'] for t in thresholds]
    recalls = [threshold_results[t]['recall'] for t in thresholds]
    f1_scores = [threshold_results[t]['f1'] for t in thresholds]
    
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
    plt.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.5, 
                label=f'Best Threshold = {best_threshold:.2f}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis for Weighted Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 11: Save best model
    print("\n11. Saving best model...")
    import joblib
    joblib.dump(best_model, "models/best_weighted_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    # Save results summary
    results_summary = {
        'distribution_analysis': distribution_analysis,
        'balanced_weights': balanced_weights,
        'unweighted_metrics': unweighted_results['classification_report'],
        'weighted_metrics': weighted_results['classification_report'],
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'cv_scores': {k: v.tolist() for k, v in cv_scores.items()},
        'grid_search_results': {
            'best_params': grid_results['best_params'],
            'best_score': grid_results['best_score']
        },
        'best_model_metrics': classification_report(y_test.values, y_pred_best, output_dict=True)
    }
    
    import json
    with open("models/class_weights_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Findings:")
    print(f"1. Dataset imbalance ratio: {distribution_analysis['imbalance_ratio']:.2f}")
    print(f"2. Balanced class weights: {balanced_weights}")
    print(f"3. Best threshold for F1: {best_threshold:.2f}")
    print(f"4. Best CV recall: {grid_results['best_score']:.3f}")
    print(f"5. Best class weight config: {grid_results['best_params']}")
    
    print("\nFiles saved:")
    print("- models/best_weighted_model.pkl")
    print("- models/preprocessor.pkl")
    print("- models/class_weights_results.json")
    print("- plots/class_weights_comparison.png")
    print("- plots/threshold_analysis.png")
    
    return results_summary


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Class weights demonstration completed successfully!")
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise
