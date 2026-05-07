"""
Model Comparison Demo - Professional ML Practice

This script demonstrates systematic model comparison following professional ML practices.
It shows how to:
1. Set up consistent preprocessing pipelines
2. Compare multiple models fairly with cross-validation
3. Analyze bias-variance characteristics
4. Perform statistical significance testing
5. Generate comprehensive reports and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our model comparison framework
from src.model_comparison import ModelComparisonFramework, ComparisonConfig
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split


def create_sample_dataset(dataset_type: str = "synthetic"):
    """
    Create a sample dataset for demonstration.
    
    Args:
        dataset_type: Type of dataset ('synthetic', 'breast_cancer', 'imbalanced')
        
    Returns:
        Tuple of (X, y) features and target
    """
    if dataset_type == "synthetic":
        # Create synthetic classification dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            random_state=42
        )
        
        # Convert to DataFrame with feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")
        
    elif dataset_type == "breast_cancer":
        # Use real breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        
    elif dataset_type == "imbalanced":
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=3000,
            n_features=15,
            n_informative=10,
            n_redundant=5,
            weights=[0.9, 0.1],  # 90% majority, 10% minority
            random_state=42
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Created {dataset_type} dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print(f"  Class imbalance ratio: {y.value_counts().max() / y.value_counts().min():.2f}")
    
    return X, y


def demonstrate_basic_comparison():
    """Demonstrate basic model comparison workflow."""
    print("\n" + "="*80)
    print("BASIC MODEL COMPARISON DEMONSTRATION")
    print("="*80)
    
    # Create dataset
    X, y = create_sample_dataset("synthetic")
    
    # Setup comparison framework
    config = ComparisonConfig(
        cv_folds=5,
        scoring_metric="f1",
        n_iter_tuning=20,  # Reduced for demo speed
        verbose=True
    )
    
    framework = ModelComparisonFramework(config)
    
    # Setup data (auto-detect feature types)
    framework.setup_data(X, y)
    
    # Compare models
    results_df = framework.compare_models(tune_hyperparameters=True)
    
    # Generate visualizations
    framework.plot_model_comparison()
    framework.plot_bias_variance_analysis()
    
    # Generate report
    report = framework.generate_report()
    print("\n" + report)
    
    return framework, results_df


def demonstrate_imbalanced_comparison():
    """Demonstrate model comparison on imbalanced data."""
    print("\n" + "="*80)
    print("IMBALANCED DATA MODEL COMPARISON DEMONSTRATION")
    print("="*80)
    
    # Create imbalanced dataset
    X, y = create_sample_dataset("imbalanced")
    
    # Setup comparison framework with recall as metric
    config = ComparisonConfig(
        cv_folds=5,
        scoring_metric="recall",  # Focus on minority class detection
        n_iter_tuning=20,
        verbose=True
    )
    
    framework = ModelComparisonFramework(config)
    framework.setup_data(X, y)
    
    # Compare models
    results_df = framework.compare_models(tune_hyperparameters=True)
    
    # Multi-metric evaluation for top models
    print("\nMulti-Metric Evaluation for Top 3 Models:")
    print("-" * 50)
    
    top_models = results_df.head(3)['Model'].tolist()
    for model_name in top_models:
        metrics = framework.multi_metric_evaluation(model_name)
        print(f"\n{model_name}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    # Bias-variance analysis
    interpretations = framework.analyze_bias_variance()
    print("\nBias-Variance Diagnostics:")
    print("-" * 50)
    for model_name, interpretation in interpretations.items():
        print(f"{model_name:25}: {interpretation}")
    
    return framework, results_df


def demonstrate_statistical_testing():
    """Demonstrate statistical significance testing between models."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING DEMONSTRATION")
    print("="*80)
    
    # Create dataset
    X, y = create_sample_dataset("breast_cancer")
    
    # Setup framework
    config = ComparisonConfig(
        cv_folds=10,  # More folds for better statistical power
        scoring_metric="f1",
        n_iter_tuning=15,
        verbose=True
    )
    
    framework = ModelComparisonFramework(config)
    framework.setup_data(X, y)
    
    # Compare models
    results_df = framework.compare_models(tune_hyperparameters=True)
    
    # Statistical tests between top models
    top_models = results_df.head(3)['Model'].tolist()
    
    print("\nStatistical Significance Tests:")
    print("-" * 50)
    
    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):
            model1, model2 = top_models[i], top_models[j]
            test_result = framework.statistical_significance_test(model1, model2)
            
            print(f"\n{model1} vs {model2}:")
            print(f"  Mean difference: {test_result['mean_difference']:.4f}")
            print(f"  P-value: {test_result['p_value']:.4f}")
            print(f"  Significant: {test_result['significant']}")
            print(f"  Effect size: {test_result['cohens_d']:.3f} ({test_result['interpretation']})")
    
    return framework, results_df


def demonstrate_model_selection_criteria():
    """Demonstrate different model selection criteria."""
    print("\n" + "="*80)
    print("MODEL SELECTION CRITERIA DEMONSTRATION")
    print("="*80)
    
    # Create dataset
    X, y = create_sample_dataset("synthetic")
    
    # Setup framework
    config = ComparisonConfig(
        cv_folds=5,
        scoring_metric="f1",
        n_iter_tuning=20,
        verbose=True
    )
    
    framework = ModelComparisonFramework(config)
    framework.setup_data(X, y)
    
    # Compare models
    results_df = framework.compare_models(tune_hyperparameters=True)
    
    # Select models based on different criteria
    criteria = ["performance", "stability", "efficiency"]
    
    print("\nModel Selection by Different Criteria:")
    print("-" * 50)
    
    for criterion in criteria:
        try:
            model_name, result = framework.select_best_model(criteria=criterion)
            print(f"\n{criterion.capitalize()} criterion:")
            print(f"  Selected: {model_name}")
            print(f"  CV Score: {result.cv_mean:.3f} ± {result.cv_std:.3f}")
            print(f"  Gap: {result.gap:.3f}")
            if result.training_time:
                print(f"  Training Time: {result.training_time:.1f}s")
        except Exception as e:
            print(f"  Error with {criterion}: {e}")
    
    # Practical recommendation
    print("\nPractical Recommendations:")
    print("-" * 50)
    
    best_performance = max(framework.results, key=lambda r: r.cv_mean)
    most_stable = min(framework.results, key=lambda r: r.cv_std)
    
    print(f"Best Performance: {best_performance.model_name}")
    print(f"  - Use when: Maximum accuracy is required")
    print(f"  - Consider: {best_performance.cv_std:.3f} std deviation indicates stability")
    
    print(f"\nMost Stable: {most_stable.model_name}")
    print(f"  - Use when: Consistent performance is critical")
    print(f"  - Consider: {most_stable.cv_mean:.3f} mean score may be sufficient")
    
    # Production considerations
    print(f"\nProduction Considerations:")
    print("-" * 50)
    
    for result in framework.results[:5]:  # Top 5 models
        considerations = []
        
        # Performance
        if result.cv_mean > 0.85:
            considerations.append("High accuracy")
        elif result.cv_mean > 0.75:
            considerations.append("Good accuracy")
        
        # Stability
        if result.cv_std < 0.02:
            considerations.append("Very stable")
        elif result.cv_std < 0.05:
            considerations.append("Stable")
        
        # Overfitting
        if result.gap < 0.03:
            considerations.append("Well-fitted")
        elif result.gap > 0.08:
            considerations.append("Overfitting risk")
        
        # Efficiency
        if result.training_time and result.training_time < 5:
            considerations.append("Fast training")
        if result.inference_time and result.inference_time < 0.01:
            considerations.append("Fast inference")
        
        if considerations:
            print(f"{result.model_name:25}: {', '.join(considerations)}")
    
    return framework, results_df


def create_comprehensive_example():
    """Create a comprehensive example combining all features."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON WORKFLOW")
    print("="*80)
    
    # Create dataset
    print("1. Loading Dataset")
    print("-" * 30)
    X, y = create_sample_dataset("breast_cancer")
    
    # Setup framework with comprehensive configuration
    print("\n2. Setting Up Comparison Framework")
    print("-" * 30)
    config = ComparisonConfig(
        cv_folds=5,
        scoring_metric="f1",
        n_iter_tuning=25,
        verbose=True
    )
    
    framework = ModelComparisonFramework(config)
    framework.setup_data(X, y)
    
    # Compare models
    print("\n3. Comparing Models")
    print("-" * 30)
    results_df = framework.compare_models(tune_hyperparameters=True)
    
    # Analysis
    print("\n4. Bias-Variance Analysis")
    print("-" * 30)
    interpretations = framework.analyze_bias_variance()
    for model_name, interpretation in interpretations.items():
        print(f"{model_name:25}: {interpretation}")
    
    # Statistical testing
    print("\n5. Statistical Significance Testing")
    print("-" * 30)
    top_2 = results_df.head(2)['Model'].tolist()
    test_result = framework.statistical_significance_test(top_2[0], top_2[1])
    print(f"\n{top_2[0]} vs {top_2[1]}:")
    print(f"  Difference: {test_result['mean_difference']:.4f}")
    print(f"  P-value: {test_result['p_value']:.4f}")
    print(f"  Significant: {test_result['significant']}")
    
    # Multi-metric evaluation
    print("\n6. Multi-Metric Evaluation")
    print("-" * 30)
    best_model = results_df.iloc[0]['Model']
    metrics = framework.multi_metric_evaluation(best_model)
    print(f"\n{best_model} Performance:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    # Visualizations
    print("\n7. Generating Visualizations")
    print("-" * 30)
    framework.plot_model_comparison()
    framework.plot_bias_variance_analysis()
    
    # Final report
    print("\n8. Final Report")
    print("-" * 30)
    report = framework.generate_report()
    print(report)
    
    return framework, results_df


def main():
    """Main function to run all demonstrations."""
    print("MODEL COMPARISON FRAMEWORK DEMONSTRATIONS")
    print("=" * 80)
    print("This demo showcases professional ML model comparison practices.")
    print("Each demonstration focuses on different aspects of the workflow.")
    
    # Run demonstrations
    try:
        # Basic comparison
        framework1, results1 = demonstrate_basic_comparison()
        
        # Imbalanced data comparison
        framework2, results2 = demonstrate_imbalanced_comparison()
        
        # Statistical testing
        framework3, results3 = demonstrate_statistical_testing()
        
        # Model selection criteria
        framework4, results4 = demonstrate_model_selection_criteria()
        
        # Comprehensive example
        framework5, results5 = create_comprehensive_example()
        
        print("\n" + "="*80)
        print("DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Takeaways:")
        print("1. Always use consistent preprocessing in pipelines")
        print("2. Report both mean and standard deviation from CV")
        print("3. Analyze train/CV gaps for bias-variance diagnosis")
        print("4. Apply fair hyperparameter tuning to all models")
        print("5. Consider statistical significance for small differences")
        print("6. Evaluate multiple metrics for imbalanced problems")
        print("7. Balance performance with practical constraints")
        print("8. Document selection rationale beyond just scores")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
