"""
Hyperparameter Optimization Tutorial

Comprehensive guide to efficient hyperparameter optimization techniques
that scale beyond traditional grid search.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.hyperparameter_optimization import (
    RandomizedSearchOptimizer, ParameterDistribution, UniformDistribution, 
    LogUniformDistribution, IntegerDistribution, CategoricalDistribution,
    create_parameter_distributions, demonstrate_randomized_search,
    demonstrate_distribution_comparison, demonstrate_efficiency_analysis,
    demonstrate_bayesian_optimization, print_optimization_strategies
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_basic_randomized_search():
    """Example 1: Basic RandomizedSearchCV usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC RANDOMIZEDSEARCHCV")
    print("="*60)
    
    print("""
📖 SCENARIO: Optimizing Random Forest Classifier

🎯 GOAL: Find optimal hyperparameters using RandomizedSearchCV
📊 DATASET: Synthetic classification with 1500 samples
🔧 APPROACH: Randomized search with intelligent parameter distributions
    """)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1500, n_features=12, n_informative=8,
        weights=[0.65, 0.35], flip_y=0.01, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Create optimizer
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_distributions = create_parameter_distributions("random_forest")
    
    optimizer = RandomizedSearchOptimizer(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    
    # Run optimization
    print("Running optimization...")
    result = optimizer.optimize(X_train, y_train)
    
    # Print results
    result.print_summary()
    optimizer.print_parameter_analysis()
    
    # Plot optimization history
    optimizer.plot_optimization_history("plots/example1_optimization.png")
    
    # Final evaluation
    print("Final evaluation...")
    best_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **result.best_params
    )
    
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    final_f1 = f1_score(y_test, y_pred)
    
    print(f"Final Test F1-Score: {final_f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    return result


def example_2_efficiency_comparison():
    """Example 2: Efficiency comparison between search methods."""
    print("\n" + "="*60)
    print("EXAMPLE 2: EFFICIENCY COMPARISON")
    print("="*60)
    
    print("""
📖 SCENARIO: Comparing Grid Search vs RandomizedSearchCV efficiency

🎯 GOAL: Demonstrate scaling advantages of RandomizedSearchCV
📊 DATASET: Varying complexity to show scaling differences
🔧 APPROACH: Time both methods, compare results
    """)
    
    complexities = [
        {'n_params': 2, 'n_estimators': [50, 100]},
        {'n_params': 3, 'n_estimators': [50, 100, 200]},
        {'n_params': 4, 'n_estimators': [50, 100, 200, 400]}
    ]
    
    results = {}
    
    for complexity in complexities:
        print(f"\nTesting {complexity['n_params']} parameters, {complexity['n_estimators']} max estimators...")
        
        # Create data
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=8, n_informative=6,
            weights=[0.7, 0.3], flip_y=0.01, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Grid search
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': complexity['n_estimators']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("Running GridSearchCV...")
        start_time = time.time()
        grid = GridSearchCV(
            rf, param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        # Randomized search
        param_distributions = {
            'n_estimators': IntegerDistribution(50, max(complexity['n_estimators']))
        }
        
        optimizer = RandomizedSearchOptimizer(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
            scoring="f1",
            random_state=42,
            n_jobs=-1
        )
        
        print("Running RandomizedSearchCV...")
        start_time = time.time()
        random_result = optimizer.optimize(X_train, y_train)
        random_time = time.time() - start_time
        
        # Store results
        results[complexity['n_params']] = {
            'grid_time': grid_time,
            'grid_score': grid.best_score_,
            'random_time': random_time,
            'random_score': random_result.best_score_,
            'speedup': (grid_time - random_time) / grid_time * 100
        }
    
    # Print comparison
    print(f"\nResults for {complexity['n_params']} parameters:")
    for complexity in complexities:
        result = results[complexity['n_params']]
        print(f"  Grid Search: {result['grid_time']:.2f}s, F1: {result['grid_score']:.4f}")
        print(f"  Random Search: {result['random_time']:.2f}s, F1: {result['random_score']:.4f}")
        if result['speedup'] > 0:
            print(f"  Speedup: {result['speedup']:.1f}% faster")
    
    return results


def example_3_distribution_selection():
    """Example 3: Choosing appropriate parameter distributions."""
    print("\n" + "="*60)
    print("EXAMPLE 3: PARAMETER DISTRIBUTION SELECTION")
    print("="*60)
    
    print("""
📖 SCENARIO: Choosing the right distribution for each parameter type

🎯 GOAL: Demonstrate how distribution choice affects optimization
📊 DATASET: Simple 2D optimization problem
🔧 APPROACH: Compare different distributions for the same parameter
    """)
    
    # Create simple 2D optimization problem
    def objective_function(params):
        """Simple 2D objective: minimize (x - a)² + (y - b)²"""
        x, y = params['x'], params['y']
        return (x - 2.7) ** 2 + (y - 1.5) ** 2
    
    # Test different distributions for parameter 'x'
    x_values_uniform = UniformDistribution(0, 10).sample(n_samples=1000, random_state=42)
    x_values_normal = np.random.normal(5, 1, 1000)  # Approximate normal
    x_values_lognorm = LogUniformDistribution(0.1, 10).sample(n_samples=1000, random_state=42)
    
    # Fixed parameter 'y'
    y_fixed = 1.5
    
    # Calculate objective values
    objectives = {
        'uniform': [objective_function({'x': x, 'y': y_fixed}) for x in x_values_uniform],
        'normal': [objective_function({'x': x, 'y': y_fixed}) for x in x_values_normal],
        'loguniform': [objective_function({'x': x, 'y': y_fixed}) for x in x_values_lognorm]
    }
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distributions = ['uniform', 'normal', 'loguniform']
    colors = ['skyblue', 'orange', 'green']
    
    for i, (dist_name, values) in enumerate(zip(distributions, [x_values_uniform, x_values_normal, x_values_lognorm])):
        axes[i].hist(values, bins=30, alpha=0.7, color=colors[i], label=dist_name)
        axes[i].set_title(f"{dist_name.capitalize()} Distribution")
        axes[i].set_xlabel("x value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/distribution_selection_example.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nObjective Statistics (y = {y_fixed}):")
    for dist_name, values in objectives.items():
        print(f"  {dist_name}: mean = {np.mean(values):.3f}, std = {np.std(values):.3f}")
    
    print("""
💡 KEY INSIGHT:
• Uniform distribution explores entire range equally
• Normal distribution concentrates around mean value
• Log-uniform explores orders of magnitude uniformly
• Choice depends on your prior knowledge about parameter behavior
    """)
    
    return objectives


def example_4_hybrid_strategy():
    """Example 4: Hybrid coarse-to-fine optimization strategy."""
    print("\n" + "="*60)
    print("EXAMPLE 4: HYBRID COARSE-TO-FINE STRATEGY")
    print("="*60)
    
    print("""
📖 SCENARIO: Two-phase optimization combining random search and grid search

🎯 GOAL: Demonstrate practical hybrid optimization workflow
📊 DATASET: Medium complexity problem requiring balanced exploration
🔧 APPROACH: Phase 1 (coarse) + Phase 2 (fine) optimization
    """)
    
    # Create medium complexity data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=7,
        weights=[0.6, 0.4], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Phase 1: Coarse exploration
    print("Phase 1: Coarse exploration with RandomizedSearchCV")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    coarse_param_distributions = {
        'n_estimators': IntegerDistribution(50, 300),
        'max_depth': IntegerDistribution(3, 15),
        'min_samples_leaf': IntegerDistribution(1, 20)
    }
    
    coarse_optimizer = RandomizedSearchOptimizer(
        estimator=rf,
        param_distributions=coarse_param_distributions,
        n_iter=30,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    
    coarse_result = coarse_optimizer.optimize(X_train, y_train)
    
    print(f"Coarse search completed. Best F1: {coarse_result.best_score_:.4f}")
    print(f"Best parameters: {coarse_result.best_params_}")
    
    # Phase 2: Fine refinement around best region
    print("Phase 2: Fine refinement with GridSearchCV")
    
    # Extract best region and create fine grid around it
    best_n_estimators = coarse_result.best_params_['n_estimators']
    best_max_depth = coarse_result.best_params_['max_depth']
    
    fine_param_grid = {
        'n_estimators': [best_n_estimators - 50, best_n_estimators, best_n_estimators + 50, best_n_estimators + 100],
        'max_depth': [best_max_depth - 2, best_max_depth - 1, best_max_depth, best_max_depth + 1, best_max_depth + 2]
    }
    
    from sklearn.model_selection import GridSearchCV
    fine_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        fine_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    
    fine_grid.fit(X_train, y_train)
    
    print(f"Fine search completed. Best F1: {fine_grid.best_score_:.4f}")
    print(f"Best parameters: {fine_grid.best_params_}")
    
    # Comparison
    improvement = fine_grid.best_score_ - coarse_result.best_score_
    print(f"Hybrid strategy improvement: {improvement:+.4f} F1 points")
    
    # Final evaluation
    best_rf = RandomForestClassifier(
        random_state=42, n_jobs=-1,
        **fine_grid.best_params_
    )
    
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    final_f1 = f1_score(y_test, y_pred)
    
    print(f"Final test F1: {final_f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    return {
        'coarse_result': coarse_result,
        'fine_result': fine_grid.best_score_,
        'improvement': improvement,
        'final_f1': final_f1
    }


def example_5_practical_checklist():
    """Example 5: Practical optimization checklist and best practices."""
    print("\n" + "="*60)
    print("EXAMPLE 5: PRACTICAL OPTIMIZATION CHECKLIST")
    print("="*60)
    
    print("""
✅ PRE-OPTIMIZATION CHECKLIST:
□ Define clear optimization objective (accuracy, F1, custom metric)
□ Understand model complexity and parameter sensitivity
□ Choose appropriate search method based on problem complexity
□ Set meaningful parameter ranges based on domain knowledge
□ Use appropriate parameter distributions for each hyperparameter
□ Set random_state for reproducible results
□ Choose CV strategy appropriate for data size and problem
□ Set reasonable n_iter based on exploration budget
□ Monitor convergence and implement early stopping
□ Use parallel processing (n_jobs=-1) when possible
□ Validate final configuration on holdout set
□ Document optimization process and reasoning

🔍 OPTIMIZATION CHECKLIST:
□ Train/test split performed before optimization
□ All preprocessing inside pipeline/estimator
□ No data leakage between CV folds
□ Parameter distributions match parameter semantics
□ Search space covers important dimensions
□ Sufficient iterations for convergence
□ Results reproducible with same random_state
□ Performance compared to baseline
□ Computational efficiency considered
□ Final model validated on unseen data

📊 REPORTING TEMPLATE:
Model: [Model Name and Type]
Search Method: [GridSearchCV/RandomizedSearchCV/Hybrid]
Parameter Space: [Ranges and distributions]
Iterations: [Number used]
Best CV Score: [Best cross-validation score]
Test Score: [Final test performance]
Improvement: [Gain over baseline]
Optimization Time: [Total computational time]
Best Parameters: [Final chosen configuration]
Convergence: [Convergence behavior observed]
Recommendations: [Next steps or deployment notes]

💡 PRO TIPS:
• Start with wide distributions, narrow based on results
• Use log-uniform for scale-free parameters (regularization)
• Use uniform for parameters with equal probability across ranges
• Use integer distributions for discrete parameters
• Plot optimization history to diagnose convergence issues
• Save optimization seeds and parameter ranges for reproducibility
• Consider multi-objective optimization for real-world problems
• Document why certain parameters were chosen over others
    """)
    
    return True


def main():
    """Main tutorial function."""
    print("""
🎯 HYPERPARAMETER OPTIMIZATION TUTORIAL
============================================

This tutorial teaches efficient hyperparameter optimization
techniques that scale beyond traditional grid search.

📚 TUTORIAL STRUCTURE:
1. RandomizedSearchCV Fundamentals
2. Parameter Distributions Deep Dive
3. Efficiency Analysis and Scaling
4. Bayesian Optimization Concepts
5. Hybrid Optimization Strategies
6. Practical Examples with Code
7. Best Practices and Checklist

⏱️  EXPECTED DURATION: 75 minutes
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Run examples
    examples = [
        ("Basic RandomizedSearchCV", example_1_basic_randomized_search),
        ("Efficiency Comparison", example_2_efficiency_comparison),
        ("Distribution Selection", example_3_distribution_selection),
        ("Hybrid Strategy", example_4_hybrid_strategy),
        ("Practical Checklist", example_5_practical_checklist)
    ]
    
    for i, (title, example_func) in enumerate(examples, 1):
        print(f"\n{'='*20} {i+1}. {title} {'='*20}")
        example_func()
        
        if i < len(examples) - 1:
            input("\nPress Enter to continue to next example...")
            input()
    
    print("\n✅ Tutorial completed!")
    print(f"\n📁 Files created: plots/")
    print("\n🎯 Key takeaways:")
    print("  • RandomizedSearchCV provides 80-90% of Bayesian benefits at 1% cost")
    print("  • Parameter distribution choice significantly impacts search efficiency")
    print("  • Hybrid strategies combine exploration speed with precision")
    print("  • Always consider computational budget vs optimization quality")
    print("  • Use appropriate distributions for different parameter types")
    print("  • Monitor convergence and use early stopping when beneficial")
    
    return True


if __name__ == "__main__":
    main()
