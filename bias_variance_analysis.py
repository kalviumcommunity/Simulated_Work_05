"""
Bias and Variance Analysis for Model Behavior

Comprehensive implementation of bias-variance analysis tools for understanding
model behavior, diagnosing problems, and guiding improvements.

This module covers:
- Mathematical foundation of bias-variance trade-off
- Learning curves for diagnosing underfitting/overfitting
- Cross-validation diagnostics for variance detection
- Practical strategies for reducing bias and variance
- Real-world examples with different algorithm families
- Visualization tools for comprehensive analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time

# Set up warnings filter
warnings.filterwarnings('ignore')


class BiasVarianceAnalyzer:
    """
    Comprehensive bias-variance analysis system for understanding model behavior.
    
    This class provides tools to diagnose whether models suffer from high bias
    (underfitting) or high variance (overfitting), and guides appropriate
    remedies for each situation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the bias-variance analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
    def demonstrate_bias_variance_concepts(self):
        """Demonstrate fundamental bias-variance concepts."""
        print("\n" + "="*70)
        print("BIAS-VARIANCE FUNDAMENTALS")
        print("="*70)
        
        print("""
📖 SCENARIO: Understanding the bias-variance trade-off in machine learning

🎯 GOAL: Learn what bias and variance mean conceptually and mathematically
📊 THEORY: Mathematical decomposition of prediction error
🔧 APPROACH: Visual examples with different model complexities
        """)
        
        # Create simple data to demonstrate concepts
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (100, 2))
        
        # True underlying relationship: y = x² + noise
        true_y = X[:, 0]**2 + np.random.normal(0, 0.1, 100)
        
        # Create models with different complexities
        models = {
            'High Bias (Underfit)': LinearRegression(fit_intercept=False),
            'Balanced': LinearRegression(),
            'High Variance (Overfit)': self._create_high_variance_model(X, true_y)
        }
        
        print("Generated Data: y = x² + noise")
        print(f"X shape: {X.shape}")
        
        # Analyze each model
        results = {}
        
        for name, model in models.items():
            if name == 'High Variance (Overfit)':
                # This model overfits by design
                y_pred = model.predict(X)
                train_error = mean_squared_error(true_y, y_pred)
                
                results[name] = {
                    'train_error': train_error,
                    'model_type': name,
                    'description': 'Memorizes training data including noise'
                }
            else:
                # Fit linear models properly
                model.fit(X, true_y)
                y_pred = model.predict(X)
                train_error = mean_squared_error(true_y, y_pred)
                
                results[name] = {
                    'train_error': train_error,
                    'model_type': name,
                    'description': self._get_model_description(name, model)
                }
        
        # Visualize the concepts
        self._plot_bias_variance_concepts(X, true_y, models, results)
        
        # Mathematical explanation
        self._explain_bias_variance_mathematics()
        
        return results
    
    def _create_high_variance_model(self, X: np.ndarray, y: np.ndarray):
        """Create a high-variance model that overfits training data."""
        # Use a complex model that will overfit
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # Create polynomial features to increase model capacity
        poly_features = PolynomialFeatures(degree=10, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        # Fit high-capacity model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        return model
    
    def _get_model_description(self, name: str, model) -> str:
        """Get description of model behavior."""
        if name == 'High Bias (Underfit)':
            return "Too simple: y = 0 (horizontal line)"
        elif name == 'Balanced':
            coef = model.coef_[0] if len(model.coef_) > 0 else 0
            intercept = model.intercept_ if hasattr(model, 'intercept_') else 0
            return f"Moderate complexity: y = {coef:.2f}x + {intercept:.2f}"
        else:
            return "Very complex: fits noise and patterns"
    
    def _plot_bias_variance_concepts(self, X: np.ndarray, y: np.ndarray, 
                                  models: Dict, results: Dict):
        """Plot bias-variance concepts with visual examples."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: True relationship and data
        axes[0, 0].scatter(X[:, 0], y, alpha=0.6, label='True Relationship')
        axes[0, 0].plot(X[:, 0], X[:, 0]**2, 'r-', linewidth=2, label='True Function: y = x²')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('True Relationship: y = x² + noise')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: High bias model
        if 'High Bias (Underfit)' in models:
            model = models['High Bias (Underfit)']
            y_pred_high_bias = np.zeros_like(y)  # Predicts mean
            
            axes[0, 1].scatter(X[:, 0], y, alpha=0.6, label='True Relationship')
            axes[0, 1].plot(X[:, 0], y_pred_high_bias, 'b-', linewidth=2, label='High Bias Model')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('y')
            axes[0, 1].set_title('High Bias Model: Underfitting')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Balanced model
        if 'Balanced' in models:
            model = models['Balanced']
            y_pred_balanced = model.predict(X)
            
            axes[1, 0].scatter(X[:, 0], y, alpha=0.6, label='True Relationship')
            axes[1, 0].plot(X[:, 0], y_pred_balanced, 'g-', linewidth=2, label='Balanced Model')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            axes[1, 0].set_title('Balanced Model: Good Fit')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: High variance model
        if 'High Variance (Overfit)' in models:
            model = models['High Variance (Overfit)']
            y_pred_high_var = model.predict(X)
            
            axes[1, 1].scatter(X[:, 0], y, alpha=0.6, label='True Relationship')
            axes[1, 1].plot(X[:, 0], y_pred_high_var, 'r-', linewidth=2, label='High Variance Model')
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('y')
            axes[1, 1].set_title('High Variance Model: Overfitting')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Error comparison
        model_names = list(results.keys())
        train_errors = [results[name]['train_error'] for name in model_names]
        
        axes[1, 2].bar(model_names, train_errors, 
                     color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 2].set_ylabel('Training Error (MSE)')
        axes[1, 2].set_title('Model Complexity vs. Training Error')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/bias_variance_concepts.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _explain_bias_variance_mathematics(self):
        """Explain the mathematical foundation of bias-variance trade-off."""
        print("\n" + "="*70)
        print("MATHEMATICAL FOUNDATION")
        print("="*70)
        
        math_explanation = """
MATHEMATICAL DECOMPOSITION OF PREDICTION ERROR:

The total expected prediction error can be decomposed into three fundamental components:

Bias² + Variance + Irreducible Noise = Total Error²

Where:

BIAS² (Bias Error):
Error from wrong assumptions about the relationship form.
Systematic and repeatable across different training sets.
Reducible by increasing model flexibility.

VARIANCE (Variance Error):
Error from model sensitivity to specific training examples.
Changes dramatically with small changes in training data.
Reducible by constraining model complexity or adding more data.

IRREDUCIBLE NOISE:
Inherent randomness in the data-generating process.
Cannot be reduced by any model.
Minimum achievable error = noise level.

KEY INSIGHTS:
1. Bias and variance are NOT independent - they trade off mathematically
2. Reducing one typically increases the other by a comparable amount
3. The optimal point balances both types of error for minimum total
4. This creates the characteristic U-shaped bias-variance curve

PRACTICAL IMPLICATIONS:
• High bias models need more capacity or better features
• High variance models need constraints or more data
• The "best" model depends on your dataset size and noise level
• Cross-validation measures variance directly (std of CV scores)
• Learning curves visualize the trade-off empirically
        """
        
        print(math_explanation)
    
    def demonstrate_learning_curves_diagnosis(self):
        """Demonstrate learning curves for bias-variance diagnosis."""
        print("\n" + "="*70)
        print("LEARNING CURVES FOR BIAS-VARIANCE DIAGNOSIS")
        print("="*70)
        
        print("""
📖 SCENARIO: Using learning curves to diagnose model behavior

🎯 GOAL: Learn to identify high bias vs. high variance from learning patterns
📊 DATASET: Varying complexity to show different behaviors
🔧 APPROACH: Generate and interpret learning curves for multiple models
        """)
        
        # Create datasets with different characteristics
        datasets = {
            'Simple Linear': self._create_simple_data(100, noise=0.1),
            'Complex Linear': self._create_complex_data(100, noise=0.1),
            'Small Sample': self._create_simple_data(50, noise=0.1),
            'Large Sample': self._create_simple_data(1000, noise=0.1)
        }
        
        # Models to test
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=3, 
                                           random_state=self.random_state, n_jobs=-1)
        }
        
        # Generate learning curves for each combination
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, (data_name, (X, y)) in enumerate(datasets.items()):
            for j, (model_name, model) in enumerate(models.items()):
                print(f"\nAnalyzing {model_name} on {data_name} data...")
                
                # Generate learning curve
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy',
                    random_state=self.random_state
                )
                
                # Plot learning curve
                row, col = i, j
                axes[row, col].plot(train_sizes, train_scores.mean(axis=1), 
                                 label='Training Score', linewidth=2)
                axes[row, col].plot(train_sizes, test_scores.mean(axis=1), 
                                 label='CV Score', linewidth=2)
                axes[row, col].fill_between(train_sizes, 
                                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                                        alpha=0.15)
                axes[row, col].fill_between(train_sizes,
                                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                                        test_scores.mean(axis=1) + test_scores.std(axis=1),
                                        alpha=0.15)
                axes[row, col].set_xlabel('Training Set Size')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].set_title(f'{model_name} on {data_name}')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
                
                # Diagnose pattern
                train_mean = train_scores.mean(axis=1)[-1]
                test_mean = test_scores.mean(axis=1)[-1]
                train_std = train_scores.std(axis=1)[-1]
                test_std = test_scores.std(axis=1)[-1]
                gap = train_mean - test_mean
                
                diagnosis = self._diagnose_learning_curve(
                    train_mean, test_mean, train_std, test_std, gap
                )
                
                print(f"  Diagnosis: {diagnosis}")
                print(f"  Train Score: {train_mean:.3f} ± {train_std:.3f}")
                print(f"  Test Score:  {test_mean:.3f} ± {test_std:.3f}")
                print(f"  Gap: {gap:.3f}")
        
        plt.tight_layout()
        plt.savefig("plots/learning_curves_diagnosis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return datasets, models
    
    def _create_simple_data(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple linear data for learning curves."""
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (n_samples, 1))
        y = X[:, 0] + np.random.normal(0, noise, n_samples)
        return X, y
    
    def _create_complex_data(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create complex non-linear data for learning curves."""
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (n_samples, 2))
        # Complex non-linear relationship
        y = (X[:, 0]**2 + X[:, 1]**3) + np.random.normal(0, noise, n_samples)
        return X, y
    
    def _diagnose_learning_curve(self, train_mean: float, test_mean: float, 
                           train_std: float, test_std: float, gap: float) -> str:
        """Diagnose learning curve pattern."""
        if gap < 0.05 and train_mean < 0.8 and test_mean < 0.8:
            return "HIGH BIAS: Underfitting - both scores low, small gap"
        elif gap > 0.2 and train_mean > 0.95 and test_std > 0.1:
            return "HIGH VARIANCE: Overfitting - high train score, large gap, high variance"
        elif train_std > 0.1 or test_std > 0.1:
            return "HIGH VARIANCE: Unstable - large variance in CV scores"
        elif 0.05 <= gap <= 0.15 and 0.7 <= test_mean <= 0.9:
            return "GOOD FIT: Balanced - reasonable performance, moderate gap"
        elif train_mean > 0.9 and test_mean > 0.85 and gap < 0.1:
            return "GOOD FIT: Low variance - stable performance, small gap"
        else:
            return "UNCERTAIN: Mixed signals - investigate further"
    
    def demonstrate_cross_validation_diagnostics(self):
        """Demonstrate cross-validation as a variance diagnostic tool."""
        print("\n" + "="*70)
        print("CROSS-VALIDATION VARIANCE DIAGNOSTICS")
        print("="*70)
        
        print("""
📖 SCENARIO: Using CV to detect model instability

🎯 GOAL: Learn how cross-validation reveals variance problems
📊 DATASET: Multiple models with different stability characteristics
🔧 APPROACH: Compare CV scores across models and configurations
        """)
        
        # Create test data
        X, y = self._create_simple_data(500, noise=0.2)
        
        # Models with different stability characteristics
        models = {
            'Stable': Ridge(alpha=10.0),  # High regularization
            'Moderate': Ridge(alpha=1.0),   # Moderate regularization
            'Unstable': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),  # Complex tree
            'Very Unstable': KNeighborsClassifier(n_neighbors=1)  # K=1 memorizes
        }
        
        # Run cross-validation for each model
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nTesting {name} model...")
            
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(10, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'min': cv_scores.min(),
                'max': cv_scores.max(),
                'range': cv_scores.max() - cv_scores.min()
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Range: {cv_results[name]['range']:.4f}")
        
        # Visualize CV results
        self._plot_cv_diagnostics(cv_results)
        
        # Analyze variance patterns
        self._analyze_variance_patterns(cv_results)
        
        return cv_results
    
    def _plot_cv_diagnostics(self, cv_results: Dict):
        """Plot cross-validation diagnostic results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(cv_results.keys())
        means = [cv_results[name]['mean'] for name in names]
        stds = [cv_results[name]['std'] for name in names]
        ranges = [cv_results[name]['range'] for name in names]
        
        # Plot 1: Mean scores with error bars
        axes[0, 0].bar(names, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0, 0].set_ylabel('CV Accuracy (Mean)')
        axes[0, 0].set_title('Cross-Validation Performance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Variance (range of scores)
        axes[0, 1].bar(names, ranges, color='red', alpha=0.7)
        axes[0, 1].set_ylabel('CV Range (Max-Min)')
        axes[0, 1].set_title('Model Stability (Lower is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/cv_diagnostics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_variance_patterns(self, cv_results: Dict):
        """Analyze and explain variance patterns in CV results."""
        print(f"\nVARIANCE PATTERN ANALYSIS:")
        print("-" * 50)
        
        for name, results in cv_results.items():
            stability = "Stable" if results['std'] < 0.02 else "Unstable"
            print(f"{name}: {results['mean']:.4f} ± {results['std']:.4f} ({stability})")
        
        # Find most and least stable models
        stds = {name: results['std'] for name, results in cv_results.keys()}
        most_stable = min(stds, key=stds.get)
        least_stable = max(stds, key=stds.get)
        
        print(f"\nMost Stable Model: {most_stable} (std = {stds[most_stable]:.4f})")
        print(f"Least Stable Model: {least_stable} (std = {stds[least_stable]:.4f})")
        
        # Recommendations
        print(f"\nVARIANCE REDUCTION RECOMMENDATIONS:")
        if stds[least_stable] > 0.1:
            print("• Consider simpler model or stronger regularization")
            print("• Increase training data size")
            print("• Use ensemble methods to average out variance")
        
        if stds[most_stable] < 0.02:
            print("• Model is stable - consider increasing complexity")
            print("• Test on more diverse data")
    
    def demonstrate_practical_strategies(self):
        """Demonstrate practical strategies for managing bias and variance."""
        print("\n" + "="*70)
        print("PRACTICAL BIAS-VARIANCE STRATEGIES")
        print("="*70)
        
        print("""
📖 SCENARIO: Real-world strategies for bias-variance problems

🎯 GOAL: Learn practical techniques for common model issues
📊 DATASET: Multiple scenarios requiring different approaches
🔧 APPROACH: Demonstrate fixes with before/after comparisons
        """)
        
        strategies = [
            ("Reducing High Bias", self._demonstrate_bias_reduction),
            ("Reducing High Variance", self._demonstrate_variance_reduction),
            ("Dataset Size Effects", self._demonstrate_dataset_size_effects),
            ("Algorithm Selection", self._demonstrate_algorithm_selection)
        ]
        
        results = {}
        
        for title, strategy_func in strategies:
            print(f"\n{'='*20} {title} {'='*20}")
            try:
                result = strategy_func()
                results[title] = result
            except Exception as e:
                print(f"Error in {title}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _demonstrate_bias_reduction(self):
        """Demonstrate strategies to reduce high bias (underfitting)."""
        print("STRATEGY: Reducing High Bias (Underfitting)")
        print("-" * 50)
        
        # Create underfitting scenario
        X, y = self._create_complex_data(300, noise=0.2)
        
        # High bias model (underfitting)
        underfit_model = LinearRegression()
        underfit_model.fit(X, y)
        
        # Improved model
        improved_model = self._create_high_variance_model(X, y)
        improved_model.fit(X, y)
        
        # Evaluate
        underfit_score = underfit_model.score(X, y)
        improved_score = improved_model.score(X, y)
        
        print(f"Underfitting Model Score: {underfit_score:.4f}")
        print(f"Improved Model Score:    {improved_score:.4f}")
        print(f"Improvement:            +{improved_score - underfit_score:.4f}")
        
        # Visualize
        self._plot_bias_reduction_comparison(X, y, underfit_model, improved_model)
        
        return {
            'underfit_score': underfit_score,
            'improved_score': improved_score,
            'improvement': improved_score - underfit_score
        }
    
    def _demonstrate_variance_reduction(self):
        """Demonstrate strategies to reduce high variance (overfitting)."""
        print("STRATEGY: Reducing High Variance (Overfitting)")
        print("-" * 50)
        
        # Create overfitting scenario
        X, y = self._create_simple_data(300, noise=0.1)
        
        # High variance model (overfitting)
        overfit_model = self._create_high_variance_model(X, y)
        
        # Regularized model
        regularized_model = Ridge(alpha=10.0)
        regularized_model.fit(X, y)
        
        # Ensemble model
        ensemble_model = RandomForestClassifier(
            n_estimators=50, max_depth=3,
            random_state=self.random_state, n_jobs=-1
        )
        ensemble_model.fit(X, y)
        
        # Evaluate
        overfit_score = overfit_model.score(X, y)
        regularized_score = regularized_model.score(X, y)
        ensemble_score = ensemble_model.score(X, y)
        
        print(f"Overfitting Model Score:  {overfit_score:.4f}")
        print(f"Regularized Model Score:  {regularized_score:.4f}")
        print(f"Ensemble Model Score:       {ensemble_score:.4f}")
        
        # Visualize
        self._plot_variance_reduction_comparison(X, y, overfit_model, 
                                        regularized_model, ensemble_model)
        
        return {
            'overfit_score': overfit_score,
            'regularized_score': regularized_score,
            'ensemble_score': ensemble_score
        }
    
    def _demonstrate_dataset_size_effects(self):
        """Demonstrate how dataset size affects bias-variance trade-off."""
        print("STRATEGY: Dataset Size Effects on Bias-Variance")
        print("-" * 50)
        
        # Test different dataset sizes
        sizes = [100, 500, 1000, 5000]
        model = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)
        
        results = {}
        
        for size in sizes:
            # Create data
            X, y = self._create_simple_data(size, noise=0.2)
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='accuracy'
            )
            
            results[size] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Size {size:4d}: CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualize
        self._plot_dataset_size_effects(results)
        
        return results
    
    def _demonstrate_algorithm_selection(self):
        """Demonstrate algorithm selection based on bias-variance characteristics."""
        print("STRATEGY: Algorithm Selection Guide")
        print("-" * 50)
        
        guide = """
ALGORITHM SELECTION GUIDE:

High Bias Situations (Underfitting):
✓ Linear models with polynomial features
✓ Neural networks with more layers/units
✓ Decision Trees with greater depth
✓ Ensemble methods with individual complex models

High Variance Situations (Overfitting):
✓ Regularized linear models (Ridge, Lasso)
✓ Decision Trees with depth limits
✓ KNN with larger K values
✓ Ensemble methods (Random Forest, Gradient Boosting)
✓ SVM with appropriate regularization

Small Dataset (< 500 samples):
✓ Simple models (high bias acceptable)
✓ Strong regularization
✓ Cross-validation with fewer folds
✓ Consider data augmentation

Large Dataset (> 10,000 samples):
✓ Complex models (can handle variance)
✓ Deep learning architectures
✓ Ensemble methods
✓ Feature selection to reduce dimensionality
✓ More aggressive regularization

Key Principle:
Match model capacity to data size and complexity.
Small data → simpler models (accept some bias)
Large data → complex models (control variance)
        """
        
        print(guide)
        
        return guide
    
    def _plot_bias_reduction_comparison(self, X: np.ndarray, y: np.ndarray,
                                underfit_model, improved_model):
        """Plot bias reduction strategy comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot both models
        underfit_pred = underfit_model.predict(X)
        improved_pred = improved_model.predict(X)
        
        axes[0, 0].scatter(y, underfit_pred, alpha=0.6, label='Underfitting Model')
        axes[0, 0].scatter(y, improved_pred, alpha=0.6, label='Improved Model')
        
        # Perfect prediction line
        min_val = min(y.min(), underfit_pred.min(), improved_pred.min())
        max_val = max(y.max(), underfit_pred.max(), improved_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', 
                        linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Bias Reduction Strategy Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        underfit_r2 = underfit_model.score(X, y)
        improved_r2 = improved_model.score(X, y)
        
        metrics_text = f"""
Underfitting Model:  R² = {underfit_r2:.3f}
Improved Model:      R² = {improved_r2:.3f}
Improvement:          +{improved_r2 - underfit_r2:.3f}
        """
        
        axes[0, 1].text(0.05, 0.95, metrics_text, 
                        transform=axes[0, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig("plots/bias_reduction_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_variance_reduction_comparison(self, X: np.ndarray, y: np.ndarray,
                                  overfit_model, regularized_model, ensemble_model):
        """Plot variance reduction strategy comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Get predictions
        overfit_pred = overfit_model.predict(X)
        regularized_pred = regularized_model.predict(X)
        ensemble_pred = ensemble_model.predict(X)
        
        # Plot all three models
        axes[0, 0].scatter(y, overfit_pred, alpha=0.6, label='Overfitting Model')
        axes[0, 0].scatter(y, regularized_pred, alpha=0.6, label='Regularized Model')
        axes[0, 0].scatter(y, ensemble_pred, alpha=0.6, label='Ensemble Model')
        
        # Perfect prediction line
        min_val = min(y.min(), overfit_pred.min(), regularized_pred.min(), ensemble_pred.min())
        max_val = max(y.max(), overfit_pred.max(), regularized_pred.max(), ensemble_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', 
                        linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Variance Reduction Strategy Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        overfit_score = overfit_model.score(X, y)
        regularized_score = regularized_model.score(X, y)
        ensemble_score = ensemble_model.score(X, y)
        
        metrics_text = f"""
Overfitting Model:    R² = {overfit_score:.3f}
Regularized Model:    R² = {regularized_score:.3f}
Ensemble Model:        R² = {ensemble_score:.3f}
        """
        
        axes[0, 1].text(0.05, 0.95, metrics_text, 
                        transform=axes[0, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig("plots/variance_reduction_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_dataset_size_effects(self, results: Dict):
        """Plot dataset size effects on learning."""
        sizes = list(results.keys())
        means = [results[size]['cv_mean'] for size in sizes]
        stds = [results[size]['cv_std'] for size in sizes]
        
        plt.figure(figsize=(10, 6))
        
        # Plot mean performance
        plt.subplot(1, 2, 1)
        plt.plot(sizes, means, 'bo-', label='Mean CV Score')
        plt.fill_between(sizes, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15)
        plt.xlabel('Dataset Size')
        plt.ylabel('CV Accuracy')
        plt.title('Dataset Size vs. Model Performance')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot variance (stability)
        plt.subplot(1, 2, 2)
        plt.plot(sizes, stds, 'ro-', label='CV Std Dev')
        plt.xlabel('Dataset Size')
        plt.ylabel('CV Stability')
        plt.title('Dataset Size vs. Model Stability')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("plots/dataset_size_effects.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_practical_checklist(self):
        """Print practical checklist for bias-variance analysis."""
        print("\n" + "="*70)
        print("PRACTICAL BIAS-VARIANCE CHECKLIST")
        print("="*70)
        
        checklist = """
🔍 DIAGNOSTIC CHECKLIST:
□ Always compute both training and test performance
□ Calculate train/test gap to detect overfitting
□ Use cross-validation to measure model stability
□ Plot learning curves to visualize bias-variance trade-off
□ Compare multiple algorithms before finalizing
□ Consider dataset size when choosing model complexity
□ Document both mean and standard deviation of CV scores

🎯 STRATEGY SELECTION CHECKLIST:
□ High bias → Increase model complexity or add features
□ High variance → Add regularization or reduce complexity
□ Small dataset → Use simpler models with strong regularization
□ Large dataset → Use ensemble methods or feature selection
□ Unstable performance → Use more data or ensemble methods
□ Always validate improvements with held-out test set

📊 INTERPRETATION GUIDELINES:
□ Bias indicates systematic underfitting from wrong assumptions
□ Variance indicates overfitting to training noise
□ Small train/test gap with low scores = good generalization
□ Large CV standard deviation = unstable model
□ Learning curves show if more data would help
□ Cross-validation variance directly measures model stability

🚨 COMMON MISTAKES TO AVOID:
□ Only looking at test accuracy (hides overfitting)
□ Assuming "more complex always better"
□ Ignoring dataset size effects on model choice
□ Not reporting CV standard deviations
□ Using high-variance model for critical applications
□ Adding data without confirming it reduces variance
□ Not validating that improvements generalize
        """
        
        print(checklist)
    
    def run_complete_tutorial(self):
        """Run the complete bias-variance analysis tutorial."""
        print("""
🎯 BIAS-VARIANCE ANALYSIS TUTORIAL
=========================================

This tutorial provides comprehensive coverage of bias-variance analysis for
understanding model behavior and guiding improvements.

📚 TUTORIAL STRUCTURE:
1. Bias-Variance Fundamental Concepts
2. Learning Curves Diagnosis
3. Cross-Validation Diagnostics
4. Practical Reduction Strategies
5. Algorithm Selection Guide
6. Real-World Application Examples

⏱️  EXPECTED DURATION: 60-90 minutes
        """)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Run all demonstrations
        demonstrations = [
            ("Fundamental Concepts", self.demonstrate_bias_variance_concepts),
            ("Learning Curves", self.demonstrate_learning_curves_diagnosis),
            ("CV Diagnostics", self.demonstrate_cross_validation_diagnostics),
            ("Practical Strategies", self.demonstrate_practical_strategies)
        ]
        
        results = {}
        
        for i, (title, demo_func) in enumerate(demonstrations, 1):
            print(f"\n{'='*20} {i}. {title} {'='*20}")
            
            try:
                result = demo_func()
                results[title] = result
                
                if i < len(demonstrations):
                    input("\nPress Enter to continue to next demonstration...")
                    
            except Exception as e:
                print(f"Error in {title}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print practical checklist
        self.print_practical_checklist()
        
        # Summary
        print("\n" + "="*70)
        print("TUTORIAL COMPLETED!")
        print("="*70)
        
        print(f"\n📁 Files created: plots/")
        print("\n🎯 Key Takeaways:")
        takeaways = [
            "• Bias and variance are fundamental trade-off, not independent problems",
            "• Learning curves are the primary diagnostic tool for model behavior",
            "• Cross-validation standard deviation directly measures model variance",
            "• High bias models need more capacity or better features",
            "• High variance models need constraints or ensemble methods",
            "• Dataset size dramatically affects optimal model complexity",
            "• The bias-variance trade-off cannot be eliminated, only balanced",
            "• Always diagnose before treating symptoms (bias vs. variance)",
            "• Model behavior provides clues about underlying data patterns",
            "• Cross-validation converts guesswork into evidence-based decisions"
        ]
        
        for takeaway in takeaways:
            print(f"  {takeaway}")
        
        return True


def main():
    """Main function to run the bias-variance analysis tutorial."""
    analyzer = BiasVarianceAnalyzer(random_state=42)
    return analyzer.run_complete_tutorial()


if __name__ == "__main__":
    main()
