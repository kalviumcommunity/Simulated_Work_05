"""
Comprehensive Bias-Variance Trade-Off Analysis

This module provides a complete implementation of bias-variance analysis tools
for understanding model behavior, diagnosing problems, and guiding improvements.

Key Features:
- Mathematical foundation and conceptual understanding
- Learning curves for diagnosing underfitting/overfitting
- Cross-validation diagnostics for variance detection
- Algorithm-specific bias-variance characteristics
- Practical strategies for reducing bias and variance
- Real-world examples with multiple algorithm families
- Comprehensive visualization tools

Author: Machine Learning Practitioner
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import time
import logging

# ML imports
from sklearn.model_selection import (train_test_split, learning_curve, validation_curve, 
                                   cross_val_score, StratifiedKFold, GridSearchCV)
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score, 
                             classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Set up warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ComprehensiveBiasVarianceAnalyzer:
    """
    Comprehensive bias-variance analysis system for understanding model behavior.
    
    This class provides complete tools to diagnose whether models suffer from 
    high bias (underfitting) or high variance (overfitting), and guides 
    appropriate remedies for each situation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the comprehensive bias-variance analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        self.plots_dir = "plots"
        
        # Create plots directory
        import os
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info("Comprehensive Bias-Variance Analyzer initialized")
    
    def demonstrate_bias_variance_fundamentals(self) -> Dict:
        """
        Demonstrate fundamental bias-variance concepts with mathematical foundation.
        
        Returns:
            Dictionary containing analysis results
        """
        print("\n" + "="*80)
        print("BIAS-VARIANCE FUNDAMENTALS: MATHEMATICAL FOUNDATION")
        print("="*80)
        
        print("""
📖 SCENARIO: Understanding the mathematical foundation of bias-variance trade-off

🎯 GOAL: Master the conceptual and mathematical understanding of bias and variance
📊 THEORY: Bias² + Variance + Irreducible Noise = Total Error²
🔧 APPROACH: Visual examples with different model complexities and mathematical explanations
        """)
        
        # Create synthetic data with known relationship
        X = np.random.normal(0, 1, (200, 1))
        # True underlying relationship: y = 2x² + 3x + 1 + noise
        true_y = 2 * X[:, 0]**2 + 3 * X[:, 0] + 1 + np.random.normal(0, 0.5, 200)
        
        # Split for proper evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, true_y, test_size=0.3, random_state=self.random_state
        )
        
        # Models across the bias-variance spectrum
        models = {
            'Very High Bias': LinearRegression(fit_intercept=False),  # Forces y = 0
            'High Bias': LinearRegression(),  # Simple linear
            'Moderate Bias': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),  # Can capture quadratic
            'Low Bias': Pipeline([
                ('poly', PolynomialFeatures(degree=5)),
                ('linear', LinearRegression())
            ]),  # Very flexible
            'Very Low Bias': Pipeline([
                ('poly', PolynomialFeatures(degree=15)),
                ('linear', Ridge(alpha=0.001))
            ]),  # Extremely flexible
        }
        
        print(f"Generated Data: y = 2x² + 3x + 1 + noise")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Analyze each model
        results = {}
        
        for name, model in models.items():
            print(f"\nAnalyzing {name} model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross-validation for variance assessment
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mse': cv_mse,
                'cv_std': cv_std,
                'gap': test_mse - train_mse,
                'model': model
            }
            
            print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
            print(f"  Train R²:  {train_r2:.4f}, Test R²:  {test_r2:.4f}")
            print(f"  CV MSE:    {cv_mse:.4f} ± {cv_std:.4f}")
            print(f"  Gap:       {test_mse - train_mse:.4f}")
        
        # Visualize the concepts
        self._plot_bias_variance_fundamentals(X, true_y, models, results)
        
        # Mathematical explanation
        self._explain_mathematical_foundation()
        
        # Diagnose each model
        self._diagnose_models(results)
        
        return results
    
    def _plot_bias_variance_fundamentals(self, X: np.ndarray, y: np.ndarray, 
                                       models: Dict, results: Dict):
        """Plot comprehensive bias-variance fundamentals visualization."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Sort models by complexity for better visualization
        model_order = ['Very High Bias', 'High Bias', 'Moderate Bias', 'Low Bias', 'Very Low Bias']
        
        # Plot 1: True relationship and data
        ax1 = fig.add_subplot(gs[0, 0])
        X_sorted = np.sort(X[:, 0])
        y_true_sorted = 2 * X_sorted**2 + 3 * X_sorted + 1
        ax1.scatter(X[:, 0], y, alpha=0.6, s=30, label='Data Points')
        ax1.plot(X_sorted, y_true_sorted, 'r-', linewidth=3, label='True Function')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('True Relationship: y = 2x² + 3x + 1 + noise')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plots 2-6: Different model fits
        for i, model_name in enumerate(model_order[:5]):
            ax = fig.add_subplot(gs[i//2, (i%2)+1])
            
            model = models[model_name]
            y_pred = model.predict(X)
            
            # Plot data and model prediction
            ax.scatter(X[:, 0], y, alpha=0.4, s=20, label='Data')
            
            # Sort for smooth line plot
            sort_idx = np.argsort(X[:, 0])
            ax.plot(X[sort_idx, 0], y_pred[sort_idx], linewidth=2, 
                   label=f'{model_name}', alpha=0.8)
            
            # Add true function for comparison
            ax.plot(X_sorted, y_true_sorted, 'r--', linewidth=1, alpha=0.5, 
                   label='True Function')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{model_name}\nTest R² = {results[model_name]["test_r2"]:.3f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 7: Error comparison
        ax7 = fig.add_subplot(gs[2, 0])
        model_names = list(results.keys())
        train_errors = [results[name]['train_mse'] for name in model_names]
        test_errors = [results[name]['test_mse'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        ax7.bar(x_pos - width/2, train_errors, width, label='Train MSE', alpha=0.7)
        ax7.bar(x_pos + width/2, test_errors, width, label='Test MSE', alpha=0.7)
        ax7.set_xlabel('Model Complexity')
        ax7.set_ylabel('Mean Squared Error')
        ax7.set_title('Training vs. Test Error')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(model_names, rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Bias-Variance Trade-off Curve
        ax8 = fig.add_subplot(gs[2, 1])
        complexities = np.arange(len(model_names))
        test_errors = [results[name]['test_mse'] for name in model_names]
        
        ax8.plot(complexities, test_errors, 'bo-', linewidth=2, markersize=8)
        ax8.set_xlabel('Model Complexity')
        ax8.set_ylabel('Test Error')
        ax8.set_title('Bias-Variance Trade-off Curve')
        ax8.set_xticks(complexities)
        ax8.set_xticklabels(model_names, rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)
        
        # Add annotations for bias and variance regions
        ax8.axvspan(-0.5, 1.5, alpha=0.2, color='blue', label='High Bias Region')
        ax8.axvspan(2.5, 4.5, alpha=0.2, color='red', label='High Variance Region')
        ax8.legend()
        
        # Plot 9: Cross-validation stability
        ax9 = fig.add_subplot(gs[2, 2])
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        ax9.bar(model_names, cv_stds, color='orange', alpha=0.7)
        ax9.set_ylabel('CV Standard Deviation')
        ax9.set_title('Model Stability (Lower is Better)')
        ax9.tick_params(axis='x', rotation=45)
        ax9.grid(True, alpha=0.3)
        
        # Plot 10: Gap analysis
        ax10 = fig.add_subplot(gs[2, 3])
        gaps = [results[name]['gap'] for name in model_names]
        
        colors = ['red' if gap > 1.0 else 'orange' if gap > 0.5 else 'green' 
                 for gap in gaps]
        ax10.bar(model_names, gaps, color=colors, alpha=0.7)
        ax10.set_ylabel('Test-Train Error Gap')
        ax10.set_title('Overfitting Indicator')
        ax10.tick_params(axis='x', rotation=45)
        ax10.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Bias-Variance Analysis', fontsize=16, fontweight='bold')
        plt.savefig(f"{self.plots_dir}/bias_variance_fundamentals.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _explain_mathematical_foundation(self):
        """Explain the mathematical foundation of bias-variance trade-off."""
        print("\n" + "="*80)
        print("MATHEMATICAL FOUNDATION OF BIAS-VARIANCE TRADE-OFF")
        print("="*80)
        
        math_explanation = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MATHEMATICAL DECOMPOSITION                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

The total expected prediction error can be decomposed into three fundamental components:

Bias² + Variance + Irreducible Noise = Total Error²

┌─────────────────────────────────────────────────────────────────────────────┐
│ BIAS² (Bias Error)                                                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│ • Error from wrong assumptions about the relationship form                  │
│ • Systematic and repeatable across different training sets                  │
│ • Reducible by increasing model flexibility                                 │
│ • High bias = underfitting (model too simple)                               │
│                                                                             │
│ Example: Linear model trying to fit quadratic data                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ VARIANCE (Variance Error)                                                   │
│ ─────────────────────────────────────────────────────────────────────────── │
│ • Error from model sensitivity to specific training examples               │
│ • Changes dramatically with small changes in training data                  │
│ • Reducible by constraining model complexity or adding more data           │
│ • High variance = overfitting (model too complex)                          │
│                                                                             │
│ Example: High-degree polynomial fitting noise                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ IRREDUCIBLE NOISE                                                           │
│ ─────────────────────────────────────────────────────────────────────────── │
│ • Inherent randomness in the data-generating process                        │
│ • Cannot be reduced by any model                                            │
│ • Minimum achievable error = noise level                                    │
│                                                                             │
│ Example: Measurement error, random fluctuations                            │
└─────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                      KEY MATHEMATICAL INSIGHTS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Bias and variance are NOT independent - they trade off mathematically
2. Reducing one typically increases the other by a comparable amount
3. The optimal point balances both types of error for minimum total
4. This creates the characteristic U-shaped bias-variance curve
5. The trade-off is a mathematical constraint, not a modeling failure

Bias²(x) = [E[f̂(x)] - f(x)]²
Variance(x) = E[(f̂(x) - E[f̂(x)])²]
Total Error(x) = Bias²(x) + Variance(x) + σ²

Where:
• f̂(x) = predicted function
• f(x) = true function  
• E[.] = expectation over training sets
• σ² = irreducible noise

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRACTICAL IMPLICATIONS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

• High bias models need more capacity or better features
• High variance models need constraints or more data
• The "best" model depends on dataset size and noise level
• Cross-validation measures variance directly (std of CV scores)
• Learning curves visualize the trade-off empirically
• No model can eliminate both bias and variance simultaneously

The goal is NOT to eliminate bias or variance individually.
The goal is to find the optimal balance that minimizes total error.
        """
        
        print(math_explanation)
    
    def _diagnose_models(self, results: Dict):
        """Diagnose bias-variance characteristics of each model."""
        print("\n" + "="*80)
        print("MODEL DIAGNOSIS: BIAS-VARIANCE CHARACTERISTICS")
        print("="*80)
        
        print(f"{'Model':<20} {'Diagnosis':<25} {'Train R²':<12} {'Test R²':<12} {'Gap':<10} {'CV Std':<10}")
        print("-" * 90)
        
        for name, result in results.items():
            train_r2 = result['train_r2']
            test_r2 = result['test_r2']
            gap = result['gap']
            cv_std = result['cv_std']
            
            # Diagnosis logic
            if train_r2 < 0.7 and test_r2 < 0.7 and gap < 0.5:
                diagnosis = "HIGH BIAS (Underfitting)"
            elif train_r2 > 0.9 and test_r2 < 0.7 and gap > 1.0:
                diagnosis = "HIGH VARIANCE (Overfitting)"
            elif train_r2 > 0.9 and test_r2 > 0.85 and gap < 0.2:
                diagnosis = "GOOD FIT (Low Variance)"
            elif 0.7 <= train_r2 <= 0.9 and 0.7 <= test_r2 <= 0.9 and 0.2 <= gap <= 0.5:
                diagnosis = "BALANCED (Good Trade-off)"
            else:
                diagnosis = "UNCERTAIN (Mixed Signals)"
            
            print(f"{name:<20} {diagnosis:<25} {train_r2:<12.3f} {test_r2:<12.3f} {gap:<10.3f} {cv_std:<10.3f}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR EACH MODEL")
        print("="*80)
        
        recommendations = {
            "HIGH BIAS (Underfitting)": [
                "✓ Increase model complexity (add features, use non-linear models)",
                "✓ Add polynomial or interaction features", 
                "✓ Reduce regularization strength",
                "✓ Use more flexible algorithms (trees, neural networks)",
                "✓ Feature engineering to capture more signal"
            ],
            "HIGH VARIANCE (Overfitting)": [
                "✓ Apply regularization (L1, L2, dropout)",
                "✓ Reduce model complexity (limit depth, features)",
                "✓ Collect more training data",
                "✓ Use ensemble methods (Random Forest, bagging)",
                "✓ Feature selection to remove noisy features"
            ],
            "GOOD FIT (Low Variance)": [
                "✓ Model is performing well - consider deployment",
                "✓ Monitor for drift in production",
                "✓ Consider slight complexity increase if underfitting"
            ],
            "BALANCED (Good Trade-off)": [
                "✓ Good balance found - fine-tune if needed",
                "✓ Consider cross-validation for hyperparameter optimization",
                "✓ Test on more diverse data if available"
            ]
        }
        
        for diagnosis, recs in recommendations.items():
            print(f"\n{diagnosis}:")
            for rec in recs:
                print(f"  {rec}")
    
    def demonstrate_learning_curves_diagnosis(self) -> Dict:
        """
        Demonstrate comprehensive learning curves for bias-variance diagnosis.
        
        Returns:
            Dictionary containing learning curve analysis results
        """
        print("\n" + "="*80)
        print("LEARNING CURVES: COMPREHENSIVE BIAS-VARIANCE DIAGNOSIS")
        print("="*80)
        
        print("""
📖 SCENARIO: Using learning curves to diagnose model behavior patterns

🎯 GOAL: Master identification of high bias vs. high variance from learning patterns
📊 DATASET: Multiple scenarios showing different bias-variance characteristics
🔧 APPROACH: Generate and interpret learning curves for various model-dataset combinations
        """)
        
        # Create datasets with different characteristics
        datasets = {
            'Simple Linear (Low Noise)': self._create_dataset('simple_linear', 500, 0.1),
            'Complex Non-linear': self._create_dataset('complex_nonlinear', 500, 0.2),
            'Small Sample': self._create_dataset('simple_linear', 100, 0.15),
            'Large Sample': self._create_dataset('simple_linear', 2000, 0.15),
            'High Noise': self._create_dataset('simple_linear', 500, 0.5)
        }
        
        # Models with different bias-variance characteristics
        models = {
            'High Bias (Linear)': LinearRegression(),
            'Moderate Bias (Ridge)': Ridge(alpha=1.0),
            'Low Bias (Poly3)': Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression())
            ]),
            'Very Low Bias (Poly10)': Pipeline([
                ('poly', PolynomialFeatures(degree=10)),
                ('linear', Ridge(alpha=0.01))
            ]),
            'High Variance (KNN-1)': KNeighborsRegressor(n_neighbors=1),
            'Ensemble (RF)': RandomForestRegressor(n_estimators=50, max_depth=5, 
                                                 random_state=self.random_state)
        }
        
        # Generate comprehensive learning curves
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\n{'='*20} Analyzing {data_name} {'='*20}")
            results[data_name] = {}
            
            for model_name, model in models.items():
                print(f"\nAnalyzing {model_name} on {data_name}...")
                
                # Generate learning curve
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='neg_mean_squared_error',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                # Convert to positive MSE
                train_mse = -train_scores
                test_mse = -test_scores
                
                # Calculate key metrics
                final_train_mse = train_mse.mean(axis=1)[-1]
                final_test_mse = test_mse.mean(axis=1)[-1]
                final_train_std = train_mse.std(axis=1)[-1]
                final_test_std = test_mse.std(axis=1)[-1]
                gap = final_test_mse - final_train_mse
                
                # Diagnose learning curve pattern
                diagnosis = self._diagnose_learning_curve_pattern(
                    final_train_mse, final_test_mse, final_train_std, 
                    final_test_std, gap, train_sizes, train_mse, test_mse
                )
                
                results[data_name][model_name] = {
                    'train_sizes': train_sizes,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'final_train_mse': final_train_mse,
                    'final_test_mse': final_test_mse,
                    'final_train_std': final_train_std,
                    'final_test_std': final_test_std,
                    'gap': gap,
                    'diagnosis': diagnosis
                }
                
                print(f"  Final Train MSE: {final_train_mse:.4f} ± {final_train_std:.4f}")
                print(f"  Final Test MSE:  {final_test_mse:.4f} ± {final_test_std:.4f}")
                print(f"  Gap:             {gap:.4f}")
                print(f"  Diagnosis:       {diagnosis}")
        
        # Create comprehensive visualization
        self._plot_comprehensive_learning_curves(datasets, models, results)
        
        return results
    
    def _create_dataset(self, dataset_type: str, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create different types of datasets for learning curve analysis."""
        np.random.seed(self.random_state)
        
        if dataset_type == 'simple_linear':
            # Simple linear relationship
            X = np.random.normal(0, 1, (n_samples, 1))
            y = 2 * X[:, 0] + np.random.normal(0, noise, n_samples)
            
        elif dataset_type == 'complex_nonlinear':
            # Complex non-linear relationship
            X = np.random.normal(0, 1, (n_samples, 2))
            y = (X[:, 0]**2 + X[:, 1]**3 + np.sin(X[:, 0]) * 
                 np.cos(X[:, 1]) + np.random.normal(0, noise, n_samples))
        else:
            # Default to simple linear
            X = np.random.normal(0, 1, (n_samples, 1))
            y = 2 * X[:, 0] + np.random.normal(0, noise, n_samples)
        
        return X, y
    
    def _diagnose_learning_curve_pattern(self, train_mse: float, test_mse: float, 
                                       train_std: float, test_std: float, gap: float,
                                       train_sizes: np.ndarray, train_scores: np.ndarray, 
                                       test_scores: np.ndarray) -> str:
        """Diagnose learning curve pattern for bias-variance analysis."""
        
        # Check convergence
        train_trend = train_scores.mean(axis=1)[-1] - train_scores.mean(axis=1)[0]
        test_trend = test_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[0]
        
        # High bias indicators
        if (gap < 0.5 and train_mse > 1.0 and test_mse > 1.0 and 
            abs(train_trend) < 0.1 and abs(test_trend) < 0.1):
            return "HIGH BIAS: Converged at high error - model too simple"
        
        # High variance indicators  
        elif (gap > 2.0 and train_mse < 0.1 and test_std > 0.5 and 
              test_trend < -0.5):
            return "HIGH VARIANCE: Large gap, unstable - model overfitting"
        
        # Good fit indicators
        elif (0.1 <= gap <= 0.5 and 0.1 <= test_mse <= 0.5 and 
              test_std < 0.2 and test_trend < -0.1):
            return "GOOD FIT: Balanced performance with reasonable gap"
        
        # Data limited indicators
        elif (gap > 1.0 and test_trend < -0.3 and 
             test_scores.mean(axis=1)[-1] > test_scores.mean(axis=1)[0]):
            return "DATA LIMITED: More data would help reduce variance"
        
        # Model limited indicators
        elif (gap < 0.2 and train_mse > 0.5 and test_mse > 0.5 and 
              abs(test_trend) < 0.05):
            return "MODEL LIMITED: More data won't help - need more complex model"
        
        else:
            return "UNCERTAIN: Mixed signals - investigate further"
    
    def _plot_comprehensive_learning_curves(self, datasets: Dict, models: Dict, 
                                          results: Dict):
        """Plot comprehensive learning curves analysis."""
        # Create large figure for all learning curves
        fig, axes = plt.subplots(len(datasets), len(models), 
                                figsize=(6*len(models), 4*len(datasets)))
        
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        if len(models) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (data_name, (X, y)) in enumerate(datasets.items()):
            for j, (model_name, model) in enumerate(models.items()):
                ax = axes[i, j]
                
                result = results[data_name][model_name]
                train_sizes = result['train_sizes']
                train_mse_mean = result['train_mse'].mean(axis=1)
                train_mse_std = result['train_mse'].std(axis=1)
                test_mse_mean = result['test_mse'].mean(axis=1)
                test_mse_std = result['test_mse'].std(axis=1)
                
                # Plot learning curves
                ax.plot(train_sizes, train_mse_mean, 'b-', linewidth=2, 
                       label='Training Error')
                ax.fill_between(train_sizes, 
                               train_mse_mean - train_mse_std,
                               train_mse_mean + train_mse_std,
                               alpha=0.15, color='blue')
                
                ax.plot(train_sizes, test_mse_mean, 'r-', linewidth=2, 
                       label='Validation Error')
                ax.fill_between(train_sizes,
                               test_mse_mean - test_mse_std,
                               test_mse_mean + test_mse_std,
                               alpha=0.15, color='red')
                
                # Styling
                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('Mean Squared Error')
                ax.set_title(f'{model_name}\n{data_name}\n{result["diagnosis"]}', 
                           fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/comprehensive_learning_curves.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_cross_validation_diagnostics(self) -> Dict:
        """
        Demonstrate cross-validation as a comprehensive variance diagnostic tool.
        
        Returns:
            Dictionary containing CV diagnostic results
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION: COMPREHENSIVE VARIANCE DIAGNOSTICS")
        print("="*80)
        
        print("""
📖 SCENARIO: Using cross-validation to detect model instability and variance

🎯 GOAL: Master CV techniques for variance detection and model assessment
📊 DATASET: Multiple models with different stability characteristics
🔧 APPROACH: Comprehensive CV analysis with multiple metrics and visualizations
        """)
        
        # Create test data
        X, y = self._create_dataset('complex_nonlinear', 800, 0.2)
        
        # Models with different stability characteristics
        models = {
            'Very Stable': Ridge(alpha=10.0),  # High regularization
            'Stable': Ridge(alpha=1.0),        # Moderate regularization
            'Moderate': DecisionTreeRegressor(max_depth=3, random_state=self.random_state),
            'Unstable': DecisionTreeRegressor(max_depth=10, random_state=self.random_state),
            'Very Unstable': KNeighborsRegressor(n_neighbors=1),  # K=1 memorizes
            'Ensemble Stable': RandomForestRegressor(n_estimators=100, max_depth=5,
                                                   random_state=self.random_state, n_jobs=-1)
        }
        
        # Run comprehensive cross-validation analysis
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nAnalyzing {name} model...")
            
            # Multiple CV configurations
            cv_configs = {
                '5-fold': 5,
                '10-fold': 10,
                'LOOCV': len(X) if len(X) < 200 else 20  # Leave-one-out for small datasets
            }
            
            model_results = {}
            
            for cv_name, cv_folds in cv_configs.items():
                if cv_name == 'LOOCV' and len(X) > 200:
                    # Use 20-fold for larger datasets instead of true LOOCV
                    cv_folds = 20
                    cv_display_name = '20-fold'
                else:
                    cv_display_name = cv_name
                
                # Cross-validation with multiple metrics
                scoring_metrics = ['neg_mean_squared_error', 'r2']
                
                metric_results = {}
                for metric in scoring_metrics:
                    cv_scores = cross_val_score(
                        model, X, y, 
                        cv=cv_folds, 
                        scoring=metric,
                        n_jobs=-1
                    )
                    
                    if metric == 'neg_mean_squared_error':
                        cv_scores = -cv_scores  # Convert to positive MSE
                        metric_name = 'MSE'
                    else:
                        metric_name = 'R²'
                    
                    metric_results[metric_name] = {
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std(),
                        'min': cv_scores.min(),
                        'max': cv_scores.max(),
                        'range': cv_scores.max() - cv_scores.min(),
                        'scores': cv_scores
                    }
                
                model_results[cv_display_name] = metric_results
            
            cv_results[name] = model_results
            
            # Print summary for 10-fold CV
            summary = model_results['10-fold']['MSE']
            print(f"  10-fold CV MSE: {summary['mean']:.4f} ± {summary['std']:.4f}")
            print(f"  Range: {summary['range']:.4f}")
            print(f"  Stability: {'High' if summary['std'] < 0.1 else 'Medium' if summary['std'] < 0.5 else 'Low'}")
        
        # Visualize CV diagnostics
        self._plot_cv_diagnostics(cv_results)
        
        # Analyze variance patterns
        self._analyze_comprehensive_variance_patterns(cv_results)
        
        return cv_results
    
    def _plot_cv_diagnostics(self, cv_results: Dict):
        """Plot comprehensive cross-validation diagnostic results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        names = list(cv_results.keys())
        
        # Extract 10-fold CV results for comparison
        mse_means = []
        mse_stds = []
        r2_means = []
        r2_stds = []
        
        for name in names:
            cv_10fold = cv_results[name]['10-fold']
            mse_means.append(cv_10fold['MSE']['mean'])
            mse_stds.append(cv_10fold['MSE']['std'])
            r2_means.append(cv_10fold['R²']['mean'])
            r2_stds.append(cv_10fold['R²']['std'])
        
        # Plot 1: MSE comparison with error bars
        axes[0, 0].bar(names, mse_means, yerr=mse_stds, capsize=5, 
                      alpha=0.7, color='skyblue')
        axes[0, 0].set_ylabel('CV MSE (Mean)')
        axes[0, 0].set_title('Cross-Validation Performance (MSE)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: R² comparison with error bars
        axes[0, 1].bar(names, r2_means, yerr=r2_stds, capsize=5,
                      alpha=0.7, color='lightcoral')
        axes[0, 1].set_ylabel('CV R² (Mean)')
        axes[0, 1].set_title('Cross-Validation Performance (R²)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Variance comparison (stability)
        axes[0, 2].bar(names, mse_stds, color='orange', alpha=0.7)
        axes[0, 2].set_ylabel('CV MSE Standard Deviation')
        axes[0, 2].set_title('Model Stability (Lower is Better)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs. Stability scatter
        axes[1, 0].scatter(mse_stds, mse_means, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (mse_stds[i], mse_means[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('CV MSE Std (Variance)')
        axes[1, 0].set_ylabel('CV MSE Mean (Performance)')
        axes[1, 0].set_title('Performance vs. Stability Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: CV score distributions for most and least stable models
        most_stable_idx = np.argmin(mse_stds)
        least_stable_idx = np.argmax(mse_stds)
        
        most_stable_scores = cv_results[names[most_stable_idx]]['10-fold']['MSE']['scores']
        least_stable_scores = cv_results[names[least_stable_idx]]['10-fold']['MSE']['scores']
        
        axes[1, 1].hist(most_stable_scores, alpha=0.7, label=f'Most Stable ({names[most_stable_idx]})', 
                        bins=10, color='green')
        axes[1, 1].hist(least_stable_scores, alpha=0.7, label=f'Least Stable ({names[least_stable_idx]})', 
                        bins=10, color='red')
        axes[1, 1].set_xlabel('CV MSE Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('CV Score Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Range analysis
        mse_ranges = []
        for name in names:
            mse_ranges.append(cv_results[name]['10-fold']['MSE']['range'])
        
        axes[1, 2].bar(names, mse_ranges, color='purple', alpha=0.7)
        axes[1, 2].set_ylabel('CV MSE Range (Max-Min)')
        axes[1, 2].set_title('Score Range Analysis')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/comprehensive_cv_diagnostics.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_comprehensive_variance_patterns(self, cv_results: Dict):
        """Analyze and explain variance patterns in comprehensive CV results."""
        print(f"\n" + "="*80)
        print("COMPREHENSIVE VARIANCE PATTERN ANALYSIS")
        print("="*80)
        
        print(f"{'Model':<20} {'MSE Mean':<12} {'MSE Std':<12} {'R² Mean':<12} {'R² Std':<12} {'Stability':<12}")
        print("-" * 85)
        
        stability_ratings = {}
        
        for name, results in cv_results.items():
            cv_10fold = results['10-fold']
            mse_mean = cv_10fold['MSE']['mean']
            mse_std = cv_10fold['MSE']['std']
            r2_mean = cv_10fold['R²']['mean']
            r2_std = cv_10fold['R²']['std']
            
            # Stability rating
            if mse_std < 0.1:
                stability = "Very High"
            elif mse_std < 0.3:
                stability = "High"
            elif mse_std < 0.8:
                stability = "Medium"
            else:
                stability = "Low"
            
            stability_ratings[name] = mse_std
            
            print(f"{name:<20} {mse_mean:<12.4f} {mse_std:<12.4f} {r2_mean:<12.4f} {r2_std:<12.4f} {stability:<12}")
        
        # Find most and least stable models
        most_stable = min(stability_ratings, key=stability_ratings.get)
        least_stable = max(stability_ratings, key=stability_ratings.get)
        
        print(f"\nMost Stable Model: {most_stable} (MSE Std = {stability_ratings[most_stable]:.4f})")
        print(f"Least Stable Model: {least_stable} (MSE Std = {stability_ratings[least_stable]:.4f})")
        
        # Comprehensive recommendations
        print(f"\n" + "="*80)
        print("VARIANCE MANAGEMENT RECOMMENDATIONS")
        print("="*80)
        
        high_variance_models = [name for name, std in stability_ratings.items() if std > 0.5]
        stable_models = [name for name, std in stability_ratings.items() if std < 0.2]
        
        if high_variance_models:
            print(f"\nHIGH VARIANCE MODELS (Need stabilization):")
            for model in high_variance_models:
                print(f"  • {model}: Consider regularization, simpler architecture, or ensembling")
        
        if stable_models:
            print(f"\nSTABLE MODELS (Good for deployment):")
            for model in stable_models:
                print(f"  • {model}: Low variance - suitable for production")
        
        print(f"\nGENERAL VARIANCE REDUCTION STRATEGIES:")
        print("  • Increase training data size")
        print("  • Apply regularization (L1, L2, dropout)")
        print("  • Use ensemble methods (bagging, Random Forest)")
        print("  • Reduce model complexity")
        print("  • Feature selection to remove noisy features")
        print("  • Cross-validation for hyperparameter tuning")
        print("  • Early stopping for iterative models")
    
    def demonstrate_practical_strategies(self) -> Dict:
        """
        Demonstrate practical strategies for managing bias and variance.
        
        Returns:
            Dictionary containing strategy demonstration results
        """
        print("\n" + "="*80)
        print("PRACTICAL STRATEGIES: BIAS-VARIANCE MANAGEMENT")
        print("="*80)
        
        print("""
📖 SCENARIO: Real-world strategies for common bias-variance problems

🎯 GOAL: Master practical techniques for model improvement
📊 DATASET: Multiple scenarios requiring different approaches  
🔧 APPROACH: Demonstrate fixes with comprehensive before/after comparisons
        """)
        
        strategies = [
            ("Reducing High Bias", self._demonstrate_comprehensive_bias_reduction),
            ("Reducing High Variance", self._demonstrate_comprehensive_variance_reduction),
            ("Dataset Size Effects", self._demonstrate_dataset_size_impact),
            ("Algorithm Selection Guide", self._demonstrate_algorithm_selection),
            ("Ensemble Methods", self._demonstrate_ensemble_techniques)
        ]
        
        results = {}
        
        for title, strategy_func in strategies:
            print(f"\n{'='*30} {title} {'='*30}")
            try:
                result = strategy_func()
                results[title] = result
                print(f"✓ {title} completed successfully")
            except Exception as e:
                print(f"✗ Error in {title}: {str(e)}")
                logger.error(f"Error in {title}: {str(e)}")
                results[title] = {"error": str(e)}
        
        return results
    
    def _demonstrate_comprehensive_bias_reduction(self) -> Dict:
        """Demonstrate comprehensive strategies to reduce high bias."""
        print("STRATEGY: Comprehensive High Bias (Underfitting) Reduction")
        print("-" * 60)
        
        # Create challenging underfitting scenario
        X, y = self._create_dataset('complex_nonlinear', 500, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Strategy 1: Baseline high bias model
        baseline_model = LinearRegression()
        baseline_model.fit(X_train, y_train)
        baseline_train_score = baseline_model.score(X_train, y_train)
        baseline_test_score = baseline_model.score(X_test, y_test)
        
        # Strategy 2: Add polynomial features
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        poly_model.fit(X_train, y_train)
        poly_train_score = poly_model.score(X_train, y_train)
        poly_test_score = poly_model.score(X_test, y_test)
        
        # Strategy 3: Use more flexible algorithm
        tree_model = DecisionTreeRegressor(max_depth=8, random_state=self.random_state)
        tree_model.fit(X_train, y_train)
        tree_train_score = tree_model.score(X_train, y_train)
        tree_test_score = tree_model.score(X_test, y_test)
        
        # Strategy 4: Ensemble method
        ensemble_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                            random_state=self.random_state, n_jobs=-1)
        ensemble_model.fit(X_train, y_train)
        ensemble_train_score = ensemble_model.score(X_train, y_train)
        ensemble_test_score = ensemble_model.score(X_test, y_test)
        
        # Strategy 5: Neural network
        nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000,
                               random_state=self.random_state)
        nn_model.fit(X_train, y_train)
        nn_train_score = nn_model.score(X_train, y_train)
        nn_test_score = nn_model.score(X_test, y_test)
        
        # Results summary
        strategies = {
            'Baseline (Linear)': (baseline_train_score, baseline_test_score),
            'Polynomial Features': (poly_train_score, poly_test_score),
            'Decision Tree': (tree_train_score, tree_test_score),
            'Random Forest': (ensemble_train_score, ensemble_test_score),
            'Neural Network': (nn_train_score, nn_test_score)
        }
        
        print(f"\nBias Reduction Results:")
        print(f"{'Strategy':<20} {'Train R²':<12} {'Test R²':<12} {'Improvement':<12}")
        print("-" * 60)
        
        baseline_test = baseline_test_score
        for strategy, (train_score, test_score) in strategies.items():
            improvement = test_score - baseline_test
            print(f"{strategy:<20} {train_score:<12.4f} {test_score:<12.4f} {improvement:+.4f}")
        
        # Visualization
        self._plot_bias_reduction_strategies(strategies)
        
        return {
            'baseline': (baseline_train_score, baseline_test_score),
            'strategies': strategies,
            'best_improvement': max(test_score - baseline_test for _, (_, test_score) in strategies.items())
        }
    
    def _demonstrate_comprehensive_variance_reduction(self) -> Dict:
        """Demonstrate comprehensive strategies to reduce high variance."""
        print("STRATEGY: Comprehensive High Variance (Overfitting) Reduction")
        print("-" * 60)
        
        # Create overfitting scenario
        X, y = self._create_dataset('simple_linear', 300, 0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Strategy 1: Overfitting baseline (high-degree polynomial)
        overfit_model = Pipeline([
            ('poly', PolynomialFeatures(degree=15)),
            ('linear', LinearRegression())
        ])
        overfit_model.fit(X_train, y_train)
        overfit_train_score = overfit_model.score(X_train, y_train)
        overfit_test_score = overfit_model.score(X_test, y_test)
        
        # Strategy 2: Regularization (Ridge)
        ridge_model = Pipeline([
            ('poly', PolynomialFeatures(degree=15)),
            ('ridge', Ridge(alpha=10.0))
        ])
        ridge_model.fit(X_train, y_train)
        ridge_train_score = ridge_model.score(X_train, y_train)
        ridge_test_score = ridge_model.score(X_test, y_test)
        
        # Strategy 3: Lasso regularization
        lasso_model = Pipeline([
            ('poly', PolynomialFeatures(degree=15)),
            ('lasso', Lasso(alpha=0.1))
        ])
        lasso_model.fit(X_train, y_train)
        lasso_train_score = lasso_model.score(X_train, y_train)
        lasso_test_score = lasso_model.score(X_test, y_test)
        
        # Strategy 4: Reduce model complexity
        simple_model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        simple_model.fit(X_train, y_train)
        simple_train_score = simple_model.score(X_train, y_train)
        simple_test_score = simple_model.score(X_test, y_test)
        
        # Strategy 5: Ensemble method
        ensemble_model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                            random_state=self.random_state, n_jobs=-1)
        ensemble_model.fit(X_train, y_train)
        ensemble_train_score = ensemble_model.score(X_train, y_train)
        ensemble_test_score = ensemble_model.score(X_test, y_test)
        
        # Strategy 6: More data simulation (use larger training set)
        X_large, y_large = self._create_dataset('simple_linear', 1000, 0.1)
        X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
            X_large, y_large, test_size=0.2, random_state=self.random_state
        )
        
        large_data_model = Pipeline([
            ('poly', PolynomialFeatures(degree=15)),
            ('ridge', Ridge(alpha=1.0))
        ])
        large_data_model.fit(X_train_large, y_train_large)
        large_train_score = large_data_model.score(X_train_large, y_train_large)
        large_test_score = large_data_model.score(X_test_large, y_test_large)
        
        # Results summary
        strategies = {
            'Overfit Baseline': (overfit_train_score, overfit_test_score),
            'Ridge Regularization': (ridge_train_score, ridge_test_score),
            'Lasso Regularization': (lasso_train_score, lasso_test_score),
            'Reduced Complexity': (simple_train_score, simple_test_score),
            'Ensemble Method': (ensemble_train_score, ensemble_test_score),
            'More Data': (large_train_score, large_test_score)
        }
        
        print(f"\nVariance Reduction Results:")
        print(f"{'Strategy':<20} {'Train R²':<12} {'Test R²':<12} {'Gap':<12} {'Improvement':<12}")
        print("-" * 80)
        
        baseline_test = overfit_test_score
        for strategy, (train_score, test_score) in strategies.items():
            gap = train_score - test_score
            improvement = test_score - baseline_test
            print(f"{strategy:<20} {train_score:<12.4f} {test_score:<12.4f} {gap:<12.4f} {improvement:+.4f}")
        
        # Visualization
        self._plot_variance_reduction_strategies(strategies)
        
        return {
            'baseline': (overfit_train_score, overfit_test_score),
            'strategies': strategies,
            'best_improvement': max(test_score - baseline_test for _, (_, test_score) in strategies.items())
        }
    
    def _demonstrate_dataset_size_impact(self) -> Dict:
        """Demonstrate how dataset size affects bias-variance trade-off."""
        print("STRATEGY: Dataset Size Impact on Bias-Variance Trade-off")
        print("-" * 60)
        
        # Test different dataset sizes
        sizes = [50, 100, 200, 500, 1000, 2000]
        models = {
            'High Bias': LinearRegression(),
            'High Variance': KNeighborsRegressor(n_neighbors=1),
            'Balanced': RandomForestRegressor(n_estimators=50, max_depth=5,
                                            random_state=self.random_state, n_jobs=-1)
        }
        
        results = {model_name: {} for model_name in models.keys()}
        
        for size in sizes:
            print(f"\nTesting dataset size: {size}")
            
            # Create data
            X, y = self._create_dataset('complex_nonlinear', size, 0.2)
            
            for model_name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=5, scoring='r2', n_jobs=-1
                )
                
                results[model_name][size] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_min': cv_scores.min(),
                    'cv_max': cv_scores.max()
                }
                
                print(f"  {model_name:<15}: R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualization
        self._plot_dataset_size_impact(results)
        
        return results
    
    def _demonstrate_algorithm_selection(self) -> Dict:
        """Demonstrate algorithm selection based on bias-variance characteristics."""
        print("STRATEGY: Algorithm Selection Guide")
        print("-" * 60)
        
        guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALGORITHM SELECTION GUIDE                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│ HIGH BIAS SITUATIONS (Underfitting)                                           │
│ ───────────────────────────────────────────────────────────────────────────── │
│ ✓ Linear models with polynomial features (increase capacity)                   │
│ ✓ Neural networks with more layers/units (increase flexibility)               │
│ ✓ Decision Trees with greater depth (allow more splits)                        │
│ ✓ Gradient Boosting with individual complex models                            │
│ ✓ Support Vector Machines with RBF kernel                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ HIGH VARIANCE SITUATIONS (Overfitting)                                         │
│ ───────────────────────────────────────────────────────────────────────────── │
│ ✓ Regularized linear models (Ridge, Lasso, ElasticNet)                        │
│ ✓ Decision Trees with depth limits or pruning                                 │
│ ✓ KNN with larger K values (smoother boundaries)                              │
│ ✓ Ensemble methods (Random Forest, Bagging)                                    │
│ ✓ SVM with appropriate regularization parameter C                               │
│ ✓ Dropout and weight decay in neural networks                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ DATASET SIZE CONSIDERATIONS                                                    │
│ ───────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│ Small Dataset (< 500 samples):                                                  │
│ ✓ Simple models (high bias acceptable)                                          │
│ ✓ Strong regularization                                                         │
│ ✓ Cross-validation with fewer folds                                            │
│ ✓ Consider data augmentation                                                    │
│ ✓ Linear models, shallow trees                                                 │
│                                                                                 │
│ Medium Dataset (500-5000 samples):                                             │
│ ✓ Moderate complexity models                                                    │
│ ✓ Regularization still important                                               │
│ ✓ Ensemble methods start to be effective                                       │
│ ✓ Neural networks with moderate size                                           │
│                                                                                 │
│ Large Dataset (> 5000 samples):                                               │
│ ✓ Complex models (can handle variance)                                         │
│ ✓ Deep learning architectures                                                  │
│ ✓ Ensemble methods highly effective                                            │
│ ✓ Feature selection to reduce dimensionality                                   │
│ ✓ More aggressive regularization                                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM-SPECIFIC CHARACTERISTICS                                             │
│ ───────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│ Linear Regression:                                                             │
│ • Default: High bias, low variance                                             │
│ • Fix: Add polynomial features, interaction terms                             │
│                                                                                 │
│ Logistic Regression:                                                           │
│ • Default: High bias, low variance                                             │
│ • Fix: Add features, reduce regularization, use non-linear SVM                 │
│                                                                                 │
│ K-Nearest Neighbors:                                                           │
│ • K=1: Very low bias, very high variance                                      │
│ • K=large: High bias, low variance                                            │
│ • Fix: Tune K based on cross-validation                                         │
│                                                                                 │
│ Decision Trees:                                                                │
│ • Unconstrained: Low bias, very high variance                                 │
│ • Shallow: High bias, low variance                                            │
│ • Fix: Control depth, pruning, ensemble methods                                │
│                                                                                 │
│ Random Forest:                                                                 │
│ • Default: Moderate bias, reduced variance                                    │
│ • Fix: Tune n_estimators, max_depth, min_samples_split                        │
│                                                                                 │
│ Support Vector Machines:                                                        │
│ • High C: Low bias, high variance                                              │
│ • Low C: High bias, low variance                                               │
│ • Fix: Tune C and kernel parameters                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ KEY PRINCIPLE                                                                   │
│ ───────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│ Match model capacity to data size and complexity:                               │
│                                                                                 │
│ Small data → simpler models (accept some bias)                                  │
│ Large data → complex models (control variance)                                  │
│                                                                                 │
│ The optimal model is the simplest one that captures the underlying pattern.     │
│ Complexity should be justified by data, not the other way around.              │
└─────────────────────────────────────────────────────────────────────────────────┘
        """
        
        print(guide)
        
        return {"guide": guide}
    
    def _demonstrate_ensemble_techniques(self) -> Dict:
        """Demonstrate ensemble techniques for variance reduction."""
        print("STRATEGY: Ensemble Techniques for Variance Reduction")
        print("-" * 60)
        
        # Create data
        X, y = self._create_dataset('complex_nonlinear', 800, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Single models for comparison
        single_tree = DecisionTreeRegressor(max_depth=8, random_state=self.random_state)
        single_tree.fit(X_train, y_train)
        
        # Ensemble methods
        bagging_model = RandomForestRegressor(n_estimators=100, max_depth=8,
                                            random_state=self.random_state, n_jobs=-1)
        bagging_model.fit(X_train, y_train)
        
        boosting_model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                  random_state=self.random_state)
        boosting_model.fit(X_train, y_train)
        
        # Extra Trees (more randomization)
        from sklearn.ensemble import ExtraTreesRegressor
        extra_trees = ExtraTreesRegressor(n_estimators=100, max_depth=8,
                                         random_state=self.random_state, n_jobs=-1)
        extra_trees.fit(X_train, y_train)
        
        # Evaluate all models
        models = {
            'Single Tree': single_tree,
            'Bagging RF': bagging_model,
            'Gradient Boosting': boosting_model,
            'Extra Trees': extra_trees
        }
        
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation for variance assessment
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_estimators': getattr(model, 'n_estimators', 1)
            }
            
            print(f"{name:<20}: Test R² = {test_score:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualization
        self._plot_ensemble_comparison(results)
        
        return results
    
    def _plot_bias_reduction_strategies(self, strategies: Dict):
        """Plot bias reduction strategy comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        names = list(strategies.keys())
        train_scores = [strategies[name][0] for name in names]
        test_scores = [strategies[name][1] for name in names]
        
        # Plot 1: Train vs Test scores
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_scores, width, label='Train R²', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.7)
        axes[0, 0].set_xlabel('Strategy')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Bias Reduction: Train vs Test Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement over baseline
        baseline_test = strategies['Baseline (Linear)'][1]
        improvements = [score - baseline_test for _, score in test_scores]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(names, improvements, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Improvement over Baseline')
        axes[0, 1].set_title('Bias Reduction: Test Score Improvement')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Overfitting indicator (gap)
        gaps = [train - test for train, test in zip(train_scores, test_scores)]
        
        axes[1, 0].bar(names, gaps, color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Train-Test Gap')
        axes[1, 0].set_title('Overfitting Indicator (Lower is Better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        
        summary_text = "Bias Reduction Summary:\n\n"
        for i, name in enumerate(names):
            summary_text += f"{name}:\n"
            summary_text += f"  Train: {train_scores[i]:.3f}\n"
            summary_text += f"  Test:  {test_scores[i]:.3f}\n"
            summary_text += f"  Gap:   {gaps[i]:.3f}\n\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/bias_reduction_strategies.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_variance_reduction_strategies(self, strategies: Dict):
        """Plot variance reduction strategy comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        names = list(strategies.keys())
        train_scores = [strategies[name][0] for name in names]
        test_scores = [strategies[name][1] for name in names]
        gaps = [train - test for train, test in zip(train_scores, test_scores)]
        
        # Plot 1: Train vs Test scores
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_scores, width, label='Train R²', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.7)
        axes[0, 0].set_xlabel('Strategy')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Variance Reduction: Train vs Test Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gap reduction (main variance indicator)
        baseline_gap = gaps[0]  # Overfit baseline gap
        gap_reductions = [baseline_gap - gap for gap in gaps]
        
        colors = ['green' if reduction > 0 else 'red' for reduction in gap_reductions]
        axes[0, 1].bar(names, gap_reductions, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Gap Reduction')
        axes[0, 1].set_title('Variance Reduction: Gap Improvement')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Overfitting severity
        severity_colors = ['red' if gap > 0.5 else 'orange' if gap > 0.2 else 'green' 
                         for gap in gaps]
        
        axes[1, 0].bar(names, gaps, color=severity_colors, alpha=0.7)
        axes[1, 0].set_ylabel('Train-Test Gap')
        axes[1, 0].set_title('Overfitting Severity (Lower is Better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add severity zones
        axes[1, 0].axhspan(0, 0.2, alpha=0.2, color='green', label='Low Overfitting')
        axes[1, 0].axhspan(0.2, 0.5, alpha=0.2, color='orange', label='Moderate Overfitting')
        axes[1, 0].axhspan(0.5, 1.0, alpha=0.2, color='red', label='Severe Overfitting')
        axes[1, 0].legend()
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        
        summary_text = "Variance Reduction Summary:\n\n"
        for i, name in enumerate(names):
            summary_text += f"{name}:\n"
            summary_text += f"  Train: {train_scores[i]:.3f}\n"
            summary_text += f"  Test:  {test_scores[i]:.3f}\n"
            summary_text += f"  Gap:   {gaps[i]:.3f}\n"
            if i > 0:
                reduction = gap_reductions[i]
                summary_text += f"  Gap ↓: {reduction:+.3f}\n"
            summary_text += "\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/variance_reduction_strategies.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_dataset_size_impact(self, results: Dict):
        """Plot dataset size impact on model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = sorted(next(iter(results.values())).keys())
        
        # Plot 1: Performance vs. Dataset Size
        for model_name, model_results in results.items():
            means = [model_results[size]['cv_mean'] for size in sizes]
            stds = [model_results[size]['cv_std'] for size in sizes]
            
            axes[0, 0].plot(sizes, means, 'o-', label=model_name, linewidth=2, markersize=6)
            axes[0, 0].fill_between(sizes, 
                                   [m - s for m, s in zip(means, stds)],
                                   [m + s for m, s in zip(means, stds)],
                                   alpha=0.15)
        
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('CV R² Score')
        axes[0, 0].set_title('Performance vs. Dataset Size')
        axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Stability vs. Dataset Size
        for model_name, model_results in results.items():
            stds = [model_results[size]['cv_std'] for size in sizes]
            
            axes[0, 1].plot(sizes, stds, 's-', label=model_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('CV Standard Deviation')
        axes[0, 1].set_title('Model Stability vs. Dataset Size')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance improvement rate
        for model_name, model_results in results.items():
            means = [model_results[size]['cv_mean'] for size in sizes]
            # Calculate improvement rate
            improvements = []
            for i in range(1, len(means)):
                if means[i-1] > 0:
                    improvement = (means[i] - means[i-1]) / abs(means[i-1])
                    improvements.append(improvement)
                else:
                    improvements.append(0)
            
            # Plot improvement rate vs. dataset size (skip first point)
            axes[1, 0].plot(sizes[1:], improvements, '^-', label=model_name, 
                           linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Relative Improvement Rate')
        axes[1, 0].set_title('Learning Rate vs. Dataset Size')
        axes[1, 0].set_xscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Final performance comparison at largest size
        final_means = []
        final_stds = []
        model_names = []
        
        for model_name, model_results in results.items():
            final_means.append(model_results[sizes[-1]]['cv_mean'])
            final_stds.append(model_results[sizes[-1]]['cv_std'])
            model_names.append(model_name)
        
        x_pos = np.arange(len(model_names))
        axes[1, 1].bar(x_pos, final_means, yerr=final_stds, capsize=5, alpha=0.7)
        axes[1, 1].set_xlabel('Model Type')
        axes[1, 1].set_ylabel(f'CV R² at Size {sizes[-1]}')
        axes[1, 1].set_title('Final Performance Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/dataset_size_impact.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_ensemble_comparison(self, results: Dict):
        """Plot ensemble method comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(results.keys())
        test_scores = [results[name]['test_score'] for name in names]
        cv_means = [results[name]['cv_mean'] for name in names]
        cv_stds = [results[name]['cv_std'] for name in names]
        n_estimators = [results[name]['n_estimators'] for name in names]
        
        # Plot 1: Test performance comparison
        axes[0, 0].bar(names, test_scores, alpha=0.7, color='skyblue')
        axes[0, 0].set_ylabel('Test R² Score')
        axes[0, 0].set_title('Ensemble Methods: Test Performance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CV stability comparison
        axes[0, 1].bar(names, cv_stds, alpha=0.7, color='lightcoral')
        axes[0, 1].set_ylabel('CV Standard Deviation')
        axes[0, 1].set_title('Ensemble Methods: Model Stability')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Number of estimators vs. performance
        axes[1, 0].scatter(n_estimators, test_scores, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (n_estimators[i], test_scores[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Number of Estimators')
        axes[1, 0].set_ylabel('Test R² Score')
        axes[1, 0].set_title('Ensemble Size vs. Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs. Stability trade-off
        axes[1, 1].scatter(cv_stds, test_scores, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (cv_stds[i], test_scores[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('CV Standard Deviation')
        axes[1, 1].set_ylabel('Test R² Score')
        axes[1, 1].set_title('Performance vs. Stability Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/ensemble_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run the complete comprehensive bias-variance analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE BIAS-VARIANCE ANALYSIS")
        print("="*80)
        
        print("""
🎯 COMPREHENSIVE ANALYSIS OVERVIEW
=====================================

This analysis covers:
1. Mathematical foundation and conceptual understanding
2. Learning curves for bias-variance diagnosis
3. Cross-validation diagnostics for variance detection
4. Practical strategies for reducing bias and variance
5. Algorithm selection based on bias-variance characteristics
6. Dataset size effects on model behavior
7. Ensemble techniques for variance reduction

Each section provides:
• Theoretical background and mathematical foundation
• Practical examples with real algorithms
• Comprehensive visualizations
• Diagnostic tools and recommendations
• Before/after comparisons for strategies

Let's begin the comprehensive analysis...
        """)
        
        # Run all analyses
        all_results = {}
        
        try:
            print("\n" + "="*80)
            print("1. BIAS-VARIANCE FUNDAMENTALS")
            print("="*80)
            all_results['fundamentals'] = self.demonstrate_bias_variance_fundamentals()
            
            print("\n" + "="*80)
            print("2. LEARNING CURVES DIAGNOSIS")
            print("="*80)
            all_results['learning_curves'] = self.demonstrate_learning_curves_diagnosis()
            
            print("\n" + "="*80)
            print("3. CROSS-VALIDATION DIAGNOSTICS")
            print("="*80)
            all_results['cv_diagnostics'] = self.demonstrate_cross_validation_diagnostics()
            
            print("\n" + "="*80)
            print("4. PRACTICAL STRATEGIES")
            print("="*80)
            all_results['strategies'] = self.demonstrate_practical_strategies()
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            print(f"Error occurred: {str(e)}")
        
        # Final summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _print_final_summary(self, results: Dict):
        """Print final summary of the comprehensive analysis."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        KEY TAKEAWAYS                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. MATHEMATICAL FOUNDATION
   • Bias² + Variance + Irreducible Noise = Total Error²
   • Bias and variance trade off mathematically
   • Cannot eliminate both simultaneously
   • Goal is optimal balance, not elimination

2. DIAGNOSTIC TOOLS
   • Learning curves: Identify bias vs variance patterns
   • Cross-validation: Measure model stability directly
   • Train/test gap: Primary variance indicator
   • CV standard deviation: Quantify model instability

3. HIGH BIAS (UNDERFITTING)
   • Signs: Low train and test performance, small gap
   • Solutions: Increase complexity, add features, reduce regularization
   • Algorithms: Polynomial features, neural networks, deep trees

4. HIGH VARIANCE (OVERFITTING)
   • Signs: High train performance, low test performance, large gap
   • Solutions: Regularization, more data, ensembles, simpler models
   • Algorithms: Ridge/Lasso, Random Forest, bagging

5. DATASET SIZE EFFECTS
   • Small data: Bias dominates, use simple models
   • Large data: Variance manageable, use complex models
   • Learning curves show if more data will help

6. ALGORITHM SELECTION
   • Match model capacity to data size and complexity
   • Consider default bias-variance characteristics
   • Use ensemble methods for variance reduction
   • Tune hyperparameters for optimal balance

7. PRACTICAL WORKFLOW
   • Start with simple model (high bias baseline)
   • Use learning curves to diagnose problems
   • Apply appropriate strategy based on diagnosis
   • Validate improvements with cross-validation
   • Monitor variance throughout the process

╔══════════════════════════════════════════════════════════════════════════════╗
║                    VISUALIZATIONS CREATED                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated plots in the 'plots/' directory:
• bias_variance_fundamentals.png - Core concepts visualization
• comprehensive_learning_curves.png - Learning curve patterns
• comprehensive_cv_diagnostics.png - Cross-validation analysis
• bias_reduction_strategies.png - Bias reduction techniques
• variance_reduction_strategies.png - Variance reduction techniques
• dataset_size_impact.png - Data size effects
• ensemble_comparison.png - Ensemble method analysis

These visualizations provide:
• Clear understanding of bias-variance concepts
• Practical diagnostic patterns
• Strategy effectiveness comparisons
• Algorithm behavior insights

╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEXT STEPS                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Apply these diagnostic tools to your own models
2. Use learning curves to guide model improvements
3. Monitor cross-validation stability in production
4. Choose algorithms based on your data characteristics
5. Balance complexity with data size
6. Remember: The goal is optimal balance, not perfect scores

Bias-variance understanding transforms model debugging from guesswork 
into systematic, evidence-based improvement.
        """
        
        print(summary)


def main():
    """Main function to run the comprehensive bias-variance analysis."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          COMPREHENSIVE BIAS-VARIANCE TRADE-OFF ANALYSIS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This comprehensive analysis provides:
• Mathematical foundation of bias-variance trade-off
• Learning curves for diagnosing underfitting/overfitting
• Cross-validation diagnostics for variance detection
• Practical strategies for bias and variance reduction
• Algorithm selection based on bias-variance characteristics
• Dataset size effects on model behavior
• Ensemble techniques for variance reduction
• Comprehensive visualizations and examples

Author: Machine Learning Practitioner
Date: 2026
    """)
    
    # Create analyzer and run analysis
    analyzer = ComprehensiveBiasVarianceAnalyzer(random_state=42)
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("All visualizations saved to the 'plots/' directory")
    print("Review the generated plots and analysis results")
    print("Apply these techniques to your own machine learning projects")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
