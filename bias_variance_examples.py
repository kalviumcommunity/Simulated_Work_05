"""
Bias-Variance Analysis Examples and Utilities

Additional practical examples for understanding and managing
bias-variance trade-off in machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

warnings.filterwarnings('ignore')


class BiasVarianceExamples:
    """Collection of advanced bias-variance analysis examples."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def example_model_complexity_spectrum(self):
        """Example: Compare models across the complexity spectrum."""
        print("\n" + "="*60)
        print("EXAMPLE: MODEL COMPLEXITY SPECTRUM ANALYSIS")
        print("="*60)
        
        # Create challenging dataset
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=6,
            n_redundant=2, flip_y=0.1, random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Models across complexity spectrum
        models = {
            'Very Simple': LinearRegression(),
            'Simple': Ridge(alpha=0.1),
            'Moderate': DecisionTreeClassifier(max_depth=2, random_state=self.random_state),
            'Complex': DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            'Very Complex': DecisionTreeClassifier(max_depth=15, random_state=self.random_state),
            'Ensemble': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        }
        
        # Evaluate all models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation for stability
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'test_accuracy': test_accuracy,
                'train_time': train_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Mean:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualize complexity spectrum
        self._plot_complexity_spectrum(results)
        
        return results
    
    def _plot_complexity_spectrum(self, results: Dict):
        """Plot model complexity spectrum analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        names = list(results.keys())
        test_acc = [results[name]['test_accuracy'] for name in names]
        cv_means = [results[name]['cv_mean'] for name in names]
        cv_stds = [results[name]['cv_std'] for name in names]
        
        # Plot 1: Test accuracy comparison
        axes[0, 0].bar(names, test_acc, alpha=0.7)
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Test Accuracy Across Model Complexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CV stability
        axes[0, 1].bar(names, cv_stds, color='red', alpha=0.7)
        axes[0, 1].set_ylabel('CV Standard Deviation')
        axes[0, 1].set_title('Model Stability (Lower is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Bias-variance characteristics
        # Calculate bias (underfitting) and variance (overfitting) indicators
        bias_indicators = []
        variance_indicators = []
        
        for name in names:
            cv_mean = results[name]['cv_mean']
            cv_std = results[name]['cv_std']
            
            # High bias indicator: low CV score
            bias_indicator = 1 if cv_mean < 0.7 else 0
            bias_indicators.append(bias_indicator)
            
            # High variance indicator: high CV std
            variance_indicator = 1 if cv_std > 0.1 else 0
            variance_indicators.append(variance_indicator)
        
        # Create bias-variance map
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, bias_indicators, width, label='High Bias', color='blue', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, variance_indicators, width, label='High Variance', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Model Complexity')
        axes[1, 0].set_ylabel('Indicator (1 = Problem)')
        axes[1, 0].set_title('Bias-Variance Characteristics')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/complexity_spectrum.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def example_regularization_path(self):
        """Example: Regularization path for bias-variance optimization."""
        print("\n" + "="*60)
        print("EXAMPLE: REGULARIZATION PATH OPTIMIZATION")
        print("="*60)
        
        # Create data that benefits from regularization
        X, y = make_regression(
            n_samples=500, n_features=8, n_informative=4,
            noise=0.5, random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Test different regularization strengths
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        results = {}
        
        for alpha in alphas:
            # Ridge regression with different alpha
            model = Ridge(alpha=alpha, random_state=self.random_state)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[alpha] = {
                'alpha': alpha,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'coefficients': model.coef_
            }
            
            print(f"Alpha {alpha:6.3f}: Test = {test_score:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Find optimal alpha
        best_alpha = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_result = results[best_alpha]
        
        print(f"\nOptimal Alpha: {best_alpha}")
        print(f"Best Test Score: {best_result['test_score']:.4f}")
        print(f"Best CV Score:    {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        # Visualize regularization path
        self._plot_regularization_path(results, best_alpha)
        
        return results
    
    def _plot_regularization_path(self, results: Dict, best_alpha: float):
        """Plot regularization optimization path."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        alphas = list(results.keys())
        test_scores = [results[alpha]['test_score'] for alpha in alphas]
        cv_means = [results[alpha]['cv_mean'] for alpha in alphas]
        
        # Plot 1: Test scores vs. regularization strength
        axes[0, 0].plot(alphas, test_scores, 'bo-', label='Test Score')
        axes[0, 0].axvline(x=best_alpha, color='red', linestyle='--', 
                           label=f'Optimal Alpha ({best_alpha})')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Regularization Strength (alpha)')
        axes[0, 0].set_ylabel('Test Score')
        axes[0, 0].set_title('Regularization Path: Test Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CV scores vs. regularization strength
        axes[0, 1].plot(alphas, cv_means, 'go-', label='CV Mean Score')
        axes[0, 1].fill_between(alphas, 
                                [m - s for m, s in zip(cv_means, 
                                             [results[alpha]['cv_std'] for alpha in alphas])],
                                [m + s for m, s in zip(cv_means, 
                                             [results[alpha]['cv_std'] for alpha in alphas])],
                                alpha=0.15)
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('Regularization Strength (alpha)')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].set_title('Regularization Path: Cross-Validation Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/regularization_path.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def example_ensemble_variance_reduction(self):
        """Example: How ensembles reduce variance compared to single models."""
        print("\n" + "="*60)
        print("EXAMPLE: ENSEMBLE VARIANCE REDUCTION")
        print("="*60)
        
        # Create data
        X, y = make_classification(
            n_samples=800, n_features=12, n_informative=8,
            flip_y=0.15, random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Single models
        single_tree = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
        single_rf = RandomForestClassifier(n_estimators=1, max_depth=5, 
                                   random_state=self.random_state, n_jobs=-1)
        
        # Ensemble models
        bagging_rf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                     random_state=self.random_state, n_jobs=-1)
        boosting_gb = GradientBoostingClassifier(n_estimators=50, max_depth=5,
                                        random_state=self.random_state)
        
        models = {
            'Single Tree': single_tree,
            'Single RF': single_rf,
            'Bagging RF': bagging_rf,
            'Boosting GB': boosting_gb
        }
        
        # Evaluate all models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation for variance analysis
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'test_accuracy': test_accuracy,
                'train_time': train_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_estimators': getattr(model, 'n_estimators', 1)
            }
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Mean:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Estimators:    {results[name]['n_estimators']}")
        
        # Visualize ensemble effects
        self._plot_ensemble_variance_reduction(results)
        
        return results
    
    def _plot_ensemble_variance_reduction(self, results: Dict):
        """Plot ensemble variance reduction effects."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        names = list(results.keys())
        test_acc = [results[name]['test_accuracy'] for name in names]
        cv_means = [results[name]['cv_mean'] for name in names]
        cv_stds = [results[name]['cv_std'] for name in names]
        n_estimators = [results[name]['n_estimators'] for name in names]
        
        # Plot 1: Test accuracy comparison
        axes[0, 0].bar(names, test_acc, alpha=0.7)
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Test Accuracy: Single vs. Ensemble Models')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CV stability comparison
        axes[0, 1].bar(names, cv_stds, color='red', alpha=0.7)
        axes[0, 1].set_ylabel('CV Standard Deviation')
        axes[0, 1].set_title('Model Stability: Variance Reduction')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Number of estimators vs. performance
        axes[1, 0].scatter(n_estimators, test_acc, s=100, alpha=0.6)
        axes[1, 0].plot(n_estimators, test_acc, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Estimators')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].set_title('Ensemble Size vs. Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/ensemble_variance_reduction.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def example_validation_curve_analysis(self):
        """Example: Validation curves for optimal model selection."""
        print("\n" + "="*60)
        print("EXAMPLE: VALIDATION CURVE ANALYSIS")
        print("="*60)
        
        # Create data
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=6,
            random_state=self.random_state
        )
        
        # Model with hyperparameter to optimize
        model = DecisionTreeClassifier(random_state=self.random_state)
        
        # Parameter ranges for validation curve
        param_range = np.arange(1, 21)  # max_depth from 1 to 20
        
        # Generate validation curve
        train_scores, test_scores = validation_curve(
            model, X, y, param_name='max_depth', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Analyze results
        mean_test_scores = test_scores.mean(axis=1)
        std_test_scores = test_scores.std(axis=1)
        
        # Find optimal parameter
        optimal_idx = np.argmax(mean_test_scores)
        optimal_depth = param_range[optimal_idx]
        optimal_score = mean_test_scores[optimal_idx]
        
        print(f"Optimal max_depth: {optimal_depth}")
        print(f"Optimal score: {optimal_score:.4f}")
        
        # Visualize validation curve
        self._plot_validation_curve(param_range, train_scores, test_scores, 
                               optimal_depth, optimal_score)
        
        return {
            'optimal_depth': optimal_depth,
            'optimal_score': optimal_score,
            'param_range': param_range,
            'train_scores': train_scores,
            'test_scores': test_scores
        }
    
    def _plot_validation_curve(self, param_range: np.ndarray, train_scores: np.ndarray, 
                          test_scores: np.ndarray, optimal_param: int, optimal_score: float):
        """Plot validation curve analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training scores
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        
        axes[0, 0].plot(param_range, train_mean, 'b-', label='Training Score')
        axes[0, 0].fill_between(param_range,
                                train_mean - train_std,
                                train_mean + train_std,
                                alpha=0.15)
        axes[0, 0].set_xlabel('max_depth')
        axes[0, 0].set_ylabel('Training Score')
        axes[0, 0].set_title('Training Score vs. Model Complexity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Test scores with optimal point
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        axes[0, 1].plot(param_range, test_mean, 'ro-', label='Test Score')
        axes[0, 1].fill_between(param_range,
                                test_mean - test_std,
                                test_mean + test_std,
                                alpha=0.15)
        axes[0, 1].axvline(x=optimal_param, color='red', linestyle='--',
                           label=f'Optimal ({optimal_param})')
        axes[0, 1].set_xlabel('max_depth')
        axes[0, 1].set_ylabel('Test Score')
        axes[0, 1].set_title('Validation Curve: Test Score vs. Model Complexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/validation_curve.png", dpi=300, bbox_inches='tight')
        plt.show()


def run_all_examples():
    """Run all bias-variance analysis examples."""
    print("""
🎯 BIAS-VARIANCE ANALYSIS EXAMPLES
======================================

Additional practical examples for bias-variance analysis.
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    examples = BiasVarianceExamples(random_state=42)
    
    # Run all examples
    example_funcs = [
        ("Model Complexity Spectrum", examples.example_model_complexity_spectrum),
        ("Regularization Path", examples.example_regularization_path),
        ("Ensemble Variance Reduction", examples.example_ensemble_variance_reduction),
        ("Validation Curve Analysis", examples.example_validation_curve_analysis)
    ]
    
    results = {}
    
    for title, func in example_funcs:
        print(f"\n{'='*20} {title} {'='*20}")
        try:
            result = func()
            results[title] = result
        except Exception as e:
            print(f"Error in {title}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_all_examples()
