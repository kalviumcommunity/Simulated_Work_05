"""
Bias-Variance Diagnostic Tools

Specialized diagnostic tools for identifying and analyzing bias-variance problems
in machine learning models. This module provides comprehensive diagnostic functions
that can be applied to any model to determine bias-variance characteristics.

Key Features:
- Automated bias-variance diagnosis
- Learning curve analysis with pattern recognition
- Cross-validation stability assessment
- Model complexity optimization
- Real-time diagnostic reporting
- Algorithm-specific bias-variance profiling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# ML imports
from sklearn.model_selection import (learning_curve, validation_curve, cross_val_score,
                                   train_test_split, StratifiedKFold)
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BiasVarianceDiagnosis(Enum):
    """Enumeration for bias-variance diagnosis results."""
    HIGH_BIAS = "HIGH_BIAS"
    HIGH_VARIANCE = "HIGH_VARIANCE"
    GOOD_FIT = "GOOD_FIT"
    BALANCED = "BALANCED"
    DATA_LIMITED = "DATA_LIMITED"
    MODEL_LIMITED = "MODEL_LIMITED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class DiagnosticResult:
    """Data class for storing diagnostic results."""
    diagnosis: BiasVarianceDiagnosis
    confidence: float
    train_score: float
    test_score: float
    gap: float
    train_std: float
    test_std: float
    cv_std: float
    recommendations: List[str]
    explanation: str


class BiasVarianceDiagnostics:
    """
    Comprehensive diagnostic tools for bias-variance analysis.
    
    This class provides specialized tools to automatically diagnose
    bias-variance problems in machine learning models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the bias-variance diagnostics system.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Diagnostic thresholds (can be customized)
        self.thresholds = {
            'high_bias_train_score': 0.7,
            'high_variance_gap': 0.2,
            'high_variance_cv_std': 0.1,
            'good_fit_gap': 0.1,
            'good_fit_min_score': 0.8
        }
        
        logger.info("Bias-Variance Diagnostics initialized")
    
    def diagnose_model(self, model, X, y, problem_type: str = 'regression',
                      cv_folds: int = 5, test_size: float = 0.2) -> DiagnosticResult:
        """
        Comprehensive diagnosis of model bias-variance characteristics.
        
        Args:
            model: Scikit-learn model to diagnose
            X: Feature matrix
            y: Target vector
            problem_type: 'regression' or 'classification'
            cv_folds: Number of cross-validation folds
            test_size: Test set size fraction
            
        Returns:
            DiagnosticResult with comprehensive analysis
        """
        logger.info(f"Diagnosing model: {type(model).__name__}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if problem_type == 'classification' else None
        )
        
        # Choose appropriate scoring metric
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate scores
        if problem_type == 'regression':
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
        else:
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)
        
        # Cross-validation for stability assessment
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        # Generate learning curve for pattern analysis
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=cv_folds, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=self.random_state
        )
        
        # Calculate key metrics
        gap = abs(train_score - test_score)
        cv_std = cv_scores.std()
        
        # Learning curve analysis
        final_train_mean = train_scores.mean(axis=1)[-1]
        final_test_mean = test_scores.mean(axis=1)[-1]
        final_train_std = train_scores.std(axis=1)[-1]
        final_test_std = test_scores.std(axis=1)[-1]
        
        # Learning curve trend analysis
        train_trend = train_scores.mean(axis=1)[-1] - train_scores.mean(axis=1)[0]
        test_trend = test_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[0]
        
        # Perform diagnosis
        diagnosis, confidence, explanation = self._perform_diagnosis(
            train_score, test_score, gap, cv_std,
            final_train_mean, final_test_mean, final_train_std, final_test_std,
            train_trend, test_trend, train_sizes
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(diagnosis, model)
        
        return DiagnosticResult(
            diagnosis=diagnosis,
            confidence=confidence,
            train_score=train_score,
            test_score=test_score,
            gap=gap,
            train_std=final_train_std,
            test_std=final_test_std,
            cv_std=cv_std,
            recommendations=recommendations,
            explanation=explanation
        )
    
    def _perform_diagnosis(self, train_score: float, test_score: float, gap: float,
                         cv_std: float, final_train_mean: float, final_test_mean: float,
                         final_train_std: float, final_test_std: float,
                         train_trend: float, test_trend: float,
                         train_sizes: np.ndarray) -> Tuple[BiasVarianceDiagnosis, float, str]:
        """
        Perform comprehensive bias-variance diagnosis.
        
        Returns:
            Tuple of (diagnosis, confidence, explanation)
        """
        confidence = 0.0
        explanation = ""
        
        # High Bias Diagnosis
        if (train_score < self.thresholds['high_bias_train_score'] and 
            test_score < self.thresholds['high_bias_train_score'] and 
            gap < 0.1 and abs(train_trend) < 0.05 and abs(test_trend) < 0.05):
            
            confidence = 0.9
            explanation = ("High bias detected: Both training and test scores are low "
                          "with a small gap. The model is underfitting and cannot capture "
                          "the underlying patterns in the data.")
            return BiasVarianceDiagnosis.HIGH_BIAS, confidence, explanation
        
        # High Variance Diagnosis
        elif (train_score > 0.9 and gap > self.thresholds['high_variance_gap'] and 
              cv_std > self.thresholds['high_variance_cv_std']):
            
            confidence = 0.85
            explanation = ("High variance detected: Training score is very high but "
                          "test score is significantly lower with a large gap. "
                          "The model is overfitting and memorizing training data.")
            return BiasVarianceDiagnosis.HIGH_VARIANCE, confidence, explanation
        
        # Good Fit Diagnosis
        elif (train_score > self.thresholds['good_fit_min_score'] and 
              test_score > self.thresholds['good_fit_min_score'] and 
              gap < self.thresholds['good_fit_gap'] and cv_std < 0.05):
            
            confidence = 0.95
            explanation = ("Good fit achieved: Both training and test scores are high "
                          "with a small gap and low variance. The model generalizes well.")
            return BiasVarianceDiagnosis.GOOD_FIT, confidence, explanation
        
        # Balanced Diagnosis
        elif (0.7 <= train_score <= 0.95 and 0.7 <= test_score <= 0.95 and 
              0.05 <= gap <= 0.15 and cv_std < 0.1):
            
            confidence = 0.8
            explanation = ("Balanced model: Good trade-off between bias and variance. "
                          "Performance is reasonable with moderate stability.")
            return BiasVarianceDiagnosis.BALANCED, confidence, explanation
        
        # Data Limited Diagnosis
        elif (gap > 0.15 and test_trend < -0.1 and 
              final_test_mean > final_train_mean * 0.8):
            
            confidence = 0.75
            explanation = ("Data limited: The model would benefit from more training "
                          "data. Learning curve shows improvement with more samples.")
            return BiasVarianceDiagnosis.DATA_LIMITED, confidence, explanation
        
        # Model Limited Diagnosis
        elif (gap < 0.1 and train_score < 0.8 and test_score < 0.8 and 
              abs(test_trend) < 0.02):
            
            confidence = 0.8
            explanation = ("Model limited: More data won't help significantly. "
                          "The model needs more complexity or better features.")
            return BiasVarianceDiagnosis.MODEL_LIMITED, confidence, explanation
        
        # Uncertain Diagnosis
        else:
            confidence = 0.5
            explanation = ("Uncertain diagnosis: Mixed signals detected. "
                          "Consider further analysis with different metrics or data splits.")
            return BiasVarianceDiagnosis.UNCERTAIN, confidence, explanation
    
    def _generate_recommendations(self, diagnosis: BiasVarianceDiagnosis, model) -> List[str]:
        """Generate specific recommendations based on diagnosis."""
        
        recommendations = {
            BiasVarianceDiagnosis.HIGH_BIAS: [
                "Increase model complexity (add layers, depth, or features)",
                "Add polynomial or interaction features for linear models",
                "Reduce regularization strength (increase C, decrease alpha)",
                "Use more flexible algorithms (trees, neural networks)",
                "Perform feature engineering to capture more signal",
                "Consider ensemble methods with complex base models"
            ],
            
            BiasVarianceDiagnosis.HIGH_VARIANCE: [
                "Apply regularization (L1, L2, dropout, weight decay)",
                "Reduce model complexity (limit depth, features, parameters)",
                "Collect more training data",
                "Use ensemble methods (Random Forest, bagging)",
                "Perform feature selection to remove noisy features",
                "Apply early stopping for iterative models",
                "Increase K in KNN or minimum samples in trees"
            ],
            
            BiasVarianceDiagnosis.GOOD_FIT: [
                "Model is performing well - consider deployment",
                "Monitor for data drift in production",
                "Consider slight optimization if needed",
                "Document the model configuration for reproducibility"
            ],
            
            BiasVarianceDiagnosis.BALANCED: [
                "Good balance found - fine-tune hyperparameters",
                "Consider cross-validation for optimization",
                "Test on more diverse data if available",
                "Monitor performance on validation set"
            ],
            
            BiasVarianceDiagnosis.DATA_LIMITED: [
                "Collect more training data",
                "Use data augmentation techniques",
                "Consider transfer learning if applicable",
                "Use simpler models that work better with limited data",
                "Apply stronger regularization"
            ],
            
            BiasVarianceDiagnosis.MODEL_LIMITED: [
                "Increase model complexity",
                "Add more informative features",
                "Try different algorithm families",
                "Perform feature engineering",
                "Consider non-linear transformations"
            ],
            
            BiasVarianceDiagnosis.UNCERTAIN: [
                "Analyze with different evaluation metrics",
                "Try different data splits",
                "Examine data quality and preprocessing",
                "Consider domain-specific evaluation",
                "Consult with domain experts"
            ]
        }
        
        return recommendations.get(diagnosis, ["No specific recommendations available"])
    
    def compare_models(self, models: Dict[str, Any], X, y, 
                      problem_type: str = 'regression') -> Dict[str, DiagnosticResult]:
        """
        Compare multiple models and diagnose each one.
        
        Args:
            models: Dictionary of model_name -> model_object
            X: Feature matrix
            y: Target vector
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of model_name -> DiagnosticResult
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = {}
        
        for name, model in models.items():
            try:
                result = self.diagnose_model(model, X, y, problem_type)
                results[name] = result
                logger.info(f"Diagnosed {name}: {result.diagnosis.value} (confidence: {result.confidence:.2f})")
            except Exception as e:
                logger.error(f"Error diagnosing {name}: {str(e)}")
                results[name] = None
        
        return results
    
    def optimize_model_complexity(self, model, X, y, param_name: str, 
                                 param_range: List, problem_type: str = 'regression') -> Dict:
        """
        Optimize model complexity using validation curves.
        
        Args:
            model: Scikit-learn model
            X: Feature matrix
            y: Target vector
            param_name: Parameter to optimize
            param_range: Range of parameter values
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing {param_name} for {type(model).__name__}")
        
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'
        
        # Generate validation curve
        train_scores, test_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=5, scoring=scoring, n_jobs=-1
        )
        
        # Calculate means and standard deviations
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Find optimal parameter (based on test score)
        optimal_idx = np.argmax(test_mean)
        optimal_param = param_range[optimal_idx]
        optimal_score = test_mean[optimal_idx]
        
        # Analyze bias-variance trade-off across parameter range
        analysis = self._analyze_complexity_tradeoff(
            param_range, train_mean, test_mean, train_std, test_std
        )
        
        results = {
            'optimal_param': optimal_param,
            'optimal_score': optimal_score,
            'param_range': param_range,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'train_std': train_std,
            'test_std': test_std,
            'analysis': analysis
        }
        
        # Plot validation curve
        self._plot_validation_curve(results, param_name, problem_type)
        
        return results
    
    def _analyze_complexity_tradeoff(self, param_range: List, train_mean: np.ndarray,
                                   test_mean: np.ndarray, train_std: np.ndarray,
                                   test_std: np.ndarray) -> Dict:
        """Analyze bias-variance trade-off across parameter range."""
        
        # Find regions of high bias and high variance
        high_bias_region = []
        high_variance_region = []
        good_fit_region = []
        
        for i, param in enumerate(param_range):
            gap = abs(train_mean[i] - test_mean[i])
            stability = test_std[i]
            
            if train_mean[i] < 0.7 and test_mean[i] < 0.7 and gap < 0.1:
                high_bias_region.append(param)
            elif gap > 0.2 and stability > 0.1:
                high_variance_region.append(param)
            elif train_mean[i] > 0.8 and test_mean[i] > 0.8 and gap < 0.1:
                good_fit_region.append(param)
        
        return {
            'high_bias_region': high_bias_region,
            'high_variance_region': high_variance_region,
            'good_fit_region': good_fit_region,
            'min_gap_idx': np.argmin(np.abs(train_mean - test_mean)),
            'max_test_score_idx': np.argmax(test_mean),
            'most_stable_idx': np.argmin(test_std)
        }
    
    def _plot_validation_curve(self, results: Dict, param_name: str, problem_type: str):
        """Plot validation curve with bias-variance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        param_range = results['param_range']
        train_mean = results['train_mean']
        test_mean = results['test_mean']
        train_std = results['train_std']
        test_std = results['test_std']
        analysis = results['analysis']
        
        # Plot 1: Training and validation scores
        axes[0, 0].plot(param_range, train_mean, 'b-', label='Training Score', linewidth=2)
        axes[0, 0].fill_between(param_range, train_mean - train_std, train_mean + train_std,
                               alpha=0.15, color='blue')
        
        axes[0, 0].plot(param_range, test_mean, 'r-', label='Validation Score', linewidth=2)
        axes[0, 0].fill_between(param_range, test_mean - test_std, test_mean + test_std,
                               alpha=0.15, color='red')
        
        # Mark optimal point
        optimal_idx = analysis['max_test_score_idx']
        axes[0, 0].axvline(x=param_range[optimal_idx], color='green', linestyle='--',
                           label=f'Optimal ({param_range[optimal_idx]})')
        
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title(f'Validation Curve: {param_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gap analysis
        gaps = np.abs(train_mean - test_mean)
        axes[0, 1].plot(param_range, gaps, 'g-', linewidth=2)
        axes[0, 1].axhline(y=0.1, color='orange', linestyle='--', label='Good Gap Threshold')
        axes[0, 1].axhline(y=0.2, color='red', linestyle='--', label='High Variance Threshold')
        
        axes[0, 1].set_xlabel(param_name)
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('Bias-Variance Gap Analysis')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Stability analysis
        axes[1, 0].plot(param_range, test_std, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0.05, color='green', linestyle='--', label='Good Stability')
        axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', label='Moderate Stability')
        
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel('Validation Std Dev')
        axes[1, 0].set_title('Model Stability Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Bias-variance regions
        axes[1, 1].plot(param_range, train_mean, 'b-', label='Training', linewidth=2)
        axes[1, 1].plot(param_range, test_mean, 'r-', label='Validation', linewidth=2)
        
        # Highlight regions
        if analysis['high_bias_region']:
            for param in analysis['high_bias_region']:
                axes[1, 1].axvspan(param-0.5, param+0.5, alpha=0.2, color='blue')
        
        if analysis['high_variance_region']:
            for param in analysis['high_variance_region']:
                axes[1, 1].axvspan(param-0.5, param+0.5, alpha=0.2, color='red')
        
        if analysis['good_fit_region']:
            for param in analysis['good_fit_region']:
                axes[1, 1].axvspan(param-0.5, param+0.5, alpha=0.2, color='green')
        
        axes[1, 1].set_xlabel(param_name)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Bias-Variance Regions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"plots/validation_curve_{param_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_diagnostic_report(self, results: Dict[str, DiagnosticResult]) -> str:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            results: Dictionary of model_name -> DiagnosticResult
            
        Returns:
            Formatted diagnostic report string
        """
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE BIAS-VARIANCE DIAGNOSTIC REPORT")
        report.append("="*80)
        report.append("")
        
        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 80)
        report.append(f"{'Model':<25} {'Diagnosis':<15} {'Confidence':<12} {'Train':<8} {'Test':<8} {'Gap':<8}")
        report.append("-" * 80)
        
        for name, result in results.items():
            if result:
                report.append(f"{name:<25} {result.diagnosis.value:<15} "
                            f"{result.confidence:<12.2f} {result.train_score:<8.3f} "
                            f"{result.test_score:<8.3f} {result.gap:<8.3f}")
        
        report.append("")
        
        # Detailed analysis for each model
        for name, result in results.items():
            if result:
                report.append(f"DETAILED ANALYSIS: {name}")
                report.append("=" * 50)
                report.append(f"Diagnosis: {result.diagnosis.value}")
                report.append(f"Confidence: {result.confidence:.2f}")
                report.append(f"Explanation: {result.explanation}")
                report.append("")
                report.append("Performance Metrics:")
                report.append(f"  Training Score: {result.train_score:.4f}")
                report.append(f"  Test Score: {result.test_score:.4f}")
                report.append(f"  Gap: {result.gap:.4f}")
                report.append(f"  CV Stability: {result.cv_std:.4f}")
                report.append("")
                report.append("Recommendations:")
                for i, rec in enumerate(result.recommendations, 1):
                    report.append(f"  {i}. {rec}")
                report.append("")
                report.append("-" * 50)
                report.append("")
        
        # Overall recommendations
        report.append("OVERALL RECOMMENDATIONS")
        report.append("=" * 50)
        
        # Find best performing model
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_model = max(valid_results.keys(), 
                            key=lambda k: valid_results[k].test_score)
            best_result = valid_results[best_model]
            
            report.append(f"Best Performing Model: {best_model}")
            report.append(f"Test Score: {best_result.test_score:.4f}")
            report.append(f"Diagnosis: {best_result.diagnosis.value}")
            report.append("")
            
            if best_result.diagnosis == BiasVarianceDiagnosis.GOOD_FIT:
                report.append("✓ This model is ready for deployment!")
            elif best_result.diagnosis == BiasVarianceDiagnosis.BALANCED:
                report.append("✓ This model shows good bias-variance balance.")
                report.append("  Consider fine-tuning for optimal performance.")
            else:
                report.append("⚠ This model needs improvement before deployment.")
                report.append("  Follow the specific recommendations above.")
        
        return "\n".join(report)
    
    def plot_model_comparison(self, results: Dict[str, DiagnosticResult]):
        """Plot comprehensive model comparison."""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        names = list(valid_results.keys())
        train_scores = [valid_results[name].train_score for name in names]
        test_scores = [valid_results[name].test_score for name in names]
        gaps = [valid_results[name].gap for name in names]
        cv_stds = [valid_results[name].cv_std for name in names]
        confidences = [valid_results[name].confidence for name in names]
        
        # Plot 1: Train vs Test scores
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, test_scores, width, label='Test', alpha=0.7)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Train vs Test Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gap analysis
        colors = ['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' 
                 for gap in gaps]
        axes[0, 1].bar(names, gaps, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('Overfitting Indicator')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: CV stability
        axes[0, 2].bar(names, cv_stds, color='purple', alpha=0.7)
        axes[0, 2].set_ylabel('CV Standard Deviation')
        axes[0, 2].set_title('Model Stability')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs. Stability scatter
        axes[1, 0].scatter(cv_stds, test_scores, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (cv_stds[i], test_scores[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('CV Std (Stability)')
        axes[1, 0].set_ylabel('Test Score')
        axes[1, 0].set_title('Performance vs. Stability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Diagnosis distribution
        diagnosis_counts = {}
        for result in valid_results.values():
            diagnosis = result.diagnosis.value
            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
        
        axes[1, 1].pie(diagnosis_counts.values(), labels=diagnosis_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Diagnosis Distribution')
        
        # Plot 6: Confidence levels
        axes[1, 2].bar(names, confidences, color='teal', alpha=0.7)
        axes[1, 2].set_ylabel('Confidence')
        axes[1, 2].set_title('Diagnostic Confidence')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_sample_data(data_type: str = 'regression', n_samples: int = 1000, 
                      noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for testing diagnostic tools."""
    np.random.seed(random_state)
    
    if data_type == 'regression':
        # Complex non-linear regression data
        X = np.random.normal(0, 1, (n_samples, 3))
        y = (2 * X[:, 0]**2 + 3 * X[:, 1] * X[:, 2] + 
             np.sin(X[:, 0]) + np.random.normal(0, noise, n_samples))
        
    elif data_type == 'classification':
        # Complex classification data
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=4, n_informative=3,
            n_redundant=1, flip_y=noise, random_state=random_state
        )
        
    else:
        raise ValueError("data_type must be 'regression' or 'classification'")
    
    return X, y


def main():
    """Main function to demonstrate diagnostic tools."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BIAS-VARIANCE DIAGNOSTIC TOOLS                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

This demonstration shows:
• Automated bias-variance diagnosis
• Model comparison and ranking
• Complexity optimization
• Comprehensive diagnostic reporting
    """)
    
    # Create diagnostic system
    diagnostics = BiasVarianceDiagnostics(random_state=42)
    
    # Create sample data
    X, y = create_sample_data('regression', n_samples=800, noise=0.2)
    
    # Define models to test
    models = {
        'Linear (High Bias)': LinearRegression(),
        'Ridge (Moderate)': Ridge(alpha=1.0),
        'Poly Degree 3': Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ]),
        'Poly Degree 10': Pipeline([
            ('poly', PolynomialFeatures(degree=10)),
            ('ridge', Ridge(alpha=0.01))
        ]),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, 
                                             random_state=42, n_jobs=-1),
        'KNN (k=1)': KNeighborsRegressor(n_neighbors=1),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10)
    }
    
    # Compare all models
    print("Comparing models...")
    results = diagnostics.compare_models(models, X, y, 'regression')
    
    # Generate and print report
    report = diagnostics.generate_diagnostic_report(results)
    print(report)
    
    # Plot comparison
    diagnostics.plot_model_comparison(results)
    
    # Optimize complexity for best model type
    print("\nOptimizing Random Forest complexity...")
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    optimization_results = diagnostics.optimize_model_complexity(
        rf_model, X, y, 'max_depth', list(range(1, 15)), 'regression'
    )
    
    print(f"Optimal max_depth: {optimization_results['optimal_param']}")
    print(f"Optimal score: {optimization_results['optimal_score']:.4f}")
    
    return diagnostics, results


if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    
    diagnostics, results = main()
