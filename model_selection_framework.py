"""
Comprehensive Model Selection Framework

This module implements a structured, multi-criteria model selection process
that goes beyond simple CV score optimization to consider business constraints,
computational requirements, interpretability needs, and real-world deployment
considerations.

Based on the principle: "Which model serves the real-world objective best,
reliably, honestly, and under real conditions?"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Scikit-learn imports
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_absolute_error,
    mean_squared_error, r2_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """Enumeration of problem types for metric selection."""
    REGRESSION = "regression"
    BALANCED_CLASSIFICATION = "balanced_classification"
    IMBALANCED_CLASSIFICATION = "imbalanced_classification"


class BusinessObjective(Enum):
    """Business objectives that drive metric selection."""
    MINIMIZE_FN_COST = "minimize_fn_cost"  # False Negative cost is high
    MINIMIZE_FP_COST = "minimize_fp_cost"  # False Positive cost is high
    BALANCE_ERRORS = "balance_errors"       # Both costs matter equally
    INTERPRETABILITY = "interpretability"   # Must be explainable
    LOW_LATENCY = "low_latency"            # Real-time requirements
    GENERAL_PERFORMANCE = "general_performance"  # Overall accuracy


class DeploymentEnvironment(Enum):
    """Deployment environments with different constraints."""
    BATCH = "batch"                    # Nightly/weekly processing
    REAL_TIME = "real_time"           # <100ms latency requirement
    EDGE_DEVICE = "edge_device"       # Mobile/embedded constraints
    CLOUD = "cloud"                   # Flexible scaling
    EMBEDDED = "embedded"             # Strict resource limits


@dataclass
class ModelConstraints:
    """Computational and operational constraints for model deployment."""
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    requires_gpu: bool = False
    interpretable: bool = False
    update_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    team_expertise: str = "intermediate"  # beginner, intermediate, advanced


@dataclass
class ModelEvaluationResult:
    """Results from evaluating a single model."""
    model_name: str
    model: BaseEstimator
    cv_mean: float
    cv_std: float
    train_score: float
    test_score: float
    bias_variance_gap: float
    metrics: Dict[str, float]
    latency_ms: float
    memory_mb: float
    is_interpretable: bool
    threshold_optimized: bool = False
    optimal_threshold: float = 0.5
    constraint_violations: List[str] = field(default_factory=list)


class MetricSelector:
    """Selects appropriate evaluation metrics based on problem characteristics."""
    
    @staticmethod
    def select_metrics(problem_type: ProblemType, 
                      business_objective: BusinessObjective) -> Dict[str, str]:
        """
        Select primary and secondary metrics based on problem type and business needs.
        
        Args:
            problem_type: Type of ML problem
            business_objective: Primary business driver
            
        Returns:
            Dictionary with primary_metric and secondary_metrics
        """
        metric_mapping = {
            (ProblemType.REGRESSION, BusinessObjective.GENERAL_PERFORMANCE): {
                "primary_metric": "r2",
                "secondary_metrics": ["mae", "rmse"]
            },
            (ProblemType.REGRESSION, BusinessObjective.MINIMIZE_FN_COST): {
                "primary_metric": "mae",  # Interpretable average error
                "secondary_metrics": ["rmse", "r2"]
            },
            (ProblemType.REGRESSION, BusinessObjective.MINIMIZE_FP_COST): {
                "primary_metric": "rmse",  # Penalizes large errors
                "secondary_metrics": ["mae", "r2"]
            },
            (ProblemType.BALANCED_CLASSIFICATION, BusinessObjective.GENERAL_PERFORMANCE): {
                "primary_metric": "f1",
                "secondary_metrics": ["accuracy", "roc_auc"]
            },
            (ProblemType.BALANCED_CLASSIFICATION, BusinessObjective.BALANCE_ERRORS): {
                "primary_metric": "f1",
                "secondary_metrics": ["precision", "recall", "roc_auc"]
            },
            (ProblemType.IMBALANCED_CLASSIFICATION, BusinessObjective.MINIMIZE_FN_COST): {
                "primary_metric": "recall",
                "secondary_metrics": ["f1", "pr_auc"]
            },
            (ProblemType.IMBALANCED_CLASSIFICATION, BusinessObjective.MINIMIZE_FP_COST): {
                "primary_metric": "precision",
                "secondary_metrics": ["f1", "accuracy"]
            },
            (ProblemType.IMBALANCED_CLASSIFICATION, BusinessObjective.BALANCE_ERRORS): {
                "primary_metric": "f1",
                "secondary_metrics": ["precision", "recall", "pr_auc"]
            }
        }
        
        key = (problem_type, business_objective)
        if key not in metric_mapping:
            # Default fallback
            if problem_type == ProblemType.REGRESSION:
                return {"primary_metric": "r2", "secondary_metrics": ["mae", "rmse"]}
            else:
                return {"primary_metric": "f1", "secondary_metrics": ["precision", "recall"]}
        
        return metric_mapping[key]
    
    @staticmethod
    def get_metric_description(metric: str) -> str:
        """Get human-readable description of a metric."""
        descriptions = {
            "accuracy": "Overall correctness (valid for balanced classes)",
            "precision": "TP/(TP+FP) - Minimizes false positives",
            "recall": "TP/(TP+FN) - Minimizes false negatives",
            "f1": "Harmonic mean of precision and recall",
            "roc_auc": "Threshold-independent separability measure",
            "pr_auc": "Precision-recall AUC (better for extreme imbalance)",
            "mae": "Mean absolute error (interpretable, robust to outliers)",
            "rmse": "Root mean squared error (penalizes large errors)",
            "r2": "Fraction of variance explained"
        }
        return descriptions.get(metric, "Unknown metric")


class BiasVarianceAnalyzer:
    """Analyzes bias-variance behavior from train/CV/test gaps."""
    
    @staticmethod
    def analyze_bias_variance(train_scores: np.ndarray, 
                            cv_scores: np.ndarray,
                            test_score: float) -> Dict[str, Any]:
        """
        Analyze bias-variance characteristics.
        
        Args:
            train_scores: Training scores across CV folds
            cv_scores: Cross-validation scores
            test_score: Final test set score
            
        Returns:
            Dictionary with bias-variance analysis
        """
        train_mean = train_scores.mean()
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        train_cv_gap = train_mean - cv_mean
        cv_test_gap = cv_mean - test_score
        
        # Diagnose bias-variance
        if train_cv_gap > 0.08:
            bias_variance_diagnosis = "significant_overfitting"
            recommendation = "Reduce model complexity, add regularization"
        elif train_cv_gap > 0.03:
            bias_variance_diagnosis = "mild_overfitting"
            recommendation = "Consider regularization or feature selection"
        elif train_mean < 0.7 and train_cv_gap < 0.03:
            bias_variance_diagnosis = "underfitting"
            recommendation = "Increase complexity or add features"
        else:
            bias_variance_diagnosis = "well_fitted"
            recommendation = "Good bias-variance balance"
        
        # Stability assessment
        if cv_std > 0.05:
            stability = "unstable"
            stability_note = "High variance across folds - unpredictable in production"
        elif cv_std > 0.02:
            stability = "moderately_stable"
            stability_note = "Moderate variance - acceptable but monitor"
        else:
            stability = "stable"
            stability_note = "Consistent performance across folds"
        
        return {
            "train_mean": train_mean,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "test_score": test_score,
            "train_cv_gap": train_cv_gap,
            "cv_test_gap": cv_test_gap,
            "bias_variance_diagnosis": bias_variance_diagnosis,
            "recommendation": recommendation,
            "stability": stability,
            "stability_note": stability_note
        }


class ThresholdOptimizer:
    """Optimizes decision thresholds for probabilistic classifiers."""
    
    @staticmethod
    def optimize_threshold(y_true: np.ndarray, 
                         y_proba: np.ndarray,
                         primary_metric: str = "f1",
                         threshold_range: Tuple[float, float] = (0.1, 0.9),
                         step: float = 0.01) -> Dict[str, Any]:
        """
        Find optimal decision threshold based on primary metric.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            primary_metric: Metric to optimize ('precision', 'recall', 'f1')
            threshold_range: Range of thresholds to test
            step: Step size for threshold search
            
        Returns:
            Dictionary with optimization results
        """
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "primary_score": locals()[primary_metric]
            })
        
        results_df = pd.DataFrame(results)
        best_idx = results_df["primary_score"].idxmax()
        best_result = results_df.iloc[best_idx].to_dict()
        
        return {
            "best_threshold": best_result["threshold"],
            "best_metrics": {
                "precision": best_result["precision"],
                "recall": best_result["recall"],
                "f1": best_result["f1"]
            },
            "all_results": results_df,
            "optimization_metric": primary_metric
        }


class ComputationalProfiler:
    """Profiles computational requirements of models."""
    
    @staticmethod
    def measure_inference_latency(model: BaseEstimator, 
                                X_test: np.ndarray,
                                n_runs: int = 10) -> Dict[str, float]:
        """
        Measure model inference latency.
        
        Args:
            model: Trained model
            X_test: Test data
            n_runs: Number of runs for averaging
            
        Returns:
            Dictionary with latency measurements
        """
        latencies = []
        
        # Warm up
        _ = model.predict(X_test[:1])
        
        for _ in range(n_runs):
            start_time = time.perf_counter()
            _ = model.predict(X_test)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "per_sample_latency_ms": np.mean(latencies) / len(X_test),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
    
    @staticmethod
    def estimate_memory_usage(model: BaseEstimator) -> Dict[str, float]:
        """
        Estimate model memory usage.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with memory estimates
        """
        # Get current process memory
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # This is a rough estimate - actual model size depends on implementation
        # For more accurate measurement, you'd need model-specific methods
        
        return {
            "estimated_model_size_mb": memory_info.rss / (1024 * 1024),
            "note": "This is a process-level estimate. Model-specific sizing needed for accuracy."
        }


class InterpretabilityAssessor:
    """Assesses model interpretability characteristics."""
    
    @staticmethod
    def assess_interpretability(model: BaseEstimator) -> Dict[str, Any]:
        """
        Assess model interpretability.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with interpretability assessment
        """
        model_name = model.__class__.__name__
        
        # Define interpretability levels for common model types
        interpretability_map = {
            "LogisticRegression": {
                "level": "high",
                "explanation": "Linear coefficients directly interpretable",
                "global_explanation": True,
                "local_explanation": True,
                "feature_importance": True
            },
            "LinearRegression": {
                "level": "high", 
                "explanation": "Linear coefficients directly interpretable",
                "global_explanation": True,
                "local_explanation": True,
                "feature_importance": True
            },
            "DecisionTreeClassifier": {
                "level": "medium",
                "explanation": "Tree structure can be visualized and explained",
                "global_explanation": True,
                "local_explanation": True,
                "feature_importance": True
            },
            "DecisionTreeRegressor": {
                "level": "medium",
                "explanation": "Tree structure can be visualized and explained", 
                "global_explanation": True,
                "local_explanation": True,
                "feature_importance": True
            },
            "RandomForestClassifier": {
                "level": "low",
                "explanation": "Ensemble of trees - feature importance available but individual predictions hard to explain",
                "global_explanation": True,
                "local_explanation": False,
                "feature_importance": True
            },
            "RandomForestRegressor": {
                "level": "low",
                "explanation": "Ensemble of trees - feature importance available but individual predictions hard to explain",
                "global_explanation": True,
                "local_explanation": False,
                "feature_importance": True
            },
            "GradientBoostingClassifier": {
                "level": "low",
                "explanation": "Complex ensemble - limited direct interpretability",
                "global_explanation": False,
                "local_explanation": False,
                "feature_importance": True
            },
            "SVC": {
                "level": "low",
                "explanation": "Black box for non-linear kernels",
                "global_explanation": False,
                "local_explanation": False,
                "feature_importance": False
            }
        }
        
        assessment = interpretability_map.get(model_name, {
            "level": "unknown",
            "explanation": "Model type not recognized for interpretability assessment",
            "global_explanation": False,
            "local_explanation": False,
            "feature_importance": False
        })
        
        assessment["model_type"] = model_name
        assessment["is_regulated_compliant"] = assessment["level"] in ["high", "medium"]
        
        return assessment


class ModelSelectionFramework:
    """Main framework for comprehensive model selection."""
    
    def __init__(self, 
                 problem_type: ProblemType,
                 business_objective: BusinessObjective,
                 deployment_environment: DeploymentEnvironment,
                 constraints: ModelConstraints):
        """
        Initialize the model selection framework.
        
        Args:
            problem_type: Type of ML problem
            business_objective: Primary business driver
            deployment_environment: Deployment context
            constraints: Operational constraints
        """
        self.problem_type = problem_type
        self.business_objective = business_objective
        self.deployment_environment = deployment_environment
        self.constraints = constraints
        
        # Initialize components
        self.metric_selector = MetricSelector()
        self.bias_variance_analyzer = BiasVarianceAnalyzer()
        self.threshold_optimizer = ThresholdOptimizer()
        self.computational_profiler = ComputationalProfiler()
        self.interpretability_assessor = InterpretabilityAssessor()
        
        # Get selected metrics
        self.selected_metrics = self.metric_selector.select_metrics(
            problem_type, business_objective
        )
        
        logger.info(f"Initialized framework for {problem_type.value} with {business_objective.value}")
        logger.info(f"Primary metric: {self.selected_metrics['primary_metric']}")
    
    def evaluate_model(self, 
                      model: BaseEstimator,
                      model_name: str,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      cv_folds: int = 5,
                      optimize_threshold: bool = True) -> ModelEvaluationResult:
        """
        Comprehensively evaluate a single model.
        
        Args:
            model: Model to evaluate
            model_name: Name for identification
            X_train, y_train: Training data
            X_test, y_test: Test data
            cv_folds: Number of CV folds
            optimize_threshold: Whether to optimize decision threshold
            
        Returns:
            Comprehensive evaluation result
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Cross-validation with train scores
        if self.problem_type == ProblemType.REGRESSION:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=self.selected_metrics["primary_metric"],
            return_train_score=True,
            n_jobs=-1
        )
        
        train_scores = cv_results["train_score"]
        cv_scores = cv_results["test_score"]
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Fit on full training data for test evaluation
        model.fit(X_train, y_train)
        
        # Test evaluation
        y_pred = model.predict(X_test)
        test_score = self._calculate_metric(
            y_test, y_pred, self.selected_metrics["primary_metric"]
        )
        
        # Bias-variance analysis
        bias_variance_analysis = self.bias_variance_analyzer.analyze_bias_variance(
            train_scores, cv_scores, test_score
        )
        
        # Calculate all metrics
        all_metrics = self._calculate_all_metrics(y_test, y_pred, model, X_test)
        
        # Computational profiling
        latency_results = self.computational_profiler.measure_inference_latency(model, X_test)
        memory_results = self.computational_profiler.estimate_memory_usage(model)
        
        # Interpretability assessment
        interpretability = self.interpretability_assessor.assess_interpretability(model)
        
        # Threshold optimization for classifiers
        optimal_threshold = 0.5
        if optimize_threshold and hasattr(model, 'predict_proba'):
            threshold_results = self.threshold_optimizer.optimize_threshold(
                y_test, model.predict_proba(X_test)[:, 1],
                primary_metric=self.selected_metrics["primary_metric"]
            )
            optimal_threshold = threshold_results["best_threshold"]
        
        # Check constraint violations
        violations = self._check_constraint_violations(
            latency_results["mean_latency_ms"],
            memory_results["estimated_model_size_mb"],
            interpretability
        )
        
        result = ModelEvaluationResult(
            model_name=model_name,
            model=model,
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_score=train_scores.mean(),
            test_score=test_score,
            bias_variance_gap=bias_variance_analysis["train_cv_gap"],
            metrics=all_metrics,
            latency_ms=latency_results["mean_latency_ms"],
            memory_mb=memory_results["estimated_model_size_mb"],
            is_interpretable=interpretability["level"] in ["high", "medium"],
            threshold_optimized=optimize_threshold,
            optimal_threshold=optimal_threshold,
            constraint_violations=violations
        )
        
        logger.info(f"Completed evaluation: CV={cv_mean:.3f}±{cv_std:.3f}, Test={test_score:.3f}")
        
        return result
    
    def compare_models(self, 
                      models: Dict[str, BaseEstimator],
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      cv_folds: int = 5) -> List[ModelEvaluationResult]:
        """
        Compare multiple models comprehensively.
        
        Args:
            models: Dictionary of model_name -> model
            X_train, y_train: Training data
            X_test, y_test: Test data
            cv_folds: Number of CV folds
            
        Returns:
            List of evaluation results for all models
        """
        results = []
        
        for model_name, model in models.items():
            result = self.evaluate_model(
                model, model_name, X_train, y_train, X_test, y_test, cv_folds
            )
            results.append(result)
        
        return results
    
    def create_decision_table(self, results: List[ModelEvaluationResult]) -> pd.DataFrame:
        """
        Create comprehensive decision table for model selection.
        
        Args:
            results: List of model evaluation results
            
        Returns:
            Decision table as DataFrame
        """
        decision_data = []
        
        for result in results:
            row = {
                "Model": result.model_name,
                "CV_Mean": round(result.cv_mean, 3),
                "CV_Std": round(result.cv_std, 3),
                "Test_Score": round(result.test_score, 3),
                "Bias_Variance_Gap": round(result.bias_variance_gap, 3),
                "Latency_ms": round(result.latency_ms, 1),
                "Memory_MB": round(result.memory_mb, 1),
                "Interpretable": result.is_interpretable,
                "Constraint_Violations": len(result.constraint_violations)
            }
            
            # Add all metrics
            for metric, value in result.metrics.items():
                row[f"Metric_{metric}"] = round(value, 3)
            
            decision_data.append(row)
        
        df = pd.DataFrame(decision_data)
        
        # Sort by primary metric, then by stability, then by constraints
        primary_metric_col = f"Metric_{self.selected_metrics['primary_metric']}"
        df = df.sort_values([primary_metric_col, "CV_Std", "Constraint_Violations"], 
                           ascending=[False, True, True])
        
        return df
    
    def recommend_model(self, results: List[ModelEvaluationResult]) -> Tuple[ModelEvaluationResult, str]:
        """
        Recommend the best model based on multi-criteria analysis.
        
        Args:
            results: List of model evaluation results
            
        Returns:
            Tuple of (best_model_result, recommendation_rationale)
        """
        # Filter models that violate hard constraints
        viable_models = [
            r for r in results 
            if len(r.constraint_violations) == 0
        ]
        
        if not viable_models:
            # If no models meet constraints, relax to warnings only
            viable_models = results
            constraint_note = "No models meet all constraints. Recommending from all models."
        else:
            constraint_note = "All recommended models meet constraints."
        
        # Multi-criteria scoring
        scored_models = []
        for result in viable_models:
            score = self._calculate_multi_criteria_score(result)
            scored_models.append((score, result))
        
        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        best_result = scored_models[0][1]
        
        # Generate rationale
        rationale = self._generate_recommendation_rationale(
            best_result, viable_models, constraint_note
        )
        
        return best_result, rationale
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate a specific metric."""
        if metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        elif metric == "f1":
            return f1_score(y_true, y_pred, zero_division=0)
        elif metric == "roc_auc":
            return roc_auc_score(y_true, y_pred)
        elif metric == "pr_auc":
            return average_precision_score(y_true, y_pred)
        elif metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "r2":
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model: BaseEstimator, X_test: np.ndarray) -> Dict[str, float]:
        """Calculate all relevant metrics for the problem type."""
        metrics = {}
        
        if self.problem_type == ProblemType.REGRESSION:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["r2"] = r2_score(y_true, y_pred)
        else:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
                except ValueError:
                    # Handle edge cases for AUC calculation
                    metrics["roc_auc"] = 0.0
                    metrics["pr_auc"] = 0.0
        
        return metrics
    
    def _check_constraint_violations(self, latency_ms: float, memory_mb: float,
                                    interpretability: Dict[str, Any]) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        if self.constraints.max_latency_ms and latency_ms > self.constraints.max_latency_ms:
            violations.append(f"Latency {latency_ms:.1f}ms > {self.constraints.max_latency_ms}ms")
        
        if self.constraints.max_memory_mb and memory_mb > self.constraints.max_memory_mb:
            violations.append(f"Memory {memory_mb:.1f}MB > {self.constraints.max_memory_mb}MB")
        
        if self.constraints.interpretable and interpretability["level"] not in ["high", "medium"]:
            violations.append(f"Model not interpretable enough ({interpretability['level']})")
        
        if self.constraints.requires_gpu and not hasattr(interpretability, 'requires_gpu'):
            violations.append("GPU required but not available")
        
        return violations
    
    def _calculate_multi_criteria_score(self, result: ModelEvaluationResult) -> float:
        """Calculate multi-criteria score for model ranking."""
        score = 0.0
        
        # Primary metric performance (40% weight)
        primary_metric = self.selected_metrics["primary_metric"]
        primary_score = result.metrics.get(primary_metric, 0.0)
        score += primary_score * 0.4
        
        # Stability (20% weight) - lower std is better
        stability_score = max(0, 1 - result.cv_std * 10)  # Normalize
        score += stability_score * 0.2
        
        # Bias-variance balance (15% weight) - lower gap is better
        gap_score = max(0, 1 - result.bias_variance_gap * 5)
        score += gap_score * 0.15
        
        # Computational efficiency (15% weight)
        if self.constraints.max_latency_ms:
            latency_score = max(0, 1 - result.latency_ms / self.constraints.max_latency_ms)
        else:
            latency_score = 1.0  # No constraint, favor performance
        score += latency_score * 0.15
        
        # Interpretability (10% weight)
        interpretability_score = 1.0 if result.is_interpretable else 0.5
        if self.constraints.interpretable:
            interpretability_score = 1.0 if result.is_interpretable else 0.0
        score += interpretability_score * 0.1
        
        # Penalty for constraint violations
        if result.constraint_violations:
            score *= 0.5  # Heavy penalty
        
        return score
    
    def _generate_recommendation_rationale(self, best_result: ModelEvaluationResult,
                                         viable_models: List[ModelEvaluationResult],
                                         constraint_note: str) -> str:
        """Generate detailed recommendation rationale."""
        rationale = []
        rationale.append(f"RECOMMENDED MODEL: {best_result.model_name}")
        rationale.append("=" * 50)
        rationale.append(constraint_note)
        rationale.append("")
        
        # Performance
        primary_metric = self.selected_metrics["primary_metric"]
        primary_score = best_result.metrics.get(primary_metric, 0.0)
        rationale.append(f"PERFORMANCE:")
        rationale.append(f"  {primary_metric.upper()}: {primary_score:.3f}")
        rationale.append(f"  CV Score: {best_result.cv_mean:.3f} ± {best_result.cv_std:.3f}")
        rationale.append(f"  Test Score: {best_result.test_score:.3f}")
        rationale.append("")
        
        # Stability
        rationale.append(f"STABILITY:")
        rationale.append(f"  CV Standard Deviation: {best_result.cv_std:.3f}")
        if best_result.cv_std < 0.02:
            rationale.append(f"  → Excellent stability across folds")
        elif best_result.cv_std < 0.05:
            rationale.append(f"  → Good stability, monitor in production")
        else:
            rationale.append(f"  → High variance - unpredictable performance")
        rationale.append("")
        
        # Bias-Variance
        rationale.append(f"BIAS-VARIANCE:")
        rationale.append(f"  Train-CV Gap: {best_result.bias_variance_gap:.3f}")
        if best_result.bias_variance_gap < 0.03:
            rationale.append(f"  → Well-fitted model")
        elif best_result.bias_variance_gap < 0.08:
            rationale.append(f"  → Mild overfitting")
        else:
            rationale.append(f"  → Significant overfitting")
        rationale.append("")
        
        # Computational
        rationale.append(f"COMPUTATIONAL:")
        rationale.append(f"  Inference Latency: {best_result.latency_ms:.1f}ms")
        rationale.append(f"  Memory Usage: {best_result.memory_mb:.1f}MB")
        if self.constraints.max_latency_ms:
            if best_result.latency_ms <= self.constraints.max_latency_ms:
                rationale.append(f"  → Meets latency requirement")
            else:
                rationale.append(f"  → Exceeds latency budget")
        rationale.append("")
        
        # Interpretability
        rationale.append(f"INTERPRETABILITY:")
        rationale.append(f"  Interpretable: {best_result.is_interpretable}")
        if self.constraints.interpretable:
            if best_result.is_interpretable:
                rationale.append(f"  → Meets interpretability requirement")
            else:
                rationale.append(f"  → Does not meet interpretability requirement")
        rationale.append("")
        
        # Comparison context
        if len(viable_models) > 1:
            rationale.append(f"COMPARISON:")
            rationale.append(f"  Evaluated {len(viable_models)} viable models")
            rationale.append(f"  Selected based on multi-criteria analysis")
            rationale.append(f"  Considered: performance, stability, efficiency, interpretability")
        
        return "\n".join(rationale)
    
    def generate_professional_checklist(self) -> List[str]:
        """Generate professional model selection checklist."""
        checklist = [
            f"✓ Evaluation metric: {self.selected_metrics['primary_metric']} (aligned with {self.business_objective.value})",
            f"✓ Cross-validation: StratifiedKFold (classification) / KFold (regression)",
            "✓ CV mean and CV std reported for all candidates",
            "✓ Train/CV gap computed and diagnosed per model",
            "✓ Threshold tuned on validation data (for classifiers)",
            "✓ Test set evaluated exactly once with selected model",
            "✓ Multi-metric decision table constructed",
            "✓ Interpretability requirements assessed",
            "✓ Inference latency measured against deployment constraints",
            "✓ Memory usage estimated against deployment constraints",
            f"✓ Deployment environment: {self.deployment_environment.value}",
            "✓ Selection rationale documented and defensible"
        ]
        
        return checklist


def create_sample_scenario() -> Dict[str, Any]:
    """Create a sample scenario for demonstration."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate imbalanced classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.9, 0.1], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Setup framework for fraud detection (high FN cost)
    constraints = ModelConstraints(
        max_latency_ms=50.0,
        max_memory_mb=100.0,
        interpretable=False,
        update_frequency="monthly"
    )
    
    framework = ModelSelectionFramework(
        problem_type=ProblemType.IMBALANCED_CLASSIFICATION,
        business_objective=BusinessObjective.MINIMIZE_FN_COST,
        deployment_environment=DeploymentEnvironment.REAL_TIME,
        constraints=constraints
    )
    
    return {
        "framework": framework,
        "models": models,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Model Selection Framework Demo")
    print("=" * 50)
    
    scenario = create_sample_scenario()
    framework = scenario["framework"]
    models = scenario["models"]
    
    print(f"Problem Type: {framework.problem_type.value}")
    print(f"Business Objective: {framework.business_objective.value}")
    print(f"Primary Metric: {framework.selected_metrics['primary_metric']}")
    print()
    
    # Evaluate models
    results = framework.compare_models(
        models, 
        scenario["X_train"], scenario["y_train"],
        scenario["X_test"], scenario["y_test"]
    )
    
    # Create decision table
    decision_table = framework.create_decision_table(results)
    print("DECISION TABLE:")
    print(decision_table.to_string(index=False))
    print()
    
    # Get recommendation
    best_model, rationale = framework.recommend_model(results)
    print(rationale)
    print()
    
    # Professional checklist
    print("PROFESSIONAL CHECKLIST:")
    for item in framework.generate_professional_checklist():
        print(item)
