"""
Professional Model Comparison Workflow

This module implements systematic, fair comparison of multiple machine learning models
with consistent preprocessing, cross-validation, and comprehensive evaluation.
Following professional ML practices for model selection and diagnosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, 
    RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint, loguniform, uniform

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ModelComparisonResult:
    """Data class to store model comparison results."""
    model_name: str
    cv_mean: float
    cv_std: float
    train_mean: float
    train_std: float
    gap: float
    best_params: Optional[Dict] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    scoring_metric: str = "f1"
    n_iter_tuning: int = 30
    n_jobs: int = -1
    verbose: bool = True


class ModelComparisonFramework:
    """
    Comprehensive framework for fair, systematic model comparison.
    
    Features:
    - Consistent preprocessing pipelines
    - Fair hyperparameter tuning
    - Cross-validation with uncertainty quantification
    - Train/CV gap analysis
    - Statistical significance testing
    - Multi-metric evaluation
    - Visualization and reporting
    """
    
    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()
        self.preprocessor = None
        self.results = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.numeric_features = None
        self.categorical_features = None
        
    def setup_data(self, X: pd.DataFrame, y: pd.Series, 
                   numeric_features: List[str] = None,
                   categorical_features: List[str] = None) -> None:
        """
        Setup train/test split and identify feature types.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
        """
        # Auto-detect feature types if not provided
        if numeric_features is None or categorical_features is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Stratified split for classification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            stratify=y, random_state=self.config.random_state
        )
        
        if self.config.verbose:
            print(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test samples")
            print(f"Numeric features: {len(self.numeric_features)}")
            print(f"Categorical features: {len(self.categorical_features)}")
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create a consistent preprocessing pipeline for all models.
        
        Returns:
            Configured ColumnTransformer for preprocessing
        """
        if self.numeric_features is None or self.categorical_features is None:
            raise ValueError("Must call setup_data() first")
        
        # Numeric preprocessing: imputation + scaling
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Categorical preprocessing: imputation + encoding
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features)
        ])
        
        return self.preprocessor
    
    def make_pipeline(self, model) -> Pipeline:
        """
        Create a pipeline with consistent preprocessing and the given model.
        
        Args:
            model: Scikit-learn estimator
            
        Returns:
            Pipeline with preprocessing and model
        """
        if self.preprocessor is None:
            self.create_preprocessor()
        
        return Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", model)
        ])
    
    def get_model_library(self) -> Dict[str, Any]:
        """
        Get dictionary of models to compare.
        
        Returns:
            Dictionary mapping model names to estimators
        """
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.config.random_state
            ),
            "Ridge Classifier": RidgeClassifier(random_state=self.config.random_state),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, random_state=self.config.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.config.random_state
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, random_state=self.config.random_state
            ),
            "KNN": KNeighborsClassifier(n_neighbors=11),
            "SVM": SVC(probability=True, random_state=self.config.random_state),
            "Naive Bayes": GaussianNB()
        }
    
    def get_hyperparameter_distributions(self) -> Dict[str, Dict]:
        """
        Get hyperparameter distributions for fair tuning.
        
        Returns:
            Dictionary mapping model names to parameter distributions
        """
        return {
            "Logistic Regression": {
                "model__C": loguniform(1e-3, 1e2),
                "model__penalty": ["l2", "none"]
            },
            "Ridge Classifier": {
                "model__alpha": loguniform(1e-3, 1e2)
            },
            "Decision Tree": {
                "model__max_depth": randint(3, 20),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 20)
            },
            "Random Forest": {
                "model__n_estimators": randint(50, 300),
                "model__max_depth": randint(3, 20),
                "model__min_samples_split": randint(2, 20),
                "model__max_features": ["sqrt", "log2", None]
            },
            "Gradient Boosting": {
                "model__n_estimators": randint(50, 300),
                "model__learning_rate": loguniform(1e-3, 1e-1),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.6, 0.4)
            },
            "AdaBoost": {
                "model__n_estimators": randint(50, 300),
                "model__learning_rate": loguniform(1e-3, 1e-1)
            },
            "KNN": {
                "model__n_neighbors": randint(3, 30),
                "model__weights": ["uniform", "distance"]
            },
            "SVM": {
                "model__C": loguniform(1e-3, 1e2),
                "model__gamma": ["scale", "auto"]
            },
            "Naive Bayes": {}  # No hyperparameters to tune
        }
    
    def evaluate_model_cv(self, model_name: str, model: Any, 
                         tune_hyperparameters: bool = True) -> ModelComparisonResult:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model_name: Name of the model
            model: Scikit-learn estimator
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            ModelComparisonResult with evaluation metrics
        """
        if self.X_train is None:
            raise ValueError("Must call setup_data() first")
        
        # Create pipeline
        pipeline = self.make_pipeline(model)
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, 
            shuffle=True, 
            random_state=self.config.random_state
        )
        
        # Hyperparameter tuning
        best_params = None
        training_time = None
        
        if tune_hyperparameters and model_name in self.get_hyperparameter_distributions():
            param_distributions = self.get_hyperparameter_distributions()[model_name]
            
            if param_distributions:  # Only tune if there are parameters to tune
                start_time = time.time()
                
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions,
                    n_iter=self.config.n_iter_tuning,
                    cv=cv,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                    random_state=self.config.random_state,
                    verbose=0
                )
                
                search.fit(self.X_train, self.y_train)
                pipeline = search.best_estimator_
                best_params = search.best_params_
                training_time = time.time() - start_time
                
                if self.config.verbose:
                    print(f"  {model_name}: Best params = {best_params}")
        
        # Cross-validation with train scores for gap analysis
        cv_results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=self.config.scoring_metric,
            return_train_score=True,
            n_jobs=self.config.n_jobs
        )
        
        # Calculate metrics
        train_scores = cv_results['train_score']
        test_scores = cv_results['test_score']
        
        train_mean = train_scores.mean()
        train_std = train_scores.std()
        cv_mean = test_scores.mean()
        cv_std = test_scores.std()
        gap = train_mean - cv_mean
        
        # Measure inference time
        start_time = time.time()
        _ = pipeline.predict(self.X_test[:100])  # Sample for timing
        inference_time = time.time() - start_time
        
        return ModelComparisonResult(
            model_name=model_name,
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_mean=train_mean,
            train_std=train_std,
            gap=gap,
            best_params=best_params,
            training_time=training_time,
            inference_time=inference_time
        )
    
    def compare_models(self, model_names: List[str] = None, 
                      tune_hyperparameters: bool = True) -> pd.DataFrame:
        """
        Compare multiple models systematically.
        
        Args:
            model_names: List of model names to compare (None for all)
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            DataFrame with comparison results
        """
        if self.X_train is None:
            raise ValueError("Must call setup_data() first")
        
        models = self.get_model_library()
        
        if model_names is None:
            model_names = list(models.keys())
        
        self.results = []
        
        if self.config.verbose:
            print(f"\nComparing {len(model_names)} models...")
            print("=" * 60)
        
        for model_name in model_names:
            if model_name not in models:
                print(f"Warning: Model '{model_name}' not found in library")
                continue
            
            if self.config.verbose:
                print(f"Evaluating {model_name}...")
            
            model = models[model_name]
            result = self.evaluate_model_cv(model_name, model, tune_hyperparameters)
            self.results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'Model': r.model_name,
                'CV Mean': round(r.cv_mean, 3),
                'CV Std': round(r.cv_std, 3),
                'Train Mean': round(r.train_mean, 3),
                'Train Std': round(r.train_std, 3),
                'Gap': round(r.gap, 3),
                'Training Time (s)': round(r.training_time, 1) if r.training_time else None,
                'Inference Time (ms)': round(r.inference_time * 1000, 2) if r.inference_time else None,
                'Best Params': str(r.best_params) if r.best_params else 'Default'
            }
            for r in self.results
        ])
        
        # Sort by CV mean score
        results_df = results_df.sort_values('CV Mean', ascending=False)
        
        if self.config.verbose:
            print("\nModel Comparison Results:")
            print("=" * 60)
            print(results_df.to_string(index=False))
        
        return results_df
    
    def analyze_bias_variance(self) -> Dict[str, str]:
        """
        Analyze bias-variance characteristics based on train/CV gaps.
        
        Returns:
            Dictionary mapping model names to diagnostic interpretations
        """
        interpretations = {}
        
        for result in self.results:
            gap = result.gap
            cv_mean = result.cv_mean
            
            if gap < 0.03:
                if cv_mean > 0.8:
                    interpretation = "Well-fitted: High performance with low variance"
                else:
                    interpretation = "Underfitting: Low performance, low variance - increase complexity"
            elif gap < 0.08:
                interpretation = "Mild overfitting: Good performance with moderate variance"
            else:
                interpretation = "Significant overfitting: High variance - needs regularization"
            
            interpretations[result.model_name] = interpretation
        
        return interpretations
    
    def statistical_significance_test(self, model1: str, model2: str, 
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Get CV scores for both models
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        models = self.get_model_library()
        
        # Get pipelines
        pipeline1 = self.make_pipeline(models[model1])
        pipeline2 = self.make_pipeline(models[model2])
        
        # Get CV scores
        scores1 = cross_val_score(
            pipeline1, self.X_train, self.y_train, 
            cv=cv, scoring=self.config.scoring_metric
        )
        scores2 = cross_val_score(
            pipeline2, self.X_train, self.y_train,
            cv=cv, scoring=self.config.scoring_metric
        )
        
        # Paired t-test
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d)
        mean_diff = scores1.mean() - scores2.mean()
        pooled_std = np.sqrt(((scores1.var() + scores2.var()) / 2))
        cohens_d = mean_diff / pooled_std
        
        return {
            'model1': model1,
            'model2': model2,
            'model1_mean': scores1.mean(),
            'model2_mean': scores2.mean(),
            'mean_difference': mean_diff,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def multi_metric_evaluation(self, model_name: str) -> Dict[str, float]:
        """
        Evaluate a model on multiple metrics for imbalanced problems.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            Dictionary with multiple metric scores
        """
        models = self.get_model_library()
        model = models[model_name]
        
        # Train on full training set
        pipeline = self.make_pipeline(model)
        pipeline.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = pipeline.predict(self.X_test)
        y_proba = pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_proba)
        }
        
        return metrics
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Create visualization of model comparison results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No results to plot. Run compare_models() first.")
            return
        
        # Prepare data
        model_names = [r.model_name for r in self.results]
        means = [r.cv_mean for r in self.results]
        stds = [r.cv_std for r in self.results]
        
        # Sort by mean score
        sorted_indices = np.argsort(means)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                     color="steelblue", alpha=0.8, edgecolor="black")
        
        # Customize plot
        ax.set_ylabel(f"CV {self.config.scoring_metric.upper()} Score", fontsize=12)
        ax.set_title("Model Comparison: Cross-Validation Performance (mean ± std)", 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0.5, 1.0)
        
        # Add reference line for best model
        ax.axhline(y=means[0], color="red", linestyle="--", alpha=0.4, 
                  label=f"Best: {model_names[0]}")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.config.verbose:
                print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_bias_variance_analysis(self, save_path: str = None) -> None:
        """
        Plot bias-variance analysis showing train/CV gaps.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No results to plot. Run compare_models() first.")
            return
        
        # Prepare data
        model_names = [r.model_name for r in self.results]
        train_scores = [r.train_mean for r in self.results]
        cv_scores = [r.cv_mean for r in self.results]
        
        # Sort by CV score
        sorted_indices = np.argsort(cv_scores)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        train_scores = [train_scores[i] for i in sorted_indices]
        cv_scores = [cv_scores[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, 
                      label='Train Score', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, cv_scores, width, 
                      label='CV Score', alpha=0.8, color='steelblue')
        
        # Customize plot
        ax.set_ylabel(f"{self.config.scoring_metric.upper()} Score", fontsize=12)
        ax.set_title("Bias-Variance Analysis: Train vs CV Scores", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add gap annotations
        for i, (train, cv) in enumerate(zip(train_scores, cv_scores)):
            gap = train - cv
            ax.annotate(f'Gap: {gap:.3f}', 
                       xy=(i, cv), xytext=(i, cv + 0.02),
                       ha='center', fontsize=8, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.config.verbose:
                print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.results:
            return "No results to report. Run compare_models() first."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("=" * 80)
        
        # Results table
        results_df = pd.DataFrame([
            {
                'Model': r.model_name,
                'CV Mean': round(r.cv_mean, 3),
                'CV Std': round(r.cv_std, 3),
                'Gap': round(r.gap, 3),
                'Training Time (s)': round(r.training_time, 1) if r.training_time else None
            }
            for r in self.results
        ]).sort_values('CV Mean', ascending=False)
        
        report.append("\n1. PERFORMANCE RANKINGS")
        report.append("-" * 40)
        report.append(results_df.to_string(index=False))
        
        # Bias-variance analysis
        report.append("\n\n2. BIAS-VARIANCE DIAGNOSIS")
        report.append("-" * 40)
        interpretations = self.analyze_bias_variance()
        for model_name, interpretation in interpretations.items():
            report.append(f"{model_name:25}: {interpretation}")
        
        # Recommendations
        report.append("\n\n3. RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_model = max(self.results, key=lambda r: r.cv_mean)
        most_stable = min(self.results, key=lambda r: r.cv_std)
        
        report.append(f"Best Performance: {best_model.model_name} "
                     f"(CV {best_model.cv_mean:.3f} ± {best_model.cv_std:.3f})")
        report.append(f"Most Stable: {most_stable.model_name} "
                     f"(CV {most_stable.cv_mean:.3f} ± {most_stable.cv_std:.3f})")
        
        # Selection considerations
        report.append("\n4. SELECTION CONSIDERATIONS")
        report.append("-" * 40)
        
        for result in self.results:
            considerations = []
            if result.cv_mean > 0.85:
                considerations.append("High performance")
            if result.cv_std < 0.02:
                considerations.append("Very stable")
            if result.gap < 0.03:
                considerations.append("Well-fitted")
            if result.training_time and result.training_time < 10:
                considerations.append("Fast training")
            if result.inference_time and result.inference_time < 0.01:
                considerations.append("Fast inference")
            
            if considerations:
                report.append(f"{result.model_name:25}: {', '.join(considerations)}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            if self.config.verbose:
                print(f"Report saved to {save_path}")
        
        return report_text
    
    def select_best_model(self, criteria: str = "performance") -> Tuple[str, ModelComparisonResult]:
        """
        Select the best model based on specified criteria.
        
        Args:
            criteria: Selection criteria ('performance', 'stability', 'efficiency')
            
        Returns:
            Tuple of (model_name, result)
        """
        if not self.results:
            raise ValueError("No results available. Run compare_models() first.")
        
        if criteria == "performance":
            best = max(self.results, key=lambda r: r.cv_mean)
        elif criteria == "stability":
            best = min(self.results, key=lambda r: r.cv_std)
        elif criteria == "efficiency":
            # Balance performance and training time
            best = min(self.results, 
                      key=lambda r: -r.cv_mean / (r.training_time + 1) if r.training_time else -r.cv_mean)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return best.model_name, best
