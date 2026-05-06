"""
Pipeline Leakage Prevention Module

This module provides comprehensive tools for preventing data leakage through proper
pipeline construction and evaluation methodologies. It includes detection, prevention,
and demonstration of common leakage scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder, 
                              OrdinalEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                 GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                          precision_score, recall_score, f1_score)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    Detects potential data leakage in preprocessing and evaluation workflows.
    """
    
    def __init__(self):
        self.warnings = []
        self.leakage_score = 0.0
    
    def check_scaling_leakage(self, X_train, X_test, scaler):
        """
        Check if scaler was fitted on test data.
        """
        if hasattr(scaler, 'mean_'):
            # For StandardScaler
            train_mean = np.mean(X_train, axis=0)
            test_mean = np.mean(X_test, axis=0)
            scaler_mean = scaler.mean_
            
            # Check if scaler mean is closer to combined data than training data
            combined_mean = np.mean(np.vstack([X_train, X_test]), axis=0)
            distance_to_train = np.mean(np.abs(scaler_mean - train_mean))
            distance_to_combined = np.mean(np.abs(scaler_mean - combined_mean))
            
            if distance_to_combined < distance_to_train:
                self.warnings.append("⚠️  Scaler may have been fitted on combined data")
                self.leakage_score += 0.3
                return True
        
        return False
    
    def check_feature_selection_leakage(self, X, y, selector):
        """
        Check if feature selection was done before split.
        """
        if hasattr(selector, 'scores_'):
            # Check if selector was fitted on full dataset
            n_features_original = X.shape[1]
            n_features_selected = selector.get_support().sum()
            
            if n_features_selected > n_features_original * 0.8:  # Suspicious if too many features kept
                self.warnings.append("⚠️  Feature selection may have been done before split")
                self.leakage_score += 0.4
                return True
        
        return False
    
    def check_target_encoding_leakage(self, encoder, X_train, X_test):
        """
        Check for target encoding leakage.
        """
        if hasattr(encoder, 'target_type_'):
            self.warnings.append("⚠️  Target encoding detected - ensure it's inside pipeline")
            self.leakage_score += 0.5
            return True
        
        return False
    
    def check_cv_leakage(self, cv_scores, test_score):
        """
        Check for CV vs test score discrepancy indicating leakage.
        """
        cv_mean = np.mean(cv_scores)
        
        if test_score > cv_mean + 0.05:  # 5% gap
            self.warnings.append("⚠️  Test score significantly higher than CV - possible leakage")
            self.leakage_score += 0.2
        
        return self.leakage_score
    
    def get_leakage_report(self):
        """
        Get comprehensive leakage report.
        """
        severity = "Low" if self.leakage_score < 0.3 else \
                 "Medium" if self.leakage_score < 0.7 else \
                 "High" if self.leakage_score < 1.0 else "Critical"
        
        return {
            'leakage_score': self.leakage_score,
            'severity': severity,
            'warnings': self.warnings,
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self):
        """
        Get recommendations based on detected warnings.
        """
        recommendations = []
        
        if "scaler" in str(self.warnings):
            recommendations.append("Use Pipeline to ensure scaler is fitted only on training data")
        
        if "feature selection" in str(self.warnings):
            recommendations.append("Move feature selection inside Pipeline")
        
        if "target encoding" in str(self.warnings):
            recommendations.append("Use TargetEncoder inside Pipeline with proper CV isolation")
        
        if "CV" in str(self.warnings):
            recommendations.append("Rebuild pipeline and re-evaluate with proper CV")
        
        return recommendations


class SafePipelineBuilder:
    """
    Builder for creating leakage-safe pipelines.
    """
    
    def __init__(self):
        self.steps = []
        self.column_transformer = None
    
    def add_scaler(self, scaler_type="standard", **kwargs):
        """
        Add scaling step to pipeline.
        """
        if scaler_type == "standard":
            scaler = StandardScaler(**kwargs)
        elif scaler_type == "minmax":
            scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        self.steps.append(("scaler", scaler))
        return self
    
    def add_imputer(self, strategy="mean", **kwargs):
        """
        Add imputation step to pipeline.
        """
        imputer = SimpleImputer(strategy=strategy, **kwargs)
        self.steps.append(("imputer", imputer))
        return self
    
    def add_encoder(self, encoder_type="onehot", columns=None, **kwargs):
        """
        Add encoding step to pipeline.
        """
        if encoder_type == "onehot":
            encoder = OneHotEncoder(**kwargs)
        elif encoder_type == "ordinal" and TARGET_ENCODER_AVAILABLE:
            encoder = TargetEncoder(**kwargs)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.steps.append(("encoder", encoder))
        return self
    
    def add_feature_selector(self, selector_type="kbest", **kwargs):
        """
        Add feature selection step to pipeline.
        """
        if selector_type == "kbest":
            selector = SelectKBest(**kwargs)
        elif selector_type == "rfe":
            from sklearn.feature_selection import RFE
            selector = RFE(**kwargs)
        else:
            raise ValueError(f"Unsupported selector type: {selector_type}")
        
        self.steps.append(("selector", selector))
        return self
    
    def add_dimensionality_reduction(self, method="pca", **kwargs):
        """
        Add dimensionality reduction step to pipeline.
        """
        if method == "pca":
            reducer = PCA(**kwargs)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        self.steps.append(("reducer", reducer))
        return self
    
    def add_model(self, model, model_name="model"):
        """
        Add final model step to pipeline.
        """
        self.steps.append((model_name, model))
        return self
    
    def build(self):
        """
        Build the final pipeline.
        """
        if not self.steps:
            raise ValueError("No steps added to pipeline")
        
        return Pipeline(self.steps)
    
    def build_with_column_transformer(self, numerical_columns=None, categorical_columns=None):
        """
        Build pipeline with ColumnTransformer for mixed data types.
        """
        transformers = []
        
        if numerical_columns:
            numerical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", numerical_transformer, numerical_columns))
        
        if categorical_columns:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformers.append(("cat", categorical_transformer, categorical_columns))
        
        self.column_transformer = ColumnTransformer(transformers)
        
        # Add column transformer and model
        final_steps = [("preprocessor", self.column_transformer)]
        final_steps.extend(self.steps)
        
        return Pipeline(final_steps)


def create_safe_pipeline(numerical_columns=None, categorical_columns=None, 
                     model_type="logistic", **model_params):
    """
    Create a safe pipeline with common preprocessing steps.
    
    Args:
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        model_type: Type of model to use
        **model_params: Model parameters
        
    Returns:
        Configured pipeline
    """
    builder = SafePipelineBuilder()
    
    # Add preprocessing based on column types
    if numerical_columns and categorical_columns:
        # Mixed data types - use ColumnTransformer
        transformers = []
        
        if numerical_columns:
            num_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_transformer, numerical_columns))
        
        if categorical_columns:
            cat_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformers.append(("cat", cat_transformer, categorical_columns))
        
        preprocessor = ColumnTransformer(transformers)
        
        # Add model
        if model_type == "logistic":
            model = LogisticRegression(**model_params)
        elif model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    
    else:
        # Single data type - simple pipeline
        builder.add_imputer(strategy="mean")
        builder.add_scaler("standard")
        
        if model_type == "logistic":
            model = LogisticRegression(**model_params)
        elif model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        builder.add_model(model)
        return builder.build()


def demonstrate_leakage_scenarios():
    """
    Demonstrate common leakage scenarios and their fixes.
    """
    print("="*80)
    print("DEMONSTRATING COMMON LEAKAGE SCENARIOS")
    print("="*80)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8, 
        n_redundant=2, n_clusters_per_class=1, 
        weights=[0.9, 0.1], flip_y=0.01, random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scenarios = {}
    
    # Scenario 1: Scaling before split (WRONG)
    print("\n1. SCALING BEFORE SPLIT (WRONG)")
    scaler_wrong = StandardScaler()
    X_scaled_wrong = scaler_wrong.fit_transform(X)  # Fitted on ALL data
    
    X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
        X_scaled_wrong, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_wrong = LogisticRegression(random_state=42)
    model_wrong.fit(X_train_wrong, y_train_wrong)
    wrong_score = model_wrong.score(X_test_wrong, y_test_wrong)
    
    print(f"   Wrong approach score: {wrong_score:.3f}")
    
    # Scenario 2: Scaling after split (CORRECT)
    print("\n2. SCALING AFTER SPLIT (CORRECT)")
    pipeline_correct = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    
    cv_scores = cross_val_score(pipeline_correct, X_train, y_train, cv=5)
    pipeline_correct.fit(X_train, y_train)
    correct_score = pipeline_correct.score(X_test, y_test)
    
    print(f"   Correct approach CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   Correct approach test score: {correct_score:.3f}")
    
    scenarios['scaling'] = {
        'wrong_score': wrong_score,
        'correct_cv': cv_scores.mean(),
        'correct_test': correct_score,
        'leakage_gap': wrong_score - correct_score
    }
    
    # Scenario 3: Feature selection before split (WRONG)
    print("\n3. FEATURE SELECTION BEFORE SPLIT (WRONG)")
    selector_wrong = SelectKBest(k=5, score_func=f_classif)
    X_selected_wrong = selector_wrong.fit_transform(X, y)  # Fitted on ALL data
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
        X_selected_wrong, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_fs_wrong = LogisticRegression(random_state=42)
    model_fs_wrong.fit(X_train_fs, y_train_fs)
    fs_wrong_score = model_fs_wrong.score(X_test_fs, y_test_fs)
    
    print(f"   Wrong approach score: {fs_wrong_score:.3f}")
    
    # Scenario 4: Feature selection inside pipeline (CORRECT)
    print("\n4. FEATURE SELECTION INSIDE PIPELINE (CORRECT)")
    pipeline_fs = Pipeline([
        ("selector", SelectKBest(k=5, score_func=f_classif)),
        ("model", LogisticRegression(random_state=42))
    ])
    
    cv_scores_fs = cross_val_score(pipeline_fs, X_train, y_train, cv=5)
    pipeline_fs.fit(X_train, y_train)
    fs_correct_score = pipeline_fs.score(X_test, y_test)
    
    print(f"   Correct approach CV score: {cv_scores_fs.mean():.3f} ± {cv_scores_fs.std():.3f}")
    print(f"   Correct approach test score: {fs_correct_score:.3f}")
    
    scenarios['feature_selection'] = {
        'wrong_score': fs_wrong_score,
        'correct_cv': cv_scores_fs.mean(),
        'correct_test': fs_correct_score,
        'leakage_gap': fs_wrong_score - fs_correct_score
    }
    
    # Scenario 5: Target encoding before split (if available)
    if TARGET_ENCODER_AVAILABLE:
        print("\n5. TARGET ENCODING BEFORE SPLIT (WRONG)")
        
        # Add categorical column for demonstration
        X_cat = X.copy()
        X_cat['category'] = np.random.choice(['A', 'B', 'C'], size=len(X))
        
        # Wrong approach - encode before split
        encoder_wrong = TargetEncoder()
        X_cat['category_encoded'] = encoder_wrong.fit_transform(
            X_cat['category'], y  # Uses ALL y values
        )
        
        X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
            X_cat.drop('category', axis=1), y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_enc_wrong = LogisticRegression(random_state=42)
        model_enc_wrong.fit(X_train_enc, y_train_enc)
        enc_wrong_score = model_enc_wrong.score(X_test_enc, y_test_enc)
        
        print(f"   Wrong approach score: {enc_wrong_score:.3f}")
        
        # Correct approach - encode inside pipeline
        pipeline_enc = Pipeline([
            ("encoder", TargetEncoder()),
            ("model", LogisticRegression(random_state=42))
        ])
        
        cv_scores_enc = cross_val_score(pipeline_enc, X_train, y_train, cv=5)
        pipeline_enc.fit(X_train, y_train)
        enc_correct_score = pipeline_enc.score(X_test, y_test)
        
        print(f"   Correct approach CV score: {cv_scores_enc.mean():.3f} ± {cv_scores_enc.std():.3f}")
        print(f"   Correct approach test score: {enc_correct_score:.3f}")
        
        scenarios['target_encoding'] = {
            'wrong_score': enc_wrong_score,
            'correct_cv': cv_scores_enc.mean(),
            'correct_test': enc_correct_score,
            'leakage_gap': enc_wrong_score - enc_correct_score
        }
    
    return scenarios


def demonstrate_cv_isolation():
    """
    Demonstrate how pipelines ensure CV isolation.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING CROSS-VALIDATION ISOLATION")
    print("="*80)
    
    # Create data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        weights=[0.8, 0.2], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Manual preprocessing (WRONG for CV)
    print("\n1. MANUAL PREPROCESSING BEFORE CV (WRONG)")
    scaler_manual = StandardScaler()
    X_train_manual = scaler_manual.fit_transform(X_train)  # Fitted on ALL training data
    
    cv_scores_manual = cross_val_score(
        LogisticRegression(random_state=42), 
        X_train_manual, y_train, cv=5
    )
    
    print(f"   Manual CV scores: {[f'{s:.3f}' for s in cv_scores_manual]}")
    print(f"   Manual CV mean: {cv_scores_manual.mean():.3f} ± {cv_scores_manual.std():.3f}")
    
    # Pipeline preprocessing (CORRECT for CV)
    print("\n2. PIPELINE PREPROCESSING FOR CV (CORRECT)")
    pipeline_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    
    cv_scores_pipeline = cross_val_score(pipeline_cv, X_train, y_train, cv=5)
    
    print(f"   Pipeline CV scores: {[f'{s:.3f}' for s in cv_scores_pipeline]}")
    print(f"   Pipeline CV mean: {cv_scores_pipeline.mean():.3f} ± {cv_scores_pipeline.std():.3f}")
    
    # Show the difference
    print(f"\n   Gap between approaches: {cv_scores_manual.mean() - cv_scores_pipeline.mean():.3f}")
    print("   (Manual approach shows inflated performance due to leakage)")
    
    return {
        'manual_cv': cv_scores_manual.tolist(),
        'pipeline_cv': cv_scores_pipeline.tolist(),
        'leakage_gap': cv_scores_manual.mean() - cv_scores_pipeline.mean()
    }


def demonstrate_gridsearch_safety():
    """
    Demonstrate how pipelines ensure safe GridSearchCV.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING GRIDSEARCHCV SAFETY")
    print("="*80)
    
    # Create data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        weights=[0.7, 0.3], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Wrong GridSearchCV (preprocessing before search)
    print("\n1. GRIDSEARCH WITH MANUAL PREPROCESSING (WRONG)")
    scaler_wrong = StandardScaler()
    X_train_scaled = scaler_wrong.fit_transform(X_train)  # Fitted on ALL training data
    
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    grid_wrong = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=5, scoring="f1"
    )
    grid_wrong.fit(X_train_scaled, y_train)
    
    print(f"   Wrong GridSearchCV best F1: {grid_wrong.best_score_:.3f}")
    print(f"   Wrong GridSearchCV best params: {grid_wrong.best_params_}")
    
    # Correct GridSearchCV (pipeline in search)
    print("\n2. GRIDSEARCH WITH PIPELINE (CORRECT)")
    pipeline_grid = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    
    param_grid_pipeline = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    grid_correct = GridSearchCV(
        pipeline_grid, param_grid_pipeline, cv=5, scoring="f1"
    )
    grid_correct.fit(X_train, y_train)
    
    print(f"   Correct GridSearchCV best F1: {grid_correct.best_score_:.3f}")
    print(f"   Correct GridSearchCV best params: {grid_correct.best_params_}")
    
    # Evaluate both on test set
    wrong_test_score = grid_wrong.score(X_test, y_test)
    correct_test_score = grid_correct.score(X_test, y_test)
    
    print(f"\n   Wrong approach test score: {wrong_test_score:.3f}")
    print(f"   Correct approach test score: {correct_test_score:.3f}")
    print(f"   Performance gap: {wrong_test_score - correct_test_score:.3f}")
    
    return {
        'wrong_best_score': grid_wrong.best_score_,
        'correct_best_score': grid_correct.best_score_,
        'wrong_test_score': wrong_test_score,
        'correct_test_score': correct_test_score,
        'performance_gap': wrong_test_score - correct_test_score
    }


def create_deployment_pipeline(numerical_columns=None, categorical_columns=None,
                         model_type="logistic", **model_params):
    """
    Create a deployment-ready pipeline with proper serialization.
    
    Args:
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        model_type: Type of model to use
        **model_params: Model parameters
        
    Returns:
        Deployment-ready pipeline
    """
    pipeline = create_safe_pipeline(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        model_type=model_type,
        **model_params
    )
    
    # Add deployment metadata
    pipeline.deployment_info = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'model_type': model_type,
        'model_params': model_params,
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    return pipeline


def save_deployment_pipeline(pipeline, filepath):
    """
    Save deployment pipeline with metadata.
    
    Args:
        pipeline: Fitted pipeline to save
        filepath: Path to save the pipeline
    """
    deployment_info = {
        'pipeline': pipeline,
        'metadata': getattr(pipeline, 'deployment_info', {}),
        'feature_names_in': getattr(pipeline, 'feature_names_in_', None),
        'n_features_in': getattr(pipeline, 'n_features_in_', None)
    }
    
    joblib.dump(deployment_info, filepath)
    logger.info(f"Deployment pipeline saved to {filepath}")


def load_deployment_pipeline(filepath):
    """
    Load deployment pipeline with metadata.
    
    Args:
        filepath: Path to the saved pipeline
        
    Returns:
        Pipeline with metadata
    """
    deployment_info = joblib.load(filepath)
    pipeline = deployment_info['pipeline']
    
    if 'metadata' in deployment_info:
        pipeline.deployment_info = deployment_info['metadata']
    
    logger.info(f"Deployment pipeline loaded from {filepath}")
    return pipeline


def evaluate_pipeline_safety(pipeline, X_train, X_test, y_train, y_test):
    """
    Evaluate pipeline safety and detect potential leakage.
    
    Args:
        pipeline: Pipeline to evaluate
        X_train, X_test, y_train, y_test: Split data
        
    Returns:
        Safety evaluation report
    """
    detector = LeakageDetector()
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    # Test score
    test_score = pipeline.score(X_test, y_test)
    
    # Check for leakage
    detector.check_cv_leakage(cv_scores, test_score)
    
    # Calculate metrics
    metrics = {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Add leakage detection
    leakage_report = detector.get_leakage_report()
    metrics['leakage_analysis'] = leakage_report
    
    return metrics


def print_safety_report(safety_metrics):
    """
    Print comprehensive safety evaluation report.
    """
    print("\n" + "="*80)
    print("PIPELINE SAFETY EVALUATION REPORT")
    print("="*80)
    
    print(f"\nPerformance Metrics:")
    print(f"  CV Score: {safety_metrics['cv_mean']:.3f} ± {safety_metrics['cv_std']:.3f}")
    print(f"  Test Score: {safety_metrics['test_score']:.3f}")
    print(f"  Accuracy: {safety_metrics['accuracy']:.3f}")
    print(f"  Precision: {safety_metrics['precision']:.3f}")
    print(f"  Recall: {safety_metrics['recall']:.3f}")
    print(f"  F1-Score: {safety_metrics['f1']:.3f}")
    
    leakage_analysis = safety_metrics['leakage_analysis']
    print(f"\nLeakage Analysis:")
    print(f"  Leakage Score: {leakage_analysis['leakage_score']:.2f}")
    print(f"  Severity: {leakage_analysis['severity']}")
    
    if leakage_analysis['warnings']:
        print(f"\nWarnings Detected:")
        for warning in leakage_analysis['warnings']:
            print(f"  {warning}")
    
    if leakage_analysis['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(leakage_analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
