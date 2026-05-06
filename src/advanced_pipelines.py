"""
Advanced ML Pipelines Module

This module provides comprehensive tools for building production-ready machine learning pipelines
with proper handling of mixed data types, advanced preprocessing, and deployment workflows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                              OneHotEncoder, OrdinalEncoder, LabelEncoder,
                              PolynomialFeatures, FunctionTransformer)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, SelectFromModel, RFE, 
                                  VarianceThreshold, SelectPercentile)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                          precision_score, recall_score, f1_score)
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                 GridSearchCV, RandomizedSearchCV)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from category_encoders import TargetEncoder, LeaveOneOutEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for domain-specific preprocessing.
    """
    
    def __init__(self, func=None, validate=True):
        self.func = func
        self.validate = validate
    
    def fit(self, X, y=None):
        if self.validate:
            # Validate input
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise ValueError("X must be pandas DataFrame or numpy array")
        return self
    
    def transform(self, X):
        if self.func is None:
            return X
        
        if isinstance(X, pd.DataFrame):
            return X.apply(self.func)
        else:
            # Apply function to each row
            return np.array([self.func(row) for row in X])
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class PipelineBuilder:
    """
    Advanced pipeline builder with comprehensive preprocessing options.
    """
    
    def __init__(self):
        self.steps = []
        self.column_transformer_steps = []
        self.memory = None
        self.verbose = False
    
    def set_memory(self, memory):
        """Enable memory caching for pipeline steps."""
        self.memory = memory
        return self
    
    def set_verbose(self, verbose):
        """Set verbosity for pipeline steps."""
        self.verbose = verbose
        return self
    
    def add_custom_transformer(self, name, func, validate=True):
        """Add custom transformation step."""
        transformer = CustomTransformer(func=func, validate=validate)
        self.steps.append((name, transformer))
        return self
    
    def add_polynomial_features(self, degree=2, include_bias=False):
        """Add polynomial feature expansion."""
        poly_features = PolynomialFeatures(
            degree=degree, include_bias=include_bias
        )
        self.steps.append(("poly_features", poly_features))
        return self
    
    def add_robust_scaling(self, quantile_range=(25.0, 75.0)):
        """Add robust scaling using quantiles."""
        robust_scaler = RobustScaler(quantile_range=quantile_range)
        self.steps.append(("robust_scaler", robust_scaler))
        return self
    
    def add_knn_imputer(self, n_neighbors=5, weights='uniform'):
        """Add KNN imputation for better missing value handling."""
        knn_imputer = KNNImputer(
            n_neighbors=n_neighbors, weights=weights
        )
        self.steps.append(("knn_imputer", knn_imputer))
        return self
    
    def add_variance_threshold(self, threshold=0.0):
        """Add variance threshold feature selection."""
        variance_selector = VarianceThreshold(threshold=threshold)
        self.steps.append(("variance_threshold", variance_selector))
        return self
    
    def add_model_based_selection(self, estimator, threshold=None):
        """Add model-based feature selection."""
        model_selector = SelectFromModel(
            estimator=estimator, threshold=threshold
        )
        self.steps.append(("model_selector", model_selector))
        return self
    
    def add_pca(self, n_components=None, svd_solver='auto'):
        """Add PCA dimensionality reduction."""
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        self.steps.append(("pca", pca))
        return self
    
    def add_truncated_svd(self, n_components=50):
        """Add TruncatedSVD for sparse data."""
        svd = TruncatedSVD(n_components=n_components)
        self.steps.append(("svd", svd))
        return self
    
    def add_target_encoding(self, columns=None, smoothing=1.0):
        """Add target encoding (if available)."""
        if CATEGORY_ENCODERS_AVAILABLE:
            target_encoder = TargetEncoder(
                cols=columns, smoothing=smoothing
            )
            self.steps.append(("target_encoder", target_encoder))
        else:
            logger.warning("Category encoders not available - skipping target encoding")
        return self
    
    def add_leave_one_out_encoding(self, columns=None):
        """Add Leave-One-Out encoding (if available)."""
        if CATEGORY_ENCODERS_AVAILABLE:
            loo_encoder = LeaveOneOutEncoder(cols=columns)
            self.steps.append(("loo_encoder", loo_encoder))
        else:
            logger.warning("Category encoders not available - skipping LOO encoding")
        return self
    
    def build_numerical_pipeline(self, numerical_features):
        """Build pipeline for numerical features only."""
        steps = []
        
        # Add imputation
        steps.append(("imputer", SimpleImputer(strategy="median")))
        
        # Add scaling
        steps.append(("scaler", StandardScaler()))
        
        # Add optional robust scaling
        steps.append(("robust_scaler", RobustScaler()))
        
        # Add polynomial features if needed
        steps.append(("poly_features", PolynomialFeatures(degree=2, include_bias=False)))
        
        # Add feature selection
        steps.append(("selector", SelectKBest(k=10, score_func=f_classif)))
        
        # Add dimensionality reduction
        steps.append(("pca", PCA(n_components=0.95)))
        
        return Pipeline(steps)
    
    def build_mixed_pipeline(self, numerical_features=None, categorical_features=None):
        """Build pipeline for mixed numerical and categorical features."""
        transformers = []
        
        # Numerical pipeline
        if numerical_features:
            num_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
                ("selector", SelectKBest(k=15, score_func=f_classif))
            ])
            transformers.append(("num", num_transformer, numerical_features))
        
        # Categorical pipeline
        if categorical_features:
            if CATEGORY_ENCODERS_AVAILABLE:
                cat_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_encoder", TargetEncoder()),
                    ("loo_encoder", LeaveOneOutEncoder())
                ])
            else:
                cat_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ])
            transformers.append(("cat", cat_transformer, categorical_features))
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,
            n_jobs=-1,
            verbose=self.verbose
        )
        
        # Add final model step
        return Pipeline([
            ("preprocessor", preprocessor)
        ], memory=self.memory, verbose=self.verbose)
    
    def build(self):
        """Build the final pipeline from accumulated steps."""
        if not self.steps:
            raise ValueError("No steps added to pipeline")
        
        return Pipeline(self.steps, memory=self.memory, verbose=self.verbose)


class PipelineInspector:
    """
    Tools for inspecting and debugging pipelines.
    """
    
    @staticmethod
    def get_pipeline_info(pipeline):
        """Get comprehensive information about a pipeline."""
        info = {
            'n_steps': len(pipeline.steps),
            'step_names': [name for name, _ in pipeline.steps],
            'step_types': [type(step).__name__ for _, step in pipeline.steps],
            'memory': pipeline.memory,
            'verbose': pipeline.verbose
        }
        
        # Check for ColumnTransformer
        if hasattr(pipeline, 'named_steps'):
            preprocessor_step = pipeline.named_steps.get('preprocessor')
            if preprocessor_step and hasattr(preprocessor_step, 'transformers_'):
                info['column_transformer'] = True
                info['transformers'] = [
                    (name, type(transformer).__name__) 
                    for name, transformer in preprocessor_step.transformers_
                ]
        
        return info
    
    @staticmethod
    def print_pipeline_structure(pipeline, show_params=False):
        """Print detailed pipeline structure."""
        info = PipelineInspector.get_pipeline_info(pipeline)
        
        print("="*60)
        print("PIPELINE STRUCTURE")
        print("="*60)
        print(f"Total steps: {info['n_steps']}")
        print(f"Memory enabled: {info['memory'] is not None}")
        print(f"Verbose: {info['verbose']}")
        
        for i, (name, step) in enumerate(pipeline.steps, 1):
            print(f"\nStep {i}: {name}")
            print(f"  Type: {type(step).__name__}")
            
            if show_params and hasattr(step, 'get_params'):
                params = step.get_params()
                print(f"  Parameters: {params}")
            
            # Special handling for ColumnTransformer
            if name == 'preprocessor' and hasattr(step, 'transformers_'):
                print(f"  ColumnTransformer with {len(step.transformers_)} transformers:")
                for j, (transformer_name, transformer) in enumerate(step.transformers_, 1):
                    print(f"    {j}. {transformer_name}: {type(transformer).__name__}")
        
        print("="*60)
    
    @staticmethod
    def inspect_fitted_pipeline(pipeline):
        """Inspect a fitted pipeline and show learned parameters."""
        print("\n" + "="*60)
        print("FITTED PIPELINE INSPECTION")
        print("="*60)
        
        for name, step in pipeline.steps:
            print(f"\n{name}:")
            
            if hasattr(step, 'named_steps'):  # Nested pipeline
                for sub_name, sub_step in step.named_steps.items():
                    if hasattr(sub_step, 'mean_'):
                        print(f"  Mean: {sub_step.mean_}")
                    if hasattr(sub_step, 'scale_'):
                        print(f"  Scale: {sub_step.scale_}")
                    if hasattr(sub_step, 'variance_'):
                        print(f"  Variance: {sub_step.variance_}")
                    if hasattr(sub_step, 'components_'):
                        print(f"  Components shape: {sub_step.components_.shape}")
                    if hasattr(sub_step, 'feature_names_in_'):
                        print(f"  Input features: {len(sub_step.feature_names_in_)}")
            
            elif hasattr(step, 'mean_'):
                print(f"  Mean: {step.mean_}")
            elif hasattr(step, 'scale_'):
                print(f"  Scale: {step.scale_}")
            elif hasattr(step, 'variance_'):
                print(f"  Variance: {step.variance_}")
            elif hasattr(step, 'components_'):
                print(f"  Components shape: {step.components_.shape}")
            elif hasattr(step, 'feature_names_in_'):
                print(f"  Input features: {len(step.feature_names_in_)}")
            elif hasattr(step, 'n_features_in_'):
                print(f"  Input features: {step.n_features_in_}")
        
        print("="*60)
    
    @staticmethod
    def compare_pipelines(pipelines, X, y, cv_folds=5):
        """Compare multiple pipelines using cross-validation."""
        results = {}
        
        for name, pipeline in pipelines.items():
            cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="f1")
            results[name] = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'pipeline_info': PipelineInspector.get_pipeline_info(pipeline)
            }
        
        # Print comparison
        print("\n" + "="*60)
        print("PIPELINE COMPARISON")
        print("="*60)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  CV F1: {result['mean_score']:.3f} ± {result['std_score']:.3f}")
            print(f"  Steps: {result['pipeline_info']['n_steps']}")
        
        # Find best
        best_pipeline = max(results.keys(), key=lambda k: results[k]['mean_score'])
        print(f"\nBest pipeline: {best_pipeline}")
        print(f"Best CV F1: {results[best_pipeline]['mean_score']:.3f}")
        
        return results


class PipelineDebugger:
    """
    Tools for debugging pipeline issues.
    """
    
    @staticmethod
    def check_data_leakage(pipeline, X_train, X_test):
        """Check for potential data leakage in pipeline."""
        warnings_list = []
        
        # Check if pipeline has been fitted on test data
        if hasattr(pipeline, 'named_steps'):
            for name, step in pipeline.named_steps.items():
                if hasattr(step, 'mean_'):
                    # For scalers
                    if hasattr(step, 'partial_fit_'):
                        # This would indicate leakage
                        warnings_list.append(f"{name}: May have been fitted on test data")
        
        return warnings_list
    
    @staticmethod
    def check_feature_consistency(pipeline, X_train, X_test):
        """Check if features are consistent between train and test."""
        warnings_list = []
        
        # Check column consistency
        if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
            train_cols = set(X_train.columns)
            test_cols = set(X_test.columns)
            
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            
            if missing_in_test:
                warnings_list.append(f"Features in train but not test: {missing_in_test}")
            if missing_in_train:
                warnings_list.append(f"Features in test but not train: {missing_in_train}")
        
        return warnings_list
    
    @staticmethod
    def simulate_step_failure(pipeline, step_name, failure_type="transform_error"):
        """Simulate a step failure for testing error handling."""
        # This is useful for testing pipeline robustness
        original_step = None
        
        for name, step in pipeline.steps:
            if name == step_name:
                original_step = step
                # Create a faulty step
                if failure_type == "transform_error":
                    class FaultyStep:
                        def fit(self, X, y=None):
                            return self  # Skip fitting
                        def transform(self, X):
                            raise ValueError("Simulated transformation error")
                    pipeline.steps[pipeline.steps.index((name, step))] = (name, FaultyStep())
                break
        
        return original_step


class ProductionPipeline:
    """
    Production-ready pipeline with deployment utilities.
    """
    
    def __init__(self, pipeline, metadata=None):
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.training_info = {}
        self.deployment_info = {}
    
    def fit(self, X, y):
        """Fit pipeline and store training information."""
        start_time = pd.Timestamp.now()
        
        self.pipeline.fit(X, y)
        
        # Store training information
        self.training_info = {
            'fit_time': start_time,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns) if isinstance(X, pd.DataFrame) else X.shape[1],
            'feature_names': list(X.columns) if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])],
            'target_distribution': dict(pd.Series(y).value_counts()),
            'pipeline_info': PipelineInspector.get_pipeline_info(self.pipeline)
        }
        
        return self
    
    def predict(self, X):
        """Make predictions with deployment tracking."""
        start_time = pd.Timestamp.now()
        
        predictions = self.pipeline.predict(X)
        
        # Store deployment information
        self.deployment_info = {
            'prediction_time': start_time,
            'n_predictions': len(predictions),
            'input_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns) if isinstance(X, pd.DataFrame) else X.shape[1],
            'prediction_distribution': dict(pd.Series(predictions).value_counts()) if len(predictions) > 0 else {}
        }
        
        return predictions
    
    def get_feature_names_out(self):
        """Get output feature names from pipeline."""
        if hasattr(self.pipeline, 'named_steps'):
            preprocessor = self.pipeline.named_steps.get('preprocessor')
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                return preprocessor.get_feature_names_out()
        return []
    
    def save(self, filepath):
        """Save pipeline with metadata."""
        artifact = {
            'pipeline': self.pipeline,
            'metadata': self.metadata,
            'training_info': self.training_info,
            'deployment_info': self.deployment_info,
            'feature_names_out': self.get_feature_names_out(),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(artifact, filepath)
        logger.info(f"Production pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load production pipeline."""
        artifact = joblib.load(filepath)
        
        pipeline = ProductionPipeline(
            artifact['pipeline'], 
            artifact.get('metadata', {})
        )
        pipeline.training_info = artifact.get('training_info', {})
        pipeline.deployment_info = artifact.get('deployment_info', {})
        
        logger.info(f"Production pipeline loaded from {filepath}")
        return pipeline
    
    def get_deployment_report(self):
        """Get deployment readiness report."""
        report = {
            'pipeline_ready': bool(self.pipeline),
            'has_metadata': bool(self.metadata),
            'has_training_info': bool(self.training_info),
            'has_deployment_info': bool(self.deployment_info),
            'feature_names_available': len(self.get_feature_names_out()) > 0,
            'warnings': []
        }
        
        # Check for issues
        if not self.metadata:
            report['warnings'].append("No metadata provided")
        
        if not self.training_info:
            report['warnings'].append("No training information stored")
        
        return report


def create_production_pipeline(numerical_features=None, categorical_features=None,
                        model_type="random_forest", **model_params):
    """
    Create a production-ready pipeline with comprehensive preprocessing.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        model_type: Type of model to use
        **model_params: Model parameters
        
    Returns:
        ProductionPipeline instance
    """
    builder = PipelineBuilder()
    builder.set_memory("cachedir")
    builder.set_verbose(True)
    
    # Add preprocessing steps based on data types
    if numerical_features and categorical_features:
        # Mixed data - use ColumnTransformer approach
        pipeline = builder.build_mixed_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
    elif numerical_features:
        # Numerical only
        pipeline = builder.build_numerical_pipeline(numerical_features)
    else:
        raise ValueError("Must specify at least numerical_features or categorical_features")
    
    # Add final model
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, **model_params)
    elif model_type == "svm":
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=42, **model_params)
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add model to pipeline
    pipeline.steps.append(("model", model))
    
    # Create production pipeline
    return ProductionPipeline(pipeline, metadata={
        'model_type': model_type,
        'model_params': model_params,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'created_for': 'production'
    })


def demonstrate_advanced_pipelines():
    """
    Demonstrate advanced pipeline capabilities.
    """
    print("="*80)
    print("ADVANCED PIPELINES DEMONSTRATION")
    print("="*80)
    
    # Create synthetic data with mixed types
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=10,
        n_redundant=3, n_repeated=2, weights=[0.7, 0.3],
        flip_y=0.01, random_state=42
    )
    
    # Create DataFrame with mixed types
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Add categorical features
    X_df['category_A'] = np.random.choice(['X', 'Y', 'Z'], size=len(X_df))
    X_df['category_B'] = np.random.choice(['P', 'Q', 'R'], size=len(X_df))
    X_df['target'] = y
    
    # Add some missing values
    X_df.loc[X_df.sample(50).index, 'feature_0'] = np.nan
    X_df.loc[X_df.sample(30).index, 'feature_1'] = np.nan
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.drop('target', axis=1), y, test_size=0.2, 
        random_state=42, stratify=y
    )
    
    # Define feature groups
    numerical_features = [f"feature_{i}" for i in range(10)]
    categorical_features = ['category_A', 'category_B']
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Missing values: {X_df.isnull().sum().sum()}")
    
    # Demonstrate different pipeline configurations
    pipelines = {}
    
    # 1. Simple pipeline
    print("\n1. SIMPLE PIPELINE")
    simple_pipeline = create_production_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model_type="logistic",
        max_iter=1000
    )
    
    simple_pipeline.fit(X_train, y_train)
    simple_pred = simple_pipeline.predict(X_test)
    simple_score = f1_score(y_test, simple_pred)
    
    print(f"Simple pipeline F1: {simple_score:.3f}")
    
    pipelines['simple'] = simple_pipeline
    
    # 2. Advanced pipeline with feature selection
    print("\n2. ADVANCED PIPELINE WITH FEATURE SELECTION")
    builder = PipelineBuilder()
    builder.set_memory("cachedir")
    
    advanced_pipeline = builder.build_mixed_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    
    # Add model with hyperparameters
    from sklearn.ensemble import RandomForestClassifier
    advanced_pipeline.steps.append(("model", 
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    ))
    
    advanced_pipeline.fit(X_train, y_train)
    advanced_pred = advanced_pipeline.predict(X_test)
    advanced_score = f1_score(y_test, advanced_pred)
    
    print(f"Advanced pipeline F1: {advanced_score:.3f}")
    
    pipelines['advanced'] = advanced_pipeline
    
    # 3. Pipeline with custom transformer
    print("\n3. PIPELINE WITH CUSTOM TRANSFORMER")
    
    # Add custom log transformer
    def log_transform(x):
        return np.log1p(x)
    
    custom_pipeline = create_production_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model_type="logistic",
        max_iter=1000
    )
    
    # Insert custom transformer
    custom_pipeline.steps.insert(
        -1,  # Insert before model
        ("log_transform", CustomTransformer(func=log_transform))
    )
    
    custom_pipeline.fit(X_train, y_train)
    custom_pred = custom_pipeline.predict(X_test)
    custom_score = f1_score(y_test, custom_pred)
    
    print(f"Custom transformer pipeline F1: {custom_score:.3f}")
    
    pipelines['custom'] = custom_pipeline
    
    # Compare pipelines
    print("\n4. PIPELINE COMPARISON")
    comparison = PipelineInspector.compare_pipelines(pipelines, X_train, y_train)
    
    # Inspect best pipeline
    print(f"\n5. BEST PIPELINE INSPECTION")
    PipelineInspector.inspect_fitted_pipeline(pipelines[comparison['best_pipeline']])
    
    # Save best pipeline
    print("\n6. SAVING BEST PIPELINE")
    best_pipeline = pipelines[comparison['best_pipeline']]
    best_pipeline.save("models/best_production_pipeline.pkl")
    
    # Test loading
    print("\n7. TESTING PIPELINE LOADING")
    loaded_pipeline = ProductionPipeline.load("models/best_production_pipeline.pkl")
    test_pred = loaded_pipeline.predict(X_test)
    test_score = f1_score(y_test, test_pred)
    
    print(f"Loaded pipeline F1: {test_score:.3f}")
    print(f"Predictions match: {np.array_equal(advanced_pred, test_pred)}")
    
    return {
        'simple_score': simple_score,
        'advanced_score': advanced_score,
        'custom_score': custom_score,
        'best_pipeline': comparison['best_pipeline'],
        'comparison': comparison
    }


def visualize_pipeline_comparison(pipelines, X_test, y_test):
    """
    Create visualizations for pipeline comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance comparison
    pipeline_names = list(pipelines.keys())
    scores = [f1_score(y_test, pipelines[name].predict(X_test)) for name in pipeline_names]
    
    bars = axes[0, 0].bar(pipeline_names, scores, color=['skyblue', 'lightgreen', 'orange'])
    axes[0, 0].set_title('Pipeline F1-Score Comparison')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature importance comparison
    if 'advanced' in pipelines:
        advanced_pipeline = pipelines['advanced']
        if hasattr(advanced_pipeline.pipeline.named_steps['model'], 'feature_importances_'):
            importances = advanced_pipeline.pipeline.named_steps['model'].feature_importances_
            
            # Get feature names
            feature_names = advanced_pipeline.get_feature_names_out()
            if feature_names:
                # Take only top 10 features
                top_indices = np.argsort(importances)[-10:]
                top_importances = importances[top_indices]
                top_features = [feature_names[i] for i in top_indices]
                
                axes[0, 1].barh(range(len(top_features)), top_importances, color='lightcoral')
                axes[0, 1].set_yticks(range(len(top_features)))
                axes[0, 1].set_yticklabels(top_features)
                axes[0, 1].set_xlabel('Feature Importance')
                axes[0, 1].set_title('Top 10 Feature Importances')
                axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction distribution comparison
    axes[1, 0].set_title('Prediction Distribution Comparison')
    
    colors = ['skyblue', 'lightgreen', 'orange']
    for i, (name, pipeline) in enumerate(pipelines.items()):
        predictions = pipeline.predict(X_test)
        unique, counts = np.unique(predictions, return_counts=True)
        
        axes[1, 0].pie(counts, labels=[f'{name}\n{u}' for u in unique], 
                    colors=[colors[i % len(colors)]], autopct='%1.1f%%')
    
    plt.tight_layout()
    plt.savefig("plots/advanced_pipelines_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Demonstrate advanced pipelines
    results = demonstrate_advanced_pipelines()
    
    # Create visualization
    pipelines_to_compare = {k: v for k, v in results.items() if hasattr(v, 'predict')}
    if pipelines_to_compare:
        # We need to recreate the test data for visualization
        from sklearn.datasets import make_classification
        X_vis, y_vis = make_classification(
            n_samples=400, n_features=15, weights=[0.7, 0.3],
            flip_y=0.01, random_state=42
        )
        visualize_pipeline_comparison(pipelines_to_compare, X_vis, y_vis)
    
    print("\n✅ Advanced pipelines demonstration completed!")
