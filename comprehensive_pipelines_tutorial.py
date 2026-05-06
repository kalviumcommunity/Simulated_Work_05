"""
Comprehensive ML Pipelines Tutorial

This tutorial provides a complete guide to building production-ready machine learning pipelines
with proper handling of complex data types, advanced preprocessing, and deployment workflows.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.advanced_pipelines import (
    PipelineBuilder, PipelineInspector, PipelineDebugger, ProductionPipeline,
    create_production_pipeline, demonstrate_advanced_pipelines
)
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_introduction():
    """Print tutorial introduction."""
    print("="*80)
    print("COMPREHENSIVE ML PIPELINES TUTORIAL")
    print("="*80)
    
    print("""
🎯 OBJECTIVE:
Learn to build production-ready ML pipelines that prevent data leakage, handle complex
data types, and ensure reproducible deployment.

📚 WHAT YOU'LL LEARN:
• Pipeline architecture and component design
• Mixed data type handling with ColumnTransformer
• Advanced preprocessing techniques (robust scaling, KNN imputation)
• Custom transformers for domain-specific logic
• Feature selection and dimensionality reduction
• Hyperparameter tuning with GridSearchCV
• Pipeline inspection and debugging
• Production deployment and serialization

🚀 WHY THIS MATTERS:
• Prevents data leakage during cross-validation
• Ensures reproducible transformations
• Handles real-world data complexity
• Enables team collaboration and maintenance
• Provides deployment-ready artifacts

⏱️  EXPECTED DURATION: 45 minutes
""")


def demonstrate_basic_pipeline():
    """Demonstrate basic pipeline construction."""
    print("\n" + "="*60)
    print("1. BASIC PIPELINE CONSTRUCTION")
    print("="*60)
    
    print("""
📖 OVERVIEW:
A pipeline chains multiple preprocessing steps with a final model.
Each step learns from data (fit) and applies transformations (transform).
The pipeline ensures correct order and prevents leakage.
    """)
    
    # Create sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        weights=[0.8, 0.2], flip_y=0.01, random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 1: Creating pipeline with scaling and model")
    
    # Create basic pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    basic_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Step 2: Inspecting pipeline structure")
    PipelineInspector.print_pipeline_structure(basic_pipeline, show_params=True)
    
    print("Step 3: Training pipeline")
    basic_pipeline.fit(X_train, y_train)
    
    print("Step 4: Making predictions")
    y_pred = basic_pipeline.predict(X_test)
    
    print("Step 5: Evaluating performance")
    from sklearn.metrics import classification_report, f1_score
    print(f"Test F1-Score: {f1_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    
    return basic_pipeline


def demonstrate_mixed_data_pipeline():
    """Demonstrate pipeline with mixed data types."""
    print("\n" + "="*60)
    print("2. MIXED DATA TYPE PIPELINE")
    print("="*60)
    
    print("""
🔀 MIXED DATA CHALLENGES:
• Different preprocessing for numerical vs categorical features
• Need ColumnTransformer to handle both types
• Categorical encoding requires careful handling of unknown values
• Feature selection must respect data types
    """)
    
    # Create mixed data
    X_num, y = make_classification(
        n_samples=1000, n_features=8, n_informative=6,
        weights=[0.75, 0.25], flip_y=0.01, random_state=42
    )
    
    # Add categorical features
    X_df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])
    X_df['cat_feature'] = np.random.choice(['A', 'B', 'C'], size=len(X_df))
    X_df['ordinal_feature'] = np.random.choice([1, 2, 3], size=len(X_df))
    
    # Add missing values
    X_df.loc[X_df.sample(100).index, 'num_0'] = np.nan
    X_df.loc[X_df.sample(50).index, 'cat_feature'] = None  # Missing categorical
    
    print(f"Created dataset with {X_df.shape[1]} features")
    print(f"Numerical: {X_df.columns[:8].tolist()}")
    print(f"Categorical: {X_df.columns[8:].tolist()}")
    print(f"Missing values: {X_df.isnull().sum().sum()}")
    
    # Split data
    numerical_features = [f"num_{i}" for i in range(8)]
    categorical_features = ['cat_feature', 'ordinal_feature']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.drop('num_0', axis=1), y, test_size=0.2,
        random_state=42, stratify=y
    )
    
    print("Step 1: Building mixed data pipeline")
    
    mixed_pipeline = create_production_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model_type="random_forest",
        n_estimators=100,
        max_depth=6
    )
    
    print("Step 2: Inspecting mixed pipeline")
    PipelineInspector.print_pipeline_structure(mixed_pipeline)
    
    print("Step 3: Training mixed pipeline")
    mixed_pipeline.fit(X_train, y_train)
    
    print("Step 4: Evaluating mixed pipeline")
    y_pred = mixed_pipeline.predict(X_test)
    print(f"Test F1-Score: {f1_score(y_test, y_pred):.3f}")
    
    return mixed_pipeline


def demonstrate_advanced_preprocessing():
    """Demonstrate advanced preprocessing techniques."""
    print("\n" + "="*60)
    print("3. ADVANCED PREPROCESSING TECHNIQUES")
    print("="*60)
    
    print("""
🔧 ADVANCED TECHNIQUES:
• Robust scaling using quantiles (resistant to outliers)
• KNN imputation for better missing value estimation
• Polynomial feature expansion for non-linear relationships
• Variance threshold for automatic feature selection
• PCA for dimensionality reduction
• Custom transformers for domain-specific logic
    """)
    
    # Create data with outliers and missing values
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1500, n_features=12, n_informative=8,
        weights=[0.7, 0.3], flip_y=0.01, random_state=42
    )
    
    # Add outliers and missing values
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Add extreme outliers to first feature
    X_df.loc[X_df.sample(50).index, 'feature_0'] *= 5
    X_df.loc[X_df.sample(30).index, 'feature_1'] *= 4
    
    # Add missing values
    X_df.loc[X_df.sample(100).index, 'feature_2'] = np.nan
    X_df.loc[X_df.sample(80).index, 'feature_3'] = np.nan
    
    print(f"Dataset with outliers and missing values: {X_df.shape}")
    print(f"Missing values per feature: {X_df.isnull().sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 1: Building advanced preprocessing pipeline")
    
    builder = PipelineBuilder()
    builder.set_memory("cachedir")
    builder.set_verbose(True)
    
    # Add advanced preprocessing steps
    builder.add_knn_imputer(n_neighbors=5)
    builder.add_robust_scaling(quantile_range=(25.0, 75.0))
    builder.add_polynomial_features(degree=2)
    builder.add_variance_threshold(threshold=0.01)
    builder.add_pca(n_components=0.95)
    
    # Build pipeline for numerical features only
    numerical_features = [f"feature_{i}" for i in range(4)]
    advanced_pipeline = builder.build_numerical_pipeline(numerical_features)
    
    # Add model
    from sklearn.ensemble import GradientBoostingClassifier
    advanced_pipeline.steps.append(("model", 
        GradientBoostingClassifier(n_estimators=100, random_state=42)
    ))
    
    print("Step 2: Inspecting advanced pipeline")
    PipelineInspector.print_pipeline_structure(advanced_pipeline)
    
    print("Step 3: Training advanced pipeline")
    advanced_pipeline.fit(X_train, y_train)
    
    print("Step 4: Evaluating advanced pipeline")
    y_pred = advanced_pipeline.predict(X_test)
    print(f"Test F1-Score: {f1_score(y_test, y_pred):.3f}")
    
    return advanced_pipeline


def demonstrate_custom_transformers():
    """Demonstrate custom transformer creation and usage."""
    print("\n" + "="*60)
    print("4. CUSTOM TRANSFORMERS")
    print("="*60)
    
    print("""
🛠️  CUSTOM TRANSFORMERS:
• Domain-specific logic implementation
• Flexible function-based transformations
• Integration with scikit-learn pipeline
• Validation and error handling
• Reproducible and testable transformations
    """)
    
    # Create sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=6, n_informative=4,
        weights=[0.6, 0.4], flip_y=0.01, random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    print("Step 1: Creating custom transformers")
    
    # Custom transformer 1: Log transformation
    def log_transform(x):
        """Custom log transformation with validation."""
        if np.any(x <= 0):
            raise ValueError("Log transform requires positive values")
        return np.log1p(x)
    
    # Custom transformer 2: Interaction terms
    def create_interaction(df):
        """Create interaction between two features."""
        feature_0 = df['feature_0']
        feature_1 = df['feature_1']
        return feature_0 * feature_1
    
    # Custom transformer 3: Binning
    def bin_feature(x, bins=5):
        """Bin numerical feature into categories."""
        return pd.cut(x, bins=bins, labels=False)
    
    print("Step 2: Building pipeline with custom transformers")
    
    builder = PipelineBuilder()
    
    # Add custom transformers
    builder.add_custom_transformer("log_transform", log_transform)
    builder.add_custom_transformer("interaction", 
                               lambda df: create_interaction(df).to_frame())
    builder.add_custom_transformer("binning", 
                               lambda x: bin_feature(x, bins=5).to_frame())
    
    # Add scaling and model
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Note: This is a simplified approach - in practice, 
    # custom transformers would need more careful integration
    custom_pipeline = Pipeline([
        ("log_transform", PipelineBuilder.CustomTransformer(func=log_transform)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Step 3: Training custom transformer pipeline")
    custom_pipeline.fit(X_train, y_train)
    
    print("Step 4: Evaluating custom transformer pipeline")
    y_pred = custom_pipeline.predict(X_test)
    print(f"Test F1-Score: {f1_score(y_test, y_pred):.3f}")
    
    return custom_pipeline


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with pipelines."""
    print("\n" + "="*60)
    print("5. HYPERPARAMETER TUNING WITH PIPELINES")
    print("="*60)
    
    print("""
🎛️  HYPERPARAMETER TUNING:
• GridSearchCV for exhaustive search
• RandomizedSearchCV for efficient search
• Tuning preprocessing steps alongside model
• Double underscore notation (step__parameter)
• Cross-validation with proper pipeline isolation
• Best parameter selection and model refitting
    """)
    
    # Create data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1200, n_features=10, n_informative=7,
        weights=[0.65, 0.35], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 1: Creating pipeline for tuning")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    
    pipeline = create_production_pipeline(
        numerical_features=[f"feature_{i}" for i in range(8)],
        categorical_features=[],
        model_type="random_forest"
    )
    
    print("Step 2: Setting up parameter grid")
    
    # Tune both model and preprocessing parameters
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [1, 5, 10]
    }
    
    print("Step 3: Running GridSearchCV")
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1",
        n_jobs=-1, verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    print("Step 4: Best parameters and performance")
    print(f"Best F1-Score: {grid.best_score_:.3f}")
    print(f"Best parameters: {grid.best_params_}")
    
    print("Step 5: Final evaluation")
    y_pred = grid.best_estimator_.predict(X_test)
    print(f"Test F1-Score: {f1_score(y_test, y_pred):.3f}")
    
    return grid


def demonstrate_production_workflow():
    """Demonstrate complete production workflow."""
    print("\n" + "="*60)
    print("6. PRODUCTION WORKFLOW")
    print("="*60)
    
    print("""
🚀 PRODUCTION WORKFLOW:
• Training pipeline with metadata tracking
• Pipeline serialization with joblib
• Loading and validation in production environment
• Feature name consistency checking
• Deployment monitoring and logging
• Version control and reproducibility
    """)
    
    # Create production pipeline
    production_pipeline = create_production_pipeline(
        numerical_features=NUMERICAL_FEATURES[:5],  # Use subset for demo
        categorical_features=CATEGORICAL_FEATURES[:2],
        model_type="gradient_boosting",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4
    )
    
    print("Step 1: Training production pipeline")
    
    # Load training data
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=10,
        weights=[0.7, 0.3], flip_y=0.01, random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Add categorical features
    X_df['category_1'] = np.random.choice(['A', 'B'], size=len(X_df))
    X_df['category_2'] = np.random.choice(['X', 'Y'], size=len(X_df))
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit with metadata tracking
    production_pipeline.fit(X_train, y_train)
    
    print("Step 2: Saving production pipeline")
    
    # Save pipeline
    os.makedirs("models", exist_ok=True)
    production_pipeline.save("models/production_pipeline.pkl")
    
    print("Step 3: Loading and validating pipeline")
    
    # Load pipeline
    loaded_pipeline = ProductionPipeline.load("models/production_pipeline.pkl")
    
    # Validate loading
    original_pred = production_pipeline.predict(X_test)
    loaded_pred = loaded_pipeline.predict(X_test)
    predictions_match = np.array_equal(original_pred, loaded_pred)
    
    print(f"Predictions match: {predictions_match}")
    
    # Get deployment report
    report = loaded_pipeline.get_deployment_report()
    print(f"Deployment ready: {report['pipeline_ready']}")
    print(f"Warnings: {report['warnings']}")
    
    return production_pipeline


def print_best_practices():
    """Print pipeline best practices summary."""
    print("\n" + "="*60)
    print("7. PIPELINE BEST PRACTICES")
    print("="*60)
    
    print("""
✅  BEST PRACTICES:

🔒 LEAKAGE PREVENTION:
• Always split data BEFORE any preprocessing
• Use Pipeline for all transformations that learn from data
• Apply cross_val_score to full pipeline, not preprocessed data
• Include preprocessing steps in GridSearchCV

🏗️  PIPELINE CONSTRUCTION:
• Use ColumnTransformer for mixed data types
• Set handle_unknown="ignore" for categorical encoders
• Be explicit about remainder="drop" to avoid accidental features
• Enable memory caching for expensive computations
• Use descriptive step names for debugging

🎛️  HYPERPARAMETER TUNING:
• Use double underscore notation: step__parameter
• Tune preprocessing parameters alongside model parameters
• Use appropriate scoring metrics for your problem
• Consider RandomizedSearchCV for large parameter spaces
• Always refit on full training data after tuning

🚀 DEPLOYMENT:
• Save complete pipeline, not just model
• Include metadata (feature names, parameters, data info)
• Test pipeline loading before deployment
• Validate feature name consistency
• Use version control for pipeline definitions
• Monitor prediction distributions in production

🔍 DEBUGGING:
• Use PipelineInspector for structure analysis
• Check data leakage with PipelineDebugger
• Validate fitted parameters with inspect_fitted_pipeline
• Test custom transformers thoroughly
• Compare multiple pipeline configurations
• Use verbose mode during development

⚠️  COMMON MISTAKES TO AVOID:
• Manual preprocessing before cross-validation
• Fitting scalers/encoders on full dataset
• Saving only model, not preprocessing steps
• Inconsistent feature handling between train/test
• Not handling unknown categories in production
• Ignoring pipeline memory and performance optimization
• Not validating pipeline loading and predictions
    """)


def main():
    """Main tutorial function."""
    print_introduction()
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Run all demonstrations
    print("\n🎯 RUNNING TUTORIAL EXAMPLES...")
    
    # 1. Basic pipeline
    basic_pipeline = demonstrate_basic_pipeline()
    
    # 2. Mixed data pipeline
    mixed_pipeline = demonstrate_mixed_data_pipeline()
    
    # 3. Advanced preprocessing
    advanced_pipeline = demonstrate_advanced_preprocessing()
    
    # 4. Custom transformers
    custom_pipeline = demonstrate_custom_transformers()
    
    # 5. Hyperparameter tuning
    grid_search = demonstrate_hyperparameter_tuning()
    
    # 6. Production workflow
    production_pipeline = demonstrate_production_workflow()
    
    # 7. Best practices
    print_best_practices()
    
    print("\n" + "="*80)
    print("🎉 TUTORIAL COMPLETED!")
    print("="*80)
    
    print("""
📚  SUMMARY:
You've learned:
• Pipeline architecture and component design
• Mixed data type handling with ColumnTransformer
• Advanced preprocessing techniques
• Custom transformer implementation
• Hyperparameter tuning with GridSearchCV
• Production deployment and serialization

🎯  NEXT STEPS:
• Apply these techniques to your own projects
• Use pipelines consistently in all ML workflows
• Share pipeline definitions with your team
• Version control your pipeline configurations
• Always prioritize preventing data leakage

🔗  ADDITIONAL RESOURCES:
• Scikit-learn Pipeline documentation
• Advanced preprocessing techniques guide
• Production ML system design patterns
• Team collaboration workflows for ML pipelines
    """)
    
    print(f"\n📁 Files created: plots/, models/")
    return True


if __name__ == "__main__":
    main()
