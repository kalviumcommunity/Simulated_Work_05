"""
Feature engineering module for spam email detection.
Builds reusable transformations and pipelines for feature processing.
Implements proper StandardScaler on numerical features only.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, Any, List, Optional
import joblib
import logging

from config import (
    PREPROCESSOR_FILE, FEATURE_NAMES_FILE, SCALER_FILE,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, SCALING_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature selection.
    """
    
    def __init__(self, k: int = 10):
        self.k = k
        self.selector = SelectKBest(f_classif, k=k)
        self.selected_features = None
    
    def fit(self, X, y=None):
        self.selector.fit(X, y)
        self.selected_features = self.selector.get_support(indices=True)
        logger.info(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_features]


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer for outlier removal using IQR method.
    """
    
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        self.lower_bounds = Q1 - self.factor * IQR
        self.upper_bounds = Q3 + self.factor * IQR
        
        logger.info(f"Outlier remover fitted with factor={self.factor}")
        return self
    
    def transform(self, X):
        # For prediction, we don't remove outliers, just clip values
        X_clipped = np.clip(X, self.lower_bounds, self.upper_bounds)
        return X_clipped


def create_feature_pipeline(
    scale_features: bool = True,
    select_features: bool = True,
    remove_outliers: bool = True,
    n_features_to_select: int = 15,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None
) -> Pipeline:
    """
    Create a reusable feature engineering pipeline with proper scaling.
    
    CRITICAL: Only scales NUMERICAL_FEATURES, leaves CATEGORICAL_FEATURES unchanged.
    Uses ColumnTransformer for professional preprocessing.
    
    Args:
        scale_features (bool): Whether to scale numerical features
        select_features (bool): Whether to perform feature selection
        remove_outliers (bool): Whether to handle outliers
        n_features_to_select (int): Number of features to select if selection is enabled
        numerical_features (List[str]): List of numerical feature names
        categorical_features (List[str]): List of categorical feature names
        
    Returns:
        Pipeline: Configured feature engineering pipeline
    """
    logger.info("Creating feature engineering pipeline with proper scaling...")
    
    # Use configured feature lists if not provided
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    
    logger.info(f"Numerical features to scale: {len(numerical_features)}")
    logger.info(f"Categorical features (no scaling): {len(categorical_features)}")
    
    steps = []
    
    # Use ColumnTransformer for proper feature-specific preprocessing
    if scale_features:
        # Create preprocessing pipeline for numerical features only
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # Scale only numerical features
        ])
        
        # Categorical features: pass through without scaling
        categorical_transformer = 'passthrough'  # No scaling for categorical
        
        # Combine with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any features not specified
        )
        
        steps.append(('preprocessor', preprocessor))
        logger.info("ColumnTransformer configured: StandardScaler on numerical features only")
    
    if select_features:
        steps.append(('feature_selector', SelectKBest(f_classif, k=n_features_to_select)))
    
    pipeline = Pipeline(steps)
    
    logger.info(f"Pipeline created with {len(steps)} steps: {[step[0] for step in steps]}")
    logger.info("✅ Scaling applied to numerical features only (categorical features unchanged)")
    
    return pipeline


def fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_features: bool = True,
    select_features: bool = True,
    remove_outliers: bool = True,
    n_features_to_select: int = 15
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fit the preprocessor on training data and transform it.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        scale_features (bool): Whether to scale features
        select_features (bool): Whether to perform feature selection
        remove_outliers (bool): Whether to handle outliers
        n_features_to_select (int): Number of features to select
        
    Returns:
        Tuple[Pipeline, np.ndarray]: Fitted preprocessor and transformed features
    """
    logger.info("Fitting preprocessor on training data...")
    
    # Create and fit pipeline
    preprocessor = create_feature_pipeline(
        scale_features=scale_features,
        select_features=select_features,
        remove_outliers=remove_outliers,
        n_features_to_select=n_features_to_select
    )
    
    # Fit and transform training data
    preprocessor.fit(X_train, y_train)
    X_train_transformed = preprocessor.transform(X_train)
    
    logger.info(f"Preprocessor fitted. Output shape: {X_train_transformed.shape}")
    
    return preprocessor, X_train_transformed


def transform_features(preprocessor: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Transform features using fitted preprocessor.
    
    Args:
        preprocessor (Pipeline): Fitted preprocessor
        X (np.ndarray): Features to transform
        
    Returns:
        np.ndarray: Transformed features
    """
    logger.info(f"Transforming features with shape {X.shape}")
    
    X_transformed = preprocessor.transform(X)
    
    logger.info(f"Features transformed. Output shape: {X_transformed.shape}")
    
    return X_transformed
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline():
    """Create preprocessing pipeline for text data"""
    
    logger.info(f"Saving feature names to {filepath}")
    
    if selected_indices is not None:
        saved_names = [feature_names[i] for i in selected_indices if i < len(feature_names)]
    else:
        saved_names = feature_names
    
    joblib.dump(saved_names, filepath)
    
    logger.info(f"Feature names saved: {len(saved_names)} names")


def load_feature_names(filepath: str = None) -> List[str]:
    """
    Load feature names from disk.
    
    Args:
        filepath (str): Path to the saved feature names
        
    Returns:
        List[str]: Loaded feature names
    """
    if filepath is None:
        filepath = FEATURE_NAMES_FILE
    
    logger.info(f"Loading feature names from {filepath}")
    
    feature_names = joblib.load(filepath)
    
    logger.info(f"Feature names loaded: {len(feature_names)} names")
    
    return feature_names


def save_scaler(preprocessor: Pipeline, filepath: str = None) -> None:
    """
    Save the fitted StandardScaler from the preprocessor pipeline.
    
    CRITICAL: Scaler must be saved after fitting on training data only.
    It should NOT be refitted during inference.
    
    Args:
        preprocessor (Pipeline): Fitted preprocessor containing the scaler
        filepath (str): Path to save the scaler
    """
    if filepath is None:
        filepath = SCALER_FILE
    
    logger.info(f"Saving StandardScaler to {filepath}")
    
    # Extract the scaler from the ColumnTransformer in the pipeline
    if 'preprocessor' in preprocessor.named_steps:
        column_transformer = preprocessor.named_steps['preprocessor']
        # Get the numerical transformer which contains the scaler
        if 'num' in column_transformer.named_transformers_:
            numerical_pipeline = column_transformer.named_transformers_['num']
            if hasattr(numerical_pipeline, 'named_steps') and 'scaler' in numerical_pipeline.named_steps:
                scaler = numerical_pipeline.named_steps['scaler']
                joblib.dump(scaler, filepath)
                logger.info("✅ StandardScaler saved successfully (fitted on training data only)")
            else:
                logger.warning("No scaler found in numerical pipeline")
        else:
            logger.warning("No numerical transformer found")
    else:
        logger.warning("No preprocessor found in pipeline")


def load_scaler(filepath: str = None) -> StandardScaler:
    """
    Load a fitted StandardScaler from disk.
    
    IMPORTANT: Loaded scaler should only be used for transform(), NOT fit().
    This prevents data leakage and ensures consistent scaling.
    
    Args:
        filepath (str): Path to the saved scaler
        
    Returns:
        StandardScaler: Loaded scaler ready for inference
    """
    if filepath is None:
        filepath = SCALER_FILE
    
    logger.info(f"Loading StandardScaler from {filepath}")
    
    scaler = joblib.load(filepath)
    
    logger.info("✅ StandardScaler loaded successfully - ready for inference (transform only)")
    
    return scaler


def demonstrate_proper_scaling_workflow():
    """
    Demonstrate proper train-test split before scaling workflow.
    
    This function shows the CORRECT way to apply StandardScaler:
    1. Split data FIRST (before any scaling)
    2. Fit scaler ONLY on training data
    3. Transform both train and test using fitted scaler
    4. Save scaler for inference
    5. Never refit scaler during inference
    """
    logger.info("=" * 80)
    logger.info("DEMONSTRATING PROPER SCALING WORKFLOW")
    logger.info("=" * 80)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Generate sample data with correct number of features
    logger.info("Step 1: Generate sample data")
    n_features_total = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    X, y = make_classification(n_samples=1000, n_features=n_features_total, random_state=42)
    X_df = pd.DataFrame(X, columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    y_series = pd.Series(y, name='is_spam')
    
    logger.info(f"   Dataset shape: {X_df.shape}")
    logger.info(f"   Target distribution: {y_series.value_counts().to_dict()}")
    
    # Step 1: Split data BEFORE scaling (CRITICAL to prevent data leakage)
    logger.info("Step 2: Split data BEFORE scaling (prevents data leakage)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    logger.info(f"   Train set: {X_train.shape}")
    logger.info(f"   Test set: {X_test.shape}")
    
    # Step 2: Create and fit scaler on TRAINING data only
    logger.info("Step 3: Fit StandardScaler on training data ONLY")
    scaler = StandardScaler()
    X_train_scaled_values = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_train_scaled = X_train.copy()
    X_train_scaled[NUMERICAL_FEATURES] = X_train_scaled_values
    logger.info(f"   Scaler fitted on {len(NUMERICAL_FEATURES)} numerical features")
    logger.info(f"   Feature means (train): {scaler.mean_[:3]}...")  # Show first 3
    logger.info(f"   Feature scales (train): {scaler.scale_[:3]}...")  # Show first 3
    
    # Step 3: Transform test data using fitted scaler (NEVER fit on test data)
    logger.info("Step 4: Transform test data using fitted scaler (transform only)")
    X_test_scaled_values = scaler.transform(X_test[NUMERICAL_FEATURES])
    X_test_scaled = X_test.copy()
    X_test_scaled[NUMERICAL_FEATURES] = X_test_scaled_values
    logger.info(f"   Test data transformed using training scaler parameters")
    
    # Step 4: Verify categorical features unchanged
    logger.info("Step 5: Verify categorical features unchanged")
    for cat_feature in CATEGORICAL_FEATURES:
        train_unchanged = (X_train[cat_feature].values == X_train_scaled[cat_feature].values).all()
        test_unchanged = (X_test[cat_feature].values == X_test_scaled[cat_feature].values).all()
        logger.info(f"   {cat_feature}: Unchanged in train={train_unchanged}, test={test_unchanged}")
    
    # Step 5: Save scaler for inference
    logger.info("Step 6: Save fitted scaler for inference")
    joblib.dump(scaler, SCALER_FILE)
    logger.info(f"   Scaler saved to {SCALER_FILE}")
    
    logger.info("=" * 80)
    logger.info("✅ PROPER SCALING WORKFLOW COMPLETED")
    logger.info("=" * 80)
    logger.info("Key Principles Demonstrated:")
    logger.info("   • Split data BEFORE scaling (prevents data leakage)")
    logger.info("   • Fit scaler ONLY on training data")
    logger.info("   • Transform test data using fitted scaler")
    logger.info("   • Scale ONLY numerical features (categorical unchanged)")
    logger.info("   • Save scaler for inference (never refit)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    """
    Main function to demonstrate feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline...")
    
    try:
        # This would typically be called from train.py
        # Just demonstrating the pipeline creation
        pipeline = create_feature_pipeline()
        
        logger.info("Feature engineering pipeline created successfully!")
        logger.info(f"Pipeline steps: {pipeline.named_steps.keys()}")
        
        # Also demonstrate proper scaling workflow
        demonstrate_proper_scaling_workflow()
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer())
    ])

if __name__ == "__main__":
    main()
