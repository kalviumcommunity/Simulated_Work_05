"""
Feature engineering module for spam email detection.
Builds reusable transformations and pipelines for feature processing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, Any, List, Optional
import joblib
import logging

from config import PREPROCESSOR_FILE, FEATURE_NAMES_FILE

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
    n_features_to_select: int = 15
) -> Pipeline:
    """
    Create a reusable feature engineering pipeline.
    
    Args:
        scale_features (bool): Whether to scale features
        select_features (bool): Whether to perform feature selection
        remove_outliers (bool): Whether to handle outliers
        n_features_to_select (int): Number of features to select if selection is enabled
        
    Returns:
        Pipeline: Configured feature engineering pipeline
    """
    logger.info("Creating feature engineering pipeline...")
    
    steps = []
    
    if remove_outliers:
        steps.append(('outlier_handler', OutlierRemover()))
    
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    if select_features:
        steps.append(('feature_selector', FeatureSelector(k=n_features_to_select)))
    
    pipeline = Pipeline(steps)
    
    logger.info(f"Pipeline created with {len(steps)} steps: {[step[0] for step in steps]}")
    
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
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    
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


def save_preprocessor(preprocessor: Pipeline, filepath: str = None) -> None:
    """
    Save the fitted preprocessor to disk.
    
    Args:
        preprocessor (Pipeline): Fitted preprocessor
        filepath (str): Path to save the preprocessor
    """
    if filepath is None:
        filepath = PREPROCESSOR_FILE
    
    logger.info(f"Saving preprocessor to {filepath}")
    
    joblib.dump(preprocessor, filepath)
    
    logger.info("Preprocessor saved successfully")


def load_preprocessor(filepath: str = None) -> Pipeline:
    """
    Load a fitted preprocessor from disk.
    
    Args:
        filepath (str): Path to the saved preprocessor
        
    Returns:
        Pipeline: Loaded preprocessor
    """
    if filepath is None:
        filepath = PREPROCESSOR_FILE
    
    logger.info(f"Loading preprocessor from {filepath}")
    
    preprocessor = joblib.load(filepath)
    
    logger.info("Preprocessor loaded successfully")
    
    return preprocessor


def get_feature_importance(preprocessor: Pipeline, feature_names: List[str]) -> Dict[str, float]:
    """
    Get feature importance from the preprocessor if it includes feature selection.
    
    Args:
        preprocessor (Pipeline): Fitted preprocessor
        feature_names (List[str]): Original feature names
        
    Returns:
        Dict[str, float]: Feature importance scores
    """
    logger.info("Extracting feature importance...")
    
    importance_dict = {}
    
    # Check if pipeline has feature selector
    if 'feature_selector' in preprocessor.named_steps:
        selector = preprocessor.named_steps['feature_selector']
        selected_indices = selector.selected_features
        
        # Get scores from the selector
        scores = selector.selector.scores_
        
        # Create importance dictionary for selected features
        for idx in selected_indices:
            if idx < len(feature_names):
                importance_dict[feature_names[idx]] = float(scores[idx])
    
    logger.info(f"Feature importance extracted for {len(importance_dict)} features")
    
    return importance_dict


def save_feature_names(feature_names: List[str], selected_indices: Optional[List[int]] = None, filepath: str = None) -> None:
    """
    Save feature names to disk.
    
    Args:
        feature_names (List[str]): Original feature names
        selected_indices (Optional[List[int]]): Indices of selected features
        filepath (str): Path to save the feature names
    """
    if filepath is None:
        filepath = FEATURE_NAMES_FILE
    
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
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()
