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
    
    # Use only standard scikit-learn components for reliability
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    if select_features:
        steps.append(('feature_selector', SelectKBest(f_classif, k=n_features_to_select)))
    
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
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer())
    ])

if __name__ == "__main__":
    main()
