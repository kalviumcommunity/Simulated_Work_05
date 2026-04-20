import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def preprocess_data(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocess the spam email dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        scale_features (bool): Whether to scale features
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
            X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_info
    """
    # Handle missing values
    X_clean = X.copy()
    if X_clean.isnull().any().any():
        X_clean = X_clean.fillna(X_clean.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Scale features if requested
    scaler = None
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values
    
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Store preprocessing information
    preprocessing_info = {
        'original_shape': X.shape,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'missing_values_handled': X.isnull().any().any(),
        'features_scaled': scale_features,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, preprocessing_info


def remove_outliers(
    X: np.ndarray, 
    y: np.ndarray, 
    method: str = 'iqr',
    factor: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from the dataset.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        method (str): Outlier detection method ('iqr' or 'zscore')
        factor (float): Factor for outlier detection
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned X and y
    """
    if method == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        
    elif method == 'zscore':
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        mask = (z_scores < factor).all(axis=1)
        
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return X[mask], y[mask]


def get_feature_statistics(X: np.ndarray, feature_names: list = None) -> Dict[str, Any]:
    """
    Calculate feature statistics.
    
    Args:
        X (np.ndarray): Feature matrix
        feature_names (list): Names of features
        
    Returns:
        Dict[str, Any]: Feature statistics
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    stats = {}
    for i, name in enumerate(feature_names):
        stats[name] = {
            'mean': float(np.mean(X[:, i])),
            'std': float(np.std(X[:, i])),
            'min': float(np.min(X[:, i])),
            'max': float(np.max(X[:, i])),
            'median': float(np.median(X[:, i]))
        }
    
    return stats
