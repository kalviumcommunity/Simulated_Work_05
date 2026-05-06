"""
Data preprocessing module for spam email detection.
Handles loading, cleaning, and splitting of data.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import logging

from config import (
    SYNTHETIC_DATA, N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_REDUNDANT,
    RANDOM_STATE, TEST_SIZE, FEATURE_NAMES, DATA_DIR,
    TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, 
    EXCLUDED_COLUMNS, ALL_FEATURES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_features(df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and separate features from target with professional assertions.
    
    Args:
        df (pd.DataFrame): Complete dataset with all columns
        target (pd.Series): Target variable
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Validated features (X) and target (y)
        
    Raises:
        AssertionError: If validation fails
    """
    logger.info("🔍 Validating feature types and separation...")
    
    # Validate target column exists in the dataframe
    if TARGET_COLUMN not in df.columns:
        raise AssertionError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    assert TARGET_COLUMN not in ALL_FEATURES, "Target column cannot be in feature set"
    
    # Validate excluded columns are not in features
    for excluded_col in EXCLUDED_COLUMNS:
        assert excluded_col not in ALL_FEATURES, f"Excluded column '{excluded_col}' found in feature set"
        if excluded_col in df.columns:
            assert excluded_col not in ALL_FEATURES, f"Excluded column '{excluded_col}' found in feature set"
    
    # Separate features and target
    features = df[ALL_FEATURES].copy()
    target_clean = target.copy()
    
    # Validate no target leakage in features
    for col in features.columns:
        assert col != TARGET_COLUMN, f"Target column '{TARGET_COLUMN}' found in features"
    
    # Print feature type summary
    print(f"\n📊 FEATURE TYPE VALIDATION RESULTS:")
    print(f"   Target column: {TARGET_COLUMN}")
    print(f"   Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"   Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"   Excluded columns: {len(EXCLUDED_COLUMNS)}")
    print(f"   Total features for modeling: {len(ALL_FEATURES)}")
    print(f"   Available columns in dataset: {list(df.columns)}")
    
    # Validate feature presence
    missing_features = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            missing_features.append(feature)
    
    # Additional validation: check for any column name issues
    available_columns = list(df.columns)
    for col in available_columns:
        if col not in ALL_FEATURES and col not in EXCLUDED_COLUMNS and col != TARGET_COLUMN:
            logger.warning(f"Unexpected column found: {col}")
    
    if missing_features:
        logger.warning(f"Missing expected features: {missing_features}")
    else:
        logger.info("✅ All expected features found in dataset")
    
    return features, target_clean

def load_data(data_path: Optional[str] = None, synthetic: bool = SYNTHETIC_DATA) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load spam email dataset.
    
    Args:
        data_path (Optional[str]): Path to CSV file. If None and synthetic=False, raises ValueError.
        synthetic (bool): If True, generates synthetic spam email data.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        
    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If data_path is None and synthetic=False
    """
    logger.info("Loading data...")
    
    if synthetic:
        logger.info(f"Generating synthetic data with {N_SAMPLES} samples and {N_FEATURES} features")
        
        # Generate synthetic spam email data
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=N_INFORMATIVE,
            n_redundant=N_REDUNDANT,
            n_classes=2,
            random_state=RANDOM_STATE
        )
        
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        y_series = pd.Series(y, name=TARGET_COLUMN)
        
        logger.info(f"Data loaded successfully. Shape: {X_df.shape}")
        logger.info(f"Class distribution: {y_series.value_counts().to_dict()}")
        
        # Apply feature validation
        complete_df = pd.concat([X_df, y_series], axis=1)
        return validate_features(complete_df, y_series)
    
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when synthetic=False")
            
        try:
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            # Apply feature validation
            target = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else data.iloc[:, -1]
            features = data.drop(columns=[TARGET_COLUMN] + EXCLUDED_COLUMNS, errors='ignore')
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            logger.info(f"Class distribution: {target.value_counts().to_dict()}")
            
            return validate_features(features, target)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")


def clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the dataset by handling missing values and basic validation.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned features and target
    """
    logger.info("Cleaning data...")
    
    # Handle missing values
    if X.isnull().any().any():
        logger.info("Handling missing values by filling with mean")
        X_clean = X.fillna(X.mean())
    else:
        X_clean = X.copy()
    
    # Remove any rows with NaN in target
    if y.isnull().any():
        logger.info("Removing rows with missing target values")
        valid_mask = ~y.isnull()
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]
    else:
        y_clean = y.copy()
    
    logger.info(f"Data cleaned. Final shape: {X_clean.shape}")
    
    return X_clean, y_clean


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        stratify (bool): Whether to use stratified sampling
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_data_info(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Get basic information about the dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        Dict[str, Any]: Dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(y.unique()),
        'class_distribution': y.value_counts().to_dict(),
        'feature_names': X.columns.tolist(),
        'missing_values': X.isnull().sum().sum(),
        'data_types': X.dtypes.to_dict()
    }
    
    return info


def main():
    """
    Main function to demonstrate data preprocessing pipeline.
    """
    logger.info("Starting data preprocessing pipeline...")
    
    try:
        # Load data
        X, y = load_data(synthetic=True)
        
        # Clean data
        X_clean, y_clean = clean_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
        
        # Get data info
        data_info = get_data_info(X_clean, y_clean)
        
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Dataset info: {data_info}")
        
        return X_train, X_test, y_train, y_test, data_info
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
