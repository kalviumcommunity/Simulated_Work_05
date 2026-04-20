import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple


def load_data(data_path: str = None, synthetic: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load spam email dataset.
    
    Args:
        data_path (str): Path to CSV file. If None and synthetic=False, raises ValueError.
        synthetic (bool): If True, generates synthetic spam email data.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        
    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If data_path is None and synthetic=False
    """
    if synthetic:
        # Generate synthetic spam email data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Create feature names that resemble email features
        feature_names = [
            'word_freq_free', 'word_freq_offer', 'word_freq_win', 'word_freq_money',
            'word_freq_click', 'word_freq_business', 'word_freq_email', 'word_freq_internet',
            'word_freq_order', 'word_freq_credit', 'char_freq_exclamation', 'char_freq_dollar',
            'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
            'email_length', 'subject_length', 'has_html', 'has_attachments', 'sender_reputation'
        ]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='is_spam')
        
        return X_df, y_series
    
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when synthetic=False")
            
        try:
            data = pd.read_csv(data_path)
            
            # Assume last column is target and rest are features
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            return X, y
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")


def get_data_info(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        dict: Dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(y.unique()),
        'class_distribution': y.value_counts().to_dict(),
        'feature_names': X.columns.tolist(),
        'missing_values': X.isnull().sum().sum()
    }
    
    return info
