import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple
import joblib


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest',
    **model_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a machine learning model for spam detection.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_type (str): Type of model to train
        **model_params: Additional parameters for the model
        
    Returns:
        Tuple[Any, Dict]: Trained model and training information
    """
    # Initialize model based on type
    if model_type == 'random_forest':
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10
        }
        params = {**default_params, **model_params}
        model = RandomForestClassifier(**params)
        
    elif model_type == 'svm':
        default_params = {
            'random_state': 42,
            'kernel': 'rbf',
            'C': 1.0
        }
        params = {**default_params, **model_params}
        model = SVC(**params, probability=True)
        
    elif model_type == 'logistic_regression':
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        params = {**default_params, **model_params}
        model = LogisticRegression(**params)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get training information
    training_info = {
        'model_type': model_type,
        'model_params': params,
        'training_samples': len(X_train),
        'feature_count': X_train.shape[1],
        'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
    }
    
    # Add model-specific information
    if hasattr(model, 'feature_importances_'):
        training_info['feature_importances'] = model.feature_importances_.tolist()
    
    return model, training_info


def save_model(model: Any, filepath: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Any: Loaded model object
    """
    return joblib.load(filepath)


def predict(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model object
        X (np.ndarray): Features to predict
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predictions and probabilities
    """
    predictions = model.predict(X)
    
    # Get probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    else:
        # For models without predict_proba, create dummy probabilities
        probabilities = np.zeros((len(predictions), 2))
        probabilities[np.arange(len(predictions)), predictions] = 1.0
    
    return predictions, probabilities


def get_model_summary(model: Any, training_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the trained model.
    
    Args:
        model: Trained model object
        training_info (Dict): Training information
        
    Returns:
        Dict[str, Any]: Model summary
    """
    summary = training_info.copy()
    
    # Add model-specific metrics
    if hasattr(model, 'feature_importances_'):
        summary['top_features'] = {
            'indices': np.argsort(model.feature_importances_)[-5:].tolist(),
            'importances': np.sort(model.feature_importances_)[-5:].tolist()
        }
    
    if hasattr(model, 'score'):
        summary['has_score_method'] = True
    
    return summary
