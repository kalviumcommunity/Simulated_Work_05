"""
Training module for spam email detection.
Fits the model and saves artifacts for later prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Tuple, Dict, Any

from config import (
    MODEL_TYPE, MODEL_PARAMS, MODEL_FILE, FEATURE_NAMES_FILE,
    RANDOM_STATE, FEATURE_NAMES
)
from data_preprocessing import load_data, clean_data, split_data
from feature_engineering import (
    fit_preprocessor, save_preprocessor, save_feature_names,
    get_feature_importance
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(model_type: str = MODEL_TYPE, **params) -> Any:
    """
    Create a machine learning model instance.
    
    Args:
        model_type (str): Type of model to create
        **params: Additional parameters for the model
        
    Returns:
        Any: Model instance
    """
    logger.info(f"Creating {model_type} model...")
    
    if model_type == 'random_forest':
        default_params = {
            'random_state': RANDOM_STATE,
            'max_depth': 10
        }
        params = {**default_params, **params}
        model = RandomForestClassifier(**params)
        
    elif model_type == 'svm':
        default_params = {
            'random_state': RANDOM_STATE,
            'kernel': 'rbf',
            'C': 1.0
        }
        params = {**default_params, **params}
        model = SVC(**params, probability=True)
        
    elif model_type == 'logistic_regression':
        default_params = {
            'random_state': RANDOM_STATE,
            'max_iter': 1000
        }
        params = {**default_params, **params}
        model = LogisticRegression(**params)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Model created with parameters: {params}")
    
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = MODEL_TYPE,
    **model_params
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a machine learning model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_type (str): Type of model to train
        **model_params: Additional parameters for the model
        
    Returns:
        Tuple[Any, Dict]: Trained model and training information
    """
    logger.info(f"Training {model_type} model...")
    
    # Create and train model
    model = create_model(model_type, **model_params)
    model.fit(X_train, y_train)
    
    # Get training predictions for basic metrics
    train_predictions = model.predict(X_train)
    
    # Calculate training metrics
    training_info = {
        'model_type': model_type,
        'model_params': model_params,
        'training_samples': len(X_train),
        'feature_count': X_train.shape[1],
        'training_accuracy': float(accuracy_score(y_train, train_predictions)),
        'training_precision': float(precision_score(y_train, train_predictions, average='binary')),
        'training_recall': float(recall_score(y_train, train_predictions, average='binary')),
        'training_f1': float(f1_score(y_train, train_predictions, average='binary'))
    }
    
    # Add model-specific information
    if hasattr(model, 'feature_importances_'):
        training_info['has_feature_importances'] = True
    else:
        training_info['has_feature_importances'] = False
    
    logger.info(f"Model trained successfully!")
    logger.info(f"Training accuracy: {training_info['training_accuracy']:.4f}")
    
    return model, training_info


def save_model(model: Any, filepath: str = None) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    if filepath is None:
        filepath = MODEL_FILE
    
    logger.info(f"Saving model to {filepath}")
    
    joblib.dump(model, filepath)
    
    logger.info("Model saved successfully")


def load_model(filepath: str = None) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Any: Loaded model object
    """
    if filepath is None:
        filepath = MODEL_FILE
    
    logger.info(f"Loading model from {filepath}")
    
    model = joblib.load(filepath)
    
    logger.info("Model loaded successfully")
    
    return model


def evaluate_training_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model object
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        Dict[str, float]: Test metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    test_metrics = {
        'test_accuracy': float(accuracy_score(y_test, test_predictions)),
        'test_precision': float(precision_score(y_test, test_predictions, average='binary')),
        'test_recall': float(recall_score(y_test, test_predictions, average='binary')),
        'test_f1': float(f1_score(y_test, test_predictions, average='binary'))
    }
    
    # Add ROC AUC if model supports probabilities
    if hasattr(model, 'predict_proba'):
        from sklearn.metrics import roc_auc_score
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_metrics['test_roc_auc'] = float(roc_auc_score(y_test, test_probabilities))
    
    logger.info(f"Test evaluation completed. Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    return test_metrics


def main():
    """
    Main training pipeline function.
    """
    logger.info("Starting training pipeline...")
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        X, y = load_data(synthetic=True)
        X_clean, y_clean = clean_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        preprocessor, X_train_transformed = fit_preprocessor(X_train.values, y_train.values)
        X_test_transformed = preprocessor.transform(X_test.values)
        
        # Step 3: Train model
        logger.info("Step 3: Training model...")
        model, training_info = train_model(
            X_train_transformed, 
            y_train.values, 
            model_type=MODEL_TYPE,
            **MODEL_PARAMS
        )
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        test_metrics = evaluate_training_model(model, X_test_transformed, y_test.values)
        
        # Step 5: Save artifacts
        logger.info("Step 5: Saving artifacts...")
        save_model(model)
        save_preprocessor(preprocessor)
        
        # Save feature names if feature selection was used
        if 'feature_selector' in preprocessor.named_steps:
            selector = preprocessor.named_steps['feature_selector']
            selected_indices = selector.get_support(indices=True)
            save_feature_names(FEATURE_NAMES, selected_indices)
        else:
            save_feature_names(FEATURE_NAMES)
        
        # Step 6: Generate training summary
        logger.info("Step 6: Generating training summary...")
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            if 'feature_selector' in preprocessor.named_steps:
                selector = preprocessor.named_steps['feature_selector']
                selected_indices = selector.get_support(indices=True)
                scores = selector.scores_
                # Create importance dictionary for selected features
                for idx in selected_indices:
                    if idx < len(FEATURE_NAMES):
                        feature_importance[FEATURE_NAMES[idx]] = float(scores[idx])
        
        # Combine all information
        training_summary = {
            'training_info': training_info,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'data_info': {
                'total_samples': len(X_clean),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'original_features': X_clean.shape[1],
                'transformed_features': X_train_transformed.shape[1]
            }
        }
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1-score: {test_metrics['test_f1']:.4f}")
        
        if 'test_roc_auc' in test_metrics:
            logger.info(f"Test ROC AUC: {test_metrics['test_roc_auc']:.4f}")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
