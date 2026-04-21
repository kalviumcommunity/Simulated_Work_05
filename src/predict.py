"""
Prediction module for spam email detection.
Loads saved artifacts and produces predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Union
import logging
import joblib

from config import MODEL_FILE, PREPROCESSOR_FILE, FEATURE_NAMES_FILE, FEATURE_NAMES
from train import load_model
from feature_engineering import load_preprocessor, load_feature_names, transform_features
from evaluate import evaluate_model, print_evaluation_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpamClassifier:
    """
    Spam classifier that loads trained artifacts and makes predictions.
    """
    
    def __init__(
        self,
        model_path: str = None,
        preprocessor_path: str = None,
        feature_names_path: str = None
    ):
        """
        Initialize the classifier with saved artifacts.
        
        Args:
            model_path (str): Path to saved model
            preprocessor_path (str): Path to saved preprocessor
            feature_names_path (str): Path to saved feature names
        """
        logger.info("Initializing SpamClassifier...")
        
        # Load artifacts
        self.model = load_model(model_path)
        self.preprocessor = load_preprocessor(preprocessor_path)
        
        try:
            self.feature_names = load_feature_names(feature_names_path)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        except FileNotFoundError:
            logger.warning("No feature names file found, using default feature names")
            self.feature_names = FEATURE_NAMES
        
        self.is_fitted = True
        logger.info("SpamClassifier initialized successfully")
    
    def _validate_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and prepare features for prediction.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features
            
        Returns:
            np.ndarray: Validated features
            
        Raises:
            ValueError: If features don't match expected format
        """
        logger.info("Validating input features...")
        
        # For prediction, we need the original 20 features before feature selection
        original_feature_names = FEATURE_NAMES
        
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            if X.columns.tolist() != original_feature_names:
                logger.warning("Feature names don't match. Reordering columns...")
                X = X[original_feature_names]
            X_array = X.values
        else:
            X_array = X
        
        # Check feature count
        if X_array.shape[1] != len(original_feature_names):
            raise ValueError(
                f"Expected {len(original_feature_names)} features, got {X_array.shape[1]}"
            )
        
        logger.info(f"Features validated: shape {X_array.shape}")
        
        return X_array
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        logger.info(f"Making predictions on {len(X)} samples...")
        
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Load artifacts first.")
        
        # Validate features
        X_validated = self._validate_features(X)
        
        # Transform features
        X_transformed = transform_features(self.preprocessor, X_validated)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        result = {
            'predictions': predictions.tolist(),
            'sample_count': len(predictions),
            'prediction_labels': ['Not Spam' if p == 0 else 'Spam' for p in predictions]
        }
        
        # Add probabilities if requested and available
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_transformed)
            result['probabilities'] = probabilities.tolist()
            result['spam_probabilities'] = probabilities[:, 1].tolist()
        
        logger.info(f"Predictions completed: {sum(predictions)} spam, {len(predictions) - sum(predictions)} not spam")
        
        return result
    
    def predict_single(
        self,
        features: Union[Dict[str, float], List[float], np.ndarray],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction on a single sample.
        
        Args:
            features (Union[Dict, List, np.ndarray]): Single sample features
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            Dict[str, Any]: Prediction result for single sample
        """
        logger.info("Making single prediction...")
        
        # Convert features to DataFrame using original feature names
        original_feature_names = FEATURE_NAMES
        
        if isinstance(features, dict):
            # Ensure all required features are present
            missing_features = set(original_feature_names) - set(features.keys())
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Create DataFrame with correct column order
            features_df = pd.DataFrame([features], columns=original_feature_names)
        elif isinstance(features, (list, np.ndarray)):
            if len(features) != len(original_feature_names):
                raise ValueError(
                    f"Expected {len(original_feature_names)} features, got {len(features)}"
                )
            features_df = pd.DataFrame([features], columns=original_feature_names)
        else:
            raise ValueError("Features must be dict, list, or numpy array")
        
        # Make prediction
        result = self.predict(features_df, return_probabilities)
        
        # Extract single prediction
        single_result = {
            'prediction': result['predictions'][0],
            'label': result['prediction_labels'][0],
            'sample_count': 1
        }
        
        if return_probabilities and 'probabilities' in result:
            single_result['probabilities'] = result['probabilities'][0]
            single_result['spam_probability'] = result['spam_probabilities'][0]
        
        logger.info(f"Single prediction completed: {single_result['label']}")
        
        return single_result
    
    def evaluate_on_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        print_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on held-out data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Test features
            y (Union[pd.Series, np.ndarray]): True labels
            print_summary (bool): Whether to print evaluation summary
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Evaluating classifier on test data...")
        
        # Make predictions
        prediction_result = self.predict(X, return_probabilities=True)
        
        # Convert to numpy arrays
        y_pred = np.array(prediction_result['predictions'])
        y_proba = None
        
        if 'probabilities' in prediction_result:
            y_proba = np.array(prediction_result['spam_probabilities'])
        
        # Evaluate
        evaluation_results = evaluate_model(
            y_true=np.array(y),
            y_pred=y_pred,
            y_proba=y_proba
        )
        
        if print_summary:
            print_evaluation_summary(evaluation_results)
        
        logger.info("Evaluation completed")
        
        return evaluation_results


def load_classifier(
    model_path: str = None,
    preprocessor_path: str = None,
    feature_names_path: str = None
) -> SpamClassifier:
    """
    Load a pre-trained spam classifier.
    
    Args:
        model_path (str): Path to saved model
        preprocessor_path (str): Path to saved preprocessor
        feature_names_path (str): Path to saved feature names
        
    Returns:
        SpamClassifier: Loaded classifier
    """
    logger.info("Loading pre-trained classifier...")
    
    classifier = SpamClassifier(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_names_path=feature_names_path
    )
    
    logger.info("Classifier loaded successfully")
    
    return classifier


def predict_from_features(
    features: Union[Dict[str, float], List[float], pd.DataFrame],
    classifier: SpamClassifier = None
) -> Dict[str, Any]:
    """
    Convenience function to make predictions from features.
    
    Args:
        features (Union[Dict, List, DataFrame]): Input features
        classifier (SpamClassifier): Pre-loaded classifier
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    if classifier is None:
        classifier = load_classifier()
    
    # Handle different input types
    if isinstance(features, pd.DataFrame):
        return classifier.predict(features)
    else:
        return classifier.predict_single(features)


def main():
    """
    Main function to demonstrate prediction functionality.
    """
    logger.info("Starting prediction demonstration...")
    
    try:
        # Load classifier
        classifier = load_classifier()
        
        # Create sample data for demonstration
        sample_features = {
            'word_freq_free': 0.5,
            'word_freq_offer': 0.3,
            'word_freq_win': 0.2,
            'word_freq_money': 0.1,
            'word_freq_click': 0.0,
            'word_freq_business': 0.1,
            'word_freq_email': 0.0,
            'word_freq_internet': 0.0,
            'word_freq_order': 0.0,
            'word_freq_credit': 0.0,
            'char_freq_exclamation': 0.2,
            'char_freq_dollar': 0.1,
            'capital_run_length_average': 2.5,
            'capital_run_length_longest': 5.0,
            'capital_run_length_total': 10.0,
            'email_length': 100.0,
            'subject_length': 20.0,
            'has_html': 1.0,
            'has_attachments': 0.0,
            'sender_reputation': 0.3
        }
        
        # Make single prediction
        result = classifier.predict_single(sample_features)
        
        logger.info("Prediction demonstration completed!")
        logger.info(f"Sample prediction: {result['label']} (confidence: {result.get('spam_probability', 'N/A')})")
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {str(e)}")
        logger.info("Please run train.py first to create model artifacts")
        return None
    except Exception as e:
        logger.error(f"Error in prediction demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()
