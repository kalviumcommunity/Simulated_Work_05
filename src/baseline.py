"""
Baseline Model Implementation for Spam Email Detection.

This module implements a DummyClassifier baseline to establish a minimum
performance benchmark and demonstrate meaningful improvement over trivial
solutions.

The baseline uses the 'most_frequent' strategy, which always predicts the
most common class (non-spam). This represents a "do nothing" heuristic that
any useful model must beat.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import logging
from typing import Dict, Any, Tuple

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_baseline_model(strategy: str = 'most_frequent', random_state: int = RANDOM_STATE) -> DummyClassifier:
    """
    Create a baseline DummyClassifier.
    
    Args:
        strategy (str): Strategy for baseline prediction
            - 'most_frequent': Always predict most common class (default)
            - 'stratified': Predict according to class distribution
            - 'uniform': Random uniform predictions
        random_state (int): Random seed for reproducibility
        
    Returns:
        DummyClassifier: Baseline model instance
    """
    logger.info(f"Creating baseline model with strategy='{strategy}'...")
    
    baseline = DummyClassifier(strategy=strategy, random_state=random_state)
    
    logger.info(f"Baseline model created: {baseline}")
    
    return baseline


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = 'most_frequent',
    **kwargs
) -> Tuple[DummyClassifier, Dict[str, Any]]:
    """
    Train a baseline model on training data only.
    
    IMPORTANT: Baseline is fitted ONLY on training data to prevent leakage.
    
    Args:
        X_train (np.ndarray): Training features (not used by DummyClassifier but kept for API consistency)
        y_train (np.ndarray): Training labels
        strategy (str): Baseline strategy
        **kwargs: Additional parameters
        
    Returns:
        Tuple[DummyClassifier, Dict]: Trained baseline and training info
    """
    logger.info(f"Training baseline model (strategy={strategy})...")
    logger.info(f"Training samples: {len(y_train)}")
    
    # Create baseline model
    baseline = create_baseline_model(strategy=strategy)
    
    # Fit baseline ONLY on training data (X not used by DummyClassifier)
    baseline.fit(X_train, y_train)
    
    # Get training predictions for baseline metrics
    train_predictions = baseline.predict(X_train)
    
    # Calculate training metrics
    training_info = {
        'model_type': 'DummyClassifier',
        'strategy': strategy,
        'training_samples': len(X_train),
        'class_distribution': {
            'class_0': int(np.sum(y_train == 0)),
            'class_1': int(np.sum(y_train == 1))
        },
        'most_frequent_class': int(baseline.class_prior_.argmax()),
        'training_accuracy': float(accuracy_score(y_train, train_predictions)),
        'training_precision': float(precision_score(y_train, train_predictions, average='binary', zero_division=0)),
        'training_recall': float(recall_score(y_train, train_predictions, average='binary', zero_division=0)),
        'training_f1': float(f1_score(y_train, train_predictions, average='binary', zero_division=0))
    }
    
    logger.info(f"Baseline trained successfully!")
    logger.info(f"Most frequent class: {training_info['most_frequent_class']}")
    logger.info(f"Training accuracy: {training_info['training_accuracy']:.4f}")
    
    return baseline, training_info


def evaluate_baseline(
    baseline: DummyClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate baseline model on held-out test data.
    
    Args:
        baseline (DummyClassifier): Trained baseline model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        Dict[str, Any]: Comprehensive test metrics
    """
    logger.info("Evaluating baseline model on test data...")
    
    # Make predictions
    y_pred = baseline.predict(X_test)
    
    # Calculate basic metrics
    test_metrics = {
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred, average='binary', zero_division=0))
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        confusion_metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        }
    else:
        confusion_metrics = {'confusion_matrix': cm.tolist()}
    
    # Generate classification report
    class_report = classification_report(
        y_test, y_pred, 
        target_names=['Not Spam', 'Spam'],
        output_dict=True,
        zero_division=0
    )
    
    # Combine all metrics
    evaluation_results = {
        'test_metrics': test_metrics,
        'confusion_matrix': confusion_metrics,
        'classification_report': class_report,
        'test_samples': len(y_test),
        'class_distribution': {
            'class_0': int(np.sum(y_test == 0)),
            'class_1': int(np.sum(y_test == 1))
        }
    }
    
    logger.info("Baseline evaluation completed!")
    logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Test F1-score: {test_metrics['test_f1']:.4f}")
    
    return evaluation_results


def save_baseline(baseline: DummyClassifier, filepath: str) -> None:
    """
    Save the trained baseline model to disk.
    
    Args:
        baseline (DummyClassifier): Trained baseline model
        filepath (str): Path to save the baseline
    """
    logger.info(f"Saving baseline model to {filepath}")
    joblib.dump(baseline, filepath)
    logger.info("Baseline model saved successfully")


def load_baseline(filepath: str) -> DummyClassifier:
    """
    Load a trained baseline model from disk.
    
    Args:
        filepath (str): Path to the saved baseline
        
    Returns:
        DummyClassifier: Loaded baseline model
    """
    logger.info(f"Loading baseline model from {filepath}")
    baseline = joblib.load(filepath)
    logger.info("Baseline model loaded successfully")
    return baseline


def compare_models(
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    baseline_name: str = 'Baseline (DummyClassifier)',
    model_name: str = 'Primary Model (Random Forest)'
) -> Dict[str, Any]:
    """
    Compare baseline and primary model performance side-by-side.
    
    Args:
        baseline_metrics (Dict): Baseline evaluation results
        model_metrics (Dict): Primary model evaluation results
        baseline_name (str): Name for baseline model
        model_name (str): Name for primary model
        
    Returns:
        Dict[str, Any]: Comprehensive comparison results
    """
    logger.info("=" * 80)
    logger.info("BASELINE VS PRIMARY MODEL COMPARISON")
    logger.info("=" * 80)
    
    # Extract test metrics
    baseline_test = baseline_metrics.get('test_metrics', baseline_metrics)
    model_test = model_metrics.get('test_metrics', model_metrics)
    
    # Calculate improvements
    accuracy_improvement = model_test['test_accuracy'] - baseline_test['test_accuracy']
    precision_improvement = model_test['test_precision'] - baseline_test['test_precision']
    recall_improvement = model_test['test_recall'] - baseline_test['test_recall']
    f1_improvement = model_test['test_f1'] - baseline_test['test_f1']
    
    # Calculate percentage improvements
    accuracy_pct = (accuracy_improvement / baseline_test['test_accuracy'] * 100) if baseline_test['test_accuracy'] > 0 else 0
    f1_pct = (f1_improvement / baseline_test['test_f1'] * 100) if baseline_test['test_f1'] > 0 else 0
    
    # Create comparison table
    comparison = {
        'baseline_name': baseline_name,
        'model_name': model_name,
        'metrics': {
            'accuracy': {
                'baseline': baseline_test['test_accuracy'],
                'model': model_test['test_accuracy'],
                'improvement': accuracy_improvement,
                'improvement_pct': accuracy_pct
            },
            'precision': {
                'baseline': baseline_test['test_precision'],
                'model': model_test['test_precision'],
                'improvement': precision_improvement
            },
            'recall': {
                'baseline': baseline_test['test_recall'],
                'model': model_test['test_recall'],
                'improvement': recall_improvement
            },
            'f1_score': {
                'baseline': baseline_test['test_f1'],
                'model': model_test['test_f1'],
                'improvement': f1_improvement,
                'improvement_pct': f1_pct
            }
        },
        'summary': {
            'accuracy_improvement': accuracy_improvement,
            'accuracy_improvement_pct': accuracy_pct,
            'f1_improvement': f1_improvement,
            'f1_improvement_pct': f1_pct,
            'meaningful_improvement': accuracy_improvement > 0.1 or f1_improvement > 0.1
        }
    }
    
    # Log comparison results
    logger.info(f"{'Metric':<20} {'Baseline':<15} {'Model':<15} {'Improvement':<15}")
    logger.info("-" * 80)
    logger.info(f"{'Accuracy':<20} {baseline_test['test_accuracy']:<15.4f} {model_test['test_accuracy']:<15.4f} {accuracy_improvement:+.4f} ({accuracy_pct:+.1f}%)")
    logger.info(f"{'Precision':<20} {baseline_test['test_precision']:<15.4f} {model_test['test_precision']:<15.4f} {precision_improvement:+.4f}")
    logger.info(f"{'Recall':<20} {baseline_test['test_recall']:<15.4f} {model_test['test_recall']:<15.4f} {recall_improvement:+.4f}")
    logger.info(f"{'F1-Score':<20} {baseline_test['test_f1']:<15.4f} {model_test['test_f1']:<15.4f} {f1_improvement:+.4f} ({f1_pct:+.1f}%)")
    logger.info("=" * 80)
    
    if comparison['summary']['meaningful_improvement']:
        logger.info("✅ MEANINGFUL IMPROVEMENT: Primary model significantly outperforms baseline")
    else:
        logger.info("⚠️ LIMITED IMPROVEMENT: Primary model shows marginal improvement over baseline")
    
    return comparison


def run_baseline_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str = 'most_frequent'
) -> Dict[str, Any]:
    """
    Run a complete baseline experiment: train, evaluate, and return results.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        strategy (str): Baseline strategy
        
    Returns:
        Dict[str, Any]: Complete baseline experiment results
    """
    logger.info("=" * 80)
    logger.info("RUNNING BASELINE MODEL EXPERIMENT")
    logger.info("=" * 80)
    
    # Train baseline
    baseline, training_info = train_baseline(X_train, y_train, strategy=strategy)
    
    # Evaluate baseline
    evaluation_results = evaluate_baseline(baseline, X_test, y_test)
    
    # Combine results
    experiment_results = {
        'training_info': training_info,
        'evaluation_results': evaluation_results,
        'strategy': strategy
    }
    
    logger.info("=" * 80)
    logger.info("BASELINE EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    
    return experiment_results


def main():
    """
    Main function to demonstrate baseline model implementation.
    """
    logger.info("=" * 80)
    logger.info("BASELINE MODEL DEMONSTRATION")
    logger.info("=" * 80)
    
    # Import necessary modules
    from data_preprocessing import load_data, clean_data, split_data
    from feature_engineering import fit_preprocessor
    from train import train_model, evaluate_training_model
    
    # Load and preprocess data
    logger.info("Step 1: Loading data...")
    X, y = load_data(synthetic=True)
    X_clean, y_clean = clean_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Class distribution in train: {np.bincount(y_train)}")
    logger.info(f"Class distribution in test: {np.bincount(y_test)}")
    
    # Feature engineering
    logger.info("Step 2: Feature engineering...")
    preprocessor, X_train_transformed = fit_preprocessor(X_train, y_train.values)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Run baseline experiment
    logger.info("Step 3: Running baseline experiment...")
    baseline_results = run_baseline_experiment(
        X_train_transformed, X_test_transformed,
        y_train.values, y_test.values,
        strategy='most_frequent'
    )
    
    # Train primary model (Random Forest)
    logger.info("Step 4: Training primary model (Random Forest)...")
    primary_model, training_info = train_model(X_train_transformed, y_train.values)
    primary_metrics = evaluate_training_model(primary_model, X_test_transformed, y_test.values)
    
    # Compare baseline vs primary model
    logger.info("Step 5: Comparing baseline vs primary model...")
    comparison = compare_models(
        baseline_results['evaluation_results'],
        primary_metrics,
        baseline_name='Baseline (DummyClassifier - most_frequent)',
        model_name='Primary Model (Random Forest)'
    )
    
    # Summary
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Baseline Strategy: {baseline_results['strategy']}")
    logger.info(f"Baseline Accuracy: {baseline_results['evaluation_results']['test_metrics']['test_accuracy']:.4f}")
    logger.info(f"Primary Model Accuracy: {primary_metrics['test_accuracy']:.4f}")
    logger.info(f"Accuracy Improvement: {comparison['summary']['accuracy_improvement']:+.4f} ({comparison['summary']['accuracy_improvement_pct']:+.1f}%)")
    logger.info(f"F1-Score Improvement: {comparison['summary']['f1_improvement']:+.4f} ({comparison['summary']['f1_improvement_pct']:+.1f}%)")
    logger.info(f"Meaningful Improvement: {'Yes' if comparison['summary']['meaningful_improvement'] else 'No'}")
    logger.info("=" * 80)
    
    return {
        'baseline_results': baseline_results,
        'primary_metrics': primary_metrics,
        'comparison': comparison
    }


if __name__ == "__main__":
    results = main()
