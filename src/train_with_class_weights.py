"""
Training Module with Class Weights Support

This module extends the original training functionality to include class weights
for handling imbalanced datasets.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Tuple, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import fit_preprocessor, transform_features
from src.class_weights import (
    compute_balanced_weights, train_weighted_model, 
    cross_validate_with_weights, grid_search_class_weights
)
from src.evaluate import evaluate_model
from src.config import *

logger = logging.getLogger(__name__)


def train_model_with_class_weights(
    X_train, y_train, 
    model_type: str = "logistic",
    class_weight: str = "balanced",
    use_grid_search: bool = False,
    cv_folds: int = 5,
    **model_params
) -> Tuple[Any, Dict]:
    """
    Train a model with class weights.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        class_weight: Class weighting strategy
        use_grid_search: Whether to use grid search for optimal weights
        cv_folds: Number of CV folds for grid search
        **model_params: Additional model parameters
        
    Returns:
        Tuple of (trained_model, training_info)
    """
    logger.info(f"Training {model_type} model with class_weight={class_weight}")
    
    if use_grid_search and class_weight != "balanced":
        # Grid search for optimal weights
        grid_results = grid_search_class_weights(
            X_train, y_train, model_type=model_type, cv_folds=cv_folds
        )
        model = grid_results['best_estimator']
        training_info = {
            'model_type': model_type,
            'class_weight': grid_results['best_params']['class_weight'],
            'grid_search_used': True,
            'best_cv_score': grid_results['best_score'],
            'model_params': model_params
        }
    else:
        # Train with specified class weight
        model = train_weighted_model(
            X_train, y_train, model_type=model_type, 
            class_weight=class_weight, **model_params
        )
        training_info = {
            'model_type': model_type,
            'class_weight': class_weight,
            'grid_search_used': False,
            'model_params': model_params
        }
    
    return model, training_info


def compare_class_weight_approaches(
    X_train, X_test, y_train, y_test,
    model_type: str = "logistic",
    approaches: list = None
) -> Dict[str, Dict]:
    """
    Compare different class weighting approaches.
    
    Args:
        X_train, X_test, y_train, y_test: Split data
        model_type: Type of model to train
        approaches: List of approaches to compare
        
    Returns:
        Dictionary of results for each approach
    """
    if approaches is None:
        approaches = [
            {'name': 'unweighted', 'class_weight': None},
            {'name': 'balanced', 'class_weight': 'balanced'},
            {'name': 'manual_1_5', 'class_weight': {0: 1, 1: 5}},
            {'name': 'manual_1_10', 'class_weight': {0: 1, 1: 10}},
            {'name': 'manual_1_20', 'class_weight': {0: 1, 1: 20}}
        ]
    
    results = {}
    
    for approach in approaches:
        logger.info(f"Training with approach: {approach['name']}")
        
        # Train model
        model, training_info = train_model_with_class_weights(
            X_train, y_train,
            model_type=model_type,
            class_weight=approach['class_weight']
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[approach['name']] = {
            'model': model,
            'training_info': training_info,
            'predictions': y_pred,
            'probabilities': y_proba,
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'minority_recall': report['1']['recall'],  # Assuming binary classification
            'minority_precision': report['1']['precision'],
            'minority_f1': report['1']['f1-score']
        }
    
    return results


def train_with_cross_validation(
    X, y,
    model_type: str = "logistic",
    class_weight: str = "balanced",
    cv_folds: int = 5,
    test_size: float = 0.2
) -> Tuple[Any, Dict]:
    """
    Train model with cross-validation and final evaluation.
    
    Args:
        X, y: Full dataset
        model_type: Type of model
        class_weight: Class weighting strategy
        cv_folds: Number of CV folds
        test_size: Test set size
        
    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Cross-validation
    cv_scores = cross_validate_with_weights(
        X_train, y_train, model_type=model_type, 
        class_weight=class_weight, cv_folds=cv_folds
    )
    
    # Train final model on full training data
    model, training_info = train_model_with_class_weights(
        X_train, y_train, model_type=model_type, class_weight=class_weight
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    test_report = classification_report(y_test, y_pred, output_dict=True)
    test_cm = confusion_matrix(y_test, y_pred)
    
    evaluation_results = {
        'cv_scores': cv_scores,
        'test_report': test_report,
        'test_confusion_matrix': test_cm,
        'training_info': training_info,
        'test_accuracy': test_report['accuracy'],
        'test_minority_recall': test_report['1']['recall'],
        'test_minority_precision': test_report['1']['precision'],
        'test_minority_f1': test_report['1']['f1-score']
    }
    
    return model, evaluation_results


def main_training_pipeline(
    synthetic: bool = True,
    model_type: str = "logistic",
    class_weight: str = "balanced",
    use_grid_search: bool = False,
    compare_approaches: bool = True,
    save_models: bool = True
) -> Dict:
    """
    Complete training pipeline with class weights.
    
    Args:
        synthetic: Whether to use synthetic data
        model_type: Type of model to train
        class_weight: Class weighting strategy
        use_grid_search: Whether to use grid search
        compare_approaches: Whether to compare different approaches
        save_models: Whether to save models
        
    Returns:
        Dictionary with all results
    """
    logger.info("Starting class weights training pipeline...")
    
    # Step 1: Load and preprocess data
    X, y = load_data(synthetic=synthetic)
    X_clean, y_clean = clean_data(X, y)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
    
    # Step 3: Feature engineering
    preprocessor, X_train_transformed = fit_preprocessor(X_train.values, y_train.values)
    X_test_transformed = transform_features(preprocessor, X_test.values)
    
    # Step 4: Analyze class distribution
    from src.class_weights import analyze_class_distribution
    distribution_analysis = analyze_class_distribution(y_clean.values)
    
    results = {
        'distribution_analysis': distribution_analysis,
        'preprocessor': preprocessor
    }
    
    if compare_approaches:
        # Step 5a: Compare different approaches
        logger.info("Comparing different class weight approaches...")
        comparison_results = compare_class_weight_approaches(
            X_train_transformed, X_test_transformed,
            y_train.values, y_test.values,
            model_type=model_type
        )
        results['comparison_results'] = comparison_results
        
        # Find best approach based on minority F1
        best_approach = max(comparison_results.keys(), 
                          key=lambda k: comparison_results[k]['minority_f1'])
        logger.info(f"Best approach: {best_approach}")
        
    else:
        # Step 5b: Train single model
        logger.info(f"Training single model with class_weight={class_weight}...")
        model, training_info = train_model_with_class_weights(
            X_train_transformed, y_train.values,
            model_type=model_type, class_weight=class_weight,
            use_grid_search=use_grid_search
        )
        
        # Evaluate
        y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)[:, 1]
        
        test_report = classification_report(y_test.values, y_pred, output_dict=True)
        test_cm = confusion_matrix(y_test.values, y_pred)
        
        results['single_model'] = {
            'model': model,
            'training_info': training_info,
            'test_report': test_report,
            'test_confusion_matrix': test_cm,
            'test_accuracy': test_report['accuracy'],
            'test_minority_recall': test_report['1']['recall'],
            'test_minority_precision': test_report['1']['precision'],
            'test_minority_f1': test_report['1']['f1-score']
        }
        
        best_approach = 'single_model'
    
    # Step 6: Save models
    if save_models:
        logger.info("Saving models and artifacts...")
        
        # Create models directory
        import os
        os.makedirs("models", exist_ok=True)
        
        if compare_approaches:
            # Save best model from comparison
            best_model = comparison_results[best_approach]['model']
            joblib.dump(best_model, "models/best_class_weight_model.pkl")
            
            # Save all models
            for approach_name, approach_results in comparison_results.items():
                model_filename = f"models/model_{approach_name}.pkl"
                joblib.dump(approach_results['model'], model_filename)
        else:
            # Save single model
            joblib.dump(results['single_model']['model'], "models/class_weight_model.pkl")
        
        # Save preprocessor
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        
        # Save results
        import json
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'preprocessor':
                continue  # Skip preprocessor (not JSON serializable)
            elif key == 'comparison_results':
                serializable_results[key] = {}
                for approach, approach_results in value.items():
                    serializable_results[key][approach] = {
                        k: v for k, v in approach_results.items() 
                        if k not in ['model', 'predictions', 'probabilities', 'confusion_matrix']
                    }
                    # Convert confusion matrix to list
                    if 'confusion_matrix' in approach_results:
                        serializable_results[key][approach]['confusion_matrix'] = \
                            approach_results['confusion_matrix'].tolist()
            elif key == 'single_model':
                serializable_results[key] = {
                    k: v for k, v in value.items() 
                    if k not in ['model', 'test_confusion_matrix']
                }
                # Convert confusion matrix to list
                if 'test_confusion_matrix' in value:
                    serializable_results[key]['test_confusion_matrix'] = \
                        value['test_confusion_matrix'].tolist()
            else:
                serializable_results[key] = value
        
        with open("models/class_weights_training_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info("Class weights training pipeline completed!")
    
    # Print summary
    print("\n" + "="*80)
    print("CLASS WEIGHTS TRAINING SUMMARY")
    print("="*80)
    print(f"Dataset imbalance ratio: {distribution_analysis['imbalance_ratio']:.2f}")
    print(f"Model type: {model_type}")
    
    if compare_approaches:
        print(f"\nBest approach: {best_approach}")
        print(f"Best minority F1: {comparison_results[best_approach]['minority_f1']:.3f}")
        print(f"Best minority recall: {comparison_results[best_approach]['minority_recall']:.3f}")
        print(f"Best minority precision: {comparison_results[best_approach]['minority_precision']:.3f}")
        
        print(f"\nAll approaches comparison:")
        for approach, approach_results in comparison_results.items():
            print(f"  {approach:15}: F1={approach_results['minority_f1']:.3f}, "
                  f"Recall={approach_results['minority_recall']:.3f}, "
                  f"Prec={approach_results['minority_precision']:.3f}")
    else:
        print(f"\nSingle model results:")
        print(f"Minority F1: {results['single_model']['test_minority_f1']:.3f}")
        print(f"Minority recall: {results['single_model']['test_minority_recall']:.3f}")
        print(f"Minority precision: {results['single_model']['test_minority_precision']:.3f}")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run the complete training pipeline
    results = main_training_pipeline(
        synthetic=True,
        model_type="logistic",
        class_weight="balanced",
        use_grid_search=False,
        compare_approaches=True,
        save_models=True
    )
    
    print("\n✅ Class weights training completed successfully!")
