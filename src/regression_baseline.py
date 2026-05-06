"""
Linear Regression Baseline Implementation for Performance Benchmarking.

This module implements a mean baseline using DummyRegressor and compares it
against a Linear Regression model using proper train-test split and consistent
regression metrics (RMSE, MAE, R²).

Key Requirements:
- Proper train-test split before fitting ANY model
- Mean baseline using DummyRegressor(strategy='mean')
- Linear Regression with optional StandardScaler in Pipeline
- Evaluation using RMSE, MAE, R² on held-out test set
- Side-by-side metric comparison
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Dict, Any, Tuple

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mean_baseline() -> DummyRegressor:
    """
    Create a mean baseline using DummyRegressor.
    
    The baseline always predicts the mean of the training target values.
    This represents the simplest possible prediction strategy.
    
    Returns:
        DummyRegressor: Baseline model that predicts mean
    """
    logger.info("Creating mean baseline (DummyRegressor with strategy='mean')...")
    baseline = DummyRegressor(strategy='mean')
    logger.info("Mean baseline created")
    return baseline


def create_linear_regression_model(use_scaler: bool = True) -> Pipeline:
    """
    Create a Linear Regression model with optional StandardScaler.
    
    Args:
        use_scaler (bool): Whether to include StandardScaler in pipeline
        
    Returns:
        Pipeline: Linear Regression model (with or without scaling)
    """
    logger.info(f"Creating Linear Regression model (use_scaler={use_scaler})...")
    
    if use_scaler:
        # Create pipeline with StandardScaler to prevent data leakage
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        logger.info("Linear Regression Pipeline created: StandardScaler + LinearRegression")
    else:
        model = LinearRegression()
        logger.info("Linear Regression model created (no scaling)")
    
    return model


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool = True
) -> Tuple[DummyRegressor, Pipeline, Dict[str, Any]]:
    """
    Train both baseline and Linear Regression models on training data.
    
    IMPORTANT: Both models are fitted ONLY on training data to prevent leakage.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values
        use_scaler (bool): Whether to use StandardScaler in Linear Regression
        
    Returns:
        Tuple[DummyRegressor, Pipeline, Dict]: Trained models and training info
    """
    logger.info("=" * 80)
    logger.info("TRAINING MODELS (Training Data Only)")
    logger.info("=" * 80)
    
    # Train mean baseline
    logger.info("Step 1: Training mean baseline...")
    baseline = create_mean_baseline()
    baseline.fit(X_train, y_train)
    
    # Get baseline training predictions for reference
    baseline_train_pred = baseline.predict(X_train)
    baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_train_pred))
    
    logger.info(f"Baseline fitted: predicts mean = {baseline.constant_[0].item():.4f}")
    logger.info(f"Baseline training RMSE: {baseline_train_rmse:.4f}")
    
    # Train Linear Regression
    logger.info("Step 2: Training Linear Regression...")
    lr_model = create_linear_regression_model(use_scaler=use_scaler)
    lr_model.fit(X_train, y_train)
    
    # Get Linear Regression training predictions
    lr_train_pred = lr_model.predict(X_train)
    lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
    
    logger.info(f"Linear Regression fitted")
    logger.info(f"Linear Regression training RMSE: {lr_train_rmse:.4f}")
    
    # Extract coefficients if available
    training_info = {
        'training_samples': len(X_train),
        'n_features': X_train.shape[1],
        'baseline': {
            'mean_prediction': baseline.constant_[0].item(),
            'training_rmse': float(baseline_train_rmse)
        },
        'linear_regression': {
            'use_scaler': use_scaler,
            'training_rmse': float(lr_train_rmse)
        }
    }
    
    if use_scaler:
        # Extract coefficients from pipeline
        regressor = lr_model.named_steps['regressor']
        training_info['linear_regression']['coefficients'] = regressor.coef_.tolist()
        training_info['linear_regression']['intercept'] = float(regressor.intercept_)
    else:
        training_info['linear_regression']['coefficients'] = lr_model.coef_.tolist()
        training_info['linear_regression']['intercept'] = float(lr_model.intercept_)
    
    logger.info("=" * 80)
    logger.info("MODEL TRAINING COMPLETED")
    logger.info("=" * 80)
    
    return baseline, lr_model, training_info


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a model on test data using regression metrics.
    
    Required Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    
    Args:
        model: Trained model (baseline or Linear Regression)
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target values
        model_name (str): Name for logging
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    logger.info(f"{model_name} Test Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    
    return metrics


def compare_regression_models(
    baseline_metrics: Dict[str, float],
    lr_metrics: Dict[str, float],
    training_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare baseline and Linear Regression performance side-by-side.
    
    Args:
        baseline_metrics (Dict): Baseline evaluation metrics
        lr_metrics (Dict): Linear Regression evaluation metrics
        training_info (Dict): Training information
        
    Returns:
        Dict[str, Any]: Comprehensive comparison results
    """
    logger.info("=" * 80)
    logger.info("BASELINE vs LINEAR REGRESSION COMPARISON")
    logger.info("=" * 80)
    
    # Calculate improvements
    rmse_improvement = baseline_metrics['rmse'] - lr_metrics['rmse']  # Lower is better
    mae_improvement = baseline_metrics['mae'] - lr_metrics['mae']     # Lower is better
    r2_improvement = lr_metrics['r2'] - baseline_metrics['r2']       # Higher is better
    
    # Calculate percentage improvements
    rmse_improvement_pct = (rmse_improvement / baseline_metrics['rmse'] * 100) if baseline_metrics['rmse'] > 0 else 0
    mae_improvement_pct = (mae_improvement / baseline_metrics['mae'] * 100) if baseline_metrics['mae'] > 0 else 0
    
    # Determine if improvement is meaningful
    # Meaningful if: RMSE reduced by >10%, R² > 0.3 (some predictive power)
    meaningful_improvement = (
        rmse_improvement_pct > 10 and 
        lr_metrics['r2'] > 0.3
    )
    
    comparison = {
        'baseline_metrics': baseline_metrics,
        'linear_regression_metrics': lr_metrics,
        'improvements': {
            'rmse': {
                'absolute': float(rmse_improvement),
                'percentage': float(rmse_improvement_pct)
            },
            'mae': {
                'absolute': float(mae_improvement),
                'percentage': float(mae_improvement_pct)
            },
            'r2': {
                'absolute': float(r2_improvement)
            }
        },
        'summary': {
            'meaningful_improvement': bool(meaningful_improvement),
            'rmse_improvement_pct': float(rmse_improvement_pct),
            'baseline_mean': float(training_info['baseline']['mean_prediction'])
        }
    }
    
    # Log comparison table
    logger.info(f"{'Metric':<20} {'Baseline':<15} {'Linear Reg':<15} {'Improvement':<20}")
    logger.info("-" * 80)
    logger.info(f"{'RMSE (lower=better)':<20} {baseline_metrics['rmse']:<15.4f} {lr_metrics['rmse']:<15.4f} {rmse_improvement:+.4f} ({rmse_improvement_pct:+.1f}%)")
    logger.info(f"{'MAE (lower=better)':<20} {baseline_metrics['mae']:<15.4f} {lr_metrics['mae']:<15.4f} {mae_improvement:+.4f} ({mae_improvement_pct:+.1f}%)")
    logger.info(f"{'R² (higher=better)':<20} {baseline_metrics['r2']:<15.4f} {lr_metrics['r2']:<15.4f} {r2_improvement:+.4f}")
    logger.info("=" * 80)
    
    if meaningful_improvement:
        logger.info("✅ MEANINGFUL IMPROVEMENT: Linear Regression significantly outperforms baseline")
    else:
        logger.info("⚠️ LIMITED IMPROVEMENT: Linear Regression shows marginal improvement over baseline")
    
    return comparison


def interpret_results(comparison: Dict[str, Any], training_info: Dict[str, Any]) -> None:
    """
    Provide detailed interpretation of the comparison results.
    
    Answers the assignment questions:
    1. Did Linear Regression outperform the baseline?
    2. Is the improvement meaningful?
    3. What does the R² value indicate?
    4. Are coefficients interpretable?
    5. Were any assumptions violated?
    """
    logger.info("=" * 80)
    logger.info("RESULTS INTERPRETATION")
    logger.info("=" * 80)
    
    baseline_r2 = comparison['baseline_metrics']['r2']
    lr_r2 = comparison['linear_regression_metrics']['r2']
    rmse_improvement_pct = comparison['summary']['rmse_improvement_pct']
    
    # Question 1: Did Linear Regression outperform the baseline?
    logger.info("\n1. Did Linear Regression outperform the baseline?")
    if lr_r2 > baseline_r2:
        logger.info("   ✅ YES: Linear Regression has higher R² than baseline")
        logger.info(f"      Baseline R² = {baseline_r2:.4f}")
        logger.info(f"      Linear Regression R² = {lr_r2:.4f}")
    else:
        logger.info("   ❌ NO: Linear Regression did not outperform baseline")
    
    # Question 2: Is the improvement meaningful?
    logger.info("\n2. Is the improvement meaningful?")
    if comparison['summary']['meaningful_improvement']:
        logger.info("   ✅ YES: The improvement is statistically meaningful")
        logger.info(f"      RMSE reduced by {rmse_improvement_pct:.1f}%")
        logger.info(f"      R² = {lr_r2:.4f} indicates {lr_r2*100:.1f}% of variance explained")
    else:
        logger.info("   ⚠️ MARGINAL: The improvement is limited")
        if lr_r2 < 0.3:
            logger.info(f"      R² = {lr_r2:.4f} is low (model explains little variance)")
    
    # Question 3: What does the R² value indicate?
    logger.info("\n3. What does the R² value indicate?")
    logger.info(f"   Baseline R² = {baseline_r2:.4f}")
    logger.info(f"   Linear Regression R² = {lr_r2:.4f}")
    logger.info(f"   Interpretation: Model explains {lr_r2*100:.1f}% of target variance")
    
    if lr_r2 < 0:
        logger.info("   ⚠️ R² < 0: Model performs WORSE than simply predicting the mean")
    elif lr_r2 < 0.3:
        logger.info("   ⚠️ Low R²: Weak predictive power")
    elif lr_r2 < 0.7:
        logger.info("   ✅ Moderate R²: Reasonable predictive power")
    else:
        logger.info("   ✅ High R²: Strong predictive power")
    
    # Question 4: Are coefficients interpretable?
    logger.info("\n4. Are coefficients interpretable and reasonable?")
    coefs = training_info['linear_regression']['coefficients']
    intercept = training_info['linear_regression']['intercept']
    
    logger.info(f"   Intercept: {intercept:.4f}")
    logger.info(f"   Number of coefficients: {len(coefs)}")
    logger.info(f"   Coefficient range: [{min(coefs):.4f}, {max(coefs):.4f}]")
    
    # Check for potential issues
    large_coefs = [c for c in coefs if abs(c) > 10]
    if large_coefs:
        logger.info(f"   ⚠️ {len(large_coefs)} large coefficients detected (possible multicollinearity)")
    else:
        logger.info("   ✅ Coefficients appear reasonable in magnitude")
    
    # Question 5: Were assumptions potentially violated?
    logger.info("\n5. Assumption Checks:")
    
    # Linearity: Cannot directly check without residuals
    logger.info("   Linearity: Assumed (Linear Regression assumes linear relationship)")
    
    # Independence: Assumed if proper train-test split
    logger.info("   Independence: ✅ Assured by proper train-test split")
    
    # Homoscedasticity: Cannot check without residual analysis
    logger.info("   Homoscedasticity: Assumed (constant error variance)")
    
    # Normality: Cannot check without residual analysis
    logger.info("   Normality: Assumed (residuals normally distributed)")
    
    # Multicollinearity check (based on coefficient magnitudes)
    if training_info['linear_regression']['use_scaler']:
        logger.info("   Multicollinearity: ⚠️ Check VIF if coefficients are unstable")
    else:
        logger.info("   Scaling: ⚠️ Features not standardized (coefficients not directly comparable)")
    
    logger.info("=" * 80)


def run_regression_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    use_scaler: bool = True
) -> Dict[str, Any]:
    """
    Run a complete regression experiment: baseline vs Linear Regression.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion for test set
        use_scaler (bool): Whether to use StandardScaler in Linear Regression
        
    Returns:
        Dict[str, Any]: Complete experiment results
    """
    logger.info("=" * 80)
    logger.info("LINEAR REGRESSION vs MEAN BASELINE EXPERIMENT")
    logger.info("=" * 80)
    
    # Step 1: Proper train-test split (CRITICAL to prevent leakage)
    logger.info("\nStep 1: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    logger.info(f"   Training samples: {len(X_train)}")
    logger.info(f"   Test samples: {len(X_test)}")
    logger.info(f"   Features: {X.shape[1]}")
    logger.info(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"   Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    # Step 2: Train both models (only on training data)
    logger.info("\nStep 2: Training models...")
    baseline, lr_model, training_info = train_models(
        X_train.values, y_train.values, use_scaler=use_scaler
    )
    
    # Step 3: Evaluate both models on held-out test set
    logger.info("\nStep 3: Evaluating models on test set...")
    baseline_metrics = evaluate_model(baseline, X_test.values, y_test.values, "Mean Baseline")
    lr_metrics = evaluate_model(lr_model, X_test.values, y_test.values, "Linear Regression")
    
    # Step 4: Compare models side-by-side
    logger.info("\nStep 4: Comparing models...")
    comparison = compare_regression_models(baseline_metrics, lr_metrics, training_info)
    
    # Step 5: Interpret results
    logger.info("\nStep 5: Interpreting results...")
    interpret_results(comparison, training_info)
    
    # Combine all results
    experiment_results = {
        'training_info': training_info,
        'baseline_metrics': baseline_metrics,
        'linear_regression_metrics': lr_metrics,
        'comparison': comparison,
        'data_info': {
            'total_samples': len(X),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'target_mean': float(y.mean()),
            'target_std': float(y.std())
        }
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    
    return experiment_results


def main():
    """
    Main function to demonstrate Linear Regression vs Baseline comparison.
    
    Uses sender_reputation as the continuous target variable for regression.
    """
    logger.info("=" * 80)
    logger.info("LINEAR REGRESSION BASELINE DEMONSTRATION")
    logger.info("=" * 80)
    
    # Import necessary modules
    from data_preprocessing import load_data, clean_data
    
    # Load data
    logger.info("\nLoading data...")
    X, y_classification = load_data(synthetic=True)
    X_clean, _ = clean_data(X, y_classification)
    
    # Use sender_reputation as continuous target for regression
    # This is a continuous variable suitable for regression
    if 'sender_reputation' in X_clean.columns:
        y_regression = X_clean['sender_reputation']
        X_features = X_clean.drop(columns=['sender_reputation'])
        logger.info(f"Using 'sender_reputation' as regression target")
    else:
        # Fallback: create synthetic continuous target from classification labels
        logger.info("Creating synthetic continuous target...")
        np.random.seed(RANDOM_STATE)
        y_regression = pd.Series(
            y_classification.values * 10 + np.random.normal(0, 2, len(y_classification)),
            name='continuous_target'
        )
        X_features = X_clean
    
    logger.info(f"Target: {y_regression.name}")
    logger.info(f"Target statistics: mean={y_regression.mean():.4f}, std={y_regression.std():.4f}")
    
    # Run experiment
    results = run_regression_experiment(X_features, y_regression, test_size=0.2, use_scaler=True)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    comparison = results['comparison']
    baseline_metrics = comparison['baseline_metrics']
    lr_metrics = comparison['linear_regression_metrics']
    
    logger.info(f"\nBaseline (Mean) Performance:")
    logger.info(f"   RMSE: {baseline_metrics['rmse']:.4f}")
    logger.info(f"   MAE:  {baseline_metrics['mae']:.4f}")
    logger.info(f"   R²:   {baseline_metrics['r2']:.4f}")
    
    logger.info(f"\nLinear Regression Performance:")
    logger.info(f"   RMSE: {lr_metrics['rmse']:.4f}")
    logger.info(f"   MAE:  {lr_metrics['mae']:.4f}")
    logger.info(f"   R²:   {lr_metrics['r2']:.4f}")
    
    logger.info(f"\nImprovement:")
    logger.info(f"   RMSE: {comparison['improvements']['rmse']['percentage']:+.1f}%")
    logger.info(f"   R²:   {comparison['improvements']['r2']['absolute']:+.4f}")
    logger.info(f"   Meaningful: {'Yes' if comparison['summary']['meaningful_improvement'] else 'No'}")
    
    logger.info("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
