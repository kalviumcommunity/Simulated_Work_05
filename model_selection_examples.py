"""
Model Selection Examples and Worked Scenarios

This module demonstrates the model selection framework with real-world scenarios
including fraud detection, medical screening, spam filtering, and regression problems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import logging

from model_selection_framework import (
    ModelSelectionFramework, ProblemType, BusinessObjective, 
    DeploymentEnvironment, ModelConstraints, create_sample_scenario
)

logger = logging.getLogger(__name__)


def fraud_detection_scenario():
    """
    Scenario 1: Fraud Detection
    - High false negative cost (missed fraud = direct financial loss)
    - Real-time requirements (transaction time)
    - Moderate interpretability needed for audit
    """
    print("=" * 60)
    print("SCENARIO 1: FRAUD DETECTION")
    print("=" * 60)
    print("Business Context: Credit card transaction fraud detection")
    print("Cost Structure: FN cost >> FP cost (missed fraud = direct loss)")
    print("Requirements: Real-time (<100ms), auditable decisions")
    print()
    
    # Generate fraud-like data (highly imbalanced)
    X, y = make_classification(
        n_samples=5000, n_features=30, n_informative=25,
        n_redundant=5, weights=[0.98, 0.02], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )
    }
    
    # Setup framework for fraud detection
    constraints = ModelConstraints(
        max_latency_ms=100.0,  # Real-time transaction processing
        max_memory_mb=200.0,
        interpretable=True,  # Required for audit/regulation
        update_frequency="weekly"
    )
    
    framework = ModelSelectionFramework(
        problem_type=ProblemType.IMBALANCED_CLASSIFICATION,
        business_objective=BusinessObjective.MINIMIZE_FN_COST,
        deployment_environment=DeploymentEnvironment.REAL_TIME,
        constraints=constraints
    )
    
    print(f"Selected Primary Metric: {framework.selected_metrics['primary_metric']}")
    print(f"Secondary Metrics: {framework.selected_metrics['secondary_metrics']}")
    print()
    
    # Evaluate models
    results = framework.compare_models(
        models, X_train, y_train, X_test, y_test
    )
    
    # Create decision table
    decision_table = framework.create_decision_table(results)
    print("DECISION TABLE:")
    print(decision_table.to_string(index=False))
    print()
    
    # Get recommendation
    best_model, rationale = framework.recommend_model(results)
    print(rationale)
    print()
    
    # Show confusion matrix for best model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print("BEST MODEL CONFUSION MATRIX:")
    print(f"Predicted → 0    1")
    print(f"Actual 0 → {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"Actual 1 → {cm[1,0]:4d} {cm[1,1]:4d}")
    print()
    
    # Calculate business impact
    fn_count = cm[1,0]  # Missed fraud
    fp_count = cm[0,1]  # False alarms
    
    print("BUSINESS IMPACT ANALYSIS:")
    print(f"False Negatives (Missed Fraud): {fn_count}")
    print(f"False Positives (False Alarms): {fp_count}")
    
    # Assume average transaction value of $100
    avg_transaction_value = 100
    fn_cost = fn_count * avg_transaction_value
    fp_cost = fp_count * 5  # Customer friction cost
    
    print(f"Estimated FN Cost: ${fn_cost:,}")
    print(f"Estimated FP Cost: ${fp_cost:,}")
    print(f"Total Cost: ${fn_cost + fp_cost:,}")
    
    return framework, results


def medical_screening_scenario():
    """
    Scenario 2: Medical Disease Screening
    - Very high false negative cost (missed disease = life-threatening)
    - Interpretability required for clinical decisions
    - Batch processing acceptable
    """
    print("=" * 60)
    print("SCENARIO 2: MEDICAL DISEASE SCREENING")
    print("=" * 60)
    print("Business Context: Disease screening from patient data")
    print("Cost Structure: FN cost >> FP cost (missed diagnosis = preventable harm)")
    print("Requirements: High interpretability, batch processing acceptable")
    print()
    
    # Generate medical screening data (moderately imbalanced)
    X, y = make_classification(
        n_samples=3000, n_features=25, n_informative=20,
        n_redundant=5, weights=[0.85, 0.15], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # Define candidate models (focus on interpretable models)
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=50, max_depth=8, random_state=42
        ),
        "Linear SVM": SVC(
            class_weight="balanced", kernel="linear", probability=True, random_state=42
        )
    }
    
    # Setup framework for medical screening
    constraints = ModelConstraints(
        max_latency_ms=1000.0,  # Batch processing, not time-critical
        max_memory_mb=500.0,
        interpretable=True,  # Clinicians must understand decisions
        update_frequency="monthly"
    )
    
    framework = ModelSelectionFramework(
        problem_type=ProblemType.IMBALANCED_CLASSIFICATION,
        business_objective=BusinessObjective.MINIMIZE_FN_COST,
        deployment_environment=DeploymentEnvironment.BATCH,
        constraints=constraints
    )
    
    print(f"Selected Primary Metric: {framework.selected_metrics['primary_metric']}")
    print(f"Secondary Metrics: {framework.selected_metrics['secondary_metrics']}")
    print()
    
    # Evaluate models
    results = framework.compare_models(
        models, X_train, y_train, X_test, y_test
    )
    
    # Create decision table
    decision_table = framework.create_decision_table(results)
    print("DECISION TABLE:")
    print(decision_table.to_string(index=False))
    print()
    
    # Get recommendation
    best_model, rationale = framework.recommend_model(results)
    print(rationale)
    print()
    
    # Sensitivity analysis at different thresholds
    best_model.fit(X_train, y_train)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("THRESHOLD SENSITIVITY ANALYSIS:")
    print("Threshold  Recall  Precision  F1-Score")
    print("-" * 40)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        recall = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_test == 1)
        precision = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_pred_thresh == 1) if np.sum(y_pred_thresh == 1) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>8.1f}  {recall:>6.3f}  {precision:>8.3f}  {f1:>8.3f}")
    
    return framework, results


def spam_filtering_scenario():
    """
    Scenario 3: Email Spam Filtering
    - High false positive cost (blocking legitimate email)
    - Real-time requirements
    - Low latency critical for user experience
    """
    print("=" * 60)
    print("SCENARIO 3: EMAIL SPAM FILTERING")
    print("=" * 60)
    print("Business Context: Email spam detection")
    print("Cost Structure: FP cost >> FN cost (blocking legitimate email = user frustration)")
    print("Requirements: Very low latency (<10ms), high precision")
    print()
    
    # Generate spam-like data (moderately imbalanced)
    X, y = make_classification(
        n_samples=8000, n_features=50, n_informative=40,
        n_redundant=10, weights=[0.7, 0.3], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define candidate models (focus on low-latency models)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Linear SVM": SVC(
            kernel="linear", probability=True, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42
        )
    }
    
    # Setup framework for spam filtering
    constraints = ModelConstraints(
        max_latency_ms=10.0,  # Very low latency for user experience
        max_memory_mb=100.0,
        interpretable=False,  # Not required for spam
        update_frequency="daily"
    )
    
    framework = ModelSelectionFramework(
        problem_type=ProblemType.IMBALANCED_CLASSIFICATION,
        business_objective=BusinessObjective.MINIMIZE_FP_COST,
        deployment_environment=DeploymentEnvironment.REAL_TIME,
        constraints=constraints
    )
    
    print(f"Selected Primary Metric: {framework.selected_metrics['primary_metric']}")
    print(f"Secondary Metrics: {framework.selected_metrics['secondary_metrics']}")
    print()
    
    # Evaluate models
    results = framework.compare_models(
        models, X_train, y_train, X_test, y_test
    )
    
    # Create decision table
    decision_table = framework.create_decision_table(results)
    print("DECISION TABLE:")
    print(decision_table.to_string(index=False))
    print()
    
    # Get recommendation
    best_model, rationale = framework.recommend_model(results)
    print(rationale)
    print()
    
    # Show precision-focused metrics
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    
    print("PRECISION-FOCUSED PERFORMANCE:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"False Positive Rate: {(1-precision):.3f}")
    print(f"Expected false positives per 1000 emails: {(1-precision)*1000:.1f}")
    
    return framework, results


def house_price_prediction_scenario():
    """
    Scenario 4: House Price Prediction
    - Regression problem
    - Large errors are unacceptable (mortgage decisions)
    - Interpretability valuable for real estate agents
    """
    print("=" * 60)
    print("SCENARIO 4: HOUSE PRICE PREDICTION")
    print("=" * 60)
    print("Business Context: Real estate price estimation")
    print("Cost Structure: Large errors unacceptable (mortgage/valuation decisions)")
    print("Requirements: Good interpretability, batch processing")
    print()
    
    # Generate regression data
    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=15,
        noise=20, random_state=42
    )
    
    # Scale target to represent house prices ($100K - $1M)
    y = np.interp(y, (y.min(), y.max()), (100000, 1000000))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Define candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Convert GradientBoostingClassifier to Regressor equivalent
    from sklearn.ensemble import GradientBoostingRegressor
    models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
    
    # Setup framework for house price prediction
    constraints = ModelConstraints(
        max_latency_ms=500.0,  # Batch processing
        max_memory_mb=300.0,
        interpretable=True,  # Real estate agents need explanations
        update_frequency="monthly"
    )
    
    framework = ModelSelectionFramework(
        problem_type=ProblemType.REGRESSION,
        business_objective=BusinessObjective.MINIMIZE_FP_COST,  # Large errors
        deployment_environment=DeploymentEnvironment.BATCH,
        constraints=constraints
    )
    
    print(f"Selected Primary Metric: {framework.selected_metrics['primary_metric']}")
    print(f"Secondary Metrics: {framework.selected_metrics['secondary_metrics']}")
    print()
    
    # Evaluate models
    results = framework.compare_models(
        models, X_train, y_train, X_test, y_test
    )
    
    # Create decision table
    decision_table = framework.create_decision_table(results)
    print("DECISION TABLE:")
    print(decision_table.to_string(index=False))
    print()
    
    # Get recommendation
    best_model, rationale = framework.recommend_model(results)
    print(rationale)
    print()
    
    # Show error distribution
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    errors = y_test - y_pred
    
    print("ERROR ANALYSIS:")
    print(f"Mean Absolute Error: ${np.mean(np.abs(errors)):,.0f}")
    print(f"Root Mean Squared Error: ${np.sqrt(np.mean(errors**2)):,.0f}")
    print(f"Max Error: ${np.max(np.abs(errors)):,.0f}")
    print(f"95th Percentile Error: ${np.percentile(np.abs(errors), 95):,.0f}")
    print()
    
    # Business impact
    print("BUSINESS IMPACT:")
    print(f"Average prediction error: ${np.mean(np.abs(errors)):,.0f}")
    print(f"Large errors (> $100K): {np.sum(np.abs(errors) > 100000)} out of {len(errors)}")
    print(f"Percentage of large errors: {np.sum(np.abs(errors) > 100000) / len(errors) * 100:.1f}%")
    
    return framework, results


def compare_all_scenarios():
    """
    Compare model selection across different business scenarios.
    """
    print("=" * 60)
    print("CROSS-SCENARIO COMPARISON")
    print("=" * 60)
    print("Comparing how business objectives drive model selection...")
    print()
    
    scenarios = {
        "Fraud Detection": {
            "objective": BusinessObjective.MINIMIZE_FN_COST,
            "environment": DeploymentEnvironment.REAL_TIME,
            "primary_metric": "recall"
        },
        "Medical Screening": {
            "objective": BusinessObjective.MINIMIZE_FN_COST,
            "environment": DeploymentEnvironment.BATCH,
            "primary_metric": "recall"
        },
        "Spam Filtering": {
            "objective": BusinessObjective.MINIMIZE_FP_COST,
            "environment": DeploymentEnvironment.REAL_TIME,
            "primary_metric": "precision"
        },
        "House Prices": {
            "objective": BusinessObjective.MINIMIZE_FP_COST,
            "environment": DeploymentEnvironment.BATCH,
            "primary_metric": "rmse"
        }
    }
    
    comparison_data = []
    
    for scenario_name, config in scenarios.items():
        comparison_data.append({
            "Scenario": scenario_name,
            "Business Objective": config["objective"].value,
            "Deployment": config["environment"].value,
            "Primary Metric": config["primary_metric"],
            "Key Constraint": "Low Latency" if config["environment"] == DeploymentEnvironment.REAL_TIME else "Interpretability"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    print("KEY INSIGHTS:")
    print("1. Same business objective (minimize FN) can lead to different models based on deployment constraints")
    print("2. Real-time requirements favor simpler, faster models even if slightly less accurate")
    print("3. Interpretability requirements can override performance advantages")
    print("4. Cost structure directly determines the primary evaluation metric")
    print("5. No single 'best' model exists - optimal choice depends on full business context")


def threshold_tuning_demonstration():
    """
    Demonstrate how threshold tuning can change model selection outcomes.
    """
    print("=" * 60)
    print("THRESHOLD TUNING IMPACT DEMONSTRATION")
    print("=" * 60)
    print("Showing how optimal thresholds can change which model wins...")
    print()
    
    # Generate imbalanced data
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=12,
        n_redundant=3, weights=[0.8, 0.2], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # Two models with similar F1 at default threshold
    models = {
        "Model A (High Recall)": LogisticRegression(
            class_weight={0: 1, 1: 3}, max_iter=1000, random_state=42
        ),
        "Model B (Balanced)": RandomForestClassifier(
            class_weight="balanced", n_estimators=50, random_state=42
        )
    }
    
    # Evaluate at default threshold
    print("PERFORMANCE AT DEFAULT THRESHOLD (0.5):")
    print("-" * 50)
    
    default_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        default_results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "probabilities": y_proba
        }
        
        print(f"{name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print()
    
    # Find optimal thresholds for recall optimization
    print("PERFORMANCE AT OPTIMIZED THRESHOLDS (Recall Focus):")
    print("-" * 50)
    
    optimized_results = {}
    for name, result in default_results.items():
        y_proba = result["probabilities"]
        
        # Find threshold that maximizes recall
        thresholds = np.arange(0.1, 0.8, 0.05)
        best_recall = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            precision = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_pred_thresh == 1) if np.sum(y_pred_thresh == 1) > 0 else 0
            recall = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_test == 1)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
                best_metrics = {"precision": precision, "recall": recall, "f1": f1}
        
        optimized_results[name] = {
            "threshold": best_threshold,
            "metrics": best_metrics
        }
        
        print(f"{name} (optimal threshold = {best_threshold:.2f}):")
        print(f"  Precision: {best_metrics['precision']:.3f}")
        print(f"  Recall: {best_metrics['recall']:.3f}")
        print(f"  F1: {best_metrics['f1']:.3f}")
        print()
    
    # Show how the winner changes
    default_f1_winner = max(default_results.keys(), key=lambda k: default_results[k]["f1"])
    optimized_recall_winner = max(optimized_results.keys(), key=lambda k: optimized_results[k]["metrics"]["recall"])
    
    print("THRESHOLD TUNING IMPACT:")
    print(f"Default threshold F1 winner: {default_f1_winner}")
    print(f"Optimized threshold recall winner: {optimized_recall_winner}")
    
    if default_f1_winner != optimized_recall_winner:
        print("→ Threshold tuning CHANGED the winning model!")
    else:
        print("→ Same model wins, but performance gap changed")


def stability_analysis_demonstration():
    """
    Demonstrate why CV stability matters as much as mean performance.
    """
    print("=" * 60)
    print("STABILITY ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("Showing why high variance models are risky for production...")
    print()
    
    # Simulate two models with same mean but different variance
    np.random.seed(42)
    
    # Model A: High mean, high variance
    model_a_scores = np.random.normal(0.88, 0.08, 100)  # Mean=0.88, Std=0.08
    model_a_scores = np.clip(model_a_scores, 0, 1)
    
    # Model B: Slightly lower mean, low variance
    model_b_scores = np.random.normal(0.86, 0.01, 100)  # Mean=0.86, Std=0.01
    model_b_scores = np.clip(model_b_scores, 0, 1)
    
    print("MODEL COMPARISON:")
    print("-" * 30)
    print(f"Model A: Mean = {model_a_scores.mean():.3f}, Std = {model_a_scores.std():.3f}")
    print(f"Model B: Mean = {model_b_scores.mean():.3f}, Std = {model_b_scores.std():.3f}")
    print()
    
    # Performance ranges
    print("EXPECTED PERFORMANCE RANGES (95% confidence):")
    print("-" * 50)
    a_lower, a_upper = np.percentile(model_a_scores, [2.5, 97.5])
    b_lower, b_upper = np.percentile(model_b_scores, [2.5, 97.5])
    
    print(f"Model A: [{a_lower:.3f}, {a_upper:.3f}]")
    print(f"Model B: [{b_lower:.3f}, {b_upper:.3f}]")
    print()
    
    # Risk analysis
    print("PRODUCTION RISK ANALYSIS:")
    print("-" * 30)
    
    # Probability of poor performance (< 0.80)
    a_poor_prob = np.mean(model_a_scores < 0.80)
    b_poor_prob = np.mean(model_b_scores < 0.80)
    
    print(f"Probability of poor performance (< 0.80):")
    print(f"  Model A: {a_poor_prob:.1%}")
    print(f"  Model B: {b_poor_prob:.1%}")
    print()
    
    # Worst-case scenarios
    print("WORST-CASE SCENARIOS:")
    print("-" * 25)
    print(f"Model A worst 5%: {np.percentile(model_a_scores, 5):.3f}")
    print(f"Model B worst 5%: {np.percentile(model_b_scores, 5):.3f}")
    print()
    
    # Business recommendation
    print("BUSINESS RECOMMENDATION:")
    if a_poor_prob > 0.1:
        print("→ Model A has significant risk of poor performance")
        print("→ Model B is the safer choice for production")
        print("→ The 0.02 mean advantage is not worth the risk")
    else:
        print("→ Both models have acceptable risk profiles")
        print("→ Choose based on other factors (latency, interpretability, etc.)")


def run_all_examples():
    """Run all model selection examples."""
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for examples
    
    print("MODEL SELECTION FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    print()
    
    # Run all scenarios
    fraud_detection_scenario()
    print("\n" + "="*60 + "\n")
    
    medical_screening_scenario()
    print("\n" + "="*60 + "\n")
    
    spam_filtering_scenario()
    print("\n" + "="*60 + "\n")
    
    house_price_prediction_scenario()
    print("\n" + "="*60 + "\n")
    
    threshold_tuning_demonstration()
    print("\n" + "="*60 + "\n")
    
    stability_analysis_demonstration()
    print("\n" + "="*60 + "\n")
    
    compare_all_scenarios()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Business objectives drive metric selection")
    print("2. Deployment constraints can override performance advantages")
    print("3. Stability matters as much as mean performance")
    print("4. Threshold tuning can change which model wins")
    print("5. The 'best' model depends on the full business context")


if __name__ == "__main__":
    run_all_examples()
