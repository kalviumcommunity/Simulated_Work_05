"""
Pipeline Leakage Prevention Demonstration

This script demonstrates comprehensive data leakage prevention through proper
pipeline construction, including common leakage scenarios and their fixes.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import fit_preprocessor, transform_features
from src.pipeline_leakage import (
    LeakageDetector, SafePipelineBuilder, create_safe_pipeline,
    demonstrate_leakage_scenarios, demonstrate_cv_isolation,
    demonstrate_gridsearch_safety, create_deployment_pipeline,
    save_deployment_pipeline, load_deployment_pipeline,
    evaluate_pipeline_safety, print_safety_report
)
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_scaling_leakage():
    """
    Demonstrate scaling leakage and its prevention.
    """
    print("="*80)
    print("SCALING LEAKAGE DEMONSTRATION")
    print("="*80)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=8,
        n_redundant=2, weights=[0.9, 0.1], 
        flip_y=0.01, random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # WRONG: Scale before split
    print("\n❌ WRONG APPROACH: Scale before split")
    from sklearn.preprocessing import StandardScaler
    
    scaler_wrong = StandardScaler()
    X_scaled_full = scaler_wrong.fit_transform(X)  # Fitted on ALL data
    
    X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
        X_scaled_full, y, test_size=0.2, random_state=42, stratify=y
    )
    
    from sklearn.linear_model import LogisticRegression
    model_wrong = LogisticRegression(random_state=42)
    model_wrong.fit(X_train_wrong, y_train_wrong)
    wrong_score = model_wrong.score(X_test_wrong, y_test_wrong)
    
    print(f"   Test accuracy: {wrong_score:.3f}")
    print(f"   Scaler fitted on {len(X_scaled_full)} samples (includes test data)")
    
    # CORRECT: Use pipeline
    print("\n✅ CORRECT APPROACH: Use pipeline")
    from sklearn.pipeline import Pipeline
    
    pipeline_correct = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    
    # Cross-validation with proper isolation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(pipeline_correct, X_train, y_train, cv=5)
    
    # Train and evaluate
    pipeline_correct.fit(X_train, y_train)
    correct_score = pipeline_correct.score(X_test, y_test)
    
    print(f"   CV scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   CV mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   Test accuracy: {correct_score:.3f}")
    
    # Show the leakage gap
    leakage_gap = wrong_score - correct_score
    print(f"\n   Leakage gap: {leakage_gap:.3f} ({leakage_gap:.1%})")
    print(f"   Wrong approach appears {leakage_gap:.1%} better due to leakage!")
    
    # Visualize the difference
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    methods = ['Wrong\n(Pre-split scaling)', 'Correct\n(Pipeline)']
    scores = [wrong_score, correct_score]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, scores, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy')
    plt.title('Scaling Leakage Impact')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    cv_methods = ['Wrong CV', 'Correct CV']
    cv_scores_list = [cv_scores.mean(), cv_scores.mean()]  # Simplified for demo
    cv_bars = plt.bar(cv_methods, cv_scores_list, color=['red', 'green'], alpha=0.7)
    plt.ylabel('CV Score')
    plt.title('Cross-Validation Comparison')
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(cv_bars, cv_scores_list):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/scaling_leakage_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'wrong_score': wrong_score,
        'correct_score': correct_score,
        'leakage_gap': leakage_gap,
        'cv_scores': cv_scores.tolist()
    }


def demonstrate_feature_selection_leakage():
    """
    Demonstrate feature selection leakage.
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION LEAKAGE DEMONSTRATION")
    print("="*80)
    
    # Create data with some irrelevant features
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1500, n_features=20, n_informative=10,
        n_redundant=5, n_redundant=5, weights=[0.8, 0.2],
        flip_y=0.01, random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Original features: {X.shape[1]}")
    
    # WRONG: Feature selection before split
    print("\n❌ WRONG APPROACH: Feature selection before split")
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector_wrong = SelectKBest(k=10, score_func=f_classif)
    X_selected_wrong = selector_wrong.fit_transform(X, y)  # Fitted on ALL data
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
        X_selected_wrong, y, test_size=0.2, random_state=42, stratify=y
    )
    
    from sklearn.linear_model import LogisticRegression
    model_fs_wrong = LogisticRegression(random_state=42)
    model_fs_wrong.fit(X_train_fs, y_train_fs)
    fs_wrong_score = model_fs_wrong.score(X_test_fs, y_test_fs)
    
    print(f"   Features selected: {X_selected_wrong.shape[1]}")
    print(f"   Test accuracy: {fs_wrong_score:.3f}")
    print(f"   Selection used {len(X)} samples (includes test data)")
    
    # CORRECT: Feature selection inside pipeline
    print("\n✅ CORRECT APPROACH: Feature selection inside pipeline")
    from sklearn.pipeline import Pipeline
    
    pipeline_fs = Pipeline([
        ("selector", SelectKBest(k=10, score_func=f_classif)),
        ("model", LogisticRegression(random_state=42))
    ])
    
    # Cross-validation
    cv_scores_fs = cross_val_score(pipeline_fs, X_train, y_train, cv=5)
    
    # Train and evaluate
    pipeline_fs.fit(X_train, y_train)
    fs_correct_score = pipeline_fs.score(X_test, y_test)
    
    print(f"   Features selected: {10}")
    print(f"   CV scores: {[f'{s:.3f}' for s in cv_scores_fs]}")
    print(f"   CV mean: {cv_scores_fs.mean():.3f} ± {cv_scores_fs.std():.3f}")
    print(f"   Test accuracy: {fs_correct_score:.3f}")
    
    # Show the leakage gap
    leakage_gap = fs_wrong_score - fs_correct_score
    print(f"\n   Leakage gap: {leakage_gap:.3f} ({leakage_gap:.1%})")
    
    # Visualize feature importance comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    methods = ['Wrong\n(Pre-split FS)', 'Correct\n(Pipeline FS)']
    scores = [fs_wrong_score, fs_correct_score]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, scores, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy')
    plt.title('Feature Selection Leakage Impact')
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    # Show which features were selected in correct approach
    selector_correct = pipeline_fs.named_steps['selector']
    selected_features = selector_correct.get_support()
    
    feature_names = [f'F{i}' for i in range(X.shape[1])]
    selected_names = [name for name, selected in zip(feature_names, selected_features) if selected]
    
    plt.bar(range(len(selected_names)), [1]*len(selected_names), color='green', alpha=0.7)
    plt.xlabel('Selected Feature Index')
    plt.title('Features Selected by Pipeline')
    plt.xticks(range(len(selected_names)), selected_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/feature_selection_leakage_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'wrong_score': fs_wrong_score,
        'correct_score': fs_correct_score,
        'leakage_gap': leakage_gap,
        'cv_scores': cv_scores_fs.tolist()
    }


def demonstrate_gridsearch_safety():
    """
    Demonstrate GridSearchCV safety with pipelines.
    """
    print("\n" + "="*80)
    print("GRIDSEARCHCV SAFETY DEMONSTRATION")
    print("="*80)
    
    # Create data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1200, n_features=15, n_informative=10,
        n_redundant=5, weights=[0.75, 0.25],
        flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # WRONG: Manual preprocessing before GridSearchCV
    print("\n❌ WRONG APPROACH: Manual preprocessing before GridSearchCV")
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    
    scaler_wrong = StandardScaler()
    X_train_scaled = scaler_wrong.fit_transform(X_train)  # Fitted on ALL training data
    
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    grid_wrong = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=5, scoring="f1"
    )
    grid_wrong.fit(X_train_scaled, y_train)
    
    print(f"   Best CV F1: {grid_wrong.best_score_:.3f}")
    print(f"   Best params: {grid_wrong.best_params_}")
    print(f"   Test F1: {grid_wrong.score(X_test, y_test):.3f}")
    
    # CORRECT: Pipeline in GridSearchCV
    print("\n✅ CORRECT APPROACH: Pipeline in GridSearchCV")
    from sklearn.pipeline import Pipeline
    
    pipeline_grid = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    
    param_grid_pipeline = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    grid_correct = GridSearchCV(
        pipeline_grid, param_grid_pipeline, cv=5, scoring="f1"
    )
    grid_correct.fit(X_train, y_train)
    
    print(f"   Best CV F1: {grid_correct.best_score_:.3f}")
    print(f"   Best params: {grid_correct.best_params_}")
    print(f"   Test F1: {grid_correct.score(X_test, y_test):.3f}")
    
    # Show the difference
    wrong_test_f1 = grid_wrong.score(X_test, y_test)
    correct_test_f1 = grid_correct.score(X_test, y_test)
    leakage_gap = wrong_test_f1 - correct_test_f1
    
    print(f"\n   Leakage gap: {leakage_gap:.3f}")
    
    # Visualize GridSearchCV comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    methods = ['Wrong\n(Manual preprocessing)', 'Correct\n(Pipeline)']
    scores = [grid_wrong.best_score_, grid_correct.best_score_]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, scores, color=colors, alpha=0.7)
    plt.ylabel('Best CV F1-Score')
    plt.title('GridSearchCV Leakage Impact')
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    # Show parameter paths
    wrong_params = grid_wrong.best_params_
    correct_params = grid_correct.best_params_
    
    param_comparison = pd.DataFrame({
        'Wrong': [wrong_params.get('C', 'N/A')],
        'Correct': [correct_params.get('model__C', 'N/A')]
    })
    
    param_comparison.plot(kind='bar', figsize=(8, 4))
    plt.title('Best Parameter Comparison')
    plt.ylabel('C Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("plots/gridsearch_leakage_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'wrong_best_score': grid_wrong.best_score_,
        'correct_best_score': grid_correct.best_score_,
        'wrong_test_score': wrong_test_f1,
        'correct_test_score': correct_test_f1,
        'leakage_gap': leakage_gap
    }


def demonstrate_comprehensive_safety():
    """
    Demonstrate comprehensive pipeline safety evaluation.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PIPELINE SAFETY EVALUATION")
    print("="*80)
    
    # Load and prepare data
    X, y = load_data(synthetic=True)
    X_clean, y_clean = clean_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
    
    # Create safe pipeline
    pipeline = create_safe_pipeline(
        numerical_columns=NUMERICAL_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
        model_type="logistic",
        max_iter=1000,
        random_state=42
    )
    
    print(f"Pipeline created with {len(pipeline.steps)} steps:")
    for i, (name, step) in enumerate(pipeline.steps):
        print(f"  {i+1}. {name}: {type(step).__name__}")
    
    # Evaluate safety
    safety_metrics = evaluate_pipeline_safety(
        pipeline, X_train, X_test, y_train, y_test
    )
    
    # Print detailed report
    print_safety_report(safety_metrics)
    
    return safety_metrics


def demonstrate_deployment_workflow():
    """
    Demonstrate deployment-ready pipeline workflow.
    """
    print("\n" + "="*80)
    print("DEPLOYMENT WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Create deployment pipeline
    pipeline = create_deployment_pipeline(
        numerical_columns=NUMERICAL_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
        model_type="logistic",
        max_iter=1000,
        random_state=42
    )
    
    # Load and train data
    X, y = load_data(synthetic=True)
    X_clean, y_clean = clean_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
    
    print("Training deployment pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Save pipeline
    os.makedirs("models", exist_ok=True)
    save_deployment_pipeline(pipeline, "models/deployment_pipeline.pkl")
    
    print("Pipeline saved to models/deployment_pipeline.pkl")
    
    # Load and test
    print("\nTesting loaded pipeline...")
    loaded_pipeline = load_deployment_pipeline("models/deployment_pipeline.pkl")
    
    # Test that loaded pipeline works
    original_pred = pipeline.predict(X_test)
    loaded_pred = loaded_pipeline.predict(X_test)
    
    predictions_match = np.array_equal(original_pred, loaded_pred)
    test_score = loaded_pipeline.score(X_test, y_test)
    
    print(f"Loaded pipeline test score: {test_score:.3f}")
    print(f"Predictions match original: {predictions_match}")
    
    # Show deployment info
    if hasattr(loaded_pipeline, 'deployment_info'):
        info = loaded_pipeline.deployment_info
        print(f"\nDeployment Info:")
        print(f"  Model type: {info.get('model_type', 'Unknown')}")
        print(f"  Created: {info.get('created_at', 'Unknown')}")
        print(f"  Numerical features: {len(info.get('numerical_columns', []))}")
        print(f"  Categorical features: {len(info.get('categorical_columns', []))}")
    
    return {
        'test_score': test_score,
        'predictions_match': predictions_match,
        'deployment_info': getattr(loaded_pipeline, 'deployment_info', {})
    }


def main():
    """
    Main demonstration function.
    """
    print("="*80)
    print("PIPELINE LEAKAGE PREVENTION DEMONSTRATION")
    print("="*80)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    results = {}
    
    # 1. Scaling leakage demonstration
    print("\n1. SCALING LEAKAGE DEMONSTRATION")
    results['scaling'] = demonstrate_scaling_leakage()
    
    # 2. Feature selection leakage demonstration
    print("\n2. FEATURE SELECTION LEAKAGE DEMONSTRATION")
    results['feature_selection'] = demonstrate_feature_selection_leakage()
    
    # 3. GridSearchCV safety demonstration
    print("\n3. GRIDSEARCHCV SAFETY DEMONSTRATION")
    results['gridsearch'] = demonstrate_gridsearch_safety()
    
    # 4. Comprehensive safety evaluation
    print("\n4. COMPREHENSIVE SAFETY EVALUATION")
    results['safety'] = demonstrate_comprehensive_safety()
    
    # 5. Deployment workflow demonstration
    print("\n5. DEPLOYMENT WORKFLOW DEMONSTRATION")
    results['deployment'] = demonstrate_deployment_workflow()
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    
    print("\n🔍 LEAKAGE GAPS DETECTED:")
    for scenario, result in results.items():
        if 'leakage_gap' in result:
            gap = result['leakage_gap']
            print(f"  {scenario.title()}: {gap:.3f} ({gap:.1%}) performance inflation")
    
    print("\n✅ KEY TAKEAWAYS:")
    print("  • Always split data BEFORE any preprocessing")
    print("  • Use Pipeline to ensure proper fit/transform order")
    print("  • Cross-validation requires fresh preprocessing per fold")
    print("  • GridSearchCV must include preprocessing steps")
    print("  • Save complete pipeline for deployment")
    print("  • Test pipeline loading to ensure reproducibility")
    
    print("\n📊 PROFESSIONAL WORKFLOW:")
    print("  1. Load data")
    print("  2. Split train/test (test goes in 'drawer')")
    print("  3. Build Pipeline with all preprocessing")
    print("  4. Cross-validate Pipeline on training data only")
    print("  5. Tune hyperparameters using Pipeline")
    print("  6. Evaluate on test set exactly once")
    print("  7. Save complete Pipeline for deployment")
    
    # Save results
    import json
    with open("pipeline_leakage_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Files created:")
    print(f"  • plots/scaling_leakage_demo.png")
    print(f"  • plots/feature_selection_leakage_demo.png")
    print(f"  • plots/gridsearch_leakage_demo.png")
    print(f"  • models/deployment_pipeline.pkl")
    print(f"  • pipeline_leakage_demo_results.json")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Pipeline leakage prevention demonstration completed successfully!")
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise
