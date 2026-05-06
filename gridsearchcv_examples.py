"""
GridSearchCV Practical Examples

Additional practical examples and utilities for GridSearchCV hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import make_classification, load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time


class GridSearchExamples:
    """Collection of practical GridSearchCV examples."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def example_multiclass_gridsearch(self):
        """Example: GridSearchCV with multi-class classification."""
        print("\n" + "="*60)
        print("MULTICLASS GRIDSEARCHCV EXAMPLE")
        print("="*60)
        
        # Load wine dataset (3 classes)
        wine = load_wine()
        X, y = wine.data, wine.target
        
        print(f"Dataset: {X.shape}, Classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Pipeline with different scalers
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=self.random_state, n_jobs=-1))
        ])
        
        # Parameter grid including scaler choice
        param_grid = {
            "scaler": [StandardScaler(), MinMaxScaler(), RobustScaler(), None],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 7, None],
            "classifier__min_samples_leaf": [1, 5, 10]
        }
        
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring="accuracy",
            n_jobs=-1
        )
        
        print("Running GridSearchCV...")
        start_time = time.time()
        grid.fit(X, y)
        grid_time = time.time() - start_time
        
        print(f"Completed in {grid_time:.2f} seconds")
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        
        # Cross-validation scores analysis
        cv_scores = cross_val_score(grid.best_estimator_, X, y, 
                                   cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state))
        
        print(f"10-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return grid
    
    def example_custom_scorer(self):
        """Example: GridSearchCV with custom scoring function."""
        print("\n" + "="*60)
        print("CUSTOM SCORER EXAMPLE")
        print("="*60)
        
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=2000, n_features=10, n_informative=8,
            weights=[0.9, 0.1], flip_y=0.01, random_state=self.random_state
        )
        
        print(f"Imbalanced dataset: {np.bincount(y)}")
        
        # Define custom scorer (e.g., F2 score - emphasizes recall)
        def f2_score(y_true, y_pred):
            """Custom F2 score (beta=2, emphasizes recall)."""
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if precision == 0 and recall == 0:
                return 0
            
            f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
            return f2
        
        f2_scorer = make_scorer(f2_score)
        
        # Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        }
        
        # Compare different scorers
        scorers = {
            "accuracy": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "f2_custom": f2_scorer
        }
        
        results = {}
        
        for scorer_name, scorer in scorers.items():
            print(f"\nTesting with {scorer_name} scorer...")
            
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring=scorer,
                n_jobs=-1
            )
            
            grid.fit(X, y)
            
            # Evaluate with all metrics
            y_pred = grid.best_estimator_.predict(X)
            
            results[scorer_name] = {
                "best_params": grid.best_params_,
                "best_cv_score": grid.best_score_,
                "accuracy": accuracy_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f2": f2_score(y, y_pred)
            }
            
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV score: {grid.best_score_:.4f}")
            print(f"  Test metrics: Acc={results[scorer_name]['accuracy']:.3f}, "
                  f"F1={results[scorer_name]['f1']:.3f}, Rec={results[scorer_name]['recall']:.3f}")
        
        # Create comparison table
        comparison_df = pd.DataFrame(results).T
        print(f"\nComparison Table:")
        print(comparison_df.round(4))
        
        return results
    
    def example_nested_cv(self):
        """Example: Nested cross-validation for unbiased performance estimation."""
        print("\n" + "="*60)
        print("NESTED CROSS-VALIDATION EXAMPLE")
        print("="*60)
        
        # Create dataset
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=15,
            weights=[0.7, 0.3], random_state=self.random_state
        )
        
        print(f"Dataset: {X.shape}")
        
        # Outer CV loop
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Inner CV setup
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        # Parameter grid
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 5]
        }
        
        outer_scores = []
        best_params_list = []
        
        print("Running nested cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            print(f"Outer fold {fold}/5...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                param_grid,
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1
            )
            
            grid.fit(X_train, y_train)
            
            # Evaluate on outer test fold
            y_pred = grid.predict(X_test)
            outer_score = f1_score(y_test, y_pred)
            
            outer_scores.append(outer_score)
            best_params_list.append(grid.best_params_)
            
            print(f"  Fold {fold}: F1 = {outer_score:.4f}, Best params: {grid.best_params_}")
        
        # Results
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        print(f"\nNested CV Results:")
        print(f"Mean F1: {mean_score:.4f} ± {std_score:.4f}")
        print(f"All scores: {[f'{s:.4f}' for s in outer_scores]}")
        
        # Analyze parameter stability
        param_counts = {}
        for params in best_params_list:
            for param, value in params.items():
                key = f"{param}={value}"
                param_counts[key] = param_counts.get(key, 0) + 1
        
        print(f"\nParameter selection frequency:")
        for param, count in param_counts.items():
            print(f"  {param}: {count}/5 folds")
        
        return {
            "outer_scores": outer_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "best_params_list": best_params_list
        }
    
    def example_feature_selection_gridsearch(self):
        """Example: GridSearchCV with feature selection."""
        print("\n" + "="*60)
        print("FEATURE SELECTION GRIDSEARCHCV EXAMPLE")
        print("="*60)
        
        # Create dataset with many features
        X, y = make_classification(
            n_samples=1000, n_features=50, n_informative=10,
            n_redundant=10, n_repeated=5, random_state=self.random_state
        )
        
        print(f"Dataset: {X.shape} (10 informative, 40 noise features)")
        
        from sklearn.feature_selection import SelectKBest, f_classif, RFE
        from sklearn.linear_model import LogisticRegression
        
        # Pipeline with feature selection
        pipeline = Pipeline([
            ("feature_selection", SelectKBest(f_classif)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        # Parameter grid including feature selection
        param_grid = {
            "feature_selection__k": [5, 10, 15, 20, 25, 30],
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"]
        }
        
        # Note: l1 penalty requires saga solver
        pipeline.set_params(classifier__solver='saga')
        
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        
        print("Running GridSearchCV with feature selection...")
        start_time = time.time()
        grid.fit(X, y)
        grid_time = time.time() - start_time
        
        print(f"Completed in {grid_time:.2f} seconds")
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        
        # Analyze feature selection
        best_k = grid.best_params_["feature_selection__k"]
        print(f"\nOptimal number of features: {best_k}")
        
        # Get selected features
        best_pipeline = grid.best_estimator_
        selector = best_pipeline.named_steps["feature_selection"]
        selected_features = selector.get_support(indices=True)
        
        print(f"Selected feature indices: {selected_features}")
        print(f"Number of selected features: {len(selected_features)}")
        
        # Compare with baseline (no feature selection)
        baseline_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        baseline_param_grid = {
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"]
        }
        
        baseline_pipeline.set_params(classifier__solver='saga')
        
        baseline_grid = GridSearchCV(
            baseline_pipeline,
            baseline_param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        
        baseline_grid.fit(X, y)
        
        print(f"\nBaseline (all features):")
        print(f"Best CV accuracy: {baseline_grid.best_score_:.4f}")
        print(f"Best parameters: {baseline_grid.best_params_}")
        
        improvement = grid.best_score_ - baseline_grid.best_score_
        print(f"Feature selection improvement: {improvement:+.4f}")
        
        return {
            "feature_selection_grid": grid,
            "baseline_grid": baseline_grid,
            "improvement": improvement,
            "selected_features": selected_features
        }
    
    def example_ensemble_gridsearch(self):
        """Example: GridSearchCV with ensemble methods."""
        print("\n" + "="*60)
        print("ENSEMBLE METHODS GRIDSEARCHCV EXAMPLE")
        print("="*60)
        
        # Create dataset
        X, y = make_classification(
            n_samples=1500, n_features=15, n_informative=10,
            weights=[0.6, 0.4], random_state=self.random_state
        )
        
        print(f"Dataset: {X.shape}")
        
        # Compare different ensemble methods
        models = {
            "RandomForest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
            "ExtraTrees": None  # Will be imported if needed
        }
        
        # Parameter grids for each model
        param_grids = {
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 5, 10]
            },
            "GradientBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        }
        
        results = {}
        
        for model_name, model in models.items():
            if model_name == "ExtraTrees":
                try:
                    from sklearn.ensemble import ExtraTreesClassifier
                    model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
                    param_grids["ExtraTrees"] = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, None],
                        "min_samples_leaf": [1, 5, 10]
                    }
                except ImportError:
                    print("ExtraTreesClassifier not available, skipping...")
                    continue
            
            print(f"\nTuning {model_name}...")
            
            grid = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )
            
            start_time = time.time()
            grid.fit(X, y)
            grid_time = time.time() - start_time
            
            results[model_name] = {
                "best_params": grid.best_params_,
                "best_score": grid.best_score_,
                "grid_time": grid_time,
                "grid": grid
            }
            
            print(f"  Best F1: {grid.best_score_:.4f}")
            print(f"  Best params: {grid.best_params_}")
            print(f"  Time: {grid_time:.2f}s")
        
        # Create comparison
        print(f"\nEnsemble Methods Comparison:")
        print(f"{'Model':<20} {'Best F1':<10} {'Time (s)':<10}")
        print("-" * 45)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['best_score']:<10.4f} {result['grid_time']:<10.2f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]["best_score"])
        print(f"\nBest model: {best_model} (F1: {results[best_model]['best_score']:.4f})")
        
        return results
    
    def create_hyperparameter_importance_plot(self, grid_result, model_name="Model"):
        """Create a plot showing hyperparameter importance."""
        results_df = pd.DataFrame(grid_result.cv_results_)
        
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        
        if len(param_cols) == 0:
            print("No parameters found for plotting")
            return
        
        # Calculate parameter importance based on score variance
        param_importance = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            scores_by_param = results_df.groupby(param_col)['mean_test_score']
            
            # Calculate variance across parameter values
            variance = scores_by_param.var().mean()
            param_importance[param_name] = variance
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        params = list(param_importance.keys())
        importance = list(param_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        params = [params[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        plt.bar(params, importance)
        plt.title(f'{model_name} - Hyperparameter Importance')
        plt.xlabel('Hyperparameter')
        plt.ylabel('Score Variance (Importance)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{model_name.lower()}_param_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return param_importance


def run_all_examples():
    """Run all GridSearchCV examples."""
    examples = GridSearchExamples(random_state=42)
    
    print("""
🎯 GRIDSEARCHCV PRACTICAL EXAMPLES
===================================

Additional practical examples for GridSearchCV hyperparameter tuning.
        """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Run examples
    example_funcs = [
        ("Multiclass Classification", examples.example_multiclass_gridsearch),
        ("Custom Scorer", examples.example_custom_scorer),
        ("Nested Cross-Validation", examples.example_nested_cv),
        ("Feature Selection", examples.example_feature_selection_gridsearch),
        ("Ensemble Methods", examples.example_ensemble_gridsearch)
    ]
    
    results = {}
    
    for title, func in example_funcs:
        print(f"\n{'='*20} {title} {'='*20}")
        try:
            result = func()
            results[title] = result
            
            # Create importance plot if applicable
            if hasattr(result, 'best_params_') or 'grid' in result:
                if 'grid' in result:
                    examples.create_hyperparameter_importance_plot(result['grid'], title)
                
        except Exception as e:
            print(f"Error in {title}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_all_examples()
