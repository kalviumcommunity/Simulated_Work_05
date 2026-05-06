"""
GridSearchCV Hyperparameter Tuning Tutorial

Comprehensive implementation of GridSearchCV for systematic hyperparameter optimization.
This tutorial covers the complete workflow from basic concepts to advanced strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
import logging

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split, 
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.datasets import make_classification, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import randint, uniform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class GridSearchTutorial:
    """Comprehensive GridSearchCV tutorial implementation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the tutorial.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
    def create_sample_data(self, n_samples: int = 1000, n_features: int = 10, 
                          imbalance: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sample classification data with optional imbalance.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            imbalance: Ratio of minority class (0.3 = 30% minority)
            
        Returns:
            Tuple of (X, y) features and labels
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(n_features // 2, 2),
            n_redundant=max(n_features // 4, 1),
            n_clusters_per_class=1,
            weights=[1 - imbalance, imbalance],
            flip_y=0.01,
            random_state=self.random_state
        )
        
        logger.info(f"Created dataset: {X.shape} with class distribution: {np.bincount(y)}")
        return X, y
    
    def demonstrate_basic_gridsearch_knn(self) -> Dict:
        """
        Demonstrate basic GridSearchCV with KNN classifier.
        
        Returns:
            Dictionary with results and best model
        """
        print("\n" + "="*70)
        print("EXAMPLE 1: BASIC GRIDSEARCHCV WITH KNN")
        print("="*70)
        
        print("""
📖 SCENARIO: Finding optimal K for K-Nearest Neighbors

🎯 GOAL: Systematically search for the best K and weighting strategy
📊 DATASET: Binary classification with moderate imbalance
🔧 APPROACH: GridSearchCV with pipeline to prevent data leakage
        """)
        
        # Create data
        X, y = self.create_sample_data(n_samples=1500, n_features=8, imbalance=0.35)
        
        # Train/test split - CRITICAL: Do this BEFORE any tuning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Data split: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        
        # Create pipeline - ESSENTIAL to prevent data leakage
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])
        
        # Define parameter grid
        param_grid = {
            "knn__n_neighbors": range(1, 21),
            "knn__weights": ["uniform", "distance"]
        }
        
        print(f"Parameter grid: {len(param_grid['knn__n_neighbors']) * len(param_grid['knn__weights'])} combinations")
        
        # Create GridSearchCV
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )
        
        # Fit the grid search
        print("Running GridSearchCV...")
        start_time = time.time()
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        # Print results
        print(f"\nGridSearchCV completed in {grid_time:.2f} seconds")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")
        
        # Evaluate on test set - ONLY ONCE, after tuning is complete
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results = {
            'grid': grid,
            'best_model': best_model,
            'best_params': grid.best_params_,
            'best_cv_score': grid.best_score_,
            'test_score': test_accuracy,
            'grid_time': grid_time,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        self.results['knn_basic'] = results
        return results
    
    def demonstrate_gridsearch_decision_tree(self) -> Dict:
        """
        Demonstrate GridSearchCV with Decision Tree for complexity control.
        
        Returns:
            Dictionary with results
        """
        print("\n" + "="*70)
        print("EXAMPLE 2: DECISION TREE COMPLEXITY TUNING")
        print("="*70)
        
        print("""
📖 SCENARIO: Controlling Decision Tree complexity to prevent overfitting

🎯 GOAL: Find optimal tree depth and leaf constraints
📊 DATASET: Multi-class classification with varying complexity
🔧 APPROACH: GridSearchCV with F1 scoring for balanced performance
        """)
        
        # Create more complex data
        X, y = self.create_sample_data(n_samples=1200, n_features=12, imbalance=0.25)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Decision Tree parameter grid
        param_grid = {
            "max_depth": [2, 4, 6, 8, 10, None],
            "min_samples_leaf": [1, 5, 10, 20],
            "criterion": ["gini", "entropy"]
        }
        
        print(f"Parameter grid: {len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion'])} combinations")
        
        # Create GridSearchCV with F1 scoring (better for potential imbalance)
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring="f1",
            return_train_score=True,
            n_jobs=-1
        )
        
        # Fit grid search
        print("Running GridSearchCV for Decision Tree...")
        start_time = time.time()
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        # Results
        print(f"\nGridSearchCV completed in {grid_time:.2f} seconds")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV F1: {grid.best_score_:.4f}")
        
        # Test evaluation
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        print(f"Test F1: {test_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results = {
            'grid': grid,
            'best_model': best_model,
            'best_params': grid.best_params_,
            'best_cv_score': grid.best_score_,
            'test_score': test_f1,
            'grid_time': grid_time
        }
        
        self.results['decision_tree'] = results
        return results
    
    def analyze_cv_results(self, grid: GridSearchCV, model_name: str = "Model") -> pd.DataFrame:
        """
        Analyze and display cross-validation results from GridSearchCV.
        
        Args:
            grid: Fitted GridSearchCV object
            model_name: Name of the model for display
            
        Returns:
            DataFrame with results summary
        """
        print(f"\n{model_name} - Cross-Validation Results Analysis")
        print("-" * 50)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(grid.cv_results_)
        
        # Select relevant columns
        display_cols = []
        for col in results_df.columns:
            if col.startswith('param_') or col in ['mean_test_score', 'std_test_score', 
                                                   'mean_train_score', 'rank_test_score']:
                display_cols.append(col)
        
        results_summary = results_df[display_cols].copy()
        
        # Sort by rank
        results_summary = results_summary.sort_values('rank_test_score')
        
        # Display top 10 configurations
        print("Top 10 Configurations:")
        print(results_summary.head(10).to_string(index=False))
        
        # Analysis of best configuration
        best_idx = results_summary['rank_test_score'].idxmin()
        best_config = results_summary.iloc[best_idx]
        
        print(f"\nBest Configuration Analysis:")
        print(f"CV Score: {best_config['mean_test_score']:.4f} ± {best_config['std_test_score']:.4f}")
        print(f"Train Score: {best_config['mean_train_score']:.4f}")
        print(f"Train-Test Gap: {best_config['mean_train_score'] - best_config['mean_test_score']:.4f}")
        
        # Check for overfitting indicators
        gap = best_config['mean_train_score'] - best_config['mean_test_score']
        if gap > 0.1:
            print("⚠️  WARNING: Large train-test gap suggests overfitting")
        elif gap < 0.02:
            print("✅ Good: Small train-test gap suggests good generalization")
        
        return results_summary
    
    def visualize_hyperparameter_effects(self, results: Dict, model_name: str = "Model"):
        """
        Create visualizations of hyperparameter effects on performance.
        
        Args:
            results: Results dictionary from grid search
            model_name: Name of the model
        """
        print(f"\nCreating visualizations for {model_name}...")
        
        grid = results['grid']
        results_df = pd.DataFrame(grid.cv_results_)
        
        # Create plots directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        if model_name == "KNN":
            self._plot_knn_results(results_df)
        elif model_name == "Decision Tree":
            self._plot_decision_tree_results(results_df)
    
    def _plot_knn_results(self, results_df: pd.DataFrame):
        """Plot KNN-specific results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Filter for uniform weights for clean visualization
        uniform_mask = results_df["param_knn__weights"] == "uniform"
        distance_mask = results_df["param_knn__weights"] == "distance"
        
        k_values_uniform = results_df[uniform_mask]["param_knn__n_neighbors"].astype(int)
        k_values_distance = results_df[distance_mask]["param_knn__n_neighbors"].astype(int)
        
        # Plot 1: Accuracy vs K for both weighting schemes
        axes[0, 0].plot(k_values_uniform, results_df[uniform_mask]["mean_test_score"],
                        label="Uniform", marker="o", linewidth=2)
        axes[0, 0].plot(k_values_distance, results_df[distance_mask]["mean_test_score"],
                        label="Distance", marker="s", linewidth=2)
        axes[0, 0].set_xlabel("K (Number of Neighbors)")
        axes[0, 0].set_ylabel("CV Accuracy")
        axes[0, 0].set_title("KNN: K vs. Cross-Validation Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test scores (overfitting analysis)
        axes[0, 1].plot(k_values_uniform, results_df[uniform_mask]["mean_train_score"],
                        label="Train (Uniform)", linestyle="--", marker="o", alpha=0.7)
        axes[0, 1].plot(k_values_uniform, results_df[uniform_mask]["mean_test_score"],
                        label="CV (Uniform)", marker="o")
        axes[0, 1].set_xlabel("K (Number of Neighbors)")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("KNN: Train vs. CV Accuracy (Uniform Weights)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Standard deviation (stability analysis)
        axes[1, 0].fill_between(
            k_values_uniform,
            results_df[uniform_mask]["mean_test_score"] - results_df[uniform_mask]["std_test_score"],
            results_df[uniform_mask]["mean_test_score"] + results_df[uniform_mask]["std_test_score"],
            alpha=0.2, label="±1 std"
        )
        axes[1, 0].plot(k_values_uniform, results_df[uniform_mask]["mean_test_score"],
                        label="Mean CV Score", linewidth=2)
        axes[1, 0].set_xlabel("K (Number of Neighbors)")
        axes[1, 0].set_ylabel("CV Accuracy")
        axes[1, 0].set_title("KNN: CV Score Stability")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Train-Test Gap
        train_test_gap = results_df[uniform_mask]["mean_train_score"] - results_df[uniform_mask]["mean_test_score"]
        axes[1, 1].plot(k_values_uniform, train_test_gap, marker="o", linewidth=2, color="red")
        axes[1, 1].set_xlabel("K (Number of Neighbors)")
        axes[1, 1].set_ylabel("Train-Test Gap")
        axes[1, 1].set_title("KNN: Overfitting Indicator")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.1, color="orange", linestyle="--", label="Warning threshold")
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("plots/knn_hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_decision_tree_results(self, results_df: pd.DataFrame):
        """Plot Decision Tree-specific results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Max Depth vs F1 Score
        for criterion in ['gini', 'entropy']:
            mask = results_df['param_criterion'] == criterion
            depth_scores = results_df[mask].groupby('param_max_depth')['mean_test_score'].mean()
            axes[0, 0].plot(depth_scores.index, depth_scores.values, marker='o', 
                           label=f'{criterion}', linewidth=2)
        
        axes[0, 0].set_xlabel("Max Depth")
        axes[0, 0].set_ylabel("CV F1 Score")
        axes[0, 0].set_title("Decision Tree: Max Depth vs. F1 Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Min Samples Leaf vs F1 Score
        for criterion in ['gini', 'entropy']:
            mask = results_df['param_criterion'] == criterion
            leaf_scores = results_df[mask].groupby('param_min_samples_leaf')['mean_test_score'].mean()
            axes[0, 1].plot(leaf_scores.index, leaf_scores.values, marker='s',
                           label=f'{criterion}', linewidth=2)
        
        axes[0, 1].set_xlabel("Min Samples Leaf")
        axes[0, 1].set_ylabel("CV F1 Score")
        axes[0, 1].set_title("Decision Tree: Min Samples Leaf vs. F1 Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of depth vs min_samples_leaf
        pivot_data = results_df.pivot_table(
            values='mean_test_score', 
            index='param_max_depth', 
            columns='param_min_samples_leaf',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_title("Decision Tree: F1 Score Heatmap")
        axes[1, 0].set_xlabel("Min Samples Leaf")
        axes[1, 0].set_ylabel("Max Depth")
        
        # Plot 4: Overfitting analysis
        results_df['train_test_gap'] = results_df['mean_train_score'] - results_df['mean_test_score']
        pivot_gap = results_df.pivot_table(
            values='train_test_gap',
            index='param_max_depth',
            columns='param_min_samples_leaf',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_gap, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 1])
        axes[1, 1].set_title("Decision Tree: Train-Test Gap (Overfitting)")
        axes[1, 1].set_xlabel("Min Samples Leaf")
        axes[1, 1].set_ylabel("Max Depth")
        
        plt.tight_layout()
        plt.savefig("plots/decision_tree_hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_scoring_metrics_comparison(self) -> Dict:
        """
        Demonstrate how different scoring metrics affect hyperparameter selection.
        
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*70)
        print("EXAMPLE 3: SCORING METRICS IMPACT ANALYSIS")
        print("="*70)
        
        print("""
📖 SCENARIO: Comparing different scoring metrics for hyperparameter selection

🎯 GOAL: Show how metric choice affects the "best" hyperparameters
📊 DATASET: Imbalanced binary classification
🔧 APPROACH: GridSearchCV with different scoring metrics
        """)
        
        # Create imbalanced data
        X, y = self.create_sample_data(n_samples=2000, n_features=10, imbalance=0.15)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Class distribution: {np.bincount(y)} (Imbalance ratio: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1)")
        
        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [1, 5, 10]
        }
        
        # Test different scoring metrics
        scoring_metrics = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
        results = {}
        
        for metric in scoring_metrics:
            print(f"\nTesting with {metric} scoring...")
            
            grid = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                param_grid,
                cv=5,
                scoring=metric,
                n_jobs=-1
            )
            
            start_time = time.time()
            grid.fit(X_train, y_train)
            grid_time = time.time() - start_time
            
            # Evaluate on test set
            y_pred = grid.best_estimator_.predict(X_test)
            y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
            
            # Calculate all metrics for comparison
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            results[metric] = {
                'best_params': grid.best_params_,
                'best_cv_score': grid.best_score_,
                'test_metrics': test_metrics,
                'grid_time': grid_time
            }
            
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV {metric}: {grid.best_score_:.4f}")
            print(f"  Test metrics: {test_metrics}")
        
        # Create comparison table
        print(f"\n{'Scoring Metric':<15} {'Best CV':<10} {'Test Acc':<10} {'Test F1':<10} {'Test Rec':<10} {'Test Prec':<10}")
        print("-" * 70)
        
        for metric in scoring_metrics:
            result = results[metric]
            print(f"{metric:<15} {result['best_cv_score']:<10.4f} "
                  f"{result['test_metrics']['accuracy']:<10.4f} "
                  f"{result['test_metrics']['f1']:<10.4f} "
                  f"{result['test_metrics']['recall']:<10.4f} "
                  f"{result['test_metrics']['precision']:<10.4f}")
        
        self.results['scoring_comparison'] = results
        return results
    
    def demonstrate_randomized_search(self) -> Dict:
        """
        Demonstrate RandomizedSearchCV as an alternative to GridSearchCV.
        
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*70)
        print("EXAMPLE 4: RANDOMIZEDSEARCHCV VS GRIDSEARCHCV")
        print("="*70)
        
        print("""
📖 SCENARIO: Comparing exhaustive vs. randomized hyperparameter search

🎯 GOAL: Show efficiency trade-offs between GridSearchCV and RandomizedSearchCV
📊 DATASET: Medium-sized classification problem
🔧 APPROACH: Compare both methods on same parameter space
        """)
        
        # Create data
        X, y = self.create_sample_data(n_samples=1500, n_features=15, imbalance=0.3)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Parameter distributions for RandomizedSearchCV
        param_distributions = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": uniform(0.3, 0.7)  # Between 0.3 and 1.0
        }
        
        # Parameter grid for GridSearchCV (smaller for comparison)
        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_leaf": [1, 5, 10, 20],
            "max_features": [0.3, 0.5, 0.7, 1.0]
        }
        
        results = {}
        
        # GridSearchCV
        print("Running GridSearchCV...")
        grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        
        start_time = time.time()
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        grid_score = f1_score(y_test, grid.best_estimator_.predict(X_test))
        
        print(f"GridSearchCV: {grid_time:.2f}s, Best CV F1: {grid.best_score_:.4f}, Test F1: {grid_score:.4f}")
        
        # RandomizedSearchCV with different n_iter values
        n_iter_values = [20, 50, 100]
        
        for n_iter in n_iter_values:
            print(f"Running RandomizedSearchCV with n_iter={n_iter}...")
            
            random_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                param_distributions,
                n_iter=n_iter,
                cv=5,
                scoring="f1",
                random_state=self.random_state,
                n_jobs=-1
            )
            
            start_time = time.time()
            random_search.fit(X_train, y_train)
            random_time = time.time() - start_time
            
            random_score = f1_score(y_test, random_search.best_estimator_.predict(X_test))
            
            speedup = (grid_time - random_time) / grid_time * 100
            performance_gap = grid_score - random_score
            
            print(f"  RandomSearchCV: {random_time:.2f}s, Best CV F1: {random_search.best_score_:.4f}, Test F1: {random_score:.4f}")
            print(f"  Speedup: {speedup:.1f}%, Performance gap: {performance_gap:.4f}")
            
            results[f'random_{n_iter}'] = {
                'best_params': random_search.best_params_,
                'best_cv_score': random_search.best_score_,
                'test_score': random_score,
                'time': random_time,
                'speedup': speedup,
                'performance_gap': performance_gap
            }
        
        # Store grid results
        results['grid'] = {
            'best_params': grid.best_params_,
            'best_cv_score': grid.best_score_,
            'test_score': grid_score,
            'time': grid_time
        }
        
        self.results['randomized_comparison'] = results
        return results
    
    def demonstrate_coarse_to_fine_strategy(self) -> Dict:
        """
        Demonstrate coarse-to-fine hyperparameter tuning strategy.
        
        Returns:
            Dictionary with strategy results
        """
        print("\n" + "="*70)
        print("EXAMPLE 5: COARSE-TO-FINE TUNING STRATEGY")
        print("="*70)
        
        print("""
📖 SCENARIO: Two-phase optimization combining exploration and refinement

🎯 GOAL: Demonstrate practical coarse-to-fine optimization workflow
📊 DATASET: Complex problem requiring balanced exploration
🔧 APPROACH: Phase 1 (coarse) + Phase 2 (fine) optimization
        """)
        
        # Create challenging data
        X, y = self.create_sample_data(n_samples=2000, n_features=20, imbalance=0.25)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        results = {}
        
        # Phase 1: Coarse exploration
        print("Phase 1: Coarse exploration with wide parameter ranges")
        
        coarse_param_grid = {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [3, 7, 15, None],
            "min_samples_leaf": [1, 10, 20, 50],
            "max_features": [0.3, 0.5, 0.7, 1.0]
        }
        
        coarse_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            coarse_param_grid,
            cv=3,  # Fewer folds for speed in exploration
            scoring="f1",
            n_jobs=-1
        )
        
        start_time = time.time()
        coarse_grid.fit(X_train, y_train)
        coarse_time = time.time() - start_time
        
        print(f"Coarse search completed in {coarse_time:.2f}s")
        print(f"Best parameters: {coarse_grid.best_params_}")
        print(f"Best CV F1: {coarse_grid.best_score_:.4f}")
        
        # Phase 2: Fine refinement around best region
        print("\nPhase 2: Fine refinement around promising region")
        
        # Extract best parameters and create fine grid
        best_coarse = coarse_grid.best_params_
        
        # Create fine grid around best parameters
        fine_param_grid = {
            "n_estimators": [
                max(50, best_coarse["n_estimators"] - 50),
                best_coarse["n_estimators"],
                best_coarse["n_estimators"] + 50,
                min(400, best_coarse["n_estimators"] + 100)
            ],
            "max_depth": self._create_fine_range(best_coarse["max_depth"], [3, 7, 15, None]),
            "min_samples_leaf": self._create_fine_range(best_coarse["min_samples_leaf"], [1, 10, 20, 50]),
            "max_features": self._create_fine_range(best_coarse["max_features"], [0.3, 0.5, 0.7, 1.0])
        }
        
        fine_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            fine_param_grid,
            cv=5,  # More folds for final tuning
            scoring="f1",
            n_jobs=-1
        )
        
        start_time = time.time()
        fine_grid.fit(X_train, y_train)
        fine_time = time.time() - start_time
        
        print(f"Fine search completed in {fine_time:.2f}s")
        print(f"Best parameters: {fine_grid.best_params_}")
        print(f"Best CV F1: {fine_grid.best_score_:.4f}")
        
        # Comparison
        improvement = fine_grid.best_score_ - coarse_grid.best_score_
        total_time = coarse_time + fine_time
        
        print(f"\nHybrid Strategy Results:")
        print(f"  Coarse CV F1: {coarse_grid.best_score_:.4f}")
        print(f"  Fine CV F1: {fine_grid.best_score_:.4f}")
        print(f"  Improvement: {improvement:+.4f} F1 points")
        print(f"  Total time: {total_time:.2f}s")
        
        # Final evaluation
        final_model = fine_grid.best_estimator_
        y_pred = final_model.predict(X_test)
        final_f1 = f1_score(y_test, y_pred)
        
        print(f"Final test F1: {final_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results = {
            'coarse_result': {
                'best_params': coarse_grid.best_params_,
                'best_score': coarse_grid.best_score_,
                'time': coarse_time
            },
            'fine_result': {
                'best_params': fine_grid.best_params_,
                'best_score': fine_grid.best_score_,
                'time': fine_time
            },
            'improvement': improvement,
            'total_time': total_time,
            'final_test_f1': final_f1
        }
        
        self.results['coarse_to_fine'] = results
        return results
    
    def _create_fine_range(self, best_value, all_values):
        """Create a fine range around the best value."""
        if best_value is None:
            return [None, 15]
        
        # Find index of best value
        try:
            idx = all_values.index(best_value)
        except ValueError:
            # If best_value not in list, return nearby values
            if isinstance(best_value, (int, float)):
                return [best_value - 2, best_value, best_value + 2]
            else:
                return all_values
        
        # Create fine range around best
        if idx == 0:
            fine_range = all_values[:3]
        elif idx == len(all_values) - 1:
            fine_range = all_values[-3:]
        else:
            fine_range = all_values[idx-1:idx+2]
        
        return fine_range
    
    def demonstrate_data_leakage_prevention(self) -> Dict:
        """
        Demonstrate proper data leakage prevention in hyperparameter tuning.
        
        Returns:
            Dictionary with leakage comparison results
        """
        print("\n" + "="*70)
        print("EXAMPLE 6: DATA LEAKAGE PREVENTION")
        print("="*70)
        
        print("""
📖 SCENARIO: Demonstrating the critical importance of preventing data leakage

🎯 GOAL: Show the difference between proper and improper hyperparameter tuning
📊 DATASET: Standard classification problem
🔧 APPROACH: Compare correct vs. incorrect workflows
        """)
        
        # Create data
        X, y = self.create_sample_data(n_samples=1000, n_features=10, imbalance=0.3)
        
        # Correct workflow (proper)
        print("CORRECT WORKFLOW:")
        print("1. Train/test split first")
        print("2. GridSearchCV on training data only")
        print("3. Evaluate on test set once")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Correct: Pipeline prevents leakage
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=self.random_state, n_jobs=-1))
        ])
        
        param_grid = {
            "rf__n_estimators": [50, 100, 200],
            "rf__max_depth": [3, 7, None]
        }
        
        correct_grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        
        correct_grid.fit(X_train, y_train)
        correct_score = f1_score(y_test, correct_grid.predict(X_test))
        
        print(f"Correct workflow CV F1: {correct_grid.best_score_:.4f}")
        print(f"Correct workflow Test F1: {correct_score:.4f}")
        
        # Incorrect workflow (leakage)
        print("\nINCORRECT WORKFLOW (WITH LEAKAGE):")
        print("1. Scale entire dataset before splitting")
        print("2. GridSearchCV sees information from test set")
        print("3. Overly optimistic performance estimate")
        
        # Incorrect: Scale before splitting (leakage)
        X_scaled = StandardScaler().fit_transform(X)
        
        X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # GridSearchCV on leaked data
        incorrect_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            {"n_estimators": [50, 100, 200], "max_depth": [3, 7, None]},
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        
        incorrect_grid.fit(X_train_leaky, y_train_leaky)
        incorrect_score = f1_score(y_test_leaky, incorrect_grid.predict(X_test_leaky))
        
        print(f"Incorrect workflow CV F1: {incorrect_grid.best_score_:.4f}")
        print(f"Incorrect workflow Test F1: {incorrect_score:.4f}")
        
        # Analysis
        leakage_inflation = incorrect_score - correct_score
        print(f"\nLeakage Analysis:")
        print(f"  Performance inflation: {leakage_inflation:.4f} F1 points")
        print(f"  Relative inflation: {(leakage_inflation/correct_score)*100:.1f}%")
        
        if leakage_inflation > 0.02:
            print("  ⚠️  SIGNIFICANT LEAKAGE DETECTED!")
        elif leakage_inflation > 0.01:
            print("  ⚠️  Moderate leakage detected")
        else:
            print("  ✅ Minimal leakage")
        
        results = {
            'correct_cv_score': correct_grid.best_score_,
            'correct_test_score': correct_score,
            'incorrect_cv_score': incorrect_grid.best_score_,
            'incorrect_test_score': incorrect_score,
            'leakage_inflation': leakage_inflation,
            'relative_inflation': (leakage_inflation/correct_score)*100
        }
        
        self.results['data_leakage'] = results
        return results
    
    def print_practical_checklist(self):
        """Print a comprehensive practical checklist for hyperparameter tuning."""
        print("\n" + "="*70)
        print("PRACTICAL HYPERPARAMETER TUNING CHECKLIST")
        print("="*70)
        
        checklist = """
🔍 PRE-OPTIMIZATION CHECKLIST:
□ Define clear optimization objective (accuracy, F1, custom metric)
□ Understand model complexity and parameter sensitivity
□ Choose appropriate search method based on problem complexity
□ Set meaningful parameter ranges based on domain knowledge
□ Set random_state for reproducible results
□ Choose CV strategy appropriate for data size and problem
□ Monitor computational budget and time constraints
□ Plan for proper evaluation on holdout set

⚡ OPTIMIZATION EXECUTION CHECKLIST:
□ Train/test split performed BEFORE any optimization
□ All preprocessing inside pipeline/estimator
□ No data leakage between CV folds
□ Parameter ranges cover important dimensions
□ Sufficient iterations/combinations for convergence
□ Results reproducible with same random_state
□ Performance compared to baseline model
□ Computational efficiency considered

📊 REPORTING CHECKLIST:
□ Baseline performance documented
□ Untuned model performance shown
□ Best CV score AND standard deviation reported
□ Final test score evaluated exactly once
□ Best hyperparameters clearly listed
□ Computational time documented
□ Train-test gap analyzed for overfitting
□ Comparison to baseline/untuned performance
□ Reproducibility information included

🚨 COMMON MISTAKES TO AVOID:
□ Tuning on the test set (MOST CRITICAL ERROR)
□ Not using pipelines (causes CV-level leakage)
□ Optimizing wrong metric for business problem
□ Building grids that are too coarse or too fine
□ Ignoring computational cost vs. benefit
□ Reporting only best score without variance
□ Not checking for overfitting indicators
□ Forgetting to set random_state
        """
        
        print(checklist)
        
        print("\n💡 PRO TIPS:")
        tips = [
            "• Start with wide ranges, narrow based on results",
            "• Use log-uniform distributions for scale-free parameters",
            "• Always include baseline for context",
            "• Monitor train-test gap for overfitting",
            "• Save random seeds for reproducibility",
            "• Consider computational budget vs optimization quality",
            "• Use parallel processing (n_jobs=-1) when possible",
            "• Document why certain parameters were chosen"
        ]
        
        for tip in tips:
            print(tip)
    
    def run_complete_tutorial(self):
        """Run the complete GridSearchCV tutorial."""
        print("""
🎯 GRIDSEARCHCV HYPERPARAMETER TUNING TUTORIAL
==============================================

This tutorial provides comprehensive coverage of GridSearchCV for
systematic hyperparameter optimization in machine learning.

📚 TUTORIAL STRUCTURE:
1. Basic GridSearchCV with KNN
2. Decision Tree complexity tuning
3. Scoring metrics impact analysis
4. RandomizedSearchCV comparison
5. Coarse-to-fine tuning strategy
6. Data leakage prevention
7. Practical checklist and best practices

⏱️  EXPECTED DURATION: 60-90 minutes
        """)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Run all examples
        examples = [
            ("Basic KNN GridSearchCV", self.demonstrate_basic_gridsearch_knn),
            ("Decision Tree Tuning", self.demonstrate_gridsearch_decision_tree),
            ("Scoring Metrics Comparison", self.demonstrate_scoring_metrics_comparison),
            ("RandomizedSearchCV vs GridSearchCV", self.demonstrate_randomized_search),
            ("Coarse-to-Fine Strategy", self.demonstrate_coarse_to_fine_strategy),
            ("Data Leakage Prevention", self.demonstrate_data_leakage_prevention)
        ]
        
        for i, (title, example_func) in enumerate(examples, 1):
            print(f"\n{'='*20} {i}. {title} {'='*20}")
            
            try:
                result = example_func()
                
                # Create visualizations for relevant examples
                if title == "Basic KNN GridSearchCV":
                    self.visualize_hyperparameter_effects(result, "KNN")
                elif title == "Decision Tree Tuning":
                    self.visualize_hyperparameter(result, "Decision Tree")
                
                # Analyze CV results
                if 'grid' in result:
                    self.analyze_cv_results(result['grid'], title)
                    
            except Exception as e:
                print(f"Error in {title}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            if i < len(examples):
                input("\nPress Enter to continue to next example...")
        
        # Print final checklist
        self.print_practical_checklist()
        
        # Summary
        print("\n" + "="*70)
        print("TUTORIAL COMPLETED!")
        print("="*70)
        
        print(f"\n📁 Files created: plots/")
        print("\n🎯 Key Takeaways:")
        takeaways = [
            "• GridSearchCV transforms hyperparameter selection from guesswork to systematic optimization",
            "• Always use pipelines to prevent data leakage during cross-validation",
            "• The scoring metric choice is as important as the grid itself",
            "• RandomizedSearchCV provides 80-90% of benefits at fraction of cost",
            "• Coarse-to-fine strategy balances exploration with efficiency",
            "• Never tune on the test set - this is the most critical rule",
            "• Always report both mean CV score and standard deviation",
            "• Monitor train-test gaps to detect overfitting"
        ]
        
        for takeaway in takeaways:
            print(f"  {takeaway}")
        
        return True


def main():
    """Main function to run the tutorial."""
    tutorial = GridSearchTutorial(random_state=42)
    return tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
