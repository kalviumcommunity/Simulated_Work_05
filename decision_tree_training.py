"""
Decision Tree Training and Interpretation

Comprehensive implementation of Decision Tree training, visualization, and interpretation
for both classification and regression problems.

This module covers:
- How Decision Trees work conceptually with recursive partitioning
- Impurity measures (Gini, Entropy) for classification
- Variance reduction for regression
- Tree growth algorithm and stopping criteria
- Overfitting detection and control methods
- Cross-validation for hyperparameter tuning
- Tree visualization and interpretation
- Comparison against baseline models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, precision_score, recall_score, f1_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

# Set up warnings filter
warnings.filterwarnings('ignore')


class DecisionTreeTrainer:
    """
    Comprehensive Decision Tree training and interpretation system.
    
    This class provides methods to train, evaluate, and interpret Decision Trees
    for both classification and regression problems.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Decision Tree trainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def create_classification_data(self, n_samples: int = 1000, n_features: int = 8,
                             n_informative: int = 4, class_sep: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create synthetic classification data for Decision Tree demonstration.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            class_sep: Class separation factor
            
        Returns:
            Tuple of (X, y) as pandas DataFrame and Series
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(n_features // 4, 1),
            n_clusters_per_class=1,
            class_sep=class_sep,
            random_state=self.random_state
        )
        
        # Convert to DataFrame with meaningful names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")
        
        return X, y
    
    def create_regression_data(self, n_samples: int = 1000, n_features: int = 8,
                          n_informative: int = 4, noise: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create synthetic regression data for Decision Tree demonstration.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            noise: Noise level
            
        Returns:
            Tuple of (X, y) as pandas DataFrame and Series
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=self.random_state
        )
        
        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")
        
        return X, y
    
    def calculate_gini_impurity(self, class_counts: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of class counts.
        
        Args:
            class_counts: Array of class counts
            
        Returns:
            Gini impurity value
        """
        total = class_counts.sum()
        if total == 0:
            return 0.0
        
        proportions = class_counts / total
        gini = 1.0 - np.sum(proportions ** 2)
        return gini
    
    def calculate_entropy_impurity(self, class_counts: np.ndarray) -> float:
        """
        Calculate entropy impurity for a set of class counts.
        
        Args:
            class_counts: Array of class counts
            
        Returns:
            Entropy impurity value
        """
        total = class_counts.sum()
        if total == 0:
            return 0.0
        
        proportions = class_counts / total
        # Avoid log(0) by adding small epsilon
        proportions = proportions[proportions > 0]
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy
    
    def demonstrate_impurity_calculations(self):
        """Demonstrate Gini and Entropy impurity calculations."""
        print("\n" + "="*70)
        print("IMPURITY MEASURES DEMONSTRATION")
        print("="*70)
        
        print("""
📖 SCENARIO: Understanding impurity measures for Decision Tree splits

🎯 GOAL: Learn how Gini and Entropy measure node purity
📊 EXAMPLE: Different class distributions in tree nodes
🔧 APPROACH: Calculate both measures and compare
        """)
        
        # Example class distributions
        examples = [
            np.array([60, 40]),    # 60/40 split
            np.array([80, 20]),    # 80/20 split
            np.array([50, 50]),    # 50/50 split (maximum impurity)
            np.array([100, 0]),   # Pure node
            np.array([95, 5])     # Nearly pure
        ]
        
        print("\nClass Distribution | Gini Impurity | Entropy Impurity | Purity Level")
        print("-" * 65)
        
        for i, counts in enumerate(examples):
            gini = self.calculate_gini_impurity(counts)
            entropy = self.calculate_entropy_impurity(counts)
            
            if gini == 0:
                purity = "Perfect"
            elif gini < 0.3:
                purity = "High"
            elif gini < 0.5:
                purity = "Medium"
            else:
                purity = "Low"
            
            print(f"{counts[0]:>3}/{counts[1]:<3}         | {gini:>11.4f}     | {entropy:>13.4f}     | {purity}")
        
        # Visualize impurity curves
        self._plot_impurity_curves()
    
    def _plot_impurity_curves(self):
        """Plot Gini and Entropy curves for binary classification."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Range of class probabilities
        p = np.linspace(0.01, 0.99, 100)
        
        # Gini curve
        gini = 1 - p**2 - (1-p)**2
        
        # Entropy curve
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        # Plot Gini
        axes[0].plot(p, gini, 'b-', linewidth=2, label='Gini')
        axes[0].set_xlabel('Class 1 Proportion (p)')
        axes[0].set_ylabel('Gini Impurity')
        axes[0].set_title('Gini Impurity vs. Class Distribution')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot Entropy
        axes[1].plot(p, entropy, 'r-', linewidth=2, label='Entropy')
        axes[1].set_xlabel('Class 1 Proportion (p)')
        axes[1].set_ylabel('Entropy Impurity')
        axes[1].set_title('Entropy Impurity vs. Class Distribution')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig("plots/impurity_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_tree_growth_algorithm(self):
        """Demonstrate the Decision Tree growth algorithm step by step."""
        print("\n" + "="*70)
        print("DECISION TREE GROWTH ALGORITHM DEMONSTRATION")
        print("="*70)
        
        print("""
📖 SCENARIO: Step-by-step tree growth on simple data

🎯 GOAL: Understand how Decision Trees recursively partition data
📊 DATASET: Small classification problem for visualization
🔧 APPROACH: Show greedy splitting decisions at each step
        """)
        
        # Create simple 2D data for visualization
        np.random.seed(self.random_state)
        n_samples = 100
        
        # Create two clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
        cluster2 = np.random.multivariate_normal([6, 6], [[1, 0.5], [0.5, 1]], n_samples//2)
        
        X = np.vstack([cluster1, cluster2])
        y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        y_series = pd.Series(y, name='target')
        
        # Train shallow tree for step-by-step visualization
        tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        tree.fit(X_df, y_series)
        
        # Visualize the growth process
        self._visualize_tree_growth(X_df, y_series, tree)
        
        return X_df, y_series, tree
    
    def _visualize_tree_growth(self, X: pd.DataFrame, y: pd.Series, tree):
        """Visualize the tree growth process."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Original data
        axes[0, 0].scatter(X['feature_1'], X['feature_2'], c=y, cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].set_title('Original Data with Classes')
        
        # Plot 2: Tree structure
        plot_tree(tree, feature_names=X.columns, class_names=['Class 0', 'Class 1'],
                 filled=True, rounded=True, ax=axes[0, 1])
        axes[0, 1].set_title('Decision Tree Structure')
        
        # Plot 3: Decision boundaries
        self._plot_decision_boundaries(tree, X, y, axes[1, 0])
        
        # Plot 4: Feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig("plots/tree_growth_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Tree Growth Summary:")
        print(f"Tree depth: {tree.get_depth()}")
        print(f"Number of leaves: {tree.get_n_leaves()}")
        print(f"Number of nodes: {tree.tree_.node_count}")
        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))
    
    def _plot_decision_boundaries(self, tree, X: pd.DataFrame, y: pd.Series, ax):
        """Plot decision boundaries of the tree."""
        # Create mesh grid
        x_min, x_max = X['feature_1'].min() - 1, X['feature_1'].max() + 1
        y_min, y_max = X['feature_2'].min() - 1, X['feature_2'].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Predict on mesh grid
        mesh_data = pd.DataFrame({'feature_1': xx.ravel(), 'feature_2': yy.ravel()})
        Z = tree.predict(mesh_data).reshape(xx.shape)
        
        # Plot decision boundaries
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        ax.scatter(X['feature_1'], X['feature_2'], c=y, cmap='viridis', edgecolors='black')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundaries')
    
    def demonstrate_overfitting_control(self):
        """Demonstrate overfitting in Decision Trees and how to control it."""
        print("\n" + "="*70)
        print("OVERFITTING CONTROL DEMONSTRATION")
        print("="*70)
        
        print("""
📖 SCENARIO: Understanding and controlling overfitting in Decision Trees

🎯 GOAL: Show how unconstrained trees overfit and how to control it
📊 DATASET: Classification with clear patterns but some noise
🔧 APPROACH: Compare trees with different complexity constraints
        """)
        
        # Create data with some noise
        X, y = self.create_classification_data(n_samples=500, n_features=6, class_sep=0.8)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Train trees with different constraints
        trees = {}
        train_scores = {}
        test_scores = {}
        
        configurations = [
            ("Unconstrained", {}),
            ("max_depth=3", {"max_depth": 3}),
            ("max_depth=6", {"max_depth": 6}),
            ("min_samples_leaf=5", {"min_samples_leaf": 5}),
            ("max_depth=3, min_samples_leaf=5", {"max_depth": 3, "min_samples_leaf": 5})
        ]
        
        for name, params in configurations:
            tree = DecisionTreeClassifier(random_state=self.random_state, **params)
            tree.fit(X_train, y_train)
            
            trees[name] = tree
            train_scores[name] = tree.score(X_train, y_train)
            test_scores[name] = tree.score(X_test, y_test)
            
            gap = train_scores[name] - test_scores[name]
            
            print(f"\n{name}:")
            print(f"  Train Accuracy: {train_scores[name]:.3f}")
            print(f"  Test Accuracy:  {test_scores[name]:.3f}")
            print(f"  Train/Test Gap: {gap:.3f}")
            print(f"  Tree Depth:     {tree.get_depth()}")
            print(f"  Number of Leaves: {tree.get_n_leaves()}")
        
        # Visualize overfitting
        self._plot_overfitting_comparison(trees, train_scores, test_scores, X_train, y_train)
        
        return trees, train_scores, test_scores
    
    def _plot_overfitting_comparison(self, trees: Dict, train_scores: Dict, 
                                 test_scores: Dict, X_train: pd.DataFrame, y_train: pd.Series):
        """Plot overfitting comparison across different tree configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Train vs Test accuracy
        names = list(trees.keys())
        train_acc = [train_scores[name] for name in names]
        test_acc = [test_scores[name] for name in names]
        gaps = [train_scores[name] - test_scores[name] for name in names]
        
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_acc, width, label='Train Accuracy', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, test_acc, width, label='Test Accuracy', alpha=0.7)
        axes[0, 0].set_xlabel('Tree Configuration')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Train vs Test Accuracy')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Train/Test gaps
        axes[0, 1].bar(names, gaps, color='red', alpha=0.7)
        axes[0, 1].set_ylabel('Train/Test Gap')
        axes[0, 1].set_title('Overfitting Indicator (Gap)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Tree complexity
        depths = [trees[name].get_depth() for name in names]
        leaves = [trees[name].get_n_leaves() for name in names]
        
        axes[1, 0].plot(names, depths, 'bo-', label='Tree Depth')
        axes[1, 0].set_ylabel('Tree Depth')
        axes[1, 0].set_title('Tree Complexity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Number of leaves
        axes[1, 1].bar(names, leaves, color='green', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Leaves')
        axes[1, 1].set_title('Tree Size')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/overfitting_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_cross_validation_tuning(self):
        """Demonstrate cross-validation for hyperparameter tuning."""
        print("\n" + "="*70)
        print("CROSS-VALIDATION HYPERPARAMETER TUNING")
        print("="*70)
        
        print("""
📖 SCENARIO: Finding optimal tree depth using cross-validation

🎯 GOAL: Learn how to use GridSearchCV for hyperparameter tuning
📊 DATASET: Classification with varying complexity needs
🔧 APPROACH: Systematic search with bias-variance analysis
        """)
        
        # Create data
        X, y = self.create_classification_data(n_samples=800, n_features=8, class_sep=1.2)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Define parameter grid
        param_grid = {
            "max_depth": range(1, 21),
            "min_samples_leaf": [1, 5, 10, 20],
            "criterion": ["gini", "entropy"]
        }
        
        # Grid search with cross-validation
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )
        
        print("Running GridSearchCV...")
        start_time = time.time()
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        print(f"GridSearchCV completed in {grid_time:.2f} seconds")
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        
        # Analyze bias-variance trade-off
        results_df = pd.DataFrame(grid.cv_results_)
        
        # Plot bias-variance trade-off
        self._plot_bias_variance_tradeoff(results_df, grid.best_params_)
        
        # Evaluate best model on test set
        best_tree = grid.best_estimator_
        test_accuracy = best_tree.score(X_test, y_test)
        
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Train/Test gap: {grid.best_score_ - test_accuracy:.4f}")
        
        # Visualize best tree
        self._visualize_best_tree(best_tree, X_train, y_train)
        
        return grid, best_tree, test_accuracy
    
    def _plot_bias_variance_tradeoff(self, results_df: pd.DataFrame, best_params: Dict):
        """Plot bias-variance trade-off across hyperparameters."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Filter for best criterion
        best_criterion = best_params['criterion']
        criterion_results = results_df[results_df['param_criterion'] == best_criterion]
        
        # Plot 1: Depth vs. Train accuracy
        depths = criterion_results['param_max_depth']
        train_scores = criterion_results['mean_train_score']
        test_scores = criterion_results['mean_test_score']
        
        axes[0, 0].plot(depths, train_scores, 'b--o', label='Train Accuracy')
        axes[0, 0].plot(depths, test_scores, 'r-o', label='CV Accuracy')
        axes[0, 0].axvline(x=best_params['max_depth'], color='green', 
                           linestyle='--', label=f'Best Depth ({best_params["max_depth"]})')
        axes[0, 0].set_xlabel('Max Depth')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'Bias-Variance Trade-off ({best_criterion.capitalize()} Criterion)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Depth vs. Test score standard deviation
        test_stds = criterion_results['std_test_score']
        axes[0, 1].plot(depths, test_stds, 'g-o')
        axes[0, 1].set_xlabel('Max Depth')
        axes[0, 1].set_ylabel('Test Score Std Dev')
        axes[0, 1].set_title('Model Stability Across Depths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Min samples leaf effect
        min_leaf_results = results_df[results_df['param_min_samples_leaf'] == best_params['min_samples_leaf']]
        leaf_depths = min_leaf_results['param_max_depth']
        leaf_scores = min_leaf_results['mean_test_score']
        
        axes[1, 0].plot(leaf_depths, leaf_scores, 'mo-')
        axes[1, 0].set_xlabel('Max Depth')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].set_title(f'Effect of min_samples_leaf={best_params["min_samples_leaf"]}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Criterion comparison
        criterion_means = results_df.groupby('param_criterion')['mean_test_score'].mean()
        criterion_names = list(criterion_means.index)
        criterion_scores = list(criterion_means.values)
        
        axes[1, 1].bar(criterion_names, criterion_scores, 
                         color=['blue', 'orange'], alpha=0.7)
        axes[1, 1].set_ylabel('Mean CV Accuracy')
        axes[1, 1].set_title('Criterion Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/bias_variance_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_best_tree(self, tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series):
        """Visualize the best tree from cross-validation."""
        plt.figure(figsize=(20, 10))
        plot_tree(
            tree,
            feature_names=X.columns.tolist(),
            class_names=[f'Class {i}' for i in sorted(y.unique())],
            filled=True,
            rounded=True,
            fontsize=10,
            impurity=True
        )
        plt.title(f'Best Decision Tree (Depth={tree.get_depth()}, Leaves={tree.get_n_leaves()})')
        plt.savefig("plots/best_decision_tree.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_classification_workflow(self):
        """Demonstrate complete classification workflow."""
        print("\n" + "="*70)
        print("COMPLETE CLASSIFICATION WORKFLOW DEMONSTRATION")
        print("="*70)
        
        print("""
📖 SCENARIO: End-to-end Decision Tree classification workflow

🎯 GOAL: Complete workflow from training to interpretation
📊 DATASET: Customer churn prediction scenario
🔧 APPROACH: Train, evaluate, visualize, and interpret
        """)
        
        # Create realistic churn-like data
        X, y = self.create_classification_data(n_samples=1000, n_features=10, class_sep=1.0)
        
        # Rename features to be more realistic
        feature_mapping = {
            'feature_0': 'tenure',
            'feature_1': 'monthly_charges',
            'feature_2': 'contract_type',
            'feature_3': 'payment_method',
            'feature_4': 'internet_service',
            'feature_5': 'tech_support',
            'feature_6': 'total_charges',
            'feature_7': 'avg_daily_usage',
            'feature_8': 'customer_age',
            'feature_9': 'region'
        }
        
        X = X.rename(columns=feature_mapping)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Features: {list(X.columns)}")
        
        # Train Decision Tree
        tree = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            criterion="gini",
            random_state=self.random_state
        )
        
        print("Training Decision Tree...")
        start_time = time.time()
        tree.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.3f} seconds")
        
        # Evaluate
        y_pred = tree.predict(X_test)
        
        train_accuracy = tree.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, y_pred)
        train_test_gap = train_accuracy - test_accuracy
        
        print(f"\nPerformance Metrics:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Train/Test Gap: {train_test_gap:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, ['No Churn', 'Churn'])
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df.to_string(index=False))
        
        # Visualize tree
        self._visualize_best_tree(tree, X_train, y_train)
        
        # Compare to baseline
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_accuracy = baseline.score(X_test, y_test)
        
        print(f"\nBaseline Comparison:")
        print(f"  Majority Class Baseline: {baseline_accuracy:.4f}")
        print(f"  Decision Tree:         {test_accuracy:.4f}")
        print(f"  Improvement:            +{test_accuracy - baseline_accuracy:.4f}")
        
        return {
            'model': tree,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_test_gap': train_test_gap,
            'importance': importance_df,
            'baseline_accuracy': baseline_accuracy,
            'improvement': test_accuracy - baseline_accuracy
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_regression_workflow(self):
        """Demonstrate complete regression workflow."""
        print("\n" + "="*70)
        print("COMPLETE REGRESSION WORKFLOW DEMONSTRATION")
        print("="*70)
        
        print("""
📖 SCENARIO: Decision Tree regression for continuous target prediction

🎯 GOAL: Complete regression workflow with interpretation
📊 DATASET: House price prediction scenario
🔧 APPROACH: Train, evaluate, and analyze regression tree
        """)
        
        # Create regression data
        X, y = self.create_regression_data(n_samples=800, n_features=8, noise=0.2)
        
        # Rename features
        feature_mapping = {
            'feature_0': 'square_footage',
            'feature_1': 'num_bedrooms',
            'feature_2': 'num_bathrooms',
            'feature_3': 'age_of_house',
            'feature_4': 'distance_to_center',
            'feature_5': 'school_quality',
            'feature_6': 'crime_rate',
            'feature_7': 'tax_rate'
        }
        
        X = X.rename(columns=feature_mapping)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        
        # Train Decision Tree Regressor
        tree_reg = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=5,
            random_state=self.random_state
        )
        
        print("Training Decision Tree Regressor...")
        tree_reg.fit(X_train, y_train)
        
        # Evaluate
        y_pred = tree_reg.predict(X_test)
        
        train_r2 = tree_reg.score(X_train, y_train)
        test_r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nRegression Metrics:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  Train/Test Gap: {train_r2 - test_r2:.4f}")
        
        # Compare to baseline
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_r2 = r2_score(y_test, baseline.predict(X_test))
        
        print(f"\nBaseline Comparison:")
        print(f"  Mean Baseline R²: {baseline_r2:.4f}")
        print(f"  Tree R²:          {test_r2:.4f}")
        print(f"  Improvement:         +{test_r2 - baseline_r2:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': tree_reg.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df.to_string(index=False))
        
        # Visualize predictions
        self._plot_regression_predictions(y_test, y_pred)
        
        # Visualize tree
        plt.figure(figsize=(16, 8))
        plot_tree(
            tree_reg,
            feature_names=X.columns.tolist(),
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=3  # Limit depth for readability
        )
        plt.title(f'Decision Tree Regressor (Depth={tree_reg.get_depth()})')
        plt.savefig("plots/regression_tree.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'model': tree_reg,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'importance': importance_df,
            'baseline_r2': baseline_r2,
            'improvement': test_r2 - baseline_r2
        }
    
    def _plot_regression_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot regression predictions vs actual values."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of predictions
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Regression: Actual vs. Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.savefig("plots/regression_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_practical_checklist(self):
        """Print a practical checklist for Decision Tree implementation."""
        print("\n" + "="*70)
        print("PRACTICAL DECISION TREE CHECKLIST")
        print("="*70)
        
        checklist = """
🔍 PRE-TRAINING CHECKLIST:
□ Define clear problem objective (classification vs. regression)
□ Understand data quality and missing values
□ Check for class imbalance (classification)
□ Identify categorical vs. numerical features
□ Plan proper train/test split with stratification
□ Choose appropriate evaluation metrics

🌳 TRAINING CHECKLIST:
□ Set max_depth to control overfitting
□ Set min_samples_leaf for stable predictions
□ Choose criterion (gini vs. entropy) based on needs
□ Use random_state for reproducible results
□ Consider cross-validation for hyperparameter tuning

📊 EVALUATION CHECKLIST:
□ Compute both train and test accuracy
□ Calculate train/test gap to detect overfitting
□ Compare against baseline (majority class or mean)
□ Generate confusion matrix for classification
□ Calculate appropriate metrics (accuracy, F1, R², RMSE)
□ Check model stability with cross-validation

🎯 INTERPRETATION CHECKLIST:
□ Extract and analyze feature importance
□ Visualize tree structure for rule extraction
□ Validate important features with domain knowledge
□ Check for potential data leakage
□ Document tree depth and complexity
□ Compare multiple configurations before finalizing

🚨 COMMON MISTAKES TO AVOID:
□ Leaving max_depth unconstrained (causes severe overfitting)
□ Not checking train/test gap (hides overfitting)
□ Ignoring baseline comparison (no context for performance)
□ Not using stratification for imbalanced data
□ Interpreting feature importance as causation
□ Using single tree for final production (consider ensembles)
□ Not validating with domain experts
        """
        
        print(checklist)
    
    def run_complete_tutorial(self):
        """Run the complete Decision Tree training tutorial."""
        print("""
🎯 DECISION TREE TRAINING TUTORIAL
========================================

This tutorial provides comprehensive coverage of Decision Tree training,
evaluation, and interpretation for both classification and regression.

📚 TUTORIAL STRUCTURE:
1. Impurity Measures (Gini vs. Entropy)
2. Tree Growth Algorithm Demonstration
3. Overfitting Control and Visualization
4. Cross-Validation Hyperparameter Tuning
5. Complete Classification Workflow
6. Complete Regression Workflow
7. Practical Checklist and Best Practices

⏱️  EXPECTED DURATION: 60-75 minutes
        """)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Run all demonstrations
        demonstrations = [
            ("Impurity Measures", self.demonstrate_impurity_calculations),
            ("Tree Growth Algorithm", self.demonstrate_tree_growth_algorithm),
            ("Overfitting Control", self.demonstrate_overfitting_control),
            ("Cross-Validation Tuning", self.demonstrate_cross_validation_tuning),
            ("Classification Workflow", self.demonstrate_classification_workflow),
            ("Regression Workflow", self.demonstrate_regression_workflow)
        ]
        
        results = {}
        
        for i, (title, demo_func) in enumerate(demonstrations, 1):
            print(f"\n{'='*20} {i}. {title} {'='*20}")
            
            try:
                result = demo_func()
                results[title] = result
                
                if i < len(demonstrations):
                    input("\nPress Enter to continue to next demonstration...")
                    
            except Exception as e:
                print(f"Error in {title}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print practical checklist
        self.print_practical_checklist()
        
        # Summary
        print("\n" + "="*70)
        print("TUTORIAL COMPLETED!")
        print("="*70)
        
        print(f"\n📁 Files created: plots/")
        print("\n🎯 Key Takeaways:")
        takeaways = [
            "• Decision Trees partition feature space recursively using binary questions",
            "• Gini and Entropy measure node purity differently but produce similar results",
            "• Unconstrained trees always overfit - control complexity with max_depth",
            "• Cross-validation is essential for finding optimal hyperparameters",
            "• Always compare against baseline to understand real performance",
            "• Feature importance shows what the tree used, not what causes outcomes",
            "• Decision Trees are interpretable but often outperformed by ensembles",
            "• Train/test gap is your best overfitting indicator",
            "• Tree visualization helps validate model logic with domain experts"
        ]
        
        for takeaway in takeaways:
            print(f"  {takeaway}")
        
        return True


def main():
    """Main function to run the Decision Tree training tutorial."""
    trainer = DecisionTreeTrainer(random_state=42)
    return trainer.run_complete_tutorial()


if __name__ == "__main__":
    main()
