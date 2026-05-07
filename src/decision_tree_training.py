"""
Decision Tree Training and Interpretation - Complete Implementation

This module provides a comprehensive implementation of Decision Tree training,
visualization, and interpretation for both classification and regression problems.
It covers the complete workflow from understanding tree mechanics to practical
application and interpretation.

Key Features:
- Impurity measure calculations (Gini and Entropy)
- Tree growth algorithm visualization
- Overfitting control and hyperparameter tuning
- Cross-validation for optimal parameter selection
- Tree visualization and rule extraction
- Baseline comparison for performance context
- Practical workflows for classification and regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, precision_recall_curve, roc_curve, auc)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import make_classification, make_regression
from typing import Dict, Tuple, Any, Optional, List
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DecisionTreeTrainer:
    """
    Comprehensive Decision Tree training and interpretation system.
    
    This class provides methods for training, evaluating, and interpreting
    Decision Trees for both classification and regression tasks.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Decision Tree trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.classification_model = None
        self.regression_model = None
        self.feature_names = None
        self.class_names = None
        
    def calculate_gini_impurity(self, class_counts: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of class counts.
        
        Gini = 1 - Σ(p_i²)
        where p_i is the proportion of class i
        
        Args:
            class_counts: Array of class counts
            
        Returns:
            Gini impurity value
        """
        total_samples = np.sum(class_counts)
        if total_samples == 0:
            return 0.0
        
        class_proportions = class_counts / total_samples
        gini = 1.0 - np.sum(class_proportions ** 2)
        return gini
    
    def calculate_entropy_impurity(self, class_counts: np.ndarray) -> float:
        """
        Calculate entropy impurity for a set of class counts.
        
        Entropy = -Σ(p_i * log₂(p_i))
        where p_i is the proportion of class i
        
        Args:
            class_counts: Array of class counts
            
        Returns:
            Entropy impurity value
        """
        total_samples = np.sum(class_counts)
        if total_samples == 0:
            return 0.0
        
        class_proportions = class_counts / total_samples
        # Avoid log(0) by filtering out zero proportions
        class_proportions = class_proportions[class_proportions > 0]
        entropy = -np.sum(class_proportions * np.log2(class_proportions))
        return entropy
    
    def demonstrate_impurity_calculations(self) -> Dict[str, Any]:
        """
        Demonstrate impurity calculations across different class distributions.
        
        Returns:
            Dictionary containing impurity calculations and visualizations
        """
        print("=" * 60)
        print("IMPURITY MEASURES DEMONSTRATION")
        print("=" * 60)
        
        # Create different class distributions
        distributions = [
            np.array([100, 0]),    # Perfectly pure
            np.array([80, 20]),    # Mostly pure
            np.array([60, 40]),    # Mixed
            np.array([50, 50]),    # Maximum impurity (binary)
            np.array([70, 20, 10]), # Multi-class
            np.array([33, 33, 34]), # Balanced multi-class
        ]
        
        results = {
            'distributions': [],
            'gini_values': [],
            'entropy_values': [],
            'comparison_data': []
        }
        
        print("\nImpurity Calculations for Different Class Distributions:")
        print("-" * 60)
        
        for i, class_counts in enumerate(distributions):
            gini = self.calculate_gini_impurity(class_counts)
            entropy = self.calculate_entropy_impurity(class_counts)
            
            results['distributions'].append(class_counts.tolist())
            results['gini_values'].append(gini)
            results['entropy_values'].append(entropy)
            
            print(f"Distribution {i+1}: {class_counts}")
            print(f"  Gini: {gini:.4f}")
            print(f"  Entropy: {entropy:.4f}")
            print()
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Binary classification comparison
        binary_range = np.arange(0, 101, 2)
        gini_binary = []
        entropy_binary = []
        
        for p in binary_range:
            if p == 0 or p == 100:
                gini_binary.append(0)
                entropy_binary.append(0)
            else:
                class_counts = np.array([p, 100-p])
                gini_binary.append(self.calculate_gini_impurity(class_counts))
                entropy_binary.append(self.calculate_entropy_impurity(class_counts))
        
        ax1.plot(binary_range, gini_binary, 'b-', linewidth=2, label='Gini')
        ax1.plot(binary_range, entropy_binary, 'r-', linewidth=2, label='Entropy')
        ax1.set_xlabel('Class 0 Percentage (%)')
        ax1.set_ylabel('Impurity')
        ax1.set_title('Impurity Measures - Binary Classification')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Comparison plot
        ax2.scatter(results['gini_values'], results['entropy_values'], 
                   c=range(len(results['gini_values'])), cmap='viridis', s=100)
        ax2.plot([0, 0.5], [0, 1], 'k--', alpha=0.5, label='Theoretical Max')
        ax2.set_xlabel('Gini Impurity')
        ax2.set_ylabel('Entropy Impurity')
        ax2.set_title('Gini vs. Entropy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for each point
        for i, (g, e) in enumerate(zip(results['gini_values'], results['entropy_values'])):
            ax2.annotate(f'{i+1}', (g, e), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
        
        results['comparison_data'] = {
            'binary_range': binary_range.tolist(),
            'gini_binary': gini_binary,
            'entropy_binary': entropy_binary
        }
        
        return results
    
    def demonstrate_tree_growth_algorithm(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Demonstrate the Decision Tree growth algorithm step by step.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary containing tree growth visualization and analysis
        """
        print("=" * 60)
        print("DECISION TREE GROWTH ALGORITHM DEMONSTRATION")
        print("=" * 60)
        
        # Generate synthetic data for visualization
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=self.random_state,
            n_clusters_per_class=1,
            class_sep=1.0
        )
        
        # Create DataFrame for easier handling
        feature_names = ['Feature_1', 'Feature_2']
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        print(f"Generated dataset: {n_samples} samples, 2 features")
        print(f"Class distribution: {np.bincount(y)}")
        print()
        
        # Train trees with different depths to show growth
        depths = [1, 2, 3, 4, None]  # None means unconstrained
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        results = {
            'trees': [],
            'depths': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'train_test_gaps': []
        }
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        for i, depth in enumerate(depths):
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )
            tree.fit(X_train, y_train)
            
            # Calculate metrics
            train_acc = tree.score(X_train, y_train)
            test_acc = tree.score(X_test, y_test)
            gap = train_acc - test_acc
            
            results['trees'].append(tree)
            results['depths'].append(depth if depth is not None else 'unconstrained')
            results['train_accuracies'].append(train_acc)
            results['test_accuracies'].append(test_acc)
            results['train_test_gaps'].append(gap)
            
            # Plot decision boundary
            ax = axes[i]
            
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = tree.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                      cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                      cmap='RdYlBu', marker='x', s=100, alpha=0.8)
            
            depth_label = depth if depth is not None else '∞'
            ax.set_title(f'Depth = {depth_label}\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}, Gap: {gap:.3f}')
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
        
        # Remove empty subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print growth analysis
        print("Tree Growth Analysis:")
        print("-" * 40)
        for i, depth in enumerate(results['depths']):
            print(f"Depth {depth}:")
            print(f"  Train Accuracy: {results['train_accuracies'][i]:.3f}")
            print(f"  Test Accuracy:  {results['test_accuracies'][i]:.3f}")
            print(f"  Train/Test Gap: {results['train_test_gaps'][i]:.3f}")
            print()
        
        return results
    
    def demonstrate_overfitting_control(self) -> Dict[str, Any]:
        """
        Demonstrate overfitting problems and control strategies.
        
        Returns:
            Dictionary containing overfitting analysis and control strategies
        """
        print("=" * 60)
        print("OVERFITTING CONTROL DEMONSTRATION")
        print("=" * 60)
        
        # Generate more complex data
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=self.random_state,
            flip_y=0.1,  # Add some noise
            class_sep=0.8
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Test different configurations
        configurations = [
            {'max_depth': None, 'min_samples_leaf': 1, 'name': 'Unconstrained'},
            {'max_depth': 10, 'min_samples_leaf': 1, 'name': 'Deep Tree'},
            {'max_depth': 5, 'min_samples_leaf': 1, 'name': 'Medium Tree'},
            {'max_depth': 3, 'min_samples_leaf': 5, 'name': 'Conservative'},
            {'max_depth': 2, 'min_samples_leaf': 10, 'name': 'Very Conservative'},
        ]
        
        results = {
            'configurations': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'gaps': [],
            'tree_sizes': []
        }
        
        print("Overfitting Analysis Across Configurations:")
        print("-" * 60)
        
        for config in configurations:
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=config['max_depth'],
                min_samples_leaf=config['min_samples_leaf'],
                random_state=self.random_state
            )
            tree.fit(X_train, y_train)
            
            # Calculate metrics
            train_acc = tree.score(X_train, y_train)
            test_acc = tree.score(X_test, y_test)
            gap = train_acc - test_acc
            tree_size = tree.get_n_leaves()
            
            results['configurations'].append(config['name'])
            results['train_accuracies'].append(train_acc)
            results['test_accuracies'].append(test_acc)
            results['gaps'].append(gap)
            results['tree_sizes'].append(tree_size)
            
            print(f"{config['name']}:")
            print(f"  Train: {train_acc:.3f}, Test: {test_acc:.3f}, Gap: {gap:.3f}")
            print(f"  Tree Size (leaves): {tree_size}")
            print()
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        x_pos = np.arange(len(results['configurations']))
        width = 0.35
        
        ax1.bar(x_pos - width/2, results['train_accuracies'], width, 
                label='Train Accuracy', alpha=0.8)
        ax1.bar(x_pos + width/2, results['test_accuracies'], width, 
                label='Test Accuracy', alpha=0.8)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Train vs Test Accuracy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(results['configurations'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Train/Test Gap
        ax2.bar(results['configurations'], results['gaps'], alpha=0.8, color='red')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Train/Test Gap')
        ax2.set_title('Overfitting Indicator (Gap)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Tree Complexity
        ax3.bar(results['configurations'], results['tree_sizes'], alpha=0.8, color='green')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Number of Leaves')
        ax3.set_title('Tree Complexity')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Bias-Variance Trade-off
        ax4.scatter(results['gaps'], results['test_accuracies'], 
                   s=100, alpha=0.8, c=range(len(results['gaps'])), cmap='viridis')
        for i, config in enumerate(results['configurations']):
            ax4.annotate(config, (results['gaps'][i], results['test_accuracies'][i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Train/Test Gap (Variance)')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Bias-Variance Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def demonstrate_cross_validation_tuning(self) -> Dict[str, Any]:
        """
        Demonstrate cross-validation for hyperparameter tuning.
        
        Returns:
            Dictionary containing CV results and optimal parameters
        """
        print("=" * 60)
        print("CROSS-VALIDATION HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Generate data
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            random_state=self.random_state,
            flip_y=0.15
        )
        
        # Define parameter grid
        param_grid = {
            'max_depth': range(1, 11),
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'criterion': ['gini', 'entropy']
        }
        
        # Perform grid search
        tree = DecisionTreeClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            tree,
            param_grid,
            cv=5,
            scoring='accuracy',
            return_train_score=True,
            n_jobs=-1
        )
        
        print("Performing grid search...")
        grid_search.fit(X, y)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"Best Parameters: {results['best_params']}")
        print(f"Best CV Score: {results['best_score']:.4f}")
        print()
        
        # Analyze depth vs performance
        depth_results = {}
        for depth in range(1, 11):
            mask = grid_search.cv_results_['param_max_depth'] == depth
            depth_results[depth] = {
                'mean_test_score': np.mean(grid_search.cv_results_['mean_test_score'][mask]),
                'mean_train_score': np.mean(grid_search.cv_results_['mean_train_score'][mask])
            }
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Depth vs Accuracy
        depths = list(depth_results.keys())
        test_scores = [depth_results[d]['mean_test_score'] for d in depths]
        train_scores = [depth_results[d]['mean_train_score'] for d in depths]
        
        ax1.plot(depths, train_scores, 'o-', label='Train Score', linewidth=2)
        ax1.plot(depths, test_scores, 'o-', label='CV Test Score', linewidth=2)
        ax1.set_xlabel('Max Depth')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Depth vs Accuracy (Bias-Variance Trade-off)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Min samples leaf analysis
        leaf_sizes = [1, 2, 5, 10, 20]
        leaf_results = {}
        for leaf_size in leaf_sizes:
            mask = grid_search.cv_results_['param_min_samples_leaf'] == leaf_size
            leaf_results[leaf_size] = np.mean(grid_search.cv_results_['mean_test_score'][mask])
        
        ax2.bar(leaf_sizes, [leaf_results[l] for l in leaf_sizes], alpha=0.8)
        ax2.set_xlabel('Min Samples Leaf')
        ax2.set_ylabel('Mean CV Accuracy')
        ax2.set_title('Min Samples Leaf vs Performance')
        ax2.grid(True, alpha=0.3)
        
        # Criterion comparison
        gini_mask = grid_search.cv_results_['param_criterion'] == 'gini'
        entropy_mask = grid_search.cv_results_['param_criterion'] == 'entropy'
        
        gini_scores = grid_search.cv_results_['mean_test_score'][gini_mask]
        entropy_scores = grid_search.cv_results_['mean_test_score'][entropy_mask]
        
        ax3.boxplot([gini_scores, entropy_scores], labels=['Gini', 'Entropy'])
        ax3.set_ylabel('CV Accuracy')
        ax3.set_title('Gini vs Entropy Performance')
        ax3.grid(True, alpha=0.3)
        
        # Heatmap of best combinations
        pivot_data = {}
        for depth in range(1, 11):
            pivot_data[depth] = {}
            for leaf in leaf_sizes:
                mask = (grid_search.cv_results_['param_max_depth'] == depth) & \
                       (grid_search.cv_results_['param_min_samples_leaf'] == leaf)
                if np.any(mask):
                    pivot_data[depth][leaf] = np.max(grid_search.cv_results_['mean_test_score'][mask])
                else:
                    pivot_data[depth][leaf] = 0
        
        pivot_df = pd.DataFrame(pivot_data).T
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
        ax4.set_title('Performance Heatmap (Depth vs Min Samples Leaf)')
        ax4.set_xlabel('Min Samples Leaf')
        ax4.set_ylabel('Max Depth')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def demonstrate_classification_workflow(self) -> Dict[str, Any]:
        """
        Complete classification workflow with baseline comparison.
        
        Returns:
            Dictionary containing complete classification results
        """
        print("=" * 60)
        print("CLASSIFICATION WORKFLOW DEMONSTRATION")
        print("=" * 60)
        
        # Generate realistic data
        X, y = make_classification(
            n_samples=1000,
            n_features=15,
            n_informative=8,
            n_redundant=3,
            n_repeated=0,
            random_state=self.random_state,
            flip_y=0.1,
            class_sep=0.9
        )
        
        # Create feature names
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        self.feature_names = feature_names
        self.class_names = ['Class_0', 'Class_1']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Class distribution: {np.bincount(y)}")
        print()
        
        # Train decision tree with optimal parameters
        tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            criterion='gini',
            random_state=self.random_state
        )
        tree.fit(X_train, y_train)
        self.classification_model = tree
        
        # Make predictions
        y_pred = tree.predict(X_test)
        y_proba = tree.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_acc = tree.score(X_train, y_train)
        test_acc = accuracy_score(y_test, y_pred)
        gap = train_acc - test_acc
        
        # Baseline comparison
        baseline = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        baseline.fit(X_train, y_train)
        baseline_acc = baseline.score(X_test, y_test)
        
        results = {
            'model': tree,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_test_gap': gap,
            'baseline_accuracy': baseline_acc,
            'improvement': test_acc - baseline_acc,
            'predictions': y_pred,
            'probabilities': y_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self._extract_feature_importance(tree, feature_names)
        }
        
        # Print results
        print("Classification Results:")
        print("-" * 40)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Train/Test Gap: {gap:.4f}")
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"Improvement:     +{results['improvement']:.4f}")
        print()
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Cross-validation
        cv_scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
        results['cv_scores'] = cv_scores
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print()
        
        # Feature importance
        print("Top 10 Feature Importances:")
        print("-" * 30)
        importance_df = results['feature_importance'].head(10)
        print(importance_df.to_string(index=False))
        print()
        
        # Visualizations
        self._visualize_classification_results(results, X_test, y_test)
        
        return results
    
    def demonstrate_regression_workflow(self) -> Dict[str, Any]:
        """
        Complete regression workflow with baseline comparison.
        
        Returns:
            Dictionary containing complete regression results
        """
        print("=" * 60)
        print("REGRESSION WORKFLOW DEMONSTRATION")
        print("=" * 60)
        
        # Generate regression data
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=6,
            noise=0.1,
            random_state=self.random_state
        )
        
        # Create feature names
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print()
        
        # Train decision tree regressor
        tree_reg = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=5,
            random_state=self.random_state
        )
        tree_reg.fit(X_train, y_train)
        self.regression_model = tree_reg
        
        # Make predictions
        y_pred = tree_reg.predict(X_test)
        
        # Calculate metrics
        train_r2 = tree_reg.score(X_train, y_train)
        test_r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        gap = train_r2 - test_r2
        
        # Baseline comparison
        baseline = DummyRegressor(strategy='mean')
        baseline.fit(X_train, y_train)
        baseline_r2 = baseline.score(X_test, y_test)
        
        results = {
            'model': tree_reg,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'train_test_gap': gap,
            'baseline_r2': baseline_r2,
            'improvement': test_r2 - baseline_r2,
            'predictions': y_pred,
            'feature_importance': self._extract_feature_importance(tree_reg, feature_names)
        }
        
        # Print results
        print("Regression Results:")
        print("-" * 30)
        print(f"Train R²:      {train_r2:.4f}")
        print(f"Test R²:       {test_r2:.4f}")
        print(f"RMSE:          {rmse:.4f}")
        print(f"Train/Test Gap:{gap:.4f}")
        print(f"Baseline R²:   {baseline_r2:.4f}")
        print(f"Improvement:   +{results['improvement']:.4f}")
        print()
        
        # Feature importance
        print("Top 10 Feature Importances:")
        print("-" * 30)
        importance_df = results['feature_importance'].head(10)
        print(importance_df.to_string(index=False))
        print()
        
        # Visualizations
        self._visualize_regression_results(results, y_test, y_pred)
        
        return results
    
    def _extract_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract and format feature importance."""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    
    def _visualize_classification_results(self, results: Dict, X_test: np.ndarray, y_test: np.ndarray):
        """Visualize classification results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature Importance
        top_features = results['feature_importance'].head(10)
        ax3.barh(top_features['Feature'], top_features['Importance'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 10 Feature Importances')
        ax3.grid(True, alpha=0.3)
        
        # Tree Visualization (simplified)
        tree = results['model']
        plot_tree(tree, max_depth=3, feature_names=self.feature_names,
                 class_names=self.class_names, filled=True, rounded=True,
                 fontsize=8, ax=ax4)
        ax4.set_title('Decision Tree Structure (Top 3 Levels)')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_regression_results(self, results: Dict, y_test: np.ndarray, y_pred: np.ndarray):
        """Visualize regression results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # Feature Importance
        top_features = results['feature_importance'].head(10)
        ax3.barh(top_features['Feature'], top_features['Importance'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 10 Feature Importances')
        ax3.grid(True, alpha=0.3)
        
        # Tree Visualization (simplified)
        tree = results['model']
        plot_tree(tree, max_depth=3, feature_names=self.feature_names,
                 filled=True, rounded=True, fontsize=8, ax=ax4)
        ax4.set_title('Decision Tree Structure (Top 3 Levels)')
        
        plt.tight_layout()
        plt.show()
    
    def export_tree_rules(self, model_type: str = 'classification') -> str:
        """
        Export tree rules in human-readable format.
        
        Args:
            model_type: 'classification' or 'regression'
            
        Returns:
            String containing tree rules
        """
        if model_type == 'classification' and self.classification_model is not None:
            model = self.classification_model
        elif model_type == 'regression' and self.regression_model is not None:
            model = self.regression_model
        else:
            raise ValueError(f"No trained {model_type} model available")
        
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        
        rules = export_text(model, feature_names=self.feature_names)
        
        print("=" * 60)
        print(f"DECISION TREE RULES - {model_type.upper()}")
        print("=" * 60)
        print(rules)
        
        return rules


def main():
    """
    Main function to run the complete Decision Tree training demonstration.
    """
    # Initialize trainer
    trainer = DecisionTreeTrainer(random_state=42)
    
    print("🌳 DECISION TREE TRAINING AND INTERPRETATION DEMONSTRATION")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrations = [
        ("Impurity Measures", trainer.demonstrate_impurity_calculations),
        ("Tree Growth Algorithm", trainer.demonstrate_tree_growth_algorithm),
        ("Overfitting Control", trainer.demonstrate_overfitting_control),
        ("Cross-Validation Tuning", trainer.demonstrate_cross_validation_tuning),
        ("Classification Workflow", trainer.demonstrate_classification_workflow),
        ("Regression Workflow", trainer.demonstrate_regression_workflow),
    ]
    
    for name, demo_func in demonstrations:
        print(f"\n🔍 {name}")
        print("=" * 80)
        try:
            results = demo_func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Export tree rules
    if trainer.classification_model is not None:
        trainer.export_tree_rules('classification')
    
    if trainer.regression_model is not None:
        trainer.export_tree_rules('regression')
    
    print("\n🎉 DECISION TREE TRAINING DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Decision Trees use impurity measures (Gini/Entropy) to make splits")
    print("2. Without constraints, trees overfit severely (100% train accuracy)")
    print("3. Cross-validation helps find optimal hyperparameters")
    print("4. Always compare against baseline models")
    print("5. Tree visualization provides interpretable rules")
    print("6. Feature importance shows what drives predictions")
    print("7. Decision Trees are foundational for ensembles (Random Forest, Gradient Boosting)")


if __name__ == "__main__":
    main()
