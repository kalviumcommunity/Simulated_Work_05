"""
Decision Tree Examples and Utilities

Additional practical examples and utility functions for Decision Tree models
including advanced visualizations, comparisons, and real-world scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, classification_report, 
                             mean_squared_error, r2_score)
from sklearn.datasets import make_classification, load_iris, load_boston
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings('ignore')


class DecisionTreeExamples:
    """Collection of advanced Decision Tree examples and utilities."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def example_learning_curves(self):
        """Example: Learning curves for Decision Trees."""
        print("\n" + "="*60)
        print("EXAMPLE: DECISION TREE LEARNING CURVES")
        print("="*60)
        
        # Create data
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=6,
            random_state=self.random_state
        )
        
        # Train models with different training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        tree = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)
        
        # Generate learning curves
        train_sizes_abs, train_scores, test_scores = learning_curve(
            tree, X, y, cv=5, train_sizes=train_sizes,
            scoring='accuracy', random_state=self.random_state
        )
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', 
                label='Training Score')
        plt.fill_between(train_sizes_abs, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Training Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_sizes_abs, test_scores.mean(axis=1), 'o-', 
                label='Cross-Validation Score')
        plt.fill_between(train_sizes_abs,
                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                        test_scores.mean(axis=1) + test_scores.std(axis=1),
                        alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Validation Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/learning_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analysis
        final_train_score = train_scores.mean(axis=1)[-1]
        final_test_score = test_scores.mean(axis=1)[-1]
        overfitting_gap = final_train_score - final_test_score
        
        print(f"Final Training Score: {final_train_score:.4f}")
        print(f"Final CV Score:      {final_test_score:.4f}")
        print(f"Overfitting Gap:     {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            print("⚠️  Significant overfitting detected")
        elif overfitting_gap > 0.05:
            print("⚠️  Moderate overfitting detected")
        else:
            print("✅ Good generalization (low overfitting)")
        
        return train_sizes_abs, train_scores, test_scores
    
    def example_tree_rules_extraction(self):
        """Example: Extract human-readable rules from Decision Tree."""
        print("\n" + "="*60)
        print("EXAMPLE: DECISION TREE RULES EXTRACTION")
        print("="*60)
        
        # Create interpretable data
        X, y = make_classification(
            n_samples=500, n_features=5, n_informative=3,
            random_state=self.random_state
        )
        
        feature_names = ['age', 'income', 'education', 'experience', 'credit_score']
        X = pd.DataFrame(X, columns=feature_names)
        
        # Train shallow tree for clear rules
        tree = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=10, 
            random_state=self.random_state
        )
        tree.fit(X, y)
        
        # Extract rules using export_text
        rules = export_text(
            tree,
            feature_names=feature_names,
            class_names=['No', 'Yes'],
            show_weights=True
        )
        
        print("Extracted Decision Rules:")
        print("="*50)
        print(rules)
        
        # Parse and format rules
        formatted_rules = self._format_tree_rules(rules, feature_names)
        print("\nFormatted Rules:")
        print("="*50)
        for rule in formatted_rules:
            print(rule)
        
        # Evaluate rule quality
        y_pred = tree.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\nRule-based Model Accuracy: {accuracy:.4f}")
        
        return tree, formatted_rules
    
    def _format_tree_rules(self, rules_text: str, feature_names: List[str]) -> List[str]:
        """Format tree rules into human-readable format."""
        rules = []
        lines = rules_text.split('\n')
        
        for line in lines:
            if '|---' in line and 'class:' in line:
                # This is a leaf node rule
                rule_parts = line.strip().split('|')
                if len(rule_parts) >= 2:
                    condition = rule_parts[0].strip()
                    outcome = rule_parts[1].split('class:')[1].strip()
                    
                    # Format condition
                    formatted_condition = self._format_condition(condition, feature_names)
                    rules.append(f"IF {formatted_condition} THEN Predict = {outcome}")
        
        return rules
    
    def _format_condition(self, condition: str, feature_names: List[str]) -> str:
        """Format a single condition into readable format."""
        # Replace feature indices with names
        for i, name in enumerate(feature_names):
            condition = condition.replace(f'feature_{i}', name)
        
        # Replace operators
        condition = condition.replace('<= ', ' is less than or equal to ')
        condition = condition.replace('> ', ' is greater than ')
        
        return condition
    
    def example_feature_interactions(self):
        """Example: How Decision Trees capture feature interactions."""
        print("\n" + "="*60)
        print("EXAMPLE: FEATURE INTERACTIONS IN DECISION TREES")
        print("="*60)
        
        # Create data with clear interactions
        np.random.seed(self.random_state)
        n_samples = 1000
        
        # Create interacting features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        
        # Create target with interaction
        y = ((X1 > 0) & (X2 > 0)).astype(int)
        
        # Add noise features
        X3 = np.random.normal(0, 1, n_samples)
        X4 = np.random.normal(0, 1, n_samples)
        
        X = pd.DataFrame({
            'feature_1': X1,
            'feature_2': X2,
            'noise_feature_1': X3,
            'noise_feature_2': X4
        })
        y = pd.Series(y, name='target')
        
        print("Generated data with interaction:")
        print("Target = 1 only when BOTH feature_1 > 0 AND feature_2 > 0")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train tree
        tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        tree.fit(X, y)
        
        # Analyze how tree captures interaction
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Data with interaction
        colors = ['red' if val == 1 else 'blue' for val in y]
        axes[0, 0].scatter(X['feature_1'], X['feature_2'], c=colors, alpha=0.6)
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].set_title('Data with Feature Interaction')
        
        # Plot 2: Decision boundaries
        self._plot_interaction_boundaries(tree, X, axes[0, 1])
        
        # Plot 3: Feature importance
        axes[1, 0].barh(feature_importance['Feature'], feature_importance['Importance'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].invert_yaxis()
        
        # Plot 4: Tree structure
        from sklearn.tree import plot_tree
        plot_tree(tree, feature_names=X.columns, class_names=['0', '1'],
                 filled=True, rounded=True, ax=axes[1, 1])
        axes[1, 1].set_title('Tree Structure')
        
        plt.tight_layout()
        plt.savefig("plots/feature_interactions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return tree, feature_importance
    
    def _plot_interaction_boundaries(self, tree, X: pd.DataFrame, ax):
        """Plot decision boundaries for interaction example."""
        # Create mesh
        x_min, x_max = X['feature_1'].min() - 1, X['feature_1'].max() + 1
        y_min, y_max = X['feature_2'].min() - 1, X['feature_2'].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Predict on mesh
        mesh_data = pd.DataFrame({'feature_1': xx.ravel(), 'feature_2': yy.ravel()})
        Z = tree.predict(mesh_data).reshape(xx.shape)
        
        # Plot boundaries
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[0.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors='red')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundaries (Red Line = Decision Boundary)')
    
    def example_comparison_with_random_forest(self):
        """Example: Compare Decision Tree with Random Forest."""
        print("\n" + "="*60)
        print("EXAMPLE: DECISION TREE VS RANDOM FOREST")
        print("="*60)
        
        # Create challenging data
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=8,
            n_redundant=3, flip_y=0.1, random_state=self.random_state
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        
        # Train both models
        print("Training Decision Tree...")
        dt = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
        dt.fit(X, y)
        
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, 
            random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        
        # Evaluate both
        dt_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
        rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        
        print(f"\nCross-Validation Results:")
        print(f"Decision Tree:  {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
        print(f"Random Forest: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
        print(f"Improvement:    +{rf_scores.mean() - dt_scores.mean():.4f}")
        
        # Compare stability
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        dt_importance = pd.DataFrame({
            'Feature': feature_names,
            'DT_Importance': dt.feature_importances_
        })
        
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'RF_Importance': rf.feature_importances_
        })
        
        importance_comparison = dt_importance.merge(rf_importance, on='Feature')
        
        print(f"\nFeature Importance Comparison:")
        print(importance_comparison.round(4).to_string(index=False))
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Performance comparison
        models = ['Decision Tree', 'Random Forest']
        scores = [dt_scores.mean(), rf_scores.mean()]
        errors = [dt_scores.std(), rf_scores.std()]
        
        axes[0, 0].bar(models, scores, yerr=errors, capsize=5, alpha=0.7)
        axes[0, 0].set_ylabel('CV Accuracy')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Importance correlation
        top_features = importance_comparison.nlargest(10, 'DT_Importance')['Feature']
        dt_top = importance_comparison[importance_comparison['Feature'].isin(top_features)]['DT_Importance']
        rf_top = importance_comparison[importance_comparison['Feature'].isin(top_features)]['RF_Importance']
        
        axes[0, 1].scatter(dt_top, rf_top, alpha=0.6)
        axes[0, 1].plot([0, max(dt_top.max(), rf_top.max())], 
                           [0, max(dt_top.max(), rf_top.max())], 'r--')
        axes[0, 1].set_xlabel('Decision Tree Importance')
        axes[0, 1].set_ylabel('Random Forest Importance')
        axes[0, 1].set_title('Importance Correlation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature importance stability
        axes[1, 0].barh(range(10), dt_top, alpha=0.7, label='Decision Tree')
        axes[1, 0].barh(range(10), rf_top, alpha=0.7, label='Random Forest')
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels(top_features)
        axes[1, 0].set_title('Top 10 Features Importance')
        axes[1, 0].legend()
        axes[1, 0].invert_yaxis()
        
        # Plot 4: Tree vs Forest complexity
        axes[1, 1].bar(['Decision Tree', 'Random Forest'], 
                         [dt.get_n_leaves(), rf.estimators_[0].get_n_leaves() * 100], # Approximate total leaves
                         color=['blue', 'green'], alpha=0.7)
        axes[1, 1].set_ylabel('Total Leaves (Approximate)')
        axes[1, 1].set_title('Model Complexity')
        
        plt.tight_layout()
        plt.savefig("plots/tree_vs_forest.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'decision_tree': dt,
            'random_forest': rf,
            'dt_scores': dt_scores,
            'rf_scores': rf_scores,
            'importance_comparison': importance_comparison
        }
    
    def example_regression_vs_classification(self):
        """Example: Compare Decision Tree behavior in classification vs regression."""
        print("\n" + "="*60)
        print("EXAMPLE: CLASSIFICATION VS REGRESSION TREES")
        print("="*60)
        
        # Create similar data for both tasks
        X_class, y_class = make_classification(
            n_samples=800, n_features=6, n_informative=4,
            random_state=self.random_state
        )
        
        X_reg, y_reg = make_regression(
            n_samples=800, n_features=6, n_informative=4,
            noise=0.1, random_state=self.random_state
        )
        
        # Train both models
        dt_class = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)
        dt_reg = DecisionTreeRegressor(max_depth=4, random_state=self.random_state)
        
        dt_class.fit(X_class, y_class)
        dt_reg.fit(X_reg, y_reg)
        
        # Compare characteristics
        print("Model Characteristics Comparison:")
        print("-" * 50)
        
        class_metrics = {
            'Model Type': 'Classification',
            'Tree Depth': dt_class.get_depth(),
            'Number of Leaves': dt_class.get_n_leaves(),
            'Training Samples': len(y_class)
        }
        
        reg_metrics = {
            'Model Type': 'Regression',
            'Tree Depth': dt_reg.get_depth(),
            'Number of Leaves': dt_reg.get_n_leaves(),
            'Training Samples': len(y_reg)
        }
        
        comparison_df = pd.DataFrame([class_metrics, reg_metrics])
        print(comparison_df.to_string(index=False))
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Classification tree
        from sklearn.tree import plot_tree
        plot_tree(dt_class, filled=True, rounded=True, 
                 feature_names=[f'f_{i}' for i in range(6)], 
                 class_names=['0', '1'], ax=axes[0, 0])
        axes[0, 0].set_title('Classification Tree')
        
        # Regression tree
        plot_tree(dt_reg, filled=True, rounded=True,
                 feature_names=[f'f_{i}' for i in range(6)], ax=axes[0, 1])
        axes[0, 1].set_title('Regression Tree')
        
        # Comparison metrics
        metrics = ['Tree Depth', 'Number of Leaves']
        class_values = [class_metrics[metric] for metric in metrics]
        reg_values = [reg_metrics[metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, class_values, width, label='Classification')
        axes[1, 0].bar(x + width/2, reg_values, width, label='Regression')
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_title('Model Characteristics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/class_vs_regression.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def example_pruning_effects(self):
        """Example: Effects of pruning on Decision Trees."""
        print("\n" + "="*60)
        print("EXAMPLE: DECISION TREE PRUNING EFFECTS")
        print("="*60)
        
        # Create data
        X, y = make_classification(
            n_samples=1000, n_features=8, n_informative=5,
            random_state=self.random_state
        )
        
        # Train trees with different pruning parameters
        configurations = [
            ("No Pruning", {}),
            ("min_samples_leaf=1", {"min_samples_leaf": 1}),
            ("min_samples_leaf=5", {"min_samples_leaf": 5}),
            ("min_samples_leaf=10", {"min_samples_leaf": 10}),
            ("min_samples_leaf=20", {"min_samples_leaf": 20}),
            ("max_depth=3", {"max_depth": 3}),
            ("max_depth=6", {"max_depth": 6}),
        ]
        
        results = {}
        
        for name, params in configurations:
            tree = DecisionTreeClassifier(random_state=self.random_state, **params)
            
            # Cross-validation
            cv_scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'depth': tree.get_depth(),
                'leaves': tree.get_n_leaves(),
                'params': params
            }
            
            print(f"\n{name}:")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Tree Depth:   {tree.get_depth()}")
            print(f"  Leaves:        {tree.get_n_leaves()}")
        
        # Visualize pruning effects
        self._plot_pruning_effects(results)
        
        return results
    
    def _plot_pruning_effects(self, results: Dict):
        """Plot the effects of different pruning strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(results.keys())
        cv_means = [results[name]['cv_mean'] for name in names]
        cv_stds = [results[name]['cv_std'] for name in names]
        depths = [results[name]['depth'] for name in names]
        leaves = [results[name]['leaves'] for name in names]
        
        # Plot 1: CV accuracy
        axes[0, 0].bar(names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_ylabel('CV Accuracy')
        axes[0, 0].set_title('Pruning Effects on Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Tree depth
        axes[0, 1].bar(names, depths, color='green', alpha=0.7)
        axes[0, 1].set_ylabel('Tree Depth')
        axes[0, 1].set_title('Pruning Effects on Complexity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Number of leaves
        axes[1, 0].bar(names, leaves, color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Number of Leaves')
        axes[1, 0].set_title('Pruning Effects on Tree Size')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs. Complexity
        axes[1, 1].scatter(depths, cv_means, s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Tree Depth')
        axes[1, 1].set_ylabel('CV Accuracy')
        axes[1, 1].set_title('Accuracy vs. Complexity Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/pruning_effects.png", dpi=300, bbox_inches='tight')
        plt.show()


def run_all_examples():
    """Run all Decision Tree examples."""
    print("""
🎯 DECISION TREE EXAMPLES
========================

Additional practical examples for Decision Tree models.
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    examples = DecisionTreeExamples(random_state=42)
    
    # Run all examples
    example_funcs = [
        ("Learning Curves", examples.example_learning_curves),
        ("Rules Extraction", examples.example_tree_rules_extraction),
        ("Feature Interactions", examples.example_feature_interactions),
        ("Comparison with Random Forest", examples.example_comparison_with_random_forest),
        ("Classification vs Regression", examples.example_regression_vs_classification),
        ("Pruning Effects", examples.example_pruning_effects)
    ]
    
    results = {}
    
    for title, func in example_funcs:
        print(f"\n{'='*20} {title} {'='*20}")
        try:
            result = func()
            results[title] = result
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
