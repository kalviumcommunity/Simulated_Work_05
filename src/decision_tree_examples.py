"""
Decision Tree Advanced Examples and Practical Applications

This module provides advanced examples and practical applications of Decision Trees,
including learning curves, rule extraction, feature interactions, ensemble comparisons,
and pruning effects. These examples complement the basic training module with
real-world scenarios and advanced techniques.

Key Features:
- Learning curves analysis for data size effects
- Rule extraction and business-friendly formatting
- Feature interaction detection and visualization
- Ensemble comparison (Decision Tree vs Random Forest)
- Pruning strategies and effects
- Advanced visualization techniques
- Real-world inspired examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_curve, roc_curve, auc)
from sklearn.datasets import make_classification, make_regression
from typing import Dict, Tuple, Any, Optional, List
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DecisionTreeExamples:
    """
    Advanced Decision Tree examples and practical applications.
    
    This class provides comprehensive examples demonstrating advanced Decision Tree
    techniques and real-world applications.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the examples class.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
    def example_learning_curves(self) -> Dict[str, Any]:
        """
        Analyze learning curves to understand data size effects.
        
        Learning curves show how model performance changes with varying training
        data sizes, helping diagnose bias/variance problems and determine if
        more data would help performance.
        
        Returns:
            Dictionary containing learning curve analysis and visualizations
        """
        print("=" * 60)
        print("LEARNING CURVES ANALYSIS")
        print("=" * 60)
        
        # Generate dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=12,
            n_informative=6,
            n_redundant=3,
            random_state=self.random_state,
            flip_y=0.1,
            class_sep=0.8
        )
        
        # Create models with different complexities
        models = {
            'Simple Tree': DecisionTreeClassifier(max_depth=2, random_state=self.random_state),
            'Medium Tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            'Complex Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
        }
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        results = {}
        
        print("Generating learning curves for different model complexities...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(models.items()):
            # Calculate learning curve
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=5,
                scoring='accuracy', random_state=self.random_state, n_jobs=-1
            )
            
            # Store results
            results[name] = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': np.mean(train_scores, axis=1),
                'train_scores_std': np.std(train_scores, axis=1),
                'val_scores_mean': np.mean(val_scores, axis=1),
                'val_scores_std': np.std(val_scores, axis=1)
            }
            
            # Plot learning curve
            ax = axes[i]
            ax.plot(train_sizes_abs, results[name]['train_scores_mean'], 
                   'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes_abs, 
                           results[name]['train_scores_mean'] - results[name]['train_scores_std'],
                           results[name]['train_scores_mean'] + results[name]['train_scores_std'],
                           alpha=0.1, color='blue')
            
            ax.plot(train_sizes_abs, results[name]['val_scores_mean'], 
                   'o-', color='red', label='Cross-validation Score')
            ax.fill_between(train_sizes_abs,
                           results[name]['val_scores_mean'] - results[name]['val_scores_std'],
                           results[name]['val_scores_mean'] + results[name]['val_scores_std'],
                           alpha=0.1, color='red')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Learning Curve - {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Combined comparison plot
        ax = axes[3]
        colors = ['blue', 'green', 'red']
        for i, (name, model_results) in enumerate(results.items()):
            ax.plot(model_results['train_sizes'], model_results['val_scores_mean'],
                   'o-', color=colors[i], linewidth=2, label=name)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Cross-validation Accuracy')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analysis and recommendations
        print("\nLearning Curves Analysis:")
        print("-" * 40)
        
        for name, model_results in results.items():
            final_train_score = model_results['train_scores_mean'][-1]
            final_val_score = model_results['val_scores_mean'][-1]
            gap = final_train_score - final_val_score
            
            print(f"\n{name}:")
            print(f"  Final Training Score: {final_train_score:.4f}")
            print(f"  Final CV Score:       {final_val_score:.4f}")
            print(f"  Gap:                  {gap:.4f}")
            
            # Diagnose bias/variance
            if gap > 0.1:
                print("  Diagnosis: HIGH VARIANCE (overfitting)")
                print("  Recommendation: Reduce model complexity or get more data")
            elif final_val_score < 0.8:
                print("  Diagnosis: HIGH BIAS (underfitting)")
                print("  Recommendation: Increase model complexity")
            else:
                print("  Diagnosis: GOOD BALANCE")
                print("  Recommendation: Current configuration is reasonable")
        
        return results
    
    def example_tree_rules_extraction(self) -> Dict[str, Any]:
        """
        Extract and format Decision Tree rules for business communication.
        
        This example demonstrates how to convert tree structure into
        human-readable IF-THEN rules that can be shared with stakeholders.
        
        Returns:
            Dictionary containing extracted rules and formatting examples
        """
        print("=" * 60)
        print("TREE RULES EXTRACTION AND FORMATTING")
        print("=" * 60)
        
        # Create a business-inspired dataset (customer churn)
        np.random.seed(self.random_state)
        n_samples = 500
        
        # Generate realistic features
        data = {
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.3, 0.1]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples, p=[0.4, 0.2, 0.2, 0.2]),
            'internet_service': np.random.choice(['No', 'DSL', 'Fiber optic'], n_samples, p=[0.2, 0.3, 0.5]),
            'tech_support': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3]),
            'online_backup': np.random.choice(['No', 'Yes'], n_samples, p=[0.6, 0.4]),
        }
        
        df = pd.DataFrame(data)
        
        # Create target (churn) based on realistic patterns
        churn_prob = (
            (df['tenure_months'] < 12) * 0.3 +
            (df['monthly_charges'] > 80) * 0.2 +
            (df['contract_type'] == 'Month-to-month') * 0.25 +
            (df['payment_method'] == 'Electronic check') * 0.1 +
            (df['tech_support'] == 'No') * 0.15
        )
        df['churn'] = (churn_prob + np.random.normal(0, 0.1, n_samples) > 0.3).astype(int)
        
        # Preprocess categorical features
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop('churn', axis=1)
        y = df_encoded['churn']
        
        # Train interpretable tree
        tree = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=20,
            random_state=self.random_state
        )
        tree.fit(X, y)
        
        print(f"Dataset: {n_samples} customers, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")
        print()
        
        # Extract rules using scikit-learn's export_text
        feature_names = X.columns.tolist()
        rules_text = export_text(tree, feature_names=feature_names, max_depth=10)
        
        print("Raw Tree Rules:")
        print("-" * 40)
        print(rules_text)
        
        # Format rules for business communication
        business_rules = self._format_business_rules(rules_text, feature_names)
        
        print("\nBusiness-Friendly Rules:")
        print("-" * 40)
        for i, rule in enumerate(business_rules, 1):
            print(f"Rule {i}: {rule}")
        
        # Create rule importance analysis
        rule_analysis = self._analyze_rule_importance(tree, X, y, feature_names)
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Tree structure
        plot_tree(tree, feature_names=feature_names, class_names=['No Churn', 'Churn'],
                 filled=True, rounded=True, fontsize=8, ax=ax1)
        ax1.set_title('Decision Tree Structure')
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        ax2.barh(importance_df['Feature'], importance_df['Importance'])
        ax2.set_title('Feature Importance')
        ax2.set_xlabel('Importance')
        
        # Rule coverage analysis
        if rule_analysis:
            rule_numbers = list(range(1, len(business_rules) + 1))
            coverage = [rule_analysis.get(i, {}).get('coverage', 0) for i in rule_numbers]
            accuracy = [rule_analysis.get(i, {}).get('accuracy', 0) for i in rule_numbers]
            
            ax3.scatter(coverage, accuracy, s=100, alpha=0.7)
            ax3.set_xlabel('Rule Coverage (%)')
            ax3.set_ylabel('Rule Accuracy (%)')
            ax3.set_title('Rule Performance Analysis')
            ax3.grid(True, alpha=0.3)
            
            # Add rule labels
            for i, (cov, acc) in enumerate(zip(coverage, accuracy)):
                ax3.annotate(f'R{i}', (cov, acc), xytext=(5, 5), textcoords='offset points')
        
        # Confusion matrix
        y_pred = tree.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'tree': tree,
            'feature_names': feature_names,
            'raw_rules': rules_text,
            'business_rules': business_rules,
            'rule_analysis': rule_analysis,
            'feature_importance': importance_df
        }
        
        return results
    
    def example_feature_interactions(self) -> Dict[str, Any]:
        """
        Demonstrate how Decision Trees capture feature interactions.
        
        This example creates synthetic data with clear interactions and shows
        how trees automatically discover and utilize them.
        
        Returns:
            Dictionary containing interaction analysis and visualizations
        """
        print("=" * 60)
        print("FEATURE INTERACTIONS DEMONSTRATION")
        print("=" * 60)
        
        # Create synthetic data with interactions
        np.random.seed(self.random_state)
        n_samples = 1000
        
        # Generate features
        X1 = np.random.uniform(0, 10, n_samples)
        X2 = np.random.uniform(0, 10, n_samples)
        X3 = np.random.uniform(0, 10, n_samples)
        X4 = np.random.uniform(0, 10, n_samples)
        
        # Create target with interactions
        # Interaction 1: X1 and X2 interact
        interaction1 = (X1 > 5) & (X2 < 5)
        
        # Interaction 2: X3 and X4 interact
        interaction2 = (X3 < 3) | (X4 > 7)
        
        # Combine interactions with noise
        y = (interaction1 ^ interaction2).astype(int)
        
        # Add some noise
        noise = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        y = y ^ noise
        
        # Create DataFrame
        X = pd.DataFrame({
            'Feature_A': X1,
            'Feature_B': X2,
            'Feature_C': X3,
            'Feature_D': X4
        })
        
        print(f"Generated dataset with {n_samples} samples")
        print("Created interactions:")
        print("  - Interaction 1: (Feature_A > 5) AND (Feature_B < 5)")
        print("  - Interaction 2: (Feature_C < 3) OR (Feature_D > 7)")
        print()
        
        # Train Decision Tree
        tree = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        tree.fit(X, y)
        
        # Train linear model for comparison
        from sklearn.linear_model import LogisticRegression
        linear_model = LogisticRegression(random_state=self.random_state)
        linear_model.fit(X, y)
        
        # Compare performance
        tree_acc = tree.score(X, y)
        linear_acc = linear_model.score(X, y)
        
        print("Performance Comparison:")
        print("-" * 30)
        print(f"Decision Tree Accuracy: {tree_acc:.4f}")
        print(f"Linear Model Accuracy:  {linear_acc:.4f}")
        print(f"Improvement:           {tree_acc - linear_acc:+.4f}")
        print()
        
        # Extract and analyze discovered interactions
        discovered_interactions = self._extract_interactions(tree, X.columns.tolist())
        
        print("Discovered Interactions:")
        print("-" * 30)
        for i, interaction in enumerate(discovered_interactions, 1):
            print(f"  {i}. {interaction}")
        print()
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Decision boundaries for interaction pairs
        self._plot_interaction_boundary(X, y, 'Feature_A', 'Feature_B', ax1, 
                                       'Interaction: A > 5 AND B < 5')
        self._plot_interaction_boundary(X, y, 'Feature_C', 'Feature_D', ax2, 
                                       'Interaction: C < 3 OR D > 7')
        
        # Tree structure
        plot_tree(tree, feature_names=X.columns.tolist(), class_names=['Class 0', 'Class 1'],
                 filled=True, rounded=True, fontsize=8, ax=ax3)
        ax3.set_title('Decision Tree - Captured Interactions')
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        ax4.bar(importance_df['Feature'], importance_df['Importance'])
        ax4.set_title('Feature Importance')
        ax4.set_ylabel('Importance')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'tree': tree,
            'linear_model': linear_model,
            'tree_accuracy': tree_acc,
            'linear_accuracy': linear_acc,
            'discovered_interactions': discovered_interactions,
            'feature_importance': importance_df
        }
        
        return results
    
    def example_comparison_with_random_forest(self) -> Dict[str, Any]:
        """
        Compare Decision Tree with Random Forest ensemble.
        
        This example demonstrates the performance differences between a single
        Decision Tree and a Random Forest ensemble, showing stability and accuracy improvements.
        
        Returns:
            Dictionary containing comparison results and analysis
        """
        print("=" * 60)
        print("DECISION TREE vs RANDOM FOREST COMPARISON")
        print("=" * 60)
        
        # Generate challenging dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_repeated=0,
            random_state=self.random_state,
            flip_y=0.15,
            class_sep=0.7
        )
        
        # Create feature names
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        print()
        
        # Train models
        tree = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit models
        tree.fit(X, y)
        rf.fit(X, y)
        
        # Cross-validation comparison
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        tree_scores = []
        rf_scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            tree.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            
            tree_scores.append(tree.score(X_val, y_val))
            rf_scores.append(rf.score(X_val, y_val))
        
        # Performance comparison
        tree_mean = np.mean(tree_scores)
        tree_std = np.std(tree_scores)
        rf_mean = np.mean(rf_scores)
        rf_std = np.std(rf_scores)
        
        print("Cross-Validation Performance:")
        print("-" * 40)
        print(f"Decision Tree: {tree_mean:.4f} ± {tree_std:.4f}")
        print(f"Random Forest: {rf_mean:.4f} ± {rf_std:.4f}")
        print(f"Improvement:   {rf_mean - tree_mean:+.4f}")
        print(f"Stability:     {tree_std:.4f} vs {rf_std:.4f}")
        print()
        
        # Feature importance comparison
        tree_importance = pd.DataFrame({
            'Feature': feature_names,
            'Tree_Importance': tree.feature_importances_,
            'RF_Importance': rf.feature_importances_
        }).set_index('Feature')
        
        # Calculate correlation between importance rankings
        correlation = np.corrcoef(tree.feature_importances_, rf.feature_importances_)[0, 1]
        
        print(f"Feature Importance Correlation: {correlation:.4f}")
        print()
        
        # Stability analysis (feature importance variance)
        tree_importance_vars = []
        rf_importance_vars = []
        
        for i in range(10):  # Train 10 times with different seeds
            tree_temp = DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=10, random_state=i
            )
            rf_temp = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=10, 
                random_state=i, n_jobs=-1
            )
            
            tree_temp.fit(X, y)
            rf_temp.fit(X, y)
            
            tree_importance_vars.append(tree_temp.feature_importances_)
            rf_importance_vars.append(rf_temp.feature_importances_)
        
        tree_importance_var = np.var(tree_importance_vars, axis=0).mean()
        rf_importance_var = np.var(rf_importance_vars, axis=0).mean()
        
        print("Feature Importance Stability (Variance):")
        print("-" * 45)
        print(f"Decision Tree: {tree_importance_var:.6f}")
        print(f"Random Forest: {rf_importance_var:.6f}")
        print(f"Stability Ratio: {tree_importance_var/rf_importance_var:.2f}x")
        print()
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance comparison
        models = ['Decision Tree', 'Random Forest']
        means = [tree_mean, rf_mean]
        stds = [tree_std, rf_std]
        
        ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.8)
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_title('Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Feature importance correlation
        ax2.scatter(tree_importance['Tree_Importance'], tree_importance['RF_Importance'], 
                   alpha=0.7)
        ax2.plot([0, max(tree_importance['Tree_Importance'].max(), 
                        tree_importance['RF_Importance'].max())],
                [0, max(tree_importance['Tree_Importance'].max(), 
                        tree_importance['RF_Importance'].max())], 
                'r--', alpha=0.5)
        ax2.set_xlabel('Decision Tree Importance')
        ax2.set_ylabel('Random Forest Importance')
        ax2.set_title(f'Feature Importance Correlation ({correlation:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Top features comparison
        top_features = tree_importance.sort_values('Tree_Importance', ascending=False).head(10)
        
        x = np.arange(len(top_features))
        width = 0.35
        
        ax3.bar(x - width/2, top_features['Tree_Importance'], width, 
                label='Decision Tree', alpha=0.8)
        ax3.bar(x + width/2, top_features['RF_Importance'], width, 
                label='Random Forest', alpha=0.8)
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Importance')
        ax3.set_title('Top 10 Feature Importance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_features.index, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Stability comparison
        stability_metrics = ['Accuracy\nStability', 'Importance\nStability']
        tree_stability = [tree_std, tree_importance_var]
        rf_stability = [rf_std, rf_importance_var]
        
        # Normalize for comparison
        tree_stability_norm = np.array(tree_stability) / np.array(tree_stability).max()
        rf_stability_norm = np.array(rf_stability) / np.array(rf_stability).max()
        
        x = np.arange(len(stability_metrics))
        width = 0.35
        
        ax4.bar(x - width/2, tree_stability_norm, width, label='Decision Tree', alpha=0.8)
        ax4.bar(x + width/2, rf_stability_norm, width, label='Random Forest', alpha=0.8)
        ax4.set_xlabel('Stability Metric')
        ax4.set_ylabel('Normalized Variance')
        ax4.set_title('Stability Comparison (Lower is Better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stability_metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        results = {
            'tree': tree,
            'random_forest': rf,
            'tree_scores': tree_scores,
            'rf_scores': rf_scores,
            'tree_mean': tree_mean,
            'tree_std': tree_std,
            'rf_mean': rf_mean,
            'rf_std': rf_std,
            'feature_importance': tree_importance,
            'importance_correlation': correlation,
            'tree_importance_variance': tree_importance_var,
            'rf_importance_variance': rf_importance_var
        }
        
        return results
    
    def example_pruning_effects(self) -> Dict[str, Any]:
        """
        Demonstrate pruning effects on Decision Tree performance and complexity.
        
        This example shows different pruning strategies and their effects on
        model complexity, performance, and interpretability.
        
        Returns:
            Dictionary containing pruning analysis and results
        """
        print("=" * 60)
        print("PRUNING STRATEGIES AND EFFECTS")
        print("=" * 60)
        
        # Generate dataset
        X, y = make_classification(
            n_samples=800,
            n_features=15,
            n_informative=8,
            n_redundant=4,
            random_state=self.random_state,
            flip_y=0.12,
            class_sep=0.8
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print()
        
        # Define pruning strategies
        pruning_strategies = [
            {'name': 'No Pruning', 'max_depth': None, 'min_samples_leaf': 1},
            {'name': 'Depth Limited', 'max_depth': 3, 'min_samples_leaf': 1},
            {'name': 'Depth Limited', 'max_depth': 5, 'min_samples_leaf': 1},
            {'name': 'Depth Limited', 'max_depth': 7, 'min_samples_leaf': 1},
            {'name': 'Leaf Limited', 'max_depth': None, 'min_samples_leaf': 5},
            {'name': 'Leaf Limited', 'max_depth': None, 'min_samples_leaf': 10},
            {'name': 'Combined', 'max_depth': 5, 'min_samples_leaf': 5},
            {'name': 'Conservative', 'max_depth': 3, 'min_samples_leaf': 10},
        ]
        
        results = {
            'strategies': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'gaps': [],
            'tree_sizes': [],
            'tree_depths': [],
            'training_times': []
        }
        
        print("Testing Pruning Strategies:")
        print("-" * 50)
        
        import time
        
        for strategy in pruning_strategies:
            start_time = time.time()
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=strategy['max_depth'],
                min_samples_leaf=strategy['min_samples_leaf'],
                random_state=self.random_state
            )
            tree.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            train_acc = tree.score(X_train, y_train)
            test_acc = tree.score(X_test, y_test)
            gap = train_acc - test_acc
            tree_size = tree.get_n_leaves()
            tree_depth = tree.get_depth()
            
            # Store results
            results['strategies'].append(strategy['name'])
            results['train_accuracies'].append(train_acc)
            results['test_accuracies'].append(test_acc)
            results['gaps'].append(gap)
            results['tree_sizes'].append(tree_size)
            results['tree_depths'].append(tree_depth)
            results['training_times'].append(training_time)
            
            print(f"{strategy['name']}:")
            print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}, Gap: {gap:.4f}")
            print(f"  Size: {tree_size} leaves, Depth: {tree_depth}, Time: {training_time:.4f}s")
            print()
        
        # Find optimal strategy
        best_idx = np.argmax(results['test_accuracies'])
        best_strategy = results['strategies'][best_idx]
        best_test_acc = results['test_accuracies'][best_idx]
        
        print(f"Best Strategy: {best_strategy}")
        print(f"Best Test Accuracy: {best_test_acc:.4f}")
        print()
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance comparison
        x = np.arange(len(results['strategies']))
        width = 0.35
        
        ax1.bar(x - width/2, results['train_accuracies'], width, 
                label='Train Accuracy', alpha=0.8)
        ax1.bar(x + width/2, results['test_accuracies'], width, 
                label='Test Accuracy', alpha=0.8)
        ax1.set_xlabel('Pruning Strategy')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance vs Pruning Strategy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results['strategies'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overfitting indicator
        ax2.bar(results['strategies'], results['gaps'], alpha=0.8, color='red')
        ax2.set_xlabel('Pruning Strategy')
        ax2.set_ylabel('Train/Test Gap')
        ax2.set_title('Overfitting Indicator')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Complexity analysis
        ax3.scatter(results['tree_sizes'], results['test_accuracies'], 
                   s=100, alpha=0.7, c=range(len(results['tree_sizes'])), cmap='viridis')
        for i, (size, acc) in enumerate(zip(results['tree_sizes'], results['test_accuracies'])):
            ax3.annotate(f'{i+1}', (size, acc), xytext=(5, 5), textcoords='offset points')
        ax3.set_xlabel('Tree Size (Number of Leaves)')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Complexity vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Training time
        ax4.bar(results['strategies'], results['training_times'], alpha=0.8, color='green')
        ax4.set_xlabel('Pruning Strategy')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Efficiency')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Train and visualize best tree
        best_tree = DecisionTreeClassifier(
            max_depth=pruning_strategies[best_idx]['max_depth'],
            min_samples_leaf=pruning_strategies[best_idx]['min_samples_leaf'],
            random_state=self.random_state
        )
        best_tree.fit(X_train, y_train)
        
        plt.figure(figsize=(15, 8))
        plot_tree(best_tree, max_depth=4, filled=True, rounded=True, fontsize=8)
        plt.title(f'Best Pruned Tree: {best_strategy}')
        plt.show()
        
        results['best_tree'] = best_tree
        results['best_strategy'] = best_strategy
        results['best_test_accuracy'] = best_test_acc
        
        return results
    
    def _format_business_rules(self, rules_text: str, feature_names: List[str]) -> List[str]:
        """Format tree rules for business communication."""
        rules = []
        lines = rules_text.split('\n')
        
        current_rule = []
        for line in lines:
            if '|---' in line:
                condition = line.split('|---')[-1].strip()
                if 'class:' in condition:
                    # End of rule
                    prediction = condition.split('class:')[-1].strip()
                    if current_rule:
                        rule_text = "IF " + " AND ".join(current_rule) + f" THEN {prediction}"
                        rules.append(rule_text)
                    current_rule = []
                else:
                    # Add condition
                    current_rule.append(condition)
        
        return rules
    
    def _analyze_rule_importance(self, tree, X, y, feature_names: List[str]) -> Dict:
        """Analyze rule coverage and accuracy."""
        # This is a simplified version - in practice, you'd want to extract
        # the actual decision paths and calculate their statistics
        return {}
    
    def _plot_interaction_boundary(self, X: pd.DataFrame, y: np.ndarray, 
                                  feature1: str, feature2: str, ax, title: str):
        """Plot decision boundary for feature interaction."""
        # Create mesh
        x_min, x_max = X[feature1].min() - 0.5, X[feature1].max() + 0.5
        y_min, y_max = X[feature2].min() - 0.5, X[feature2].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Simple decision tree for visualization
        tree_vis = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
        tree_vis.fit(X[[feature1, feature2]], y)
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = tree_vis.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        scatter = ax.scatter(X[feature1], X[feature2], c=y, cmap='RdYlBu', 
                            edgecolors='black', alpha=0.7)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(title)
        
    def _extract_interactions(self, tree, feature_names: List[str]) -> List[str]:
        """Extract feature interactions from tree structure."""
        interactions = []
        
        # Get the tree structure
        tree_ = tree.tree_
        
        # Find paths that involve multiple features
        def find_interactions(node_id, current_path, features_used):
            if tree_.feature[node_id] != -2:  # Not a leaf
                feature_idx = tree_.feature[node_id]
                feature_name = feature_names[feature_idx]
                threshold = tree_.threshold[node_id]
                
                # Add to current path
                condition = f"{feature_name} <= {threshold:.2f}"
                new_path = current_path + [condition]
                new_features = features_used + [feature_name]
                
                # Check for interaction (multiple features in path)
                if len(new_features) >= 2:
                    interaction = " AND ".join(new_path[-2:])  # Last two conditions
                    interactions.append(interaction)
                
                # Recurse on children
                find_interactions(tree_.children_left[node_id], new_path, new_features)
                find_interactions(tree_.children_right[node_id], new_path, new_features)
        
        find_interactions(0, [], [])
        
        # Remove duplicates and limit
        unique_interactions = list(set(interactions))[:5]  # Top 5 unique interactions
        
        return unique_interactions


def main():
    """
    Main function to run all Decision Tree examples.
    """
    print("🌳 DECISION TREE ADVANCED EXAMPLES")
    print("=" * 80)
    
    examples = DecisionTreeExamples(random_state=42)
    
    # Run all examples
    example_functions = [
        ("Learning Curves Analysis", examples.example_learning_curves),
        ("Tree Rules Extraction", examples.example_tree_rules_extraction),
        ("Feature Interactions", examples.example_feature_interactions),
        ("Random Forest Comparison", examples.example_comparison_with_random_forest),
        ("Pruning Effects", examples.example_pruning_effects),
    ]
    
    for name, func in example_functions:
        print(f"\n🔍 {name}")
        print("=" * 80)
        try:
            results = func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n🎉 DECISION TREE EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Learning curves reveal bias/variance trade-offs")
    print("2. Tree rules can be formatted for business communication")
    print("3. Decision Trees automatically capture feature interactions")
    print("4. Random Forests improve stability and accuracy")
    print("5. Pruning is essential for controlling overfitting")
    print("6. Model complexity should be balanced with interpretability")


if __name__ == "__main__":
    main()
