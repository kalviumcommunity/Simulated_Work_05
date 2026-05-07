"""
Algorithm-Specific Bias-Variance Examples

Comprehensive examples demonstrating bias-variance characteristics across
different algorithm families. This module shows how each algorithm family
behaves in terms of bias and variance, and provides practical guidance
for algorithm selection and tuning.

Algorithms Covered:
- Linear Regression (high bias baseline)
- Logistic Regression (classification baseline)
- K-Nearest Neighbors (variance-sensitive)
- Decision Trees (complexity-controlled)
- Random Forest (variance reduction)
- Support Vector Machines (regularization-controlled)
- Neural Networks (capacity-controlled)
- Gradient Boosting (sequential ensemble)

Each algorithm includes:
• Default bias-variance characteristics
• Hyperparameter impact analysis
• Practical tuning strategies
• Visualization of behavior patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from itertools import product

# ML imports
from sklearn.model_selection import (train_test_split, learning_curve, validation_curve,
                                   cross_val_score, GridSearchCV)
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AlgorithmBiasVarianceAnalyzer:
    """
    Comprehensive analyzer for algorithm-specific bias-variance characteristics.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the algorithm analyzer."""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        logger.info("Algorithm Bias-Variance Analyzer initialized")
    
    def demonstrate_linear_regression(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Linear Regression.
        
        Linear Regression typically shows:
        - High bias (assumes linear relationship)
        - Low variance (stable predictions)
        - Sensitive to feature engineering
        """
        print("\n" + "="*80)
        print("LINEAR REGRESSION: HIGH BIAS BASELINE")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Linear Regression:
• Default: High bias, low variance
• Assumption: Linear relationship between features and target
• Strengths: Interpretable, fast, stable
• Weaknesses: Cannot capture non-linear patterns
• Bias Control: Feature engineering, polynomial features
• Variance Control: Regularization (Ridge, Lasso)
        """)
        
        # Create datasets with different complexities
        datasets = {
            'Linear Data': self._create_linear_data(500, 0.1),
            'Non-linear Data': self._create_nonlinear_data(500, 0.1),
            'High Noise': self._create_linear_data(500, 0.5),
            'Small Sample': self._create_linear_data(100, 0.1)
        }
        
        # Linear regression variants
        models = {
            'Simple Linear': LinearRegression(),
            'Ridge (α=0.1)': Ridge(alpha=0.1),
            'Ridge (α=10)': Ridge(alpha=10),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Poly Degree 2': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'Poly Degree 5': Pipeline([
                ('poly', PolynomialFeatures(degree=5)),
                ('linear', Ridge(alpha=0.1))
            ])
        }
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            for model_name, model in models.items():
                # Train and evaluate
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                gap = abs(train_r2 - test_r2)
                
                # Cross-validation for stability
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[data_name][model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'gap': gap,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  {model_name:<20}: Test R² = {test_r2:.3f}, Gap = {gap:.3f}")
        
        # Visualize results
        self._plot_linear_regression_results(results)
        
        return results
    
    def demonstrate_logistic_regression(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Logistic Regression.
        
        Logistic Regression typically shows:
        - High bias (linear decision boundary)
        - Low variance (stable predictions)
        - Regularization controls bias-variance trade-off
        """
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION: CLASSIFICATION BASELINE")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Logistic Regression:
• Default: High bias, low variance
• Assumption: Linear decision boundary
• Strengths: Probabilistic, interpretable, fast
• Weaknesses: Cannot capture complex decision boundaries
• Bias Control: Feature engineering, interaction terms
• Variance Control: Regularization parameter C
        """)
        
        # Create classification datasets
        datasets = {
            'Linear Separable': self._create_linear_classification(500, 0.05),
            'Non-linear Boundary': self._create_nonlinear_classification(500, 0.1),
            'Noisy Features': self._create_linear_classification(500, 0.3),
            'Imbalanced Classes': self._create_imbalanced_classification(500, 0.1)
        }
        
        # Logistic regression variants
        models = {
            'Default (C=1.0)': LogisticRegression(random_state=self.random_state),
            'High Regularization (C=0.1)': LogisticRegression(C=0.1, random_state=self.random_state),
            'Low Regularization (C=10)': LogisticRegression(C=10, random_state=self.random_state),
            'Poly Features': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(random_state=self.random_state))
            ]),
            'L1 Regularization': LogisticRegression(penalty='l1', solver='liblinear', 
                                                 C=1.0, random_state=self.random_state),
            'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga',
                                            l1_ratio=0.5, C=1.0, random_state=self.random_state)
        }
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            for model_name, model in models.items():
                try:
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    gap = abs(train_acc - test_acc)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    results[data_name][model_name] = {
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'gap': gap,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"  {model_name:<25}: Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:<25}: Error - {str(e)}")
        
        # Visualize results
        self._plot_logistic_regression_results(results)
        
        return results
    
    def demonstrate_knn(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of K-Nearest Neighbors.
        
        KNN typically shows:
        - Low bias with small K (can fit any pattern)
        - High variance with small K (sensitive to noise)
        - High bias with large K (over-smoothing)
        - K controls bias-variance trade-off directly
        """
        print("\n" + "="*80)
        print("K-NEAREST NEIGHBORS: VARIANCE-SENSITIVE ALGORITHM")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

K-Nearest Neighbors:
• K=1: Very low bias, very high variance (memorizes training data)
• K=large: High bias, low variance (over-smooths)
• Bias-Variance Trade-off: Directly controlled by K
• Strengths: Simple, no training time, non-parametric
• Weaknesses: Curse of dimensionality, sensitive to scaling
• Bias Control: Increase K
• Variance Control: Decrease K, distance weighting
        """)
        
        # Create datasets
        datasets = {
            'Clean Data': self._create_linear_classification(500, 0.05),
            'Noisy Data': self._create_linear_classification(500, 0.3),
            'Complex Boundary': self._create_nonlinear_classification(500, 0.1),
            'High Dimensional': self._create_high_dim_classification(500, 10, 0.1)
        }
        
        # KNN variants with different K values
        k_values = [1, 3, 5, 10, 15, 25, 50]
        models = {f'K={k}': KNeighborsClassifier(n_neighbors=k) for k in k_values}
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Scale features for KNN
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            for model_name, model in models.items():
                # Train and evaluate
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                gap = abs(train_acc - test_acc)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[data_name][model_name] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'gap': gap,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  {model_name:<6}: Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")
        
        # Visualize results
        self._plot_knn_results(results)
        
        return results
    
    def demonstrate_decision_trees(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Decision Trees.
        
        Decision Trees typically show:
        - Low bias when unconstrained (can fit any pattern)
        - Very high variance when unconstrained (overfits severely)
        - Depth controls bias-variance trade-off
        - Pruning and regularization reduce variance
        """
        print("\n" + "="*80)
        print("DECISION TREES: COMPLEXITY-CONTROLLED ALGORITHM")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Decision Trees:
• Unconstrained: Very low bias, very high variance
• Depth-limited: Controlled bias-variance trade-off
• Bias Control: Increase max_depth, min_samples_split
• Variance Control: Decrease depth, pruning, ensembling
• Strengths: Interpretable, handles non-linear patterns
• Weaknesses: High variance, unstable with small changes
• Key Insight: Single trees are rarely optimal alone
        """)
        
        # Create datasets
        datasets = {
            'Simple Classification': self._create_linear_classification(500, 0.1),
            'Complex Classification': self._create_nonlinear_classification(500, 0.1),
            'Regression Task': self._create_nonlinear_data(500, 0.1),
            'Noisy Data': self._create_linear_classification(500, 0.3)
        }
        
        # Decision tree variants with different depths
        depths = [1, 2, 3, 5, 8, 12, None]  # None = unconstrained
        models = {}
        for depth in depths:
            name = f'Depth={depth}' if depth else 'Unconstrained'
            if depth is None:
                models[name] = DecisionTreeClassifier(random_state=self.random_state)
            else:
                models[name] = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Determine if regression or classification
            is_regression = len(np.unique(y)) > 10  # Heuristic
            
            if is_regression:
                # Use regression models
                models_reg = {}
                for depth in depths:
                    name = f'Depth={depth}' if depth else 'Unconstrained'
                    if depth is None:
                        models_reg[name] = DecisionTreeRegressor(random_state=self.random_state)
                    else:
                        models_reg[name] = DecisionTreeRegressor(max_depth=depth, random_state=self.random_state)
                models_to_use = models_reg
                scoring = 'r2'
            else:
                # Use classification models
                models_to_use = models
                scoring = 'accuracy'
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state,
                stratify=y if not is_regression else None
            )
            
            for model_name, model in models_to_use.items():
                # Train and evaluate
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                if is_regression:
                    train_score = r2_score(y_train, y_train_pred)
                    test_score = r2_score(y_test, y_test_pred)
                else:
                    train_score = accuracy_score(y_train, y_train_pred)
                    test_score = accuracy_score(y_test, y_test_pred)
                
                gap = abs(train_score - test_score)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                
                results[data_name][model_name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'gap': gap,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  {model_name:<15}: Test = {test_score:.3f}, Gap = {gap:.3f}")
        
        # Visualize results
        self._plot_decision_tree_results(results)
        
        return results
    
    def demonstrate_random_forest(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Random Forest.
        
        Random Forest typically shows:
        - Moderate bias (individual trees have low bias)
        - Reduced variance (bagging reduces variance)
        - Number of trees affects stability
        - Feature selection and depth control bias
        """
        print("\n" + "="*80)
        print("RANDOM FOREST: VARIANCE REDUCTION THROUGH ENSEMBLING")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Random Forest:
• Default: Moderate bias, reduced variance
• Mechanism: Bagging + feature randomization
• Bias Control: Tree depth, number of features
• Variance Control: Number of trees, bootstrap sampling
• Strengths: Robust, handles non-linearity, feature importance
• Weaknesses: Less interpretable, more memory
• Key Insight: Variance reduction through averaging
        """)
        
        # Create datasets
        datasets = {
            'Classification': self._create_nonlinear_classification(500, 0.1),
            'Regression': self._create_nonlinear_data(500, 0.1),
            'Noisy Features': self._create_high_dim_classification(500, 20, 0.2),
            'Small Sample': self._create_linear_classification(200, 0.1)
        }
        
        # Random Forest variants
        n_estimators_list = [10, 50, 100, 200]
        max_depth_list = [3, 5, 8, None]
        
        models = {}
        for n_est, depth in product(n_estimators_list, max_depth_list):
            name = f'Trees={n_est}, Depth={depth}' if depth else f'Trees={n_est}, Depth=∞'
            models[name] = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, 
                random_state=self.random_state, n_jobs=-1
            )
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Determine task type
            is_regression = len(np.unique(y)) > 10
            
            if is_regression:
                # Use regression models
                models_reg = {}
                for n_est, depth in product(n_estimators_list, max_depth_list):
                    name = f'Trees={n_est}, Depth={depth}' if depth else f'Trees={n_est}, Depth=∞'
                    models_reg[name] = RandomForestRegressor(
                        n_estimators=n_est, max_depth=depth,
                        random_state=self.random_state, n_jobs=-1
                    )
                models_to_use = models_reg
                scoring = 'r2'
            else:
                models_to_use = models
                scoring = 'accuracy'
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state,
                stratify=y if not is_regression else None
            )
            
            # Test a subset of models to avoid too many computations
            model_subset = list(models_to_use.items())[:8]  # Test first 8 models
            
            for model_name, model in model_subset:
                # Train and evaluate
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                if is_regression:
                    train_score = r2_score(y_train, y_train_pred)
                    test_score = r2_score(y_test, y_test_pred)
                else:
                    train_score = accuracy_score(y_train, y_train_pred)
                    test_score = accuracy_score(y_test, y_test_pred)
                
                gap = abs(train_score - test_score)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                
                results[data_name][model_name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'gap': gap,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  {model_name:<20}: Test = {test_score:.3f}, Gap = {gap:.3f}")
        
        # Visualize results
        self._plot_random_forest_results(results)
        
        return results
    
    def demonstrate_svm(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Support Vector Machines.
        
        SVM typically shows:
        - Controlled by regularization parameter C
        - Kernel choice affects bias-variance trade-off
        - High C: Low bias, high variance
        - Low C: High bias, low variance
        """
        print("\n" + "="*80)
        print("SUPPORT VECTOR MACHINES: REGULARIZATION-CONTROLLED")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Support Vector Machines:
• High C: Low bias, high variance (hard margin)
• Low C: High bias, low variance (soft margin)
• Kernel Choice: RBF (flexible) vs Linear (rigid)
• Bias Control: Decrease C, use linear kernel
• Variance Control: Increase C, use RBF kernel
• Strengths: Effective in high dimensions, theoretical guarantees
• Weaknesses: Sensitive to parameters, slower training
• Key Insight: Margin maximization with slack variables
        """)
        
        # Create datasets
        datasets = {
            'Linear Separable': self._create_linear_classification(500, 0.05),
            'Non-linear Separable': self._create_nonlinear_classification(500, 0.1),
            'Noisy Data': self._create_linear_classification(500, 0.3),
            'High Dimensional': self._create_high_dim_classification(500, 15, 0.1)
        }
        
        # SVM variants
        C_values = [0.1, 1, 10, 100]
        kernels = ['linear', 'rbf']
        
        models = {}
        for C, kernel in product(C_values, kernels):
            name = f'C={C}, {kernel}'
            models[name] = SVC(C=C, kernel=kernel, random_state=self.random_state)
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Scale features for SVM
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            for model_name, model in models.items():
                try:
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    gap = abs(train_acc - test_acc)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    results[data_name][model_name] = {
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'gap': gap,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"  {model_name:<15}: Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:<15}: Error - {str(e)}")
        
        # Visualize results
        self._plot_svm_results(results)
        
        return results
    
    def demonstrate_neural_networks(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Neural Networks.
        
        Neural Networks typically show:
        - Low bias with large networks
        - High variance with large networks
        - Architecture controls capacity
        - Regularization crucial for variance control
        """
        print("\n" + "="*80)
        print("NEURAL NETWORKS: CAPACITY-CONTROLLED ALGORITHM")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Neural Networks:
• Large Networks: Low bias, high variance
• Small Networks: High bias, low variance
• Bias Control: Network size, architecture
• Variance Control: Regularization, dropout, early stopping
• Strengths: Universal approximation, feature learning
• Weaknesses: Hard to tune, black box, computationally intensive
• Key Insight: Capacity must match data complexity
        """)
        
        # Create datasets
        datasets = {
            'Simple Pattern': self._create_linear_classification(500, 0.1),
            'Complex Pattern': self._create_nonlinear_classification(500, 0.1),
            'Noisy Data': self._create_linear_classification(500, 0.3),
            'Small Sample': self._create_linear_classification(200, 0.1)
        }
        
        # Neural network variants
        layer_configs = [
            (10,),           # Small network
            (50, 25),        # Medium network
            (100, 50, 25),   # Large network
            (200, 100, 50),  # Very large network
        ]
        
        alphas = [0.0001, 0.001, 0.01]  # Regularization strengths
        
        models = {}
        for layers, alpha in product(layer_configs, alphas):
            name = f'Layers={layers}, α={alpha}'
            models[name] = MLPClassifier(
                hidden_layer_sizes=layers,
                alpha=alpha,
                max_iter=1000,
                random_state=self.random_state
            )
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Scale features for neural networks
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Test subset of models
            model_subset = list(models.items())[:6]  # Test first 6 models
            
            for model_name, model in model_subset:
                try:
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    gap = abs(train_acc - test_acc)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    results[data_name][model_name] = {
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'gap': gap,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"  {model_name:<30}: Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:<30}: Error - {str(e)}")
        
        # Visualize results
        self._plot_neural_network_results(results)
        
        return results
    
    def demonstrate_gradient_boosting(self) -> Dict:
        """
        Demonstrate bias-variance characteristics of Gradient Boosting.
        
        Gradient Boosting typically shows:
        - Sequential error reduction
        - Can achieve low bias and moderate variance
        - Learning rate and trees control trade-off
        - Prone to overfitting with too many trees
        """
        print("\n" + "="*80)
        print("GRADIENT BOOSTING: SEQUENTIAL ENSEMBLE")
        print("="*80)
        
        print("""
📖 ALGORITHM CHARACTERISTICS
============================

Gradient Boosting:
• Sequential: Builds trees to correct errors
• Low Bias: Can achieve very low training error
• Moderate Variance: Controlled by learning rate
• Bias Control: More trees, deeper trees
• Variance Control: Learning rate, regularization, early stopping
• Strengths: High performance, handles complex patterns
• Weaknesses: Sensitive to overfitting, slower training
• Key Insight: Sequential learning with shrinkage
        """)
        
        # Create datasets
        datasets = {
            'Classification': self._create_nonlinear_classification(500, 0.1),
            'Regression': self._create_nonlinear_data(500, 0.1),
            'Noisy Data': self._create_linear_classification(500, 0.3),
            'Complex Pattern': self._create_complex_classification(500, 0.1)
        }
        
        # Gradient boosting variants
        n_estimators_list = [50, 100, 200]
        learning_rates = [0.01, 0.1, 0.2]
        max_depth_list = [3, 5]
        
        models = {}
        for n_est, lr, depth in product(n_estimators_list, learning_rates, max_depth_list):
            name = f'Trees={n_est}, LR={lr}, Depth={depth}'
            models[name] = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                random_state=self.random_state
            )
        
        results = {}
        
        for data_name, (X, y) in datasets.items():
            print(f"\nAnalyzing {data_name}:")
            results[data_name] = {}
            
            # Determine task type
            is_regression = len(np.unique(y)) > 10
            
            if is_regression:
                # Use regression models
                models_reg = {}
                for n_est, lr, depth in product(n_estimators_list, learning_rates, max_depth_list):
                    name = f'Trees={n_est}, LR={lr}, Depth={depth}'
                    models_reg[name] = GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        random_state=self.random_state
                    )
                models_to_use = models_reg
                scoring = 'r2'
            else:
                models_to_use = models
                scoring = 'accuracy'
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state,
                stratify=y if not is_regression else None
            )
            
            # Test subset of models
            model_subset = list(models_to_use.items())[:6]  # Test first 6 models
            
            for model_name, model in model_subset:
                try:
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    if is_regression:
                        train_score = r2_score(y_train, y_train_pred)
                        test_score = r2_score(y_test, y_test_pred)
                    else:
                        train_score = accuracy_score(y_train, y_train_pred)
                        test_score = accuracy_score(y_test, y_test_pred)
                    
                    gap = abs(train_score - test_score)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                    
                    results[data_name][model_name] = {
                        'train_score': train_score,
                        'test_score': test_score,
                        'gap': gap,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"  {model_name:<25}: Test = {test_score:.3f}, Gap = {gap:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:<25}: Error - {str(e)}")
        
        # Visualize results
        self._plot_gradient_boosting_results(results)
        
        return results
    
    # Data creation methods
    def _create_linear_data(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create linear regression data."""
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (n_samples, 2))
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, noise, n_samples)
        return X, y
    
    def _create_nonlinear_data(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create non-linear regression data."""
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (n_samples, 3))
        y = (X[:, 0]**2 + X[:, 1] * X[:, 2] + np.sin(X[:, 0]) + 
             np.random.normal(0, noise, n_samples))
        return X, y
    
    def _create_linear_classification(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create linearly separable classification data."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=4, n_informative=3,
            n_redundant=1, flip_y=noise, random_state=self.random_state
        )
        return X, y
    
    def _create_nonlinear_classification(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create non-linear classification data."""
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=self.random_state)
        # Add extra features
        X = np.hstack([X, np.random.normal(0, 0.5, (n_samples, 2))])
        return X, y
    
    def _create_imbalanced_classification(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create imbalanced classification data."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=4, n_informative=3,
            n_redundant=1, weights=[0.9, 0.1], flip_y=noise,
            random_state=self.random_state
        )
        return X, y
    
    def _create_high_dim_classification(self, n_samples: int, n_features: int, 
                                       noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create high-dimensional classification data."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=min(10, n_features//2),
            n_redundant=n_features//4, flip_y=noise,
            random_state=self.random_state
        )
        return X, y
    
    def _create_complex_classification(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create complex classification data."""
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, 
                           random_state=self.random_state)
        # Add extra features
        X = np.hstack([X, np.random.normal(0, 0.5, (n_samples, 3))])
        return X, y
    
    # Visualization methods
    def _plot_linear_regression_results(self, results: Dict):
        """Plot linear regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Performance comparison across datasets
        dataset_names = list(results.keys())
        model_names = list(results[dataset_names[0]].keys())
        
        x = np.arange(len(dataset_names))
        width = 0.1
        
        for i, model_name in enumerate(model_names[:6]):  # Plot first 6 models
            test_scores = [results[data][model_name]['test_r2'] for data in dataset_names]
            axes[0, 0].bar(x + i*width, test_scores, width, label=model_name, alpha=0.7)
        
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Test R²')
        axes[0, 0].set_title('Linear Regression: Performance Across Datasets')
        axes[0, 0].set_xticks(x + width * 2.5)
        axes[0, 0].set_xticklabels(dataset_names, rotation=45)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gap analysis
        for i, model_name in enumerate(model_names[:6]):
            gaps = [results[data][model_name]['gap'] for data in dataset_names]
            axes[0, 1].bar(x + i*width, gaps, width, label=model_name, alpha=0.7)
        
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('Linear Regression: Overfitting Analysis')
        axes[0, 1].set_xticks(x + width * 2.5)
        axes[0, 1].set_xticklabels(dataset_names, rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Best model per dataset
        best_models = []
        best_scores = []
        for data in dataset_names:
            best_model = max(results[data].keys(), 
                           key=lambda k: results[data][k]['test_r2'])
            best_score = results[data][best_model]['test_r2']
            best_models.append(best_model)
            best_scores.append(best_score)
        
        axes[1, 0].bar(dataset_names, best_scores, alpha=0.7, color='green')
        axes[1, 0].set_ylabel('Best Test R²')
        axes[1, 0].set_title('Linear Regression: Best Performance per Dataset')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add model names as text
        for i, (data, score, model) in enumerate(zip(dataset_names, best_scores, best_models)):
            axes[1, 0].text(i, score + 0.01, model.split()[0], ha='center', fontsize=8)
        
        # Plot 4: Summary statistics
        all_gaps = []
        all_cv_stds = []
        for data in dataset_names:
            for model in results[data].keys():
                all_gaps.append(results[data][model]['gap'])
                all_cv_stds.append(results[data][model]['cv_std'])
        
        axes[1, 1].hist(all_gaps, bins=20, alpha=0.7, label='Gap', color='blue')
        axes[1, 1].hist(all_cv_stds, bins=20, alpha=0.7, label='CV Std', color='red')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Linear Regression: Distribution of Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/linear_regression_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_logistic_regression_results(self, results: Dict):
        """Plot logistic regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        model_names = list(results[dataset_names[0]].keys())
        
        # Similar structure to linear regression but with accuracy
        x = np.arange(len(dataset_names))
        width = 0.1
        
        # Plot 1: Performance comparison
        for i, model_name in enumerate(model_names[:6]):
            test_scores = [results[data][model_name]['test_acc'] for data in dataset_names]
            axes[0, 0].bar(x + i*width, test_scores, width, label=model_name, alpha=0.7)
        
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Logistic Regression: Performance Across Datasets')
        axes[0, 0].set_xticks(x + width * 2.5)
        axes[0, 0].set_xticklabels(dataset_names, rotation=45)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gap analysis
        for i, model_name in enumerate(model_names[:6]):
            gaps = [results[data][model_name]['gap'] for data in dataset_names]
            axes[0, 1].bar(x + i*width, gaps, width, label=model_name, alpha=0.7)
        
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('Logistic Regression: Overfitting Analysis')
        axes[0, 1].set_xticks(x + width * 2.5)
        axes[0, 1].set_xticklabels(dataset_names, rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Regularization impact
        c_values = []
        avg_scores = []
        for model_name in model_names:
            if 'C=' in model_name:
                # Extract C value
                c_str = model_name.split('C=')[1].split(',')[0]
                c_values.append(float(c_str))
                avg_scores.append(np.mean([results[data][model_name]['test_acc'] 
                                         for data in dataset_names 
                                         if model_name in results[data]]))
        
        if c_values:
            axes[1, 0].plot(c_values, avg_scores, 'bo-', linewidth=2, markersize=8)
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_xlabel('Regularization Parameter C')
            axes[1, 0].set_ylabel('Average Test Accuracy')
            axes[1, 0].set_title('Logistic Regression: Regularization Impact')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        all_scores = []
        all_gaps = []
        for data in dataset_names:
            for model in results[data].keys():
                all_scores.append(results[data][model]['test_acc'])
                all_gaps.append(results[data][model]['gap'])
        
        axes[1, 1].scatter(all_gaps, all_scores, alpha=0.6)
        axes[1, 1].set_xlabel('Train-Test Gap')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Logistic Regression: Performance vs. Stability')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/logistic_regression_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_knn_results(self, results: Dict):
        """Plot KNN results showing K vs performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract K values and corresponding results
        k_values = []
        for model_name in results[dataset_names[0]].keys():
            k = int(model_name.split('=')[1])
            k_values.append(k)
        
        # Plot 1: K vs Test Accuracy for each dataset
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            test_scores = [results[data_name][f'K={k}']['test_acc'] for k in k_values]
            axes[0, 0].plot(k_values, test_scores, 'o-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel('K (Number of Neighbors)')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('KNN: K vs. Test Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: K vs Gap (overfitting indicator)
        for i, data_name in enumerate(dataset_names):
            gaps = [results[data_name][f'K={k}']['gap'] for k in k_values]
            axes[0, 1].plot(k_values, gaps, 's-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xlabel('K (Number of Neighbors)')
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('KNN: K vs. Overfitting')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: K vs CV Stability
        for i, data_name in enumerate(dataset_names):
            cv_stds = [results[data_name][f'K={k}']['cv_std'] for k in k_values]
            axes[1, 0].plot(k_values, cv_stds, '^-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('K (Number of Neighbors)')
        axes[1, 0].set_ylabel('CV Standard Deviation')
        axes[1, 0].set_title('KNN: K vs. Model Stability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Optimal K for each dataset
        optimal_ks = []
        optimal_scores = []
        for data_name in dataset_names:
            best_k = max(k_values, key=lambda k: results[data_name][f'K={k}']['test_acc'])
            best_score = results[data_name][f'K={best_k}']['test_acc']
            optimal_ks.append(best_k)
            optimal_scores.append(best_score)
        
        axes[1, 1].bar(dataset_names, optimal_ks, alpha=0.7, color='purple')
        axes[1, 1].set_ylabel('Optimal K')
        axes[1, 1].set_title('KNN: Optimal K per Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add scores as text
        for i, (data, k, score) in enumerate(zip(dataset_names, optimal_ks, optimal_scores)):
            axes[1, 1].text(i, k + 1, f'{score:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("plots/knn_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_decision_tree_results(self, results: Dict):
        """Plot decision tree results showing depth vs performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract depth values
        depth_values = []
        for model_name in results[dataset_names[0]].keys():
            if 'Depth=' in model_name:
                depth = int(model_name.split('=')[1])
            elif model_name == 'Unconstrained':
                depth = 20  # Represent unconstrained as deep
            else:
                depth = int(model_name.split('=')[1])
            depth_values.append(depth)
        
        # Plot 1: Depth vs Test Score
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            test_scores = []
            depths = []
            for model_name in results[data_name].keys():
                if 'Depth=' in model_name:
                    depth = int(model_name.split('=')[1])
                elif model_name == 'Unconstrained':
                    depth = 20
                else:
                    depth = int(model_name.split('=')[1])
                depths.append(depth)
                test_scores.append(results[data_name][model_name]['test_score'])
            
            axes[0, 0].plot(depths, test_scores, 'o-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel('Tree Depth')
        axes[0, 0].set_ylabel('Test Score')
        axes[0, 0].set_title('Decision Trees: Depth vs. Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Depth vs Gap
        for i, data_name in enumerate(dataset_names):
            gaps = []
            depths = []
            for model_name in results[data_name].keys():
                if 'Depth=' in model_name:
                    depth = int(model_name.split('=')[1])
                elif model_name == 'Unconstrained':
                    depth = 20
                else:
                    depth = int(model_name.split('=')[1])
                depths.append(depth)
                gaps.append(results[data_name][model_name]['gap'])
            
            axes[0, 1].plot(depths, gaps, 's-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xlabel('Tree Depth')
        axes[0, 1].set_ylabel('Train-Test Gap')
        axes[0, 1].set_title('Decision Trees: Depth vs. Overfitting')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Depth vs CV Stability
        for i, data_name in enumerate(dataset_names):
            cv_stds = []
            depths = []
            for model_name in results[data_name].keys():
                if 'Depth=' in model_name:
                    depth = int(model_name.split('=')[1])
                elif model_name == 'Unconstrained':
                    depth = 20
                else:
                    depth = int(model_name.split('=')[1])
                depths.append(depth)
                cv_stds.append(results[data_name][model_name]['cv_std'])
            
            axes[1, 0].plot(depths, cv_stds, '^-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Tree Depth')
        axes[1, 0].set_ylabel('CV Standard Deviation')
        axes[1, 0].set_title('Decision Trees: Depth vs. Stability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Optimal depth per dataset
        optimal_depths = []
        optimal_scores = []
        for data_name in dataset_names:
            best_model = max(results[data_name].keys(), 
                           key=lambda k: results[data_name][k]['test_score'])
            if 'Depth=' in best_model:
                best_depth = int(best_model.split('=')[1])
            elif best_model == 'Unconstrained':
                best_depth = 20
            else:
                best_depth = int(best_model.split('=')[1])
            best_score = results[data_name][best_model]['test_score']
            
            optimal_depths.append(best_depth)
            optimal_scores.append(best_score)
        
        axes[1, 1].bar(dataset_names, optimal_depths, alpha=0.7, color='green')
        axes[1, 1].set_ylabel('Optimal Depth')
        axes[1, 1].set_title('Decision Trees: Optimal Depth per Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add scores as text
        for i, (data, depth, score) in enumerate(zip(dataset_names, optimal_depths, optimal_scores)):
            axes[1, 1].text(i, depth + 0.5, f'{score:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("plots/decision_tree_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_random_forest_results(self, results: Dict):
        """Plot random forest results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract number of trees and depth information
        n_trees_values = set()
        depth_values = set()
        for data_name in dataset_names:
            for model_name in results[data_name].keys():
                parts = model_name.split(', ')
                n_trees = int(parts[0].split('=')[1])
                depth_str = parts[1].split('=')[1]
                depth = int(depth_str) if depth_str != '∞' else 20
                n_trees_values.add(n_trees)
                depth_values.add(depth)
        
        n_trees_values = sorted(list(n_trees_values))
        depth_values = sorted(list(depth_values))
        
        # Plot 1: Number of trees vs performance
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            avg_scores_by_trees = []
            for n_trees in n_trees_values:
                scores = []
                for model_name in results[data_name].keys():
                    if f'Trees={n_trees},' in model_name:
                        scores.append(results[data_name][model_name]['test_score'])
                avg_scores_by_trees.append(np.mean(scores) if scores else 0)
            
            axes[0, 0].plot(n_trees_values, avg_scores_by_trees, 'o-', 
                           color=colors[i], label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel('Number of Trees')
        axes[0, 0].set_ylabel('Average Test Score')
        axes[0, 0].set_title('Random Forest: Number of Trees vs. Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Max depth vs performance
        for i, data_name in enumerate(dataset_names):
            avg_scores_by_depth = []
            for depth in depth_values:
                scores = []
                for model_name in results[data_name].keys():
                    if f'Depth={depth}' in model_name or (depth == 20 and 'Depth=∞' in model_name):
                        scores.append(results[data_name][model_name]['test_score'])
                avg_scores_by_depth.append(np.mean(scores) if scores else 0)
            
            axes[0, 1].plot(depth_values, avg_scores_by_depth, 's-', 
                           color=colors[i], label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xlabel('Max Depth')
        axes[0, 1].set_ylabel('Average Test Score')
        axes[0, 1].set_title('Random Forest: Max Depth vs. Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance vs. Stability
        all_scores = []
        all_cv_stds = []
        for data_name in dataset_names:
            for model_name in results[data_name].keys():
                all_scores.append(results[data_name][model_name]['test_score'])
                all_cv_stds.append(results[data_name][model_name]['cv_std'])
        
        axes[1, 0].scatter(all_cv_stds, all_scores, alpha=0.6)
        axes[1, 0].set_xlabel('CV Standard Deviation')
        axes[1, 0].set_ylabel('Test Score')
        axes[1, 0].set_title('Random Forest: Performance vs. Stability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best configuration per dataset
        best_models = []
        best_scores = []
        for data_name in dataset_names:
            best_model = max(results[data_name].keys(), 
                           key=lambda k: results[data_name][k]['test_score'])
            best_score = results[data_name][best_model]['test_score']
            best_models.append(best_model)
            best_scores.append(best_score)
        
        axes[1, 1].bar(dataset_names, best_scores, alpha=0.7, color='purple')
        axes[1, 1].set_ylabel('Best Test Score')
        axes[1, 1].set_title('Random Forest: Best Performance per Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/random_forest_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_svm_results(self, results: Dict):
        """Plot SVM results showing C parameter effects."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract C values for each kernel
        c_values = [0.1, 1, 10, 100]
        kernels = ['linear', 'rbf']
        
        # Plot 1: C vs Performance for RBF kernel
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            rbf_scores = []
            for c in c_values:
                model_name = f'C={c}, rbf'
                if model_name in results[data_name]:
                    rbf_scores.append(results[data_name][model_name]['test_acc'])
                else:
                    rbf_scores.append(0)
            
            axes[0, 0].plot(c_values, rbf_scores, 'o-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('C Parameter')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('SVM: C Parameter vs. Performance (RBF Kernel)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: C vs Performance for Linear kernel
        for i, data_name in enumerate(dataset_names):
            linear_scores = []
            for c in c_values:
                model_name = f'C={c}, linear'
                if model_name in results[data_name]:
                    linear_scores.append(results[data_name][model_name]['test_acc'])
                else:
                    linear_scores.append(0)
            
            axes[0, 1].plot(c_values, linear_scores, 's-', color=colors[i], 
                           label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('C Parameter')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('SVM: C Parameter vs. Performance (Linear Kernel)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Kernel comparison
        kernel_comparison = {'linear': [], 'rbf': []}
        for data_name in dataset_names:
            # Find best C for each kernel
            best_linear = 0
            best_rbf = 0
            for c in c_values:
                linear_model = f'C={c}, linear'
                rbf_model = f'C={c}, rbf'
                if linear_model in results[data_name]:
                    best_linear = max(best_linear, results[data_name][linear_model]['test_acc'])
                if rbf_model in results[data_name]:
                    best_rbf = max(best_rbf, results[data_name][rbf_model]['test_acc'])
            kernel_comparison['linear'].append(best_linear)
            kernel_comparison['rbf'].append(best_rbf)
        
        x = np.arange(len(dataset_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, kernel_comparison['linear'], width, 
                       label='Linear Kernel', alpha=0.7)
        axes[1, 0].bar(x + width/2, kernel_comparison['rbf'], width, 
                       label='RBF Kernel', alpha=0.7)
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Best Test Accuracy')
        axes[1, 0].set_title('SVM: Kernel Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dataset_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Stability analysis
        all_scores = []
        all_gaps = []
        for data_name in dataset_names:
            for model_name in results[data_name].keys():
                all_scores.append(results[data_name][model_name]['test_acc'])
                all_gaps.append(results[data_name][model_name]['gap'])
        
        axes[1, 1].scatter(all_gaps, all_scores, alpha=0.6)
        axes[1, 1].set_xlabel('Train-Test Gap')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('SVM: Performance vs. Stability')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/svm_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_neural_network_results(self, results: Dict):
        """Plot neural network results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract layer sizes and regularization
        layer_configs = [(10,), (50, 25), (100, 50, 25)]
        alphas = [0.0001, 0.001, 0.01]
        
        # Plot 1: Network size vs performance
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            size_scores = []
            size_labels = []
            for layers in layer_configs:
                scores = []
                for alpha in alphas:
                    model_name = f'Layers={layers}, α={alpha}'
                    if model_name in results[data_name]:
                        scores.append(results[data_name][model_name]['test_acc'])
                if scores:
                    size_scores.append(np.mean(scores))
                    size_labels.append(str(layers))
            
            if size_scores:
                axes[0, 0].plot(range(len(size_scores)), size_scores, 'o-', 
                               color=colors[i], label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel('Network Size (Complexity)')
        axes[0, 0].set_ylabel('Average Test Accuracy')
        axes[0, 0].set_title('Neural Networks: Network Size vs. Performance')
        axes[0, 0].set_xticks(range(len(size_labels)))
        axes[0, 0].set_xticklabels(size_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Regularization vs performance
        for i, data_name in enumerate(dataset_names):
            reg_scores = []
            for alpha in alphas:
                scores = []
                for layers in layer_configs:
                    model_name = f'Layers={layers}, α={alpha}'
                    if model_name in results[data_name]:
                        scores.append(results[data_name][model_name]['test_acc'])
                if scores:
                    reg_scores.append(np.mean(scores))
            
            if reg_scores:
                axes[0, 1].plot(alphas, reg_scores, 's-', color=colors[i], 
                               label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('Regularization (Alpha)')
        axes[0, 1].set_ylabel('Average Test Accuracy')
        axes[0, 1].set_title('Neural Networks: Regularization vs. Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Overfitting analysis
        all_gaps = []
        all_scores = []
        for data_name in dataset_names:
            for model_name in results[data_name].keys():
                all_gaps.append(results[data_name][model_name]['gap'])
                all_scores.append(results[data_name][model_name]['test_acc'])
        
        axes[1, 0].scatter(all_gaps, all_scores, alpha=0.6)
        axes[1, 0].set_xlabel('Train-Test Gap')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].set_title('Neural Networks: Performance vs. Overfitting')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best configuration per dataset
        best_models = []
        best_scores = []
        for data_name in dataset_names:
            best_model = max(results[data_name].keys(), 
                           key=lambda k: results[data_name][k]['test_acc'])
            best_score = results[data_name][best_model]['test_acc']
            best_models.append(best_model)
            best_scores.append(best_score)
        
        axes[1, 1].bar(dataset_names, best_scores, alpha=0.7, color='purple')
        axes[1, 1].set_ylabel('Best Test Accuracy')
        axes[1, 1].set_title('Neural Networks: Best Performance per Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/neural_network_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_gradient_boosting_results(self, results: Dict):
        """Plot gradient boosting results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dataset_names = list(results.keys())
        
        # Extract parameter combinations
        n_estimators_values = [50, 100, 200]
        learning_rates = [0.01, 0.1, 0.2]
        
        # Plot 1: Number of trees vs performance
        colors = ['blue', 'red', 'green', 'orange']
        for i, data_name in enumerate(dataset_names):
            tree_scores = []
            for n_est in n_estimators_values:
                scores = []
                for model_name in results[data_name].keys():
                    if f'Trees={n_est},' in model_name:
                        scores.append(results[data_name][model_name]['test_score'])
                if scores:
                    tree_scores.append(np.mean(scores))
                else:
                    tree_scores.append(0)
            
            axes[0, 0].plot(n_estimators_values, tree_scores, 'o-', 
                           color=colors[i], label=data_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel('Number of Trees')
        axes[0, 0].set_ylabel('Average Test Score')
        axes[0, 0].set_title('Gradient Boosting: Number of Trees vs. Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning rate vs performance
        for i, data_name in enumerate(dataset_names):
            lr_scores = []
            for lr in learning_rates:
                scores = []
                for model_name in results[data_name].keys():
                    if f'LR={lr},' in model_name:
                        scores.append(results[data_name][model_name]['test_score'])
                if scores:
                    lr_scores.append(np.mean(scores))
                else:
                    lr_scores.append(0)
            
            axes[0, 1].plot(learning_rates, lr_scores, 's-', 
                           color=colors[i], label=data_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Average Test Score')
        axes[0, 1].set_title('Gradient Boosting: Learning Rate vs. Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance vs. Overfitting
        all_scores = []
        all_gaps = []
        for data_name in dataset_names:
            for model_name in results[data_name].keys():
                all_scores.append(results[data_name][model_name]['test_score'])
                all_gaps.append(results[data_name][model_name]['gap'])
        
        axes[1, 0].scatter(all_gaps, all_scores, alpha=0.6)
        axes[1, 0].set_xlabel('Train-Test Gap')
        axes[1, 0].set_ylabel('Test Score')
        axes[1, 0].set_title('Gradient Boosting: Performance vs. Overfitting')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best configuration per dataset
        best_models = []
        best_scores = []
        for data_name in dataset_names:
            best_model = max(results[data_name].keys(), 
                           key=lambda k: results[data_name][k]['test_score'])
            best_score = results[data_name][best_model]['test_score']
            best_models.append(best_model)
            best_scores.append(best_score)
        
        axes[1, 1].bar(dataset_names, best_scores, alpha=0.7, color='green')
        axes[1, 1].set_ylabel('Best Test Score')
        axes[1, 1].set_title('Gradient Boosting: Best Performance per Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/gradient_boosting_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_algorithm_analyses(self) -> Dict:
        """Run comprehensive analysis of all algorithms."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ALGORITHM BIAS-VARIANCE ANALYSIS")
        print("="*80)
        
        print("""
This comprehensive analysis examines bias-variance characteristics
across major algorithm families:

1. Linear Regression - High bias baseline
2. Logistic Regression - Classification baseline  
3. K-Nearest Neighbors - Variance-sensitive
4. Decision Trees - Complexity-controlled
5. Random Forest - Variance reduction
6. Support Vector Machines - Regularization-controlled
7. Neural Networks - Capacity-controlled
8. Gradient Boosting - Sequential ensemble

Each analysis includes:
• Default bias-variance characteristics
• Hyperparameter impact analysis
• Performance across different data types
• Visualization of behavior patterns
• Practical tuning recommendations
        """)
        
        all_results = {}
        
        try:
            # Run each algorithm analysis
            algorithms = [
                ('Linear Regression', self.demonstrate_linear_regression),
                ('Logistic Regression', self.demonstrate_logistic_regression),
                ('K-Nearest Neighbors', self.demonstrate_knn),
                ('Decision Trees', self.demonstrate_decision_trees),
                ('Random Forest', self.demonstrate_random_forest),
                ('Support Vector Machines', self.demonstrate_svm),
                ('Neural Networks', self.demonstrate_neural_networks),
                ('Gradient Boosting', self.demonstrate_gradient_boosting)
            ]
            
            for name, func in algorithms:
                print(f"\n{'='*20} {name} {'='*20}")
                try:
                    result = func()
                    all_results[name] = result
                    print(f"✓ {name} analysis completed")
                except Exception as e:
                    print(f"✗ Error in {name}: {str(e)}")
                    logger.error(f"Error in {name}: {str(e)}")
                    all_results[name] = {"error": str(e)}
        
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            print(f"Error occurred: {str(e)}")
        
        # Print final summary
        self._print_algorithm_summary(all_results)
        
        return all_results
    
    def _print_algorithm_summary(self, results: Dict):
        """Print summary of algorithm analysis."""
        print("\n" + "="*80)
        print("ALGORITHM BIAS-VARIANCE SUMMARY")
        print("="*80)
        
        summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALGORITHM CHARACTERISTICS SUMMARY                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. LINEAR REGRESSION
   • Default: High bias, low variance
   • Best for: Linear relationships, interpretable models
   • Bias reduction: Polynomial features, interaction terms
   • Variance control: Regularization (Ridge, Lasso)

2. LOGISTIC REGRESSION  
   • Default: High bias, low variance
   • Best for: Linear classification, probabilistic output
   • Bias reduction: Feature engineering, non-linear features
   • Variance control: Regularization parameter C

3. K-NEAREST NEIGHBORS
   • K=1: Very low bias, very high variance
   • K=large: High bias, low variance
   • Best for: Local patterns, non-parametric learning
   • Bias-variance control: K value, distance weighting

4. DECISION TREES
   • Unconstrained: Very low bias, very high variance
   • Depth-limited: Controlled bias-variance trade-off
   • Best for: Non-linear patterns, interpretable results
   • Variance control: Depth limits, pruning, ensembling

5. RANDOM FOREST
   • Default: Moderate bias, reduced variance
   • Best for: Robust performance, feature importance
   • Bias control: Tree depth, number of features
   • Variance control: Number of trees, bootstrap sampling

6. SUPPORT VECTOR MACHINES
   • High C: Low bias, high variance
   • Low C: High bias, low variance
   • Best for: High-dimensional data, margin maximization
   • Bias-variance control: C parameter, kernel choice

7. NEURAL NETWORKS
   • Large networks: Low bias, high variance
   • Small networks: High bias, low variance
   • Best for: Complex patterns, feature learning
   • Variance control: Regularization, dropout, architecture

8. GRADIENT BOOSTING
   • Sequential: Low bias, moderate variance
   • Best for: High performance, complex patterns
   • Bias control: More trees, deeper trees
   • Variance control: Learning rate, early stopping

╔══════════════════════════════════════════════════════════════════════════════╗
║                    SELECTION RECOMMENDATIONS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SMALL DATASETS (< 500 samples):
• Start with Linear/Logistic Regression (high bias acceptable)
• Use strong regularization
• Consider KNN with appropriate K
• Avoid very complex models

MEDIUM DATASETS (500-5000 samples):
• Decision Trees with depth limits
• Random Forest with moderate complexity
• SVM with tuned parameters
• Neural networks with moderate size

LARGE DATASETS (> 5000 samples):
• Complex models (can handle variance)
• Deep neural networks
• Gradient Boosting
• Ensemble methods

HIGH-DIMENSIONAL DATA:
• Linear models with regularization
• SVM (especially linear kernel)
• Random Forest (feature selection)
• Neural networks with dropout

KEY PRINCIPLE:
Match model capacity to data size and complexity.
Simple data → simpler models (accept some bias)
Complex data → complex models (control variance)

The optimal model is the simplest one that captures the underlying pattern.
        """
        
        print(summary)


def main():
    """Main function to run all algorithm analyses."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ALGORITHM-SPECIFIC BIAS-VARIANCE ANALYSIS                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This comprehensive analysis examines bias-variance characteristics
across major machine learning algorithm families.

Each algorithm is analyzed for:
• Default bias-variance behavior
• Hyperparameter impact on trade-off
• Performance across different data types
• Practical tuning strategies
• Visualization of behavior patterns

Generated plots in 'plots/' directory:
• linear_regression_analysis.png
• logistic_regression_analysis.png
• knn_analysis.png
• decision_tree_analysis.png
• random_forest_analysis.png
• svm_analysis.png
• neural_network_analysis.png
• gradient_boosting_analysis.png

These analyses provide practical guidance for:
• Algorithm selection based on data characteristics
• Hyperparameter tuning for optimal bias-variance balance
• Understanding when each algorithm is most appropriate
• Diagnosing and fixing bias-variance problems
    """)
    
    # Create analyzer and run all analyses
    analyzer = AlgorithmBiasVarianceAnalyzer(random_state=42)
    results = analyzer.run_all_algorithm_analyses()
    
    print("\n" + "="*80)
    print("ALGORITHM ANALYSIS COMPLETE!")
    print("="*80)
    print("All visualizations saved to the 'plots/' directory")
    print("Review the generated plots and analysis results")
    print("Use these insights for algorithm selection and tuning")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
