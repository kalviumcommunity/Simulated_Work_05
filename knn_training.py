"""
K-Nearest Neighbors (KNN) Training and Interpretation

Comprehensive implementation of KNN training, evaluation, and interpretation for both 
classification and regression problems. This module covers:

- How KNN works conceptually as an instance-based learning algorithm
- Distance metrics and their impact on similarity measurement
- The bias-variance trade-off in KNN and how K controls it
- Practical implementation with scikit-learn for both problem types
- Visualization of decision boundaries and neighbor influences
- Comparison with baseline models and practical considerations
- Real-world examples and common pitfalls to avoid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import time

# Set up warnings filter
warnings.filterwarnings('ignore')


class KNNTrainer:
    """
    Comprehensive KNN training and interpretation system.
    
    This class provides methods to train, evaluate, and interpret KNN models
    for both classification and regression problems, with detailed analysis of
    distance metrics, scaling considerations, and practical recommendations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the KNN trainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def demonstrate_knn_concepts(self):
        """Demonstrate fundamental KNN concepts."""
        print("\n" + "="*70)
        print("K-NEAREST NEIGHBORS FUNDAMENTALS")
        print("="*70)
        
        print("""
📖 SCENARIO: Understanding how KNN works as an instance-based algorithm

🎯 GOAL: Learn the core principles of KNN and why it's different from parametric models
📊 THEORY: Distance metrics, similarity, and the voting mechanism
🔧 APPROACH: Visual demonstrations of distance calculations and neighbor effects
        """)
        
        # Create simple 2D dataset for visualization
        np.random.seed(self.random_state)
        
        # Create two distinct clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 50)
        cluster2 = np.random.multivariate_normal([8, 6], [[1, 0.5], [0.5, 1]], 50)
        
        X = np.vstack([cluster1, cluster2])
        y = np.array([0] * 50 + [1] * 50)  # 50 points from each cluster
        
        print(f"Generated dataset: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Demonstrate distance metrics
        self._demonstrate_distance_metrics()
        
        # Show KNN in action with different K values
        self._demonstrate_k_varying_k(X, y)
        
        # Explain the concepts
        self._explain_knn_principles()
        
        return X, y
    
    def _demonstrate_distance_metrics(self):
        """Demonstrate different distance metrics used in KNN."""
        print("\nDISTANCE METRICS DEMONSTRATION")
        print("-" * 50)
        
        # Create sample points for distance comparison
        points = np.array([
            [0, 0],    # Origin
            [3, 4],    # Point A
            [6, 2]     # Point B
            [2, 6]     # Point C (same distance to A as B)
            [9, 0]     # Point D (closer to origin than A)
        ])
        
        # Calculate different distances from origin
        origin = np.array([0, 0])
        
        # Euclidean distance
        euclidean_distances = np.sqrt(np.sum((points - origin)**2, axis=1))
        
        # Manhattan distance
        manhattan_distances = np.sum(np.abs(points - origin), axis=1)
        
        # Minkowski distance (generalization)
        p_values = [1, 2, 3]  # Different p values
        minkowski_distances = []
        for p in p_values:
            distances = np.sum(np.abs(points - origin)**p, axis=1)**(1/p)
            minkowski_distances.append(distances)
        
        # Display results
        print("Distance calculations from origin [0, 0]:")
        print("Point\tEuclidean\tManhattan\tMinkowski(p=2)")
        for i, point in enumerate(points):
            print(f"{point}\t{euclidean_distances[i]:.3f}\t{manhattan_distances[i]:.3f}\t{minkowski_distances[1][i]:.3f}")
        
        # Visualize distance circles
        self._plot_distance_comparison(points, origin, euclidean_distances, manhattan_distances)
        
        return points
    
    def _plot_distance_comparison(self, points: np.ndarray, origin: np.ndarray, 
                            euclidean_distances: np.ndarray, manhattan_distances: np.ndarray):
        """Plot distance metric comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot points and origin
        axes[0, 0].scatter(points[:, 0], points[:, 1], c='blue', s=100, label='Data Points')
        axes[0, 0].scatter(origin[0], origin[1], c='red', s=200, marker='*', label='Origin')
        
        # Plot Euclidean distance circles
        for i, point in enumerate(points):
            circle = plt.Circle(point, euclidean_distances[i], fill=False, 
                              edgecolor='blue', alpha=0.3, label='Euclidean')
            axes[0, 0].add_patch(circle)
        
        # Plot Manhattan distance circles
        for i, point in enumerate(points):
            circle = plt.Circle(point, manhattan_distances[i], fill=False, 
                              edgecolor='green', alpha=0.3, linestyle='--', label='Manhattan')
            axes[0, 0].add_patch(circle)
        
        axes[0, 0].set_xlim(-2, 12)
        axes[0, 0].set_ylim(-2, 10)
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].set_title('Distance Metrics Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("plots/distance_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _demonstrate_k_varying_k(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate KNN behavior with different K values."""
        print("\nK VALUE DEMONSTRATION")
        print("-" * 50)
        
        # Test different K values
        k_values = [1, 3, 5, 10, 15, 25]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, k in enumerate(k_values):
            print(f"\nTraining KNN with K={k}...")
            
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=k, random_state=self.random_state)
            start_time = time.time()
            knn.fit(X, y)
            train_time = time.time() - start_time
            
            # Store model
            self.models[f'K={k}'] = knn
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
            
            print(f"  Training time: {train_time:.3f}s")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Plot decision boundary for this K
            row, col = i // 3, i % 3
            
            # Create a mesh for visualization
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            
            # Predict on mesh
            mesh_data = np.c_[xx.ravel(), yy.ravel()]
            Z = knn.predict(mesh_data).reshape(xx.shape)
            
            # Plot
            axes[row, col].contourf(xx, yy, Z, alpha=0.3, levels=20)
            axes[row, col].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
            axes[row, col].set_title(f'KNN Decision Boundary (K={k})')
            axes[row, col].set_xlabel('Feature 1')
            axes[row, col].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig("plots/knn_varying_k.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return k_values
    
    def _explain_knn_principles(self):
        """Explain the fundamental principles of KNN."""
        print("\n" + "="*70)
        print("KNN FUNDAMENTAL PRINCIPLES")
        print("="*70)
        
        principles = """
CORE PRINCIPLES:

1. INSTANCE-BASED LEARNING:
   KNN stores ALL training data
   No parameters are learned during training
   Prediction = majority vote of K nearest neighbors
   Entire computation happens at prediction time

2. LAZY EVALUATION:
   "Lazy" because no work is done during training
   All computational cost is deferred to prediction time
   Makes KNN suitable for dynamic datasets where new data arrives frequently

3. DISTANCE-BASED SIMILARITY:
   Similarity = inverse of distance
   Closer neighbors have more influence on prediction
   Distance choice determines what "similar" means
   Different metrics capture different notions of closeness

4. BIAS-VARIANCE TRADE-OFF:
   K (number of neighbors) controls model complexity:
   • Small K → High bias, low variance (underfitting)
   • Large K → Low bias, high variance (overfitting)
   • Optimal K balances both for your specific dataset

5. CURSE OF DIMENSIONALITY:
   In high dimensions, "distance concentration" occurs
   All points become roughly equidistant
   KNN effectiveness degrades significantly beyond ~20 features
   Dimensionality reduction often necessary before KNN

6. FEATURE SCALING CRITICAL:
   KNN is distance-based, not scale-invariant
   Features on different scales dominate distance calculations
   ALWAYS scale features before applying KNN
   Use StandardScaler for normally distributed features
   Use MinMaxScaler for bounded features
   Scaling must be fitted on training data only

PRACTICAL IMPLICATIONS:
• KNN is most interpretable when K is small
• KNN works surprisingly well for certain problems (image classification, recommendation systems)
• KNN struggles with high-dimensional sparse data
• KNN prediction time grows linearly with dataset size
• KNN memory usage grows with both dataset size and K
        """
        
        print(principles)
    
    def demonstrate_scaling_importance(self):
        """Demonstrate why feature scaling is critical for KNN."""
        print("\n" + "="*70)
        print("FEATURE SCALING IMPORTANCE")
        print("="*70)
        
        print("""
📖 SCENARIO: Demonstrating the critical importance of feature scaling for KNN

🎯 GOAL: Show how unscaled features destroy KNN performance
📊 DATASET: Mixed-scale features with clear relationships
🔧 APPROACH: Compare KNN with and without proper scaling
        """)
        
        # Create data with different feature scales
        np.random.seed(self.random_state)
        n_samples = 200
        
        # Feature 1: Small scale (0-10)
        X1 = np.random.normal(2, 1, (n_samples, 1))
        
        # Feature 2: Large scale (100-1000)
        X2 = np.random.normal(500, 100, (n_samples, 1))
        
        # Feature 3: Binary scale (0 or 1)
        X3 = np.random.binomial(1, 0.3, (n_samples, 1)) * 100
        
        # True relationship depends only on the small-scale feature
        y = (X1 > 1).astype(int)
        
        # Combine features
        X = np.column_stack([X1, X2, X3])
        
        print(f"Dataset: {X.shape}")
        print(f"Feature scales: [0-10], [100-1000], [0-100]")
        print(f"True relationship: y depends only on Feature 1 (small scale)")
        
        # Train KNN without scaling
        print("\nTraining KNN WITHOUT scaling...")
        knn_no_scale = KNeighborsClassifier(n_neighbors=5, random_state=self.random_state)
        knn_no_scale.fit(X, y)
        y_pred_no_scale = knn_no_scale.predict(X)
        accuracy_no_scale = accuracy_score(y, y_pred_no_scale)
        
        print(f"Accuracy without scaling: {accuracy_no_scale:.4f}")
        
        # Train KNN with proper scaling
        print("\nTraining KNN WITH scaling...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, random_state=self.random_state))
        ])
        pipeline.fit(X, y)
        y_pred_scaled = pipeline.predict(X)
        accuracy_scaled = accuracy_score(y, y_pred_scaled)
        
        print(f"Accuracy with scaling: {accuracy_scaled:.4f}")
        
        # Show improvement
        improvement = accuracy_scaled - accuracy_no_scale
        print(f"Improvement from scaling: +{improvement:.4f}")
        
        # Visualize the dramatic difference
        self._plot_scaling_effects(X, y, y_pred_no_scale, y_pred_scaled, 
                               knn_no_scale, knn_no_scale, pipeline)
        
        return {
            'accuracy_no_scale': accuracy_no_scale,
            'accuracy_scaled': accuracy_scaled,
            'improvement': improvement
        }
    
    def _plot_scaling_effects(self, X: np.ndarray, y: np.ndarray, 
                        y_pred_no_scale: np.ndarray, y_pred_scaled: np.ndarray,
                        knn_no_scale, knn_no_scale, pipeline):
        """Plot the effects of feature scaling on KNN performance."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        
        # Plot 1: Data visualization
        axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, label='Data Points')
        axes[0, 0].set_xlabel('Feature 1 (Small Scale)')
        axes[0, 0].set_ylabel('Feature 2 (Large Scale)')
        axes[0, 0].set_title('Data: True relationship depends on Feature 1 only')
        
        # Plot 2: Without scaling predictions
        axes[0, 1].scatter(X[:, 0], y_pred_no_scale, alpha=0.6, 
                          label='Without Scaling', marker='o')
        axes[0, 1].scatter(X[:, 0], y_pred_scaled, alpha=0.6, 
                          label='With Scaling', marker='s')
        
        # Perfect prediction line
        min_val = min(y.min(), y_pred_no_scale.min(), y_pred_scaled.min())
        max_val = max(y.max(), y_pred_no_scale.max(), y_pred_scaled.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', 
                        linewidth=2, label='Perfect Prediction')
        
        axes[0, 1].set_xlabel('Feature 1')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Feature Scaling Impact on KNN Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Decision boundary comparison
        # Create mesh for boundary visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50)
        )
        
        mesh_data = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict without scaling
        Z_no_scale = knn_no_scale.predict(mesh_data).reshape(xx.shape)
        
        # Predict with scaling
        Z_scaled = pipeline.predict(mesh_data).reshape(xx.shape)
        
        # Plot boundaries
        axes[1, 0].contourf(xx, yy, Z_no_scale, alpha=0.3, levels=20, 
                          colors='red', linestyles='--')
        axes[1, 0].contourf(xx, yy, Z_scaled, alpha=0.3, levels=20, 
                          colors='blue', linestyles='--')
        
        axes[1, 0].set_title('Decision Boundaries: Red=Without Scaling, Blue=With Scaling')
        axes[1, 0].set_xlabel('Feature 1')
        axes[1, 0].set_ylabel('Feature 2')
        axes[1, 0].legend()
        
        plt.tight_layout()
        plt.savefig("plots/knn_scaling_effects.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_knn_classification(self):
        """Demonstrate complete KNN classification workflow."""
        print("\n" + "="*70)
        print("KNN CLASSIFICATION WORKFLOW")
        print("="*70)
        
        print("""
📖 SCENARIO: Complete KNN classification pipeline from training to evaluation

🎯 GOAL: Learn proper KNN implementation with scaling, validation, and interpretation
📊 DATASET: Multi-class classification with mixed feature types
🔧 APPROACH: End-to-end workflow with baseline comparison and visualization
        """)
        
        # Create realistic dataset
        X, y = make_classification(
            n_samples=800, n_features=10, n_informative=6,
            n_classes=3, n_clusters_per_class=1,
            random_state=self.random_state
        )
        
        # Add some categorical features
        np.random.seed(self.random_state + 1)
        cat_features = np.random.randint(0, 3, size=(X.shape[0], 3))
        X_with_cat = np.column_stack([X, np.eye(X.shape[0])[cat_features]])
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_cat, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Features: {list(X_with_cat.columns)}")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, random_state=self.random_state))
        ])
        
        # Train with cross-validation for K selection
        print("Running cross-validation for K selection...")
        param_grid = {"knn__n_neighbors": range(1, 31)}
        
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        
        start_time = time.time()
        grid.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        print(f"GridSearchCV completed in {grid_time:.2f} seconds")
        print(f"Best K: {grid.best_params_['knn__n_neighbors']}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        
        # Train final model with best K
        best_k = grid.best_params_['knn__n_neighbors']
        final_knn = KNeighborsClassifier(n_neighbors=best_k, random_state=self.random_state)
        
        start_time = time.time()
        final_knn.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate final model
        y_pred = final_knn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Baseline comparison
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_accuracy = baseline.score(X_test, y_test)
        
        print(f"\nFinal Model Performance:")
        print(f"  Training time: {train_time:.3f}s")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"  Improvement:  +{test_accuracy - baseline_accuracy:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, [f'Class {i}' for i in range(3)])
        
        # Analyze neighbor influence
        self._analyze_neighbor_influence(final_knn, X_train, y_train, X_test, y_test)
        
        # Store results
        self.results['classification'] = {
            'best_k': best_k,
            'test_accuracy': test_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'improvement': test_accuracy - baseline_accuracy,
            'grid': grid,
            'final_model': final_knn
        }
        
        return final_knn
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix with annotations."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("plots/knn_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_neighbor_influence(self, knn, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray):
        """Analyze how different neighbors influence predictions."""
        print("\nNEIGHBOR INFLUENCE ANALYSIS")
        print("-" * 50)
        
        # Get indices of nearest neighbors for test points
        distances, indices = knn.kneighbors(X_test, n_neighbors=knn.n_neighbors)
        
        # Analyze first few test points
        for i in range(min(10, len(y_test))):
            # Get neighbors for this point
            point_neighbors = indices[i]
            point_distances = distances[i]
            neighbor_labels = y_train[point_neighbors]
            neighbor_distances = point_distances[point_neighbors]
            
            print(f"\nTest Point {i}:")
            print(f"  True label: {y_test[i]}")
            print(f"  Neighbor labels: {neighbor_labels}")
            print(f"  Neighbor distances: {neighbor_distances}")
            
            # Majority vote
            from collections import Counter
            vote_counts = Counter(neighbor_labels)
            majority_vote = vote_counts.most_common(1)[0] if vote_counts else None
            
            print(f"  Majority vote: {majority_vote}")
            print(f"  Vote distribution: {dict(vote_counts)}")
    
    def demonstrate_knn_regression(self):
        """Demonstrate KNN for regression problems."""
        print("\n" + "="*70)
        print("KNN REGRESSION WORKFLOW")
        print("="*70)
        
        print("""
📖 SCENARIO: KNN regression with continuous target prediction

🎯 GOAL: Learn how KNN handles regression tasks and averaging
📊 DATASET: Non-linear relationship with noise
🔧 APPROACH: Complete workflow with evaluation and visualization
        """)
        
        # Create non-linear regression data
        X, y = make_regression(
            n_samples=600, n_features=8, n_informative=5,
            noise=0.2, random_state=self.random_state
        )
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Try different K values
        k_values = [1, 3, 5, 10, 15]
        results = {}
        
        for k in k_values:
            print(f"\nTesting K={k}...")
            
            # Train KNN regressor
            knn_reg = KNeighborsRegressor(n_neighbors=k, random_state=self.random_state)
            start_time = time.time()
            knn_reg.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred = knn_reg.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(knn_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[k] = {
                'k': k,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'train_time': train_time
            }
            
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test R²:   {test_r2:.4f}")
            print(f"  CV RMSE:    {cv_rmse:.4f}")
        
        # Visualize results
        self._plot_regression_results(k_values, results)
        
        return results
    
    def _plot_regression_results(self, k_values: List[int], results: Dict):
        """Plot KNN regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract metrics
        rmse_values = [results[k]['test_rmse'] for k in k_values]
        r2_values = [results[k]['test_r2'] for k in k_values]
        
        # Plot 1: RMSE vs K
        axes[0, 0].plot(k_values, rmse_values, 'bo-', marker='o', label='RMSE')
        axes[0, 0].set_xlabel('K (Number of Neighbors)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('KNN Regression: K vs. RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: R² vs K
        axes[0, 1].plot(k_values, r2_values, 'go-', marker='s', label='R²')
        axes[0, 1].set_xlabel('K (Number of Neighbors)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_title('KNN Regression: K vs. R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/knn_regression_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_knn_vs_baseline(self):
        """Compare KNN performance against baseline models."""
        print("\n" + "="*70)
        print("KNN VS BASELINE COMPARISON")
        print("="*70)
        
        # Create dataset
        X, y = make_classification(
            n_samples=800, n_features=8, n_informative=5,
            random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Models to compare
        models = {
            'KNN (K=5)': KNeighborsClassifier(n_neighbors=5, random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, 
                                       random_state=self.random_state, n_jobs=-1)
        }
        
        # Evaluate all models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'test_accuracy': test_accuracy,
                'train_time': train_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Baseline
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_accuracy = baseline.score(X_test, y_test)
        
        print(f"\nBaseline (Most Frequent Class): {baseline_accuracy:.4f}")
        
        # Visualize comparison
        self._plot_model_comparison(results, baseline_accuracy)
        
        return results
    
    def _plot_model_comparison(self, results: Dict, baseline_accuracy: float):
        """Plot model comparison results."""
        names = list(results.keys())
        test_acc = [results[name]['test_accuracy'] for name in names]
        cv_means = [results[name]['cv_mean'] for name in names]
        
        x = np.arange(len(names))
        width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Test accuracy comparison
        bars1 = ax1.bar(x - width/2, test_acc, width, label='Test Accuracy', alpha=0.7)
        ax1.axhline(y=baseline_accuracy, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_accuracy:.3f})')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Model Comparison: Test Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CV stability comparison
        bars2 = ax2.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.7)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('CV Accuracy')
        ax2.set_title('Model Comparison: Cross-Validation Stability')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/knn_vs_baseline.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_practical_checklist(self):
        """Print practical checklist for KNN implementation."""
        print("\n" + "="*70)
        print("PRACTICAL KNN CHECKLIST")
        print("="*70)
        
        checklist = """
🔍 PRE-TRAINING CHECKLIST:
□ Understand data characteristics and class distribution
□ Check for mixed feature types (numerical, categorical)
□ Assess need for feature scaling based on feature ranges
□ Plan appropriate train/test split with stratification
□ Consider computational cost vs. prediction speed requirements

🌳 TRAINING CHECKLIST:
□ Always scale features before applying KNN
□ Choose appropriate distance metric for your problem
□ Use cross-validation to select optimal K value
□ Set random_state for reproducible results
□ Consider using weighted voting for imbalanced datasets
□ Monitor training time vs. dataset size

📊 EVALUATION CHECKLIST:
□ Always compute both training and test accuracy
□ Calculate train/test gap to detect overfitting
□ Use cross-validation to assess model stability
□ Compare against appropriate baseline (majority class, mean target)
□ Generate confusion matrix for classification tasks
□ Compute multiple metrics (accuracy, precision, recall, F1)
□ Analyze prediction errors and patterns

🎯 INTERPRETATION CHECKLIST:
□ Analyze neighbor influence on individual predictions
□ Visualize decision boundaries for 2D/3D problems
□ Consider distance metric choice and its impact on results
□ Document K value selection process and reasoning
□ Validate model on held-out test set only once
□ Consider computational requirements for real-time prediction
□ Compare with other algorithms before finalizing KNN

🚨 COMMON MISTAKES TO AVOID:
□ Not scaling features (KNN is distance-based)
□ Choosing K arbitrarily without validation
□ Using KNN on very large datasets without optimization
□ Ignoring computational cost and memory requirements
□ Not considering the curse of dimensionality
□ Using KNN for real-time applications without optimization
□ Not comparing against baseline models
□ Over-interpreting neighbor influence as causation
□ Not validating that improvements generalize to new data
□ Using KNN when interpretability is not required
□ Ignoring class imbalance in voting
        """
        
        print(checklist)
    
    def run_complete_tutorial(self):
        """Run the complete KNN training tutorial."""
        print("""
🎯 K-NEAREST NEIGHBORS TRAINING TUTORIAL
============================================

This tutorial provides comprehensive coverage of KNN training, evaluation, and 
interpretation for both classification and regression problems.

📚 TUTORIAL STRUCTURE:
1. KNN Fundamental Concepts
2. Distance Metrics and Similarity
3. K Value Impact on Bias-Variance
4. Feature Scaling Critical Importance
5. Complete Classification Workflow
6. Complete Regression Workflow
7. Comparison with Baseline Models
8. Practical Checklist and Best Practices

⏱️  EXPECTED DURATION: 75-90 minutes
        """)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Run all demonstrations
        demonstrations = [
            ("KNN Concepts", self.demonstrate_knn_concepts),
            ("Feature Scaling", self.demonstrate_scaling_importance),
            ("KNN Classification", self.demonstrate_knn_classification),
            ("KNN Regression", self.demonstrate_knn_regression),
            ("Baseline Comparison", self.demonstrate_knn_vs_baseline)
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
            "• KNN is an instance-based, lazy learning algorithm",
            "• Feature scaling is absolutely critical for KNN performance",
            "• K controls the bias-variance trade-off (small K = high bias, large K = high variance)",
            "• Distance metric choice significantly impacts what 'similar' means",
            "• KNN is most interpretable when K is small",
            "• The curse of dimensionality severely impacts KNN effectiveness",
            "• Cross-validation is essential for selecting optimal K and assessing stability",
            "• KNN works surprisingly well for image and recommendation systems",
            "• Always compare against baseline models to validate KNN effectiveness",
            "• Consider computational costs before deploying KNN on large datasets"
        ]
        
        for takeaway in takeaways:
            print(f"  {takeaway}")
        
        return True


def main():
    """Main function to run the KNN training tutorial."""
    trainer = KNNTrainer(random_state=42)
    return trainer.run_complete_tutorial()


if __name__ == "__main__":
    main()
