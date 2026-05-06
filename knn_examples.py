"""
K-Nearest Neighbors Examples and Utilities

Additional practical examples and utilities for KNN models including
advanced techniques, optimizations, and real-world considerations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
import time

warnings.filterwarnings('ignore')


class KNNExamples:
    """Collection of advanced KNN examples and utilities."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def example_weighted_knn(self):
        """Example: Weighted KNN for imbalanced datasets."""
        print("\n" + "="*60)
        print("EXAMPLE: WEIGHTED KNN FOR IMBALANCED DATASETS")
        print("="*60)
        
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=1200, n_features=8, n_informative=6,
            weights=[0.7, 0.3], flip_y=0.01, random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}:1")
        
        # Compare weighted vs. unweighted KNN
        models = {
            'Unweighted KNN': KNeighborsClassifier(n_neighbors=5, random_state=self.random_state),
            'Weighted KNN': KNeighborsClassifier(n_neighbors=5, random_state=self.random_state, 
                                           weights='distance')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Detailed classification report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'test_accuracy': test_accuracy,
                'train_time': train_time,
                'classification_report': report
            }
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Training time: {train_time:.3f}s")
            
            # Show class-specific performance
            if name == 'Weighted KNN':
                print("  Weighted KNN Performance:")
                # Extract minority class performance
                minority_report = report['1']  # Assuming minority class is '1'
                print(f"    Minority Class (1):")
                print(f"      Precision: {minority_report['precision']:.4f}")
                print(f"      Recall:    {minority_report['recall']:.4f}")
                print(f"      F1-Score:  {minority_report['f1-score']:.4f}")
            
            print(f"\n  Unweighted KNN Performance:")
            majority_report = report['0']  # Assuming majority class is '0'
                print(f"    Majority Class (0):")
                print(f"      Precision: {majority_report['precision']:.4f}")
                print(f"      Recall:    {majority_report['recall']:.4f}")
                print(f"      F1-Score:  {majority_report['f1-score']:.4f}")
        
        # Visualize comparison
        self._plot_weighted_knn_comparison(results)
        
        return results
    
    def _plot_weighted_knn_comparison(self, results: Dict):
        """Plot weighted vs. unweighted KNN comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(results.keys())
        test_acc = [results[name]['test_accuracy'] for name in names]
        
        # Test accuracy comparison
        x = np.arange(len(names))
        width = 0.35
        
        bars = axes[0, 0].bar(x - width/2, test_acc, width, label='Test Accuracy', alpha=0.7)
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Weighted vs. Unweighted KNN')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision comparison for minority class
        minority_precision = [results['Weighted KNN']['classification_report']['1']['precision'] 
                        if '1' in results['Weighted KNN']['classification_report'] else 0]
        majority_precision = [results['Unweighted KNN']['classification_report']['0']['precision'] 
                        if '0' in results['Unweighted KNN']['classification_report'] else 0]
        
        x_pos = np.arange(len(names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, minority_precision, width, 
                         label='Weighted (Minority)', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, majority_precision, width, 
                         label='Unweighted (Minority)', alpha=0.7)
        axes[1, 0].set_ylabel('Minority Class Precision')
        axes[1, 0].set_title('Weighted vs. Unweighted: Minority Class Precision')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/weighted_knn_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def example_approximate_methods(self):
        """Example: Fast approximation methods for large-scale KNN."""
        print("\n" + "="*60)
        print("EXAMPLE: APPROXIMATE METHODS FOR LARGE-SCALE KNN")
        print("="*60)
        
        # Create large dataset
        X, y = make_classification(
            n_samples=10000, n_features=20, n_informative=10,
            random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01, random_state=self.random_state
        )
        
        # Standard KNN
        print("Training standard KNN...")
        start_time = time.time()
        knn_standard = KNeighborsClassifier(n_neighbors=5, random_state=self.random_state)
        knn_standard.fit(X_train, y_train)
        standard_time = time.time() - start_time
        y_pred_standard = knn_standard.predict(X_test)
        standard_accuracy = accuracy_score(y_test, y_pred_standard)
        
        print(f"Standard KNN: {standard_time:.3f}s, Accuracy: {standard_accuracy:.4f}")
        
        # Approximate KNN using KD-Tree
        try:
            from sklearn.neighbors import KDTree
            print("Training KD-Tree KNN...")
            start_time = time.time()
            knn_kdtree = KDTree(n_neighbors=5, leaf_size=40, random_state=self.random_state)
            knn_kdtree.fit(X_train, y_train)
            kdtree_time = time.time() - start_time
            y_pred_kdtree = knn_kdtree.predict(X_test)
            kdtree_accuracy = accuracy_score(y_test, y_pred_kdtree)
            
            print(f"KD-Tree KNN: {kdtree_time:.3f}s, Accuracy: {kdtree_accuracy:.4f}")
            
        except ImportError:
            print("KD-Tree not available. Skipping...")
            knn_kdtree = None
            kdtree_accuracy = 0
            kdtree_time = 0
        
        # Approximate KNN using Ball Tree
        try:
            from sklearn.neighbors import BallTree
            print("Training Ball Tree KNN...")
            start_time = time.time()
            knn_balltree = BallTree(n_neighbors=5, random_state=self.random_state)
            knn_balltree.fit(X_train, y_train)
            balltree_time = time.time() - start_time
            y_pred_balltree = knn_balltree.predict(X_test)
            balltree_accuracy = accuracy_score(y_test, y_pred_balltree)
            
            print(f"Ball Tree KNN: {balltree_time:.3f}s, Accuracy: {balltree_accuracy:.4f}")
            
        except ImportError:
            print("Ball Tree not available. Skipping...")
            knn_balltree = None
            balltree_accuracy = 0
            balltree_time = 0
        
        # Compare results
        print(f"\nComparison Results:")
        print(f"Standard KNN:    {standard_accuracy:.4f}")
        if kdtree_accuracy > 0:
            print(f"KD-Tree KNN:     {kdtree_accuracy:.4f} ({kdtree_accuracy-standard_accuracy:+.4f})")
        if balltree_accuracy > 0:
            print(f"Ball Tree KNN:   {balltree_accuracy:.4f} ({balltree_accuracy-standard_accuracy:+.4f})")
        
        return {
            'standard_accuracy': standard_accuracy,
            'kdtree_accuracy': kdtree_accuracy,
            'balltree_accuracy': balltree_accuracy,
            'standard_time': standard_time,
            'kdtree_time': kdtree_time,
            'balltree_time': balltree_time
        }
    
    def example_cross_validation_k_selection(self):
        """Example: Comprehensive cross-validation for K selection."""
        print("\n" + "="*60)
        print("EXAMPLE: CROSS-VALIDATION FOR OPTIMAL K SELECTION")
        print("="*60)
        
        # Create dataset
        X, y = make_classification(
            n_samples=2000, n_features=12, n_informative=8,
            random_state=self.random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Pipeline with scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(random_state=self.random_state))
        ])
        
        # Test range of K values
        k_range = range(1, 31)
        
        print("Running cross-validation for K selection...")
        
        # Custom cross-validation to track both accuracy and computational cost
        from sklearn.model_selection import cross_validate
        scoring = ['accuracy', 'neg_log_loss']  # Track both accuracy and computational cost
        
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=5, scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Analyze results
        test_scores = cv_results['test_score']
        fit_times = cv_results['fit_time']
        
        # Find optimal K
        optimal_idx = np.argmax(test_scores)
        optimal_k = k_range[optimal_idx]
        optimal_score = test_scores[optimal_idx]
        optimal_time = fit_times[optimal_idx]
        
        print(f"\nOptimal K: {optimal_k}")
        print(f"Optimal CV Accuracy: {optimal_score:.4f}")
        print(f"Optimal Training Time: {optimal_time:.3f}s")
        
        # Visualize results
        self._plot_cv_k_selection(k_range, test_scores, fit_times, 
                                optimal_k, optimal_score, optimal_time)
        
        return {
            'optimal_k': optimal_k,
            'optimal_score': optimal_score,
            'cv_results': cv_results
        }
    
    def _plot_cv_k_selection(self, k_range, test_scores, fit_times, 
                        optimal_k: int, optimal_score: float, optimal_time: float):
        """Plot cross-validation results for K selection."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Accuracy vs. K
        axes[0, 0].plot(k_range, test_scores, 'bo-', marker='o', label='CV Accuracy')
        axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--',
                           label=f'Optimal K={optimal_k}')
        axes[0, 0].set_xlabel('K (Number of Neighbors)')
        axes[0, 0].set_ylabel('CV Accuracy')
        axes[0, 0].set_title('Cross-Validation: K vs. Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training time vs. K
        axes[0, 1].plot(k_range, fit_times, 'rs-', marker='s', label='Training Time')
        axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--',
                           label=f'Optimal K={optimal_k}')
        axes[0, 1].set_xlabel('K (Number of Neighbors)')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Cross-Validation: Computational Cost vs. K')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/cv_k_selection.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def example_real_world_considerations(self):
        """Example: Real-world considerations for KNN deployment."""
        print("\n" + "="*60)
        print("REAL-WORLD KNN CONSIDERATIONS")
        print("="*60)
        
        considerations = """
MEMORY AND COMPUTATIONAL REQUIREMENTS:
• KNN stores entire training dataset in memory
• Prediction time grows linearly: O(n_samples * n_features * K)
• For 1M samples with K=100: ~100M distance calculations per prediction
• Consider using approximate methods (KD-Tree, Ball Tree) for large datasets

PREDICTION LATENCY REQUIREMENTS:
• Real-time applications: KNN may be too slow
• Batch predictions: KNN is better for batch processing
• Hardware acceleration: GPU libraries for distance calculations
• Approximate nearest neighbors: FAISS, Annoy for large-scale applications

DATA DISTRIBUTION CONSIDERATIONS:
• Stationary vs. streaming data: KNN needs static training set
• Feature drift: KNN may need periodic retraining
• Concept drift: Distance-based similarity may become meaningless
• Privacy: KNN stores all training data (GDPR/CCPA concerns)

ALGORITHM ALTERNATIVES:
• For high-speed requirements: Approximate methods, locality-sensitive hashing
• For high-dimensional data: Dimensionality reduction before KNN
• For interpretability requirements: Decision trees or rule-based systems
• For accuracy-critical: Ensemble methods or deep learning

WHEN KNN IS APPROPRIATE:
• Small datasets (< 1K samples)
• Low-dimensional spaces (< 20 features)
• When interpretability is required
• When prediction latency is acceptable
• For baseline or prototype models

WHEN KNN IS INAPPROPRIATE:
• Large datasets (> 100K samples)
• High-dimensional spaces (> 50 features)
• Real-time prediction requirements
• When computational efficiency is critical
• When more sophisticated algorithms are available

IMPLEMENTATION RECOMMENDATIONS:
• Always use feature scaling with proper pipeline
• Use cross-validation to select optimal K
• Consider weighted voting for imbalanced data
• Use approximate methods for large-scale applications
• Monitor training and prediction performance
• Consider hybrid approaches (KNN + feature selection)
• Implement early stopping for classification confidence
        """
        
        print(considerations)
        
        return considerations


def run_all_examples():
    """Run all KNN examples."""
    print("""
🎯 K-NEAREST NEIGHBORS EXAMPLES
====================================

Additional practical examples for KNN models.
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    examples = KNNExamples(random_state=42)
    
    # Run all examples
    example_funcs = [
        ("Weighted KNN", examples.example_weighted_knn),
        ("Approximate Methods", examples.example_approximate_methods),
        ("CV K Selection", examples.example_cross_validation_k_selection),
        ("Real-World Considerations", examples.example_real_world_considerations)
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
