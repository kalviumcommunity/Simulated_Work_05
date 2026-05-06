"""
Hyperparameter Optimization Module

This module provides comprehensive tools for efficient hyperparameter optimization,
including RandomizedSearchCV implementation, parameter distributions, and 
performance visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                          precision_score, recall_score)
from scipy.stats import (randint, uniform, loguniform, norm)
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


class ParameterDistribution:
    """
    Base class for parameter distributions with visualization.
    """
    
    def __init__(self, name, param_type="continuous"):
        self.name = name
        self.param_type = param_type
        self.samples = []
        self.pdf = None
    
    def sample(self, n_samples=1, random_state=None):
        """Sample from the distribution."""
        raise NotImplementedError("Subclasses must implement sample method")
    
    def plot_samples(self, ax=None, title=None):
        """Plot sampled values."""
        if len(self.samples) == 0:
            logger.warning("No samples to plot")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig, ax = ax, figsize=(8, 4)
        
        if self.param_type == "continuous":
            ax.hist(self.samples, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        else:
            unique, counts = np.unique(self.samples, return_counts=True)
            ax.bar(unique, counts, alpha=0.7, color='lightcoral')
        
        ax.set_title(f"{title} ({self.name})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        if ax is None:
            plt.show()
    
    def get_stats(self):
        """Get basic statistics of sampled values."""
        if len(self.samples) == 0:
            return {}
        
        samples_array = np.array(self.samples)
        return {
            'mean': np.mean(samples_array),
            'std': np.std(samples_array),
            'min': np.min(samples_array),
            'max': np.max(samples_array),
            'median': np.median(samples_array),
            'q25': np.percentile(samples_array, 25),
            'q75': np.percentile(samples_array, 75)
        }


class UniformDistribution(ParameterDistribution):
    """Uniform distribution for continuous parameters."""
    
    def __init__(self, low=0.0, high=1.0):
        super().__init__("uniform", "continuous")
        self.low = low
        self.high = high
        self.range = high - low
    
    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return rng.uniform(self.low, self.high, n_samples)
    
    def plot_samples(self, ax=None, title=None):
        """Plot uniform distribution samples."""
        samples = self.sample(n_samples=1000, random_state=42)
        self.samples = samples
        super().plot_samples(ax=ax, title=title)
    
    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            'range': f"[{self.low:.3f}, {self.high:.3f}]",
            'width': self.range
        })
        return stats


class LogUniformDistribution(ParameterDistribution):
    """Log-uniform distribution for scale-free parameters."""
    
    def __init__(self, low=1e-4, high=1e2):
        super().__init__("loguniform", "continuous")
        self.low = low
        self.high = high
    
    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), n_samples))
    
    def plot_samples(self, ax=None, title=None):
        """Plot log-uniform distribution samples."""
        samples = self.sample(n_samples=1000, random_state=42)
        self.samples = samples
        super().plot_samples(ax=ax, title=title)
        
        # Add log scale visualization
        if ax is not None:
            ax2 = ax.twinx()
            ax2.hist(np.log10(samples), bins=30, alpha=0.5, color='orange', edgecolor='black')
            ax2.set_ylabel('log10(Value)')
            ax2.tick_params(axis='y', labelcolor='orange')
    
    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            'log_range': f"[{self.low:.0e}, {self.high:.0e}]",
            'range_ratio': self.high / self.low
        })
        return stats


class IntegerDistribution(ParameterDistribution):
    """Integer distribution for discrete parameters."""
    
    def __init__(self, low=0, high=100):
        super().__init__("randint", "continuous")
        self.low = low
        self.high = high
    
    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return rng.randint(self.low, self.high + 1, n_samples)
    
    def plot_samples(self, ax=None, title=None):
        """Plot integer distribution samples."""
        samples = self.sample(n_samples=1000, random_state=42)
        self.samples = samples
        super().plot_samples(ax=ax, title=title)


class CategoricalDistribution(ParameterDistribution):
    """Categorical distribution for discrete parameters."""
    
    def __init__(self, choices):
        super().__init__("categorical", "continuous")
        self.choices = choices
    
    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return rng.choice(self.choices, n_samples)
    
    def plot_samples(self, ax=None, title=None):
        """Plot categorical distribution samples."""
        samples = self.sample(n_samples=1000, random_state=42)
        self.samples = samples
        super().plot_samples(ax=ax, title=title)
    
    def get_stats(self):
        stats = super().get_stats()
        unique, counts = np.unique(self.samples, return_counts=True)
        stats.update({
            'choices': self.choices,
            'probabilities': dict(zip(self.choices, counts / len(self.samples)))
        })
        return stats


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(self, best_params, best_score, cv_results=None, 
                 optimization_time=None, n_iterations=None):
        self.best_params = best_params
        self.best_score = best_score
        self.cv_results = cv_results or {}
        self.optimization_time = optimization_time
        self.n_iterations = n_iterations
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"Best Score: {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")
        
        if self.n_iterations:
            print(f"Iterations Used: {self.n_iterations}")
        
        if self.optimization_time:
            duration = self.optimization_time
            print(f"Optimization Time: {duration:.2f} seconds")
        
        print("\nBest Parameters Details:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        if self.cv_results:
            print(f"\nCV Statistics:")
            print(f"  Mean CV Score: {np.mean(list(self.cv_results.values())):.4f}")
            print(f"  Std CV Score: {np.std(list(self.cv_results.values())):.4f}")
            print(f"  Min CV Score: {np.min(list(self.cv_results.values())):.4f}")
            print(f"  Max CV Score: {np.max(list(self.cv_results.values())):.4f}")
        
        print("="*60)


class RandomizedSearchOptimizer:
    """
    Enhanced RandomizedSearchCV with additional analysis and visualization.
    """
    
    def __init__(self, estimator, param_distributions=None, n_iter=100, 
                 cv=5, scoring="f1", random_state=42, n_jobs=-1):
        self.estimator = estimator
        self.param_distributions = param_distributions or {}
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.result = None
    
    def optimize(self, X, y):
        """Run optimization with analysis."""
        print(f"Starting RandomizedSearchCV with {self.n_iter} iterations...")
        
        from sklearn.model_selection import RandomizedSearchCV
        
        search = RandomizedSearchCV(
            self.estimator,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            return_train_score=True
        )
        
        start_time = pd.Timestamp.now()
        search.fit(X, y)
        end_time = pd.Timestamp.now()
        
        # Store results
        self.result = OptimizationResult(
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_results=search.cv_results_,
            optimization_time=(end_time - start_time).total_seconds(),
            n_iterations=self.n_iter
        )
        
        return self.result
    
    def plot_optimization_history(self, save_path=None):
        """Plot optimization history."""
        if self.result is None:
            logger.warning("No optimization results to plot")
            return
        
        cv_results = self.result.cv_results
        if not cv_results:
            logger.warning("No CV results available")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Score vs iteration
        iterations = range(1, len(cv_results) + 1)
        scores = [self.result.best_score] + list(cv_results.values())
        
        axes[0, 0].plot(iterations, scores, 'o-', color='steelblue', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=self.result.best_score, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('CV Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter distributions
        if self.param_distributions:
            param_names = list(self.param_distributions.keys())
            n_params = len(param_names)
            
            for i, param_name in enumerate(param_names):
                if i < 4:  # Limit to 6 subplots
                    ax = axes[1, i]
                    
                    if param_name in self.result.best_params:
                        best_value = self.result.best_params[param_name]
                        ax.set_title(f"{param_name} Distribution (Best: {best_value})")
                    else:
                        ax.set_title(f"{param_name} Distribution")
                    
                    # Get distribution for this parameter
                    if hasattr(self, 'param_distributions'):
                        dist = self.param_distributions[param_name]
                        if hasattr(dist, 'plot_samples'):
                            dist.plot_samples(ax=ax)
                            ax.set_xlabel('Value')
                            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization plot saved to {save_path}")
        
        plt.show()
    
    def print_parameter_analysis(self):
        """Print analysis of parameter distributions."""
        if not self.param_distributions:
            return
        
        print("\n" + "="*60)
        print("PARAMETER DISTRIBUTION ANALYSIS")
        print("="*60)
        
        for param_name, dist in self.param_distributions.items():
            stats = dist.get_stats()
            print(f"\n{param_name} ({dist.name}):")
            print(f"  Range: {stats.get('range', 'N/A')}")
            print(f"  Mean: {stats.get('mean', 'N/A'):.4f}")
            print(f"  Std: {stats.get('std', 'N/A'):.4f}")
            
            # Parameter importance analysis
            if param_name in self.result.best_params:
                print(f"  → Selected: {self.result.best_params[param_name]}")
            else:
                print(f"  → Not in best configuration")


def create_parameter_distributions(model_type="random_forest"):
    """Create appropriate parameter distributions for different model types."""
    
    if model_type == "random_forest":
        return {
            "n_estimators": IntegerDistribution(50, 500),
            "max_depth": IntegerDistribution(3, 15),
            "min_samples_split": IntegerDistribution(2, 20),
            "min_samples_leaf": IntegerDistribution(1, 15),
            "max_features": CategoricalDistribution(["sqrt", "log2", None])
        }
    
    elif model_type == "logistic":
        return {
            "C": LogUniformDistribution(1e-4, 1e2),
            "penalty": CategoricalDistribution(["l1", "l2"])
        }
    
    elif model_type == "svm":
        return {
            "C": LogUniformDistribution(1e-3, 1e3),
            "gamma": LogUniformDistribution(1e-4, 1e0),
            "kernel": CategoricalDistribution(["rbf", "linear", "poly"])
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def demonstrate_randomized_search():
    """Demonstrate RandomizedSearchCV with analysis."""
    print("="*80)
    print("RANDOMIZEDSEARCHCV DEMONSTRATION")
    print("="*80)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1500, n_features=12, n_informative=8,
        weights=[0.7, 0.3], flip_y=0.01, random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Create optimizer
    rf = RandomForestClassifier(random_state=42)
    param_distributions = create_parameter_distributions("random_forest")
    
    optimizer = RandomizedSearchOptimizer(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    
    # Run optimization
    result = optimizer.optimize(X_train, y_train)
    
    # Print results
    result.print_summary()
    optimizer.print_parameter_analysis()
    
    # Plot optimization history
    optimizer.plot_optimization_history("plots/optimization_history.png")
    
    # Final evaluation
    print(f"\nFinal evaluation on test set...")
    y_pred = optimizer.result.best_params
    # Note: This is simplified - in practice, you'd refit the model
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
    return result


def demonstrate_distribution_comparison():
    """Demonstrate different parameter distributions."""
    print("\n" + "="*80)
    print("PARAMETER DISTRIBUTION COMPARISON")
    print("="*80)
    
    # Create distributions to compare
    uniform_dist = UniformDistribution(0, 10)
    loguniform_dist = LogUniformDistribution(1e-4, 1e2)
    integer_dist = IntegerDistribution(1, 100)
    
    # Sample and plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample from each distribution
    n_samples = 1000
    uniform_samples = uniform_dist.sample(n_samples=n_samples, random_state=42)
    loguniform_samples = loguniform_dist.sample(n_samples=n_samples, random_state=42)
    integer_samples = integer_dist.sample(n_samples=n_samples, random_state=42)
    
    # Plot distributions
    distributions = [
        ("Uniform", uniform_dist),
        ("Log-Uniform", loguniform_dist),
        ("Integer", integer_dist)
    ]
    
    colors = ['skyblue', 'orange', 'green']
    
    for i, (name, dist) in enumerate(distributions):
        ax = axes[i]
        dist.plot_samples(ax=ax, title=name)
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("Value")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/parameter_distributions_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\nDistribution Statistics (n={n_samples}):")
    for name, dist in distributions:
        stats = dist.get_stats()
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: {stats['range']}")
    
    return distributions


def demonstrate_efficiency_analysis():
    """Demonstrate efficiency scaling analysis."""
    print("\n" + "="*80)
    print("EFFICIENCY SCALING ANALYSIS")
    print("="*80)
    
    # Compare grid search vs random search efficiency
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.datasets import make_classification
    
    # Create data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        weights=[0.65, 0.35], flip_y=0.01, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Parameter grids
    param_grid = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [3, 5, 7, 10]
    }
    
    param_distributions = {
        "n_estimators": IntegerDistribution(50, 500),
        "max_depth": IntegerDistribution(3, 15),
        "min_samples_leaf": IntegerDistribution(1, 10)
    }
    
    # Grid search
    print("Running GridSearchCV...")
    import time
    start_time = time.time()
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    grid_time = time.time() - start_time
    
    # Randomized search
    print("Running RandomizedSearchCV...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions,
        n_iter=50,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    
    # Results
    print(f"\nResults Comparison:")
    print(f"Grid Search - Time: {grid_time:.2f}s, Best F1: {grid.best_score_:.4f}")
    print(f"Random Search - Time: {random_time:.2f}s, Best F1: {random_search.best_score_:.4f}")
    
    efficiency_gain = (random_time - grid_time) / grid_time * 100
    print(f"Efficiency: Random search achieved {random_search.best_score_:.4f} in {random_time:.2f}s")
    print(f"           vs Grid search {grid.best_score_:.4f} in {grid_time:.2f}s")
    print(f"Speedup: {efficiency_gain:.1f}% faster with same or better performance")
    
    return {
        'grid_time': grid_time,
        'random_time': random_time,
        'grid_score': grid.best_score_,
        'random_score': random_search.best_score_,
        'efficiency_gain': efficiency_gain
    }


def demonstrate_bayesian_optimization():
    """Demonstrate basic Bayesian optimization concepts."""
    print("\n" + "="*80)
    print("BAYesian OPTIMIZATION BASICS")
    print("="*80)
    
    print("""
🎯 BAYesian OPTIMIZATION CONCEPTS:
• Sequential parameter exploration based on probability
• Updates beliefs about parameter effectiveness
• Balances exploration vs exploitation
• Naturally handles continuous parameter spaces
• More sample-efficient than grid search

📊 BAYESIAN OPTIMIZATION METHODS:
• Grid Search with Bayesian Optimization
• Tree-structured Parzen Estimators
• Gaussian Process Optimization
• Sequential Model-Based Optimization
• Population-Based Methods (Genetic Algorithms)

🔧 WHEN TO USE BAYESIAN OPTIMIZATION:
• Expensive model evaluations (deep learning, complex ensembles)
• Continuous hyperparameter spaces
• When prior knowledge about parameter effectiveness exists
• When you need uncertainty quantification
• When exploration budget is very limited

⚠️  IMPLEMENTATION CONSIDERATIONS:
• Requires additional libraries (scipy-optimize, Optuna, Hyperopt)
• More complex implementation than RandomizedSearchCV
• Need careful prior distribution specification
• May require custom acquisition functions
• Computational overhead per iteration can be higher

💡 PRACTICAL RECOMMENDATION:
For most practitioners, RandomizedSearchCV provides:
✅ 80-90% of Bayesian optimization benefits
✅ Much simpler implementation
✅ Built into scikit-learn
✅ Parallel execution support
✅ Reproducible with random_state

Use Bayesian optimization when:
❌ Model training is very expensive
❌ You have strong prior knowledge
❌ You need full posterior distributions
❌ Exploration budget is large (>1000 evaluations)
    """)
    
    # Simple demonstration of Bayesian concept
    # This is a conceptual demo - full Bayesian optimization
    # would require additional libraries like scipy-optimize or Optuna
    
    print("Bayesian optimization requires specialized libraries like:")
    print("  • scipy.optimize")
    print("  • Optuna")
    print("  • Hyperopt")
    print("  • scikit-optimize")
    print("\nFor most use cases, RandomizedSearchCV is recommended.")


def print_optimization_strategies():
    """Print optimization strategy recommendations."""
    print("\n" + "="*80)
    print("OPTIMIZATION STRATEGIES GUIDE")
    print("="*80)
    
    print("""
🎯 OPTIMIZATION STRATEGY SELECTION:

🔍 EXPLORATION PHASE (Low Budget):
• RandomizedSearchCV with wide distributions
• Coarse grid search to identify promising regions
• Low n_iter (20-50) for broad coverage
• Goal: Find promising parameter combinations

🎯 REFINEMENT PHASE (Medium Budget):
• RandomizedSearchCV with narrow distributions
• Grid search around promising regions
• Medium n_iter (50-100) for focused search
• Goal: Precise optimization of best parameters

🚀 PRODUCTION PHASE (High Budget):
• RandomizedSearchCV with very narrow distributions
• Grid search with fine granularity
• High n_iter (100-200) for thorough optimization
• Goal: Extract maximum performance

📊 STRATEGY MATRIX:

Model Complexity	Recommended Approach	Iterations	Budget
Simple (Logistic, Linear SVM)	Grid search only	50-100	Low
Moderate (Random Forest, GBM)	Randomized + Grid	100-300	Medium
Complex (Deep Learning, XGBoost)	Randomized only	200-500	High
Very Complex (Ensembles)	Bayesian optimization	500+	Very High

⚡ PERFORMANCE INDICATORS:
• CV score plateau reached → Stop increasing iterations
• Large variance in CV scores → Reduce n_iter or narrow distributions
• Best parameters at distribution edges → Expand distribution range
• Consistent improvement across iterations → Current approach is working
• No improvement after 100 iterations → Strategy change needed

🔧 IMPLEMENTATION CHECKLIST:
✅ Use RandomizedSearchCV for most cases
✅ Set random_state for reproducibility
✅ Use appropriate distributions for parameter types
✅ Monitor convergence and early stopping
✅ Plot optimization history for analysis
✅ Compare multiple optimization strategies
✅ Validate final configuration on holdout set
✅ Document best parameters and reasoning
    """)


def main():
    """Main demonstration function."""
    print("""
🎯 HYPERPARAMETER OPTIMIZATION TUTORIAL
============================================

This tutorial covers efficient hyperparameter optimization techniques
that scale beyond traditional grid search.

📚 TOPICS COVERED:
1. RandomizedSearchCV Fundamentals
2. Parameter Distributions (Continuous, Discrete, Categorical)
3. Efficiency Analysis and Scaling
4. Bayesian Optimization Concepts
5. Optimization Strategies and Selection Guide
6. Practical Implementation Checklist

⏱️  EXPECTED DURATION: 60 minutes
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Run demonstrations
    print("\n🔄 Running demonstrations...")
    
    # 1. RandomizedSearchCV demonstration
    result1 = demonstrate_randomized_search()
    
    # 2. Distribution comparison
    result2 = demonstrate_distribution_comparison()
    
    # 3. Efficiency analysis
    result3 = demonstrate_efficiency_analysis()
    
    # 4. Bayesian optimization basics
    result4 = demonstrate_bayesian_optimization()
    
    # 5. Strategies guide
    print_optimization_strategies()
    
    print("\n✅ Tutorial completed!")
    print(f"\n📁 Files created: plots/")
    print("\n🎯 Key takeaways:")
    print("  • RandomizedSearchCV scales linearly with iterations")
    print("  • Choose parameter distributions based on parameter type")
    print("  • Use hybrid strategies for best results")
    print("  • Always set random_state for reproducibility")
    print("  • Monitor convergence and stop early if needed")
    print("  • Bayesian optimization for complex, expensive models")
    
    return True


if __name__ == "__main__":
    main()
