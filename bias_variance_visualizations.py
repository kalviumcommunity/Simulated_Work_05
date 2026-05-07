"""
Bias-Variance Trade-Off Visualizations

Comprehensive visualization tools for understanding and communicating
bias-variance concepts. This module provides publication-quality plots
that clearly illustrate bias-variance trade-offs, learning curves,
and model behavior patterns.

Key Visualizations:
- Bias-variance trade-off curves
- Learning curve patterns
- Cross-validation stability analysis
- Algorithm comparison charts
- Model complexity optimization
- Interactive diagnostic plots
- Real-world scenario illustrations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import logging
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ML imports for generating example data
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set up professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


class BiasVarianceVisualizer:
    """
    Comprehensive visualization system for bias-variance analysis.
    """
    
    def __init__(self, random_state: int = 42, style: str = 'professional'):
        """
        Initialize the visualizer.
        
        Args:
            random_state: Random state for reproducibility
            style: Plotting style ('professional', 'academic', 'presentation')
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Set up plotting style
        self._setup_plotting_style(style)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        logger.info("Bias-Variance Visualizer initialized")
    
    def _setup_plotting_style(self, style: str):
        """Set up plotting style based on preference."""
        if style == 'professional':
            # Clean, professional style
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = '#333333'
            plt.rcParams['axes.linewidth'] = 1.0
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['grid.linewidth'] = 0.5
            plt.rcParams['text.color'] = '#333333'
            
        elif style == 'academic':
            # Academic publication style
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['grid.alpha'] = 0.2
            plt.rcParams['text.color'] = 'black'
            
        elif style == 'presentation':
            # Presentation style with larger fonts
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = '#333333'
            plt.rcParams['axes.linewidth'] = 1.5
            plt.rcParams['grid.alpha'] = 0.4
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
    
    def plot_bias_variance_tradeoff_curve(self, save_path: str = None):
        """
        Create the classic bias-variance trade-off curve.
        
        This is the fundamental visualization showing how bias and variance
        trade off against each other as model complexity increases.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate model complexity values
        complexity = np.linspace(1, 10, 100)
        
        # Simulate bias, variance, and total error curves
        # Bias decreases with complexity (exponential decay)
        bias = 10 * np.exp(-0.5 * (complexity - 1))
        
        # Variance increases with complexity (polynomial growth)
        variance = 0.5 * (complexity - 1)**1.5
        
        # Total error is sum of bias² + variance + noise
        noise = 2.0  # Irreducible error
        total_error = bias**2 + variance + noise
        
        # Plot curves
        ax.plot(complexity, bias**2, 'b-', linewidth=3, label='Bias²', alpha=0.8)
        ax.plot(complexity, variance, 'r-', linewidth=3, label='Variance', alpha=0.8)
        ax.plot(complexity, total_error, 'k-', linewidth=3, label='Total Error', alpha=0.9)
        
        # Add noise level
        ax.axhline(y=noise, color='gray', linestyle='--', linewidth=2, 
                  label='Irreducible Noise', alpha=0.7)
        
        # Find optimal complexity
        optimal_idx = np.argmin(total_error)
        optimal_complexity = complexity[optimal_idx]
        optimal_error = total_error[optimal_idx]
        
        # Mark optimal point
        ax.plot(optimal_complexity, optimal_error, 'go', markersize=12, 
               label=f'Optimal Complexity ({optimal_complexity:.1f})', 
               markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Add shaded regions
        ax.fill_between(complexity[:optimal_idx+1], 0, total_error[:optimal_idx+1], 
                       alpha=0.1, color='blue', label='High Bias Region')
        ax.fill_between(complexity[optimal_idx:], 0, total_error[optimal_idx:], 
                       alpha=0.1, color='red', label='High Variance Region')
        
        # Styling
        ax.set_xlabel('Model Complexity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error', fontsize=12, fontweight='bold')
        ax.set_title('The Bias-Variance Trade-Off', fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits
        ax.set_xlim(1, 10)
        ax.set_ylim(0, max(total_error) * 1.1)
        
        # Legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add annotations
        ax.annotate('High Bias\n(Underfitting)', xy=(2, total_error[10]), 
                   xytext=(2.5, total_error[10] * 1.5),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                   fontsize=10, ha='center', color='blue', fontweight='bold')
        
        ax.annotate('High Variance\n(Overfitting)', xy=(8, total_error[80]), 
                   xytext=(7.5, total_error[80] * 1.5),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=10, ha='center', color='red', fontweight='bold')
        
        ax.annotate('Optimal Balance', xy=(optimal_complexity, optimal_error), 
                   xytext=(optimal_complexity + 1, optimal_error * 0.7),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                   fontsize=10, ha='center', color='green', fontweight='bold')
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/bias_variance_tradeoff_curve.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return {
            'optimal_complexity': optimal_complexity,
            'optimal_error': optimal_error,
            'bias_at_optimal': bias[optimal_idx]**2,
            'variance_at_optimal': variance[optimal_idx]
        }
    
    def plot_learning_curve_patterns(self, save_path: str = None):
        """
        Create comprehensive learning curve patterns showing different
        bias-variance scenarios.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Generate sample learning curve data
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Scenario 1: High Bias (Underfitting)
        ax1 = fig.add_subplot(gs[0, 0])
        train_score_high_bias = 0.6 + 0.05 * np.sin(train_sizes * np.pi)
        test_score_high_bias = 0.55 + 0.03 * np.sin(train_sizes * np.pi)
        
        ax1.plot(train_sizes, train_score_high_bias, 'b-', linewidth=2, label='Training Score')
        ax1.fill_between(train_sizes, 
                        train_score_high_bias - 0.02,
                        train_score_high_bias + 0.02,
                        alpha=0.2, color='blue')
        
        ax1.plot(train_sizes, test_score_high_bias, 'r-', linewidth=2, label='Validation Score')
        ax1.fill_between(train_sizes,
                        test_score_high_bias - 0.02,
                        test_score_high_bias + 0.02,
                        alpha=0.2, color='red')
        
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('Score')
        ax1.set_title('High Bias (Underfitting)', fontweight='bold', color='darkred')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.4, 0.8)
        
        # Add annotation
        ax1.annotate('Both scores low\nSmall gap', xy=(0.8, 0.6), 
                    xytext=(0.6, 0.45),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, ha='center')
        
        # Scenario 2: High Variance (Overfitting)
        ax2 = fig.add_subplot(gs[0, 1])
        train_score_high_var = 0.95 - 0.05 * np.exp(-train_sizes * 3)
        test_score_high_var = 0.65 + 0.15 * (1 - np.exp(-train_sizes * 2))
        
        ax2.plot(train_sizes, train_score_high_var, 'b-', linewidth=2, label='Training Score')
        ax2.fill_between(train_sizes,
                        train_score_high_var - 0.03,
                        train_score_high_var + 0.03,
                        alpha=0.2, color='blue')
        
        ax2.plot(train_sizes, test_score_high_var, 'r-', linewidth=2, label='Validation Score')
        ax2.fill_between(train_sizes,
                        test_score_high_var - 0.04,
                        test_score_high_var + 0.04,
                        alpha=0.2, color='red')
        
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('Score')
        ax2.set_title('High Variance (Overfitting)', fontweight='bold', color='darkred')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.0)
        
        # Add annotation
        ax2.annotate('Large gap\nHigh train, low test', xy=(0.3, 0.8), 
                    xytext=(0.7, 0.55),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, ha='center')
        
        # Scenario 3: Good Fit
        ax3 = fig.add_subplot(gs[0, 2])
        train_score_good = 0.85 + 0.05 * (1 - np.exp(-train_sizes * 3))
        test_score_good = 0.82 + 0.08 * (1 - np.exp(-train_sizes * 2))
        
        ax3.plot(train_sizes, train_score_good, 'b-', linewidth=2, label='Training Score')
        ax3.fill_between(train_sizes,
                        train_score_good - 0.02,
                        train_score_good + 0.02,
                        alpha=0.2, color='blue')
        
        ax3.plot(train_sizes, test_score_good, 'r-', linewidth=2, label='Validation Score')
        ax3.fill_between(train_sizes,
                        test_score_good - 0.02,
                        test_score_good + 0.02,
                        alpha=0.2, color='red')
        
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('Score')
        ax3.set_title('Good Fit (Balanced)', fontweight='bold', color='darkgreen')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.7, 1.0)
        
        # Add annotation
        ax3.annotate('Both scores high\nSmall gap', xy=(0.8, 0.88), 
                    xytext=(0.5, 0.75),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, ha='center')
        
        # Scenario 4: Data Limited
        ax4 = fig.add_subplot(gs[1, 0])
        train_score_limited = 0.9 - 0.1 * np.exp(-train_sizes * 2)
        test_score_limited = 0.6 + 0.2 * (1 - np.exp(-train_sizes * 4))
        
        ax4.plot(train_sizes, train_score_limited, 'b-', linewidth=2, label='Training Score')
        ax4.fill_between(train_sizes,
                        train_score_limited - 0.03,
                        train_score_limited + 0.03,
                        alpha=0.2, color='blue')
        
        ax4.plot(train_sizes, test_score_limited, 'r-', linewidth=2, label='Validation Score')
        ax4.fill_between(train_sizes,
                        test_score_limited - 0.04,
                        test_score_limited + 0.04,
                        alpha=0.2, color='red')
        
        ax4.set_xlabel('Training Set Size')
        ax4.set_ylabel('Score')
        ax4.set_title('Data Limited', fontweight='bold', color='darkorange')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.5, 1.0)
        
        # Add annotation
        ax4.annotate('Gap decreasing\nMore data would help', xy=(0.6, 0.75), 
                    xytext=(0.3, 0.55),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=9, ha='center')
        
        # Scenario 5: Model Limited
        ax5 = fig.add_subplot(gs[1, 1])
        train_score_model_limited = 0.65 + 0.05 * np.sin(train_sizes * np.pi)
        test_score_model_limited = 0.62 + 0.03 * np.sin(train_sizes * np.pi)
        
        ax5.plot(train_sizes, train_score_model_limited, 'b-', linewidth=2, label='Training Score')
        ax5.fill_between(train_sizes,
                        train_score_model_limited - 0.02,
                        train_score_model_limited + 0.02,
                        alpha=0.2, color='blue')
        
        ax5.plot(train_sizes, test_score_model_limited, 'r-', linewidth=2, label='Validation Score')
        ax5.fill_between(train_sizes,
                        test_score_model_limited - 0.02,
                        test_score_model_limited + 0.02,
                        alpha=0.2, color='red')
        
        ax5.set_xlabel('Training Set Size')
        ax5.set_ylabel('Score')
        ax5.set_title('Model Limited', fontweight='bold', color='purple')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0.5, 0.8)
        
        # Add annotation
        ax5.annotate('Converged at low score\nMore data won\'t help', xy=(0.8, 0.65), 
                    xytext=(0.4, 0.52),
                    arrowprops=dict(arrowstyle='->', color='purple'),
                    fontsize=9, ha='center')
        
        # Scenario 6: Ideal Learning
        ax6 = fig.add_subplot(gs[1, 2])
        train_score_ideal = 0.9 + 0.05 * (1 - np.exp(-train_sizes * 5))
        test_score_ideal = 0.88 + 0.07 * (1 - np.exp(-train_sizes * 4))
        
        ax6.plot(train_sizes, train_score_ideal, 'b-', linewidth=2, label='Training Score')
        ax6.fill_between(train_sizes,
                        train_score_ideal - 0.01,
                        train_score_ideal + 0.01,
                        alpha=0.2, color='blue')
        
        ax6.plot(train_sizes, test_score_ideal, 'r-', linewidth=2, label='Validation Score')
        ax6.fill_between(train_sizes,
                        test_score_ideal - 0.01,
                        test_score_ideal + 0.01,
                        alpha=0.2, color='red')
        
        ax6.set_xlabel('Training Set Size')
        ax6.set_ylabel('Score')
        ax6.set_title('Ideal Learning', fontweight='bold', color='darkgreen')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0.8, 1.0)
        
        # Add annotation
        ax6.annotate('Excellent convergence\nMinimal gap', xy=(0.8, 0.93), 
                    xytext=(0.5, 0.82),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, ha='center')
        
        plt.suptitle('Learning Curve Patterns: Bias-Variance Diagnosis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/learning_curve_patterns.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
    
    def plot_model_complexity_spectrum(self, save_path: str = None):
        """
        Visualize how different models perform across the complexity spectrum.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Generate complexity levels and corresponding performance
        complexity_levels = ['Very Simple', 'Simple', 'Moderate', 'Complex', 'Very Complex']
        complexity_numeric = np.arange(1, 6)
        
        # Simulate performance for different model types
        models = {
            'Linear Regression': [0.6, 0.65, 0.68, 0.70, 0.71],
            'Ridge Regression': [0.62, 0.67, 0.70, 0.72, 0.73],
            'Decision Tree': [0.55, 0.70, 0.78, 0.82, 0.80],
            'Random Forest': [0.58, 0.72, 0.82, 0.85, 0.83],
            'Neural Network': [0.60, 0.75, 0.85, 0.88, 0.86],
            'Gradient Boosting': [0.61, 0.76, 0.84, 0.87, 0.85]
        }
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot 1: Performance vs Complexity
        ax1 = axes[0, 0]
        for i, (model_name, scores) in enumerate(models.items()):
            ax1.plot(complexity_numeric, scores, 'o-', linewidth=2, 
                    label=model_name, color=colors[i], markersize=6)
        
        ax1.set_xlabel('Model Complexity')
        ax1.set_ylabel('Test Score')
        ax1.set_title('Performance vs. Model Complexity', fontweight='bold')
        ax1.set_xticks(complexity_numeric)
        ax1.set_xticklabels(complexity_levels, rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 0.95)
        
        # Plot 2: Variance (stability) vs Complexity
        ax2 = axes[0, 1]
        variance_data = {
            'Linear Regression': [0.02, 0.02, 0.02, 0.02, 0.02],
            'Ridge Regression': [0.02, 0.02, 0.02, 0.02, 0.02],
            'Decision Tree': [0.01, 0.05, 0.12, 0.20, 0.25],
            'Random Forest': [0.02, 0.04, 0.06, 0.08, 0.10],
            'Neural Network': [0.03, 0.06, 0.10, 0.15, 0.20],
            'Gradient Boosting': [0.02, 0.05, 0.08, 0.12, 0.15]
        }
        
        for i, (model_name, variances) in enumerate(variance_data.items()):
            ax2.plot(complexity_numeric, variances, 's--', linewidth=2, 
                    label=model_name, color=colors[i], markersize=6)
        
        ax2.set_xlabel('Model Complexity')
        ax2.set_ylabel('Model Variance (Std Dev)')
        ax2.set_title('Stability vs. Model Complexity', fontweight='bold')
        ax2.set_xticks(complexity_numeric)
        ax2.set_xticklabels(complexity_levels, rotation=45, ha='right')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bias-Variance Balance Score
        ax3 = axes[0, 2]
        balance_scores = {}
        for model_name in models.keys():
            scores = models[model_name]
            variances = variance_data[model_name]
            # Balance score = performance - variance penalty
            balance = [s - 2*v for s, v in zip(scores, variances)]
            balance_scores[model_name] = balance
        
        for i, (model_name, balance) in enumerate(balance_scores.items()):
            ax3.plot(complexity_numeric, balance, '^-', linewidth=2, 
                    label=model_name, color=colors[i], markersize=6)
        
        ax3.set_xlabel('Model Complexity')
        ax3.set_ylabel('Balance Score (Higher is Better)')
        ax3.set_title('Bias-Variance Balance Score', fontweight='bold')
        ax3.set_xticks(complexity_numeric)
        ax3.set_xticklabels(complexity_levels, rotation=45, ha='right')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Optimal Complexity for Each Model
        ax4 = axes[1, 0]
        optimal_complexities = []
        optimal_scores = []
        model_names = []
        
        for model_name, scores in models.items():
            optimal_idx = np.argmax(scores)
            optimal_complexities.append(complexity_numeric[optimal_idx])
            optimal_scores.append(scores[optimal_idx])
            model_names.append(model_name.replace(' ', '\n'))
        
        bars = ax4.bar(model_names, optimal_scores, alpha=0.7, 
                      color=[colors[i] for i in range(len(model_names))])
        ax4.set_ylabel('Optimal Test Score')
        ax4.set_title('Optimal Performance per Model', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add optimal complexity as text
        for i, (bar, complexity) in enumerate(zip(bars, optimal_complexities)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'Complexity {int(complexity)}', ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Performance Distribution
        ax5 = axes[1, 1]
        all_scores = []
        model_labels = []
        for model_name, scores in models.items():
            all_scores.extend(scores)
            model_labels.extend([model_name] * len(scores))
        
        # Create box plot
        score_data = []
        score_labels = []
        for model_name, scores in models.items():
            score_data.append(scores)
            score_labels.append(model_name.replace(' ', '\n'))
        
        bp = ax5.boxplot(score_data, labels=score_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.set_ylabel('Test Score Distribution')
        ax5.set_title('Performance Distribution Across Complexity', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Recommendations Heatmap
        ax6 = axes[1, 2]
        
        # Create recommendation matrix
        data_scenarios = ['Small\nDataset', 'Medium\nDataset', 'Large\nDataset', 
                         'Low\nNoise', 'High\nNoise', 'High\nDimensional']
        model_names_short = ['Linear', 'Ridge', 'Tree', 'RF', 'NN', 'GB']
        
        # Simulate recommendation scores (higher = more recommended)
        recommendations = np.array([
            [0.9, 0.8, 0.6, 0.8, 0.5, 0.7],  # Small dataset
            [0.7, 0.8, 0.8, 0.9, 0.8, 0.9],  # Medium dataset
            [0.5, 0.6, 0.8, 0.9, 0.9, 0.9],  # Large dataset
            [0.8, 0.9, 0.8, 0.9, 0.8, 0.9],  # Low noise
            [0.9, 0.9, 0.6, 0.8, 0.7, 0.8],  # High noise
            [0.8, 0.9, 0.7, 0.8, 0.9, 0.8]   # High dimensional
        ])
        
        im = ax6.imshow(recommendations, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax6.set_xticks(np.arange(len(model_names_short)))
        ax6.set_yticks(np.arange(len(data_scenarios)))
        ax6.set_xticklabels(model_names_short)
        ax6.set_yticklabels(data_scenarios)
        
        # Add text annotations
        for i in range(len(data_scenarios)):
            for j in range(len(model_names_short)):
                text = ax6.text(j, i, f'{recommendations[i, j]:.1f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax6.set_title('Model Recommendations by Scenario', fontweight='bold')
        plt.colorbar(im, ax=ax6, label='Recommendation Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/model_complexity_spectrum.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
    
    def plot_cross_validation_stability(self, save_path: str = None):
        """
        Visualize cross-validation stability and variance diagnostics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Generate sample CV data
        models = ['Linear', 'Ridge', 'Decision Tree', 'Random Forest', 'Neural Net', 'Gradient Boost']
        cv_folds = 10
        
        # Simulate CV scores with different stability characteristics
        np.random.seed(42)
        cv_data = {
            'Linear': np.random.normal(0.75, 0.02, cv_folds),
            'Ridge': np.random.normal(0.77, 0.02, cv_folds),
            'Decision Tree': np.random.normal(0.78, 0.15, cv_folds),
            'Random Forest': np.random.normal(0.82, 0.05, cv_folds),
            'Neural Net': np.random.normal(0.84, 0.08, cv_folds),
            'Gradient Boost': np.random.normal(0.83, 0.06, cv_folds)
        }
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        # Plot 1: CV Score Distributions
        ax1 = axes[0, 0]
        for i, (model_name, scores) in enumerate(cv_data.items()):
            ax1.hist(scores, alpha=0.6, bins=15, label=model_name, color=colors[i])
        
        ax1.set_xlabel('CV Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('CV Score Distributions', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean vs. Standard Deviation
        ax2 = axes[0, 1]
        means = [np.mean(scores) for scores in cv_data.values()]
        stds = [np.std(scores) for scores in cv_data.values()]
        
        scatter = ax2.scatter(stds, means, s=100, alpha=0.7, c=colors)
        for i, model_name in enumerate(models):
            ax2.annotate(model_name, (stds[i], means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('CV Standard Deviation')
        ax2.set_ylabel('CV Mean Score')
        ax2.set_title('Performance vs. Stability', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add stability zones
        ax2.axhspan(0.8, 1.0, 0, 0.05, alpha=0.2, color='green', label='High Performance')
        ax2.axhspan(0.6, 0.8, 0, 0.05, alpha=0.2, color='yellow', label='Medium Performance')
        ax2.axhspan(0, 0.6, 0, 0.05, alpha=0.2, color='red', label='Low Performance')
        ax2.axvspan(0, 0.05, 0, 1, alpha=0.2, color='green', label='High Stability')
        ax2.axvspan(0.05, 0.1, 0, 1, alpha=0.2, color='yellow', label='Medium Stability')
        ax2.axvspan(0.1, 0.3, 0, 1, alpha=0.2, color='red', label='Low Stability')
        
        # Plot 3: CV Score Ranges
        ax3 = axes[0, 2]
        ranges = [np.max(scores) - np.min(scores) for scores in cv_data.values()]
        
        bars = ax3.bar(models, ranges, alpha=0.7, color=colors)
        ax3.set_ylabel('CV Score Range (Max-Min)')
        ax3.set_title('Model Stability Range', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add mean values as text
        for i, (bar, mean_score) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'Mean: {mean_score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Fold-wise Performance
        ax4 = axes[1, 0]
        fold_numbers = np.arange(1, cv_folds + 1)
        
        for i, (model_name, scores) in enumerate(cv_data.items()):
            ax4.plot(fold_numbers, scores, 'o-', linewidth=2, 
                    label=model_name, color=colors[i], markersize=4, alpha=0.7)
        
        ax4.set_xlabel('CV Fold')
        ax4.set_ylabel('Score')
        ax4.set_title('Fold-wise Performance', fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confidence Intervals
        ax5 = axes[1, 1]
        confidence_intervals = []
        for scores in cv_data.values():
            mean = np.mean(scores)
            std = np.std(scores)
            ci = 1.96 * std / np.sqrt(len(scores))  # 95% CI
            confidence_intervals.append((mean, ci))
        
        x_pos = np.arange(len(models))
        means_ci = [ci[0] for ci in confidence_intervals]
        cis = [ci[1] for ci in confidence_intervals]
        
        ax5.errorbar(x_pos, means_ci, yerr=cis, fmt='o', linewidth=2, 
                    markersize=8, capsize=5, color='black', alpha=0.7)
        ax5.set_xlabel('Model')
        ax5.set_ylabel('CV Mean ± 95% CI')
        ax5.set_title('Confidence Intervals', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(models, rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Stability Classification
        ax6 = axes[1, 2]
        stability_classes = []
        for std in stds:
            if std < 0.03:
                stability_classes.append('Very Stable')
            elif std < 0.07:
                stability_classes.append('Stable')
            elif std < 0.12:
                stability_classes.append('Moderate')
            else:
                stability_classes.append('Unstable')
        
        # Count stability classes
        class_counts = {}
        for cls in stability_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        wedges, texts, autotexts = ax6.pie(class_counts.values(), labels=class_counts.keys(),
                                          autopct='%1.0f%%', startangle=90, colors=['green', 'lightgreen', 'orange', 'red'])
        ax6.set_title('Model Stability Classification', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/cross_validation_stability.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return {
            'cv_data': cv_data,
            'stability_analysis': {
                'means': means,
                'stds': stds,
                'ranges': ranges,
                'stability_classes': stability_classes
            }
        }
    
    def plot_algorithm_comparison_matrix(self, save_path: str = None):
        """
        Create a comprehensive algorithm comparison matrix.
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # Algorithm characteristics
        algorithms = ['Linear\nRegression', 'Logistic\nRegression', 'KNN', 
                     'Decision\nTree', 'Random\nForest', 'SVM', 
                     'Neural\nNetwork', 'Gradient\nBoosting', 'Ensemble']
        
        # Characteristics to compare
        characteristics = ['Bias\n(Tendency)', 'Variance\n(Tendency)', 'Interpretability',
                         'Training\nSpeed', 'Prediction\nSpeed', 'Data\nRequirements',
                         'Hyperparameter\nTuning', 'Handling\nNon-linearity', 'Overall\nPerformance']
        
        # Create characteristic matrix (1-10 scale, higher is better)
        matrix = np.array([
            # Linear Regression
            [2, 2, 9, 10, 10, 3, 3, 2, 5],
            # Logistic Regression
            [2, 2, 8, 9, 10, 3, 4, 2, 6],
            # KNN
            [1, 9, 7, 10, 6, 4, 3, 6, 6],
            # Decision Tree
            [3, 9, 8, 7, 8, 4, 5, 8, 7],
            # Random Forest
            [4, 6, 5, 4, 6, 5, 6, 8, 9],
            # SVM
            [3, 4, 6, 5, 7, 4, 7, 7, 8],
            # Neural Network
            [2, 8, 2, 3, 7, 7, 8, 9, 9],
            # Gradient Boosting
            [3, 6, 4, 3, 6, 5, 8, 8, 9],
            # Ensemble
            [4, 5, 4, 3, 6, 6, 7, 8, 10]
        ])
        
        # Create heatmap
        im = axes[0, 0].imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=10)
        
        # Set ticks and labels
        axes[0, 0].set_xticks(np.arange(len(characteristics)))
        axes[0, 0].set_yticks(np.arange(len(algorithms)))
        axes[0, 0].set_xticklabels(characteristics, rotation=45, ha='right')
        axes[0, 0].set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(characteristics)):
                text = axes[0, 0].text(j, i, f'{matrix[i, j]}',
                                      ha="center", va="center", color="black", 
                                      fontweight='bold')
        
        axes[0, 0].set_title('Algorithm Characteristics Matrix', fontweight='bold', fontsize=12)
        
        # Plot 2: Bias-Variance Quadrant
        ax2 = axes[0, 1]
        bias_scores = [10 - matrix[i, 0] for i in range(len(algorithms) - 1)]  # Reverse bias (higher = lower bias)
        variance_scores = [10 - matrix[i, 1] for i in range(len(algorithms) - 1)]  # Reverse variance
        
        scatter = ax2.scatter(variance_scores, bias_scores, s=100, alpha=0.7, 
                            c=range(len(algorithms) - 1), cmap='tab10')
        
        for i, alg in enumerate(algorithms[:-1]):
            ax2.annotate(alg.replace('\n', ' '), (variance_scores[i], bias_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Variance (Higher = More Variance)')
        ax2.set_ylabel('Bias (Higher = Less Bias)')
        ax2.set_title('Bias-Variance Quadrant', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax2.text(2.5, 7.5, 'High Bias\nLow Variance', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax2.text(7.5, 7.5, 'Low Bias\nLow Variance', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax2.text(2.5, 2.5, 'High Bias\nHigh Variance', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        ax2.text(7.5, 2.5, 'Low Bias\nHigh Variance', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        
        # Plot 3: Performance vs. Interpretability
        ax3 = axes[0, 2]
        performance = [matrix[i, -1] for i in range(len(algorithms) - 1)]
        interpretability = [matrix[i, 2] for i in range(len(algorithms) - 1)]
        
        ax3.scatter(interpretability, performance, s=100, alpha=0.7, c=range(len(algorithms) - 1), cmap='tab10')
        
        for i, alg in enumerate(algorithms[:-1]):
            ax3.annotate(alg.replace('\n', ' '), (interpretability[i], performance[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Interpretability')
        ax3.set_ylabel('Overall Performance')
        ax3.set_title('Performance vs. Interpretability', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Speed vs. Performance
        ax4 = axes[1, 0]
        training_speed = [matrix[i, 3] for i in range(len(algorithms) - 1)]
        
        ax4.scatter(training_speed, performance, s=100, alpha=0.7, c=range(len(algorithms) - 1), cmap='tab10')
        
        for i, alg in enumerate(algorithms[:-1]):
            ax4.annotate(alg.replace('\n', ' '), (training_speed[i], performance[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Training Speed')
        ax4.set_ylabel('Overall Performance')
        ax4.set_title('Training Speed vs. Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Data Requirements vs. Performance
        ax5 = axes[1, 1]
        data_reqs = [matrix[i, 4] for i in range(len(algorithms) - 1)]
        
        ax5.scatter(data_reqs, performance, s=100, alpha=0.7, c=range(len(algorithms) - 1), cmap='tab10')
        
        for i, alg in enumerate(algorithms[:-1]):
            ax5.annotate(alg.replace('\n', ' '), (data_reqs[i], performance[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax5.set_xlabel('Data Requirements')
        ax5.set_ylabel('Overall Performance')
        ax5.set_title('Data Requirements vs. Performance', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Hyperparameter Tuning Difficulty
        ax6 = axes[1, 2]
        tuning_difficulty = [matrix[i, 6] for i in range(len(algorithms) - 1)]
        
        bars = ax6.bar(algorithms[:-1], tuning_difficulty, alpha=0.7, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(algorithms) - 1)))
        ax6.set_ylabel('Tuning Difficulty (Higher = Easier)')
        ax6.set_title('Hyperparameter Tuning Difficulty', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Non-linearity Handling
        ax7 = axes[2, 0]
        non_linear = [matrix[i, 7] for i in range(len(algorithms) - 1)]
        
        bars = ax7.bar(algorithms[:-1], non_linear, alpha=0.7, 
                      color=plt.cm.Pastel1(np.linspace(0, 1, len(algorithms) - 1)))
        ax7.set_ylabel('Non-linearity Handling')
        ax7.set_title('Ability to Handle Non-linear Patterns', fontweight='bold')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Radar Chart for Top 4 Algorithms
        ax8 = axes[2, 1]
        ax8.remove()
        ax8 = fig.add_subplot(3, 3, 8, projection='polar')
        
        # Select top 4 algorithms for radar chart
        top_algs = ['Linear\nRegression', 'Random\nForest', 'Neural\nNetwork', 'Gradient\nBoosting']
        alg_indices = [algorithms.index(alg) for alg in top_algs]
        
        # Characteristics for radar chart
        radar_chars = ['Performance', 'Interpretability', 'Speed', 'Stability', 'Flexibility']
        radar_values = []
        
        for idx in alg_indices:
            # Create synthetic radar values
            perf = matrix[idx, -1] / 10
            inter = matrix[idx, 2] / 10
            speed = (matrix[idx, 3] + matrix[idx, 4]) / 20  # Average of training and prediction speed
            stability = (10 - matrix[idx, 1]) / 10  # Reverse variance
            flex = matrix[idx, 7] / 10
            radar_values.append([perf, inter, speed, stability, flex])
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_chars), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors_radar = ['blue', 'green', 'red', 'orange']
        for i, (alg, values) in enumerate(zip(top_algs, radar_values)):
            values += values[:1]  # Complete the circle
            ax8.plot(angles, values, 'o-', linewidth=2, label=alg.replace('\n', ' '), 
                    color=colors_radar[i])
            ax8.fill(angles, values, alpha=0.1, color=colors_radar[i])
        
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(radar_chars)
        ax8.set_ylim(0, 1)
        ax8.set_title('Algorithm Radar Comparison', fontweight='bold', pad=20)
        ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Plot 9: Recommendations Summary
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        recommendations = """
ALGORITHM SELECTION GUIDE:

🎯 INTERPRETABILITY CRITICAL:
• Linear/Logistic Regression
• Shallow Decision Trees

⚡ SPEED PRIORITY:
• Linear/Logistic Regression
• KNN (small datasets)

🔄 HIGH PERFORMANCE NEEDED:
• Gradient Boosting
• Neural Networks
• Random Forest

📊 SMALL DATASET:
• Linear models
• Regularized models
• Simple trees

📈 LARGE DATASET:
• Neural Networks
• Gradient Boosting
• Random Forest

🌀 NON-LINEAR PATTERNS:
• Neural Networks
• Gradient Boosting
• Random Forest
• SVM (with kernels)

⚖️ BALANCED APPROACH:
• Random Forest
• Gradient Boosting
        """
        
        ax9.text(0.05, 0.95, recommendations, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/algorithm_comparison_matrix.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return {
            'characteristics_matrix': matrix,
            'algorithms': algorithms,
            'characteristics': characteristics
        }
    
    def plot_real_world_scenarios(self, save_path: str = None):
        """
        Create visualizations for real-world bias-variance scenarios.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scenario 1: Medical Diagnosis (High Stakes)
        ax1 = axes[0, 0]
        
        # Simulate medical diagnosis scenario
        models = ['Simple\nLinear', 'Logistic\nRegression', 'Random\nForest', 'Neural\nNetwork']
        sensitivity = [0.85, 0.88, 0.92, 0.94]
        specificity = [0.90, 0.91, 0.87, 0.85]
        interpretability = [9, 8, 4, 2]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, sensitivity, width, label='Sensitivity', alpha=0.7, color='green')
        ax1.bar(x, specificity, width, label='Specificity', alpha=0.7, color='blue')
        ax1.bar(x + width, [i/10 for i in interpretability], width, 
                label='Interpretability', alpha=0.7, color='orange')
        
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Medical Diagnosis: High Stakes Scenario', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add recommendation
        ax1.text(0.5, 0.95, 'Recommendation: Logistic Regression\n(Balance of performance and interpretability)', 
                transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Scenario 2: Financial Trading (Speed Critical)
        ax2 = axes[0, 1]
        
        trading_models = ['Linear\nModel', 'Decision\nTree', 'Random\nForest', 'Deep\nLearning']
        prediction_speed = [9.5, 8.0, 6.5, 3.0]
        accuracy = [0.65, 0.72, 0.78, 0.82]
        stability = [8.5, 6.0, 7.5, 5.0]
        
        x = np.arange(len(trading_models))
        
        ax2.scatter(prediction_speed, accuracy, s=[stability*20 for stability in stability], 
                   alpha=0.6, c=range(len(trading_models)), cmap='viridis')
        
        for i, model in enumerate(trading_models):
            ax2.annotate(model, (prediction_speed[i], accuracy[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Prediction Speed')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Financial Trading: Speed Critical Scenario', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add speed zones
        ax2.axvspan(8, 10, alpha=0.2, color='green', label='High Speed')
        ax2.axvspan(5, 8, alpha=0.2, color='yellow', label='Medium Speed')
        ax2.axvspan(0, 5, alpha=0.2, color='red', label='Low Speed')
        
        # Scenario 3: Image Recognition (Complex Patterns)
        ax3 = axes[1, 0]
        
        # Simulate image recognition performance
        complexity_levels = ['Simple\nCNN', 'ResNet-18', 'ResNet-50', 'ResNet-101', 'Vision\nTransformer']
        training_time = [2, 8, 24, 48, 72]  # hours
        accuracy = [0.85, 0.92, 0.94, 0.95, 0.96]
        data_requirements = [1000, 5000, 10000, 20000, 50000]  # images
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(complexity_levels, accuracy, 'bo-', linewidth=2, 
                       label='Accuracy', markersize=8)
        line2 = ax3_twin.plot(complexity_levels, training_time, 'rs--', linewidth=2, 
                              label='Training Time', markersize=8)
        
        ax3.set_xlabel('Model Complexity')
        ax3.set_ylabel('Accuracy', color='blue')
        ax3_twin.set_ylabel('Training Time (hours)', color='red')
        ax3.set_title('Image Recognition: Complex Patterns Scenario', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # Add data requirements as text
        for i, (model, data_req) in enumerate(zip(complexity_levels, data_requirements)):
            ax3.text(i, accuracy[i] + 0.01, f'{data_req/1000:.0f}K images', 
                    ha='center', fontsize=8, color='gray')
        
        # Scenario 4: Recommendation Systems (Large Scale)
        ax4 = axes[1, 1]
        
        rec_models = ['Collaborative\nFiltering', 'Matrix\nFactorization', 'Deep\nLearning', 'Gradient\nBoosting']
        scalability = [6, 8, 7, 9]
        personalization = [5, 6, 9, 7]
        cold_start = [3, 4, 6, 8]
        
        x = np.arange(len(rec_models))
        width = 0.25
        
        ax4.bar(x - width, scalability, width, label='Scalability', alpha=0.7, color='purple')
        ax4.bar(x, personalization, width, label='Personalization', alpha=0.7, color='green')
        ax4.bar(x + width, cold_start, width, label='Cold Start', alpha=0.7, color='orange')
        
        ax4.set_xlabel('Model Type')
        ax4.set_ylabel('Capability Score')
        ax4.set_title('Recommendation Systems: Large Scale Scenario', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(rec_models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/real_world_scenarios.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
    
    def create_comprehensive_dashboard(self, save_path: str = None):
        """
        Create a comprehensive bias-variance analysis dashboard.
        """
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Comprehensive Bias-Variance Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Main Trade-off Curve (top left, spanning 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        complexity = np.linspace(1, 10, 100)
        bias = 8 * np.exp(-0.4 * (complexity - 1))
        variance = 0.8 * (complexity - 1)**1.3
        noise = 1.5
        total_error = bias**2 + variance + noise
        
        ax1.plot(complexity, bias**2, 'b-', linewidth=3, label='Bias²', alpha=0.8)
        ax1.plot(complexity, variance, 'r-', linewidth=3, label='Variance', alpha=0.8)
        ax1.plot(complexity, total_error, 'k-', linewidth=3, label='Total Error', alpha=0.9)
        ax1.axhline(y=noise, color='gray', linestyle='--', linewidth=2, 
                   label='Irreducible Noise', alpha=0.7)
        
        optimal_idx = np.argmin(total_error)
        ax1.plot(complexity[optimal_idx], total_error[optimal_idx], 'go', markersize=12)
        
        ax1.set_xlabel('Model Complexity', fontweight='bold')
        ax1.set_ylabel('Error', fontweight='bold')
        ax1.set_title('Bias-Variance Trade-Off', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning Curve Patterns (top right, spanning 2x2)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Multiple learning curves
        scenarios = [
            ('High Bias', 0.6 + 0.05 * np.sin(train_sizes * np.pi), 
             0.55 + 0.03 * np.sin(train_sizes * np.pi), 'red'),
            ('High Variance', 0.95 - 0.05 * np.exp(-train_sizes * 3), 
             0.65 + 0.15 * (1 - np.exp(-train_sizes * 2)), 'orange'),
            ('Good Fit', 0.85 + 0.05 * (1 - np.exp(-train_sizes * 3)), 
             0.82 + 0.08 * (1 - np.exp(-train_sizes * 2)), 'green')
        ]
        
        for name, train_score, test_score, color in scenarios:
            ax2.plot(train_sizes, train_score, '--', linewidth=2, 
                    color=color, alpha=0.7, label=f'{name} - Train')
            ax2.plot(train_sizes, test_score, '-', linewidth=2, 
                    color=color, alpha=0.9, label=f'{name} - Test')
        
        ax2.set_xlabel('Training Set Size', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Learning Curve Patterns', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Algorithm Comparison (middle left)
        ax3 = fig.add_subplot(gs[2, 0])
        algorithms = ['Linear', 'Ridge', 'Tree', 'RF', 'NN', 'GB']
        performance = [0.72, 0.75, 0.78, 0.85, 0.87, 0.86]
        stability = [0.95, 0.94, 0.75, 0.88, 0.80, 0.85]
        
        ax3.scatter(stability, performance, s=100, alpha=0.7, c=range(len(algorithms)), cmap='viridis')
        for i, alg in enumerate(algorithms):
            ax3.annotate(alg, (stability[i], performance[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Stability', fontweight='bold')
        ax3.set_ylabel('Performance', fontweight='bold')
        ax3.set_title('Algorithm Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Dataset Size Impact (middle center)
        ax4 = fig.add_subplot(gs[2, 1])
        sizes = [100, 500, 1000, 5000, 10000]
        simple_model = [0.65, 0.68, 0.70, 0.71, 0.71]
        complex_model = [0.55, 0.70, 0.80, 0.88, 0.90]
        
        ax4.plot(sizes, simple_model, 'o-', linewidth=2, label='Simple Model', markersize=6)
        ax4.plot(sizes, complex_model, 's-', linewidth=2, label='Complex Model', markersize=6)
        
        ax4.set_xscale('log')
        ax4.set_xlabel('Dataset Size', fontweight='bold')
        ax4.set_ylabel('Performance', fontweight='bold')
        ax4.set_title('Dataset Size Impact', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Hyperparameter Impact (middle right)
        ax5 = fig.add_subplot(gs[2, 2])
        param_values = [0.001, 0.01, 0.1, 1, 10, 100]
        train_scores = [0.95, 0.92, 0.88, 0.82, 0.75, 0.70]
        test_scores = [0.70, 0.78, 0.83, 0.81, 0.74, 0.68]
        
        ax5.plot(param_values, train_scores, 'o-', linewidth=2, label='Training', markersize=6)
        ax5.plot(param_values, test_scores, 's-', linewidth=2, label='Test', markersize=6)
        
        ax5.set_xscale('log')
        ax5.set_xlabel('Regularization Parameter', fontweight='bold')
        ax5.set_ylabel('Score', fontweight='bold')
        ax5.set_title('Hyperparameter Impact', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Key Insights (middle right, spanning 1x1)
        ax6 = fig.add_subplot(gs[2, 3])
        ax6.axis('off')
        
        insights = """
KEY INSIGHTS:

🎯 Optimal Balance
• Sweet spot between bias and variance
• Depends on data size and complexity

📊 Diagnostic Tools
• Learning curves reveal patterns
• CV stability indicates variance

⚖️ Trade-off Management
• Increase complexity → ↓ bias, ↑ variance
• Regularization → ↑ bias, ↓ variance

🔄 Algorithm Selection
• Match capacity to data
• Consider interpretability needs

📈 Continuous Improvement
• Monitor train/test gap
• Validate with cross-validation
        """
        
        ax6.text(0.05, 0.95, insights, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 7. Practical Recommendations (bottom, spanning 4x1)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        recommendations = """
PRACTICAL RECOMMENDATIONS WORKFLOW:

┌─ STEP 1: DIAGNOSE ──────────────────────────────────────────────────────────────┐
│ • Start with simple model (high bias baseline)                                 │
│ • Generate learning curves                                                    │
│ • Check train/test gap and CV stability                                        │
│ • Identify: High bias vs. High variance vs. Good fit                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ STEP 2: STRATEGY SELECTION ───────────────────────────────────────────────────┐
│ HIGH BIAS:     • Increase model complexity                                    │
│                • Add polynomial/interaction features                            │
│                • Reduce regularization                                          │
│                • Use more flexible algorithms                                  │
│                                                                                │
│ HIGH VARIANCE: • Apply regularization                                          │
│                • Reduce model complexity                                        │
│                • Collect more training data                                    │
│                • Use ensemble methods                                           │
│                                                                                │
│ GOOD FIT:      • Monitor for data drift                                        │
│                • Consider deployment                                            │
│                • Document configuration                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ STEP 3: VALIDATION ───────────────────────────────────────────────────────────┐
│ • Cross-validation with multiple metrics                                      │
│ • Learning curve confirmation                                                  │
│ • Test on holdout dataset                                                      │
│ • Monitor variance across folds                                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ STEP 4: DEPLOYMENT CONSIDERATIONS ────────────────────────────────────────────┐
│ • Model stability in production                                               │
│ • Performance monitoring                                                       │
│ • Retraining schedule                                                         │
│ • Interpretability requirements                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
        """
        
        ax7.text(0.02, 0.5, recommendations, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("plots/comprehensive_dashboard.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
    
    def run_all_visualizations(self):
        """Generate all bias-variance visualizations."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE BIAS-VARIANCE VISUALIZATIONS")
        print("="*80)
        
        visualizations = [
            ("Bias-Variance Trade-Off Curve", self.plot_bias_variance_tradeoff_curve),
            ("Learning Curve Patterns", self.plot_learning_curve_patterns),
            ("Model Complexity Spectrum", self.plot_model_complexity_spectrum),
            ("Cross-Validation Stability", self.plot_cross_validation_stability),
            ("Algorithm Comparison Matrix", self.plot_algorithm_comparison_matrix),
            ("Real-World Scenarios", self.plot_real_world_scenarios),
            ("Comprehensive Dashboard", self.create_comprehensive_dashboard)
        ]
        
        results = {}
        
        for name, func in visualizations:
            print(f"\nGenerating {name}...")
            try:
                result = func()
                results[name] = result
                print(f"✓ {name} completed successfully")
            except Exception as e:
                print(f"✗ Error in {name}: {str(e)}")
                logger.error(f"Error in {name}: {str(e)}")
                results[name] = {"error": str(e)}
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("Generated plots in 'plots/' directory:")
        for name in visualizations:
            filename = name.lower().replace(" ", "_").replace("-", "_") + ".png"
            print(f"  • {filename}")
        
        return results


def main():
    """Main function to generate all visualizations."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BIAS-VARIANCE VISUALIZATION SYSTEM                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

This system generates comprehensive visualizations for understanding
and communicating bias-variance concepts in machine learning.

Generated Visualizations:
• Bias-Variance Trade-Off Curve - Fundamental concept visualization
• Learning Curve Patterns - Diagnostic pattern recognition
• Model Complexity Spectrum - Algorithm comparison across complexity
• Cross-Validation Stability - Model reliability analysis
• Algorithm Comparison Matrix - Comprehensive algorithm evaluation
• Real-World Scenarios - Practical application examples
• Comprehensive Dashboard - Complete analysis overview

All plots are saved in high-quality PNG format suitable for:
• Academic publications
• Technical presentations
• Educational materials
• Model documentation
• Stakeholder communications

Each visualization includes:
• Clear annotations and explanations
• Professional styling and color schemes
• Diagnostic insights and recommendations
• Real-world context and applications
    """)
    
    # Create visualizer and generate all plots
    visualizer = BiasVarianceVisualizer(random_state=42, style='professional')
    results = visualizer.run_all_visualizations()
    
    print("\n" + "="*80)
    print("VISUALIZATION SYSTEM COMPLETE!")
    print("="*80)
    print("All bias-variance visualizations have been generated")
    print("Check the 'plots/' directory for all generated files")
    print("Use these visualizations for education, analysis, and communication")
    
    return visualizer, results


if __name__ == "__main__":
    visualizer, results = main()
