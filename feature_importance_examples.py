"""
Feature Importance Examples and Utilities

Additional practical examples and utility functions for feature importance
interpretation in tree-based models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

# ML imports
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelEncoder
import time

warnings.filterwarnings('ignore')


class FeatureImportanceExamples:
    """Collection of advanced feature importance examples and utilities."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def example_importance_stability(self):
        """Example: Testing importance stability across different random seeds."""
        print("\n" + "="*60)
        print("EXAMPLE: IMPORTANCE STABILITY ANALYSIS")
        print("="*60)
        
        # Create stable dataset
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=6,
            random_state=42
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        
        # Test stability across different random seeds
        seeds = [42, 123, 456, 789, 999]
        importance_results = []
        
        for seed in seeds:
            # Train model with different seed
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=seed, 
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Store importance
            importance_dict = {'seed': seed}
            for i, feature in enumerate(feature_names):
                importance_dict[feature] = rf.feature_importances_[i]
            
            importance_results.append(importance_dict)
        
        # Convert to DataFrame
        stability_df = pd.DataFrame(importance_results)
        
        # Calculate stability metrics
        feature_means = stability_df[feature_names].mean()
        feature_stds = stability_df[feature_names].std()
        feature_cv = feature_stds / feature_means
        
        stability_summary = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Importance': feature_means,
            'Std_Importance': feature_stds,
            'CV_Ratio': feature_cv
        }).sort_values('Mean_Importance', ascending=False)
        
        print("Feature Importance Stability Across Different Random Seeds:")
        print(stability_summary.round(4).to_string(index=False))
        
        # Visualize stability
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Importance distribution for top features
        top_features = stability_summary.head(5)['Feature'].tolist()
        for feature in top_features:
            axes[0, 0].hist(stability_df[feature], alpha=0.6, bins=10, label=feature)
        axes[0, 0].set_xlabel('Importance Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Importance Distribution (Top 5 Features)')
        axes[0, 0].legend()
        
        # Plot 2: Mean vs. CV ratio
        axes[0, 1].scatter(stability_summary['Mean_Importance'], 
                           stability_summary['CV_Ratio'], alpha=0.6)
        axes[0, 1].set_xlabel('Mean Importance')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_title('Importance Stability (Lower CV = More Stable)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of importance across seeds
        heatmap_data = stability_df[feature_names].T
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", 
                   cmap="YlOrRd", ax=axes[1, 0])
        axes[1, 0].set_title('Importance Heatmap Across Seeds')
        axes[1, 0].set_xlabel('Random Seed')
        axes[1, 0].set_ylabel('Feature')
        
        # Plot 4: Ranking stability
        rankings = stability_df[feature_names].rank(axis=1, ascending=False)
        ranking_stds = rankings.std()
        
        axes[1, 1].bar(range(len(feature_names)), ranking_stds)
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Ranking Std Dev')
        axes[1, 1].set_title('Ranking Stability (Lower = More Stable)')
        axes[1, 1].set_xticks(range(len(feature_names)))
        axes[1, 1].set_xticklabels([f'F{i}' for i in range(len(feature_names))], 
                                   rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/importance_stability.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return stability_summary
    
    def example_feature_selection_with_importance(self):
        """Example: Using importance for feature selection."""
        print("\n" + "="*60)
        print("EXAMPLE: FEATURE SELECTION USING IMPORTANCE")
        print("="*60)
        
        # Load real dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        
        print(f"Dataset: {X.shape} (Breast Cancer)")
        
        # Baseline model with all features
        rf_full = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_scores_full = cross_val_score(
            rf_full, X, y, 
            cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        print(f"Baseline F1 (all {X.shape[1]} features): {cv_scores_full.mean():.4f} ± {cv_scores_full.std():.4f}")
        
        # Get feature importance
        rf_full.fit(X, y)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_full.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Test different feature selection thresholds
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        results = []
        
        for threshold in thresholds:
            # Select features
            selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
            X_selected = X[selected_features]
            
            if len(selected_features) == 0:
                continue
            
            # Evaluate with selected features
            rf_selected = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            cv_scores = cross_val_score(
                rf_selected, X_selected, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='f1'
            )
            
            results.append({
                'Threshold': threshold,
                'Num_Features': len(selected_features),
                'F1_Mean': cv_scores.mean(),
                'F1_Std': cv_scores.std(),
                'Feature_Reduction': (X.shape[1] - len(selected_features)) / X.shape[1] * 100
            })
        
        # Convert to DataFrame
        selection_df = pd.DataFrame(results)
        
        print("\nFeature Selection Results:")
        print(selection_df.round(4).to_string(index=False))
        
        # Find optimal threshold
        best_idx = selection_df['F1_Mean'].idxmax()
        best_threshold = selection_df.loc[best_idx, 'Threshold']
        best_num_features = selection_df.loc[best_idx, 'Num_Features']
        best_f1 = selection_df.loc[best_idx, 'F1_Mean']
        
        print(f"\nOptimal threshold: {best_threshold}")
        print(f"Optimal number of features: {best_num_features}")
        print(f"Best F1 score: {best_f1:.4f}")
        print(f"Feature reduction: {selection_df.loc[best_idx, 'Feature_Reduction']:.1f}%")
        
        # Visualize feature selection curve
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(selection_df['Num_Features'], selection_df['F1_Mean'], 'bo-')
        plt.axhline(y=cv_scores_full.mean(), color='r', linestyle='--', 
                   label=f'All features ({cv_scores_full.mean():.4f})')
        plt.xlabel('Number of Features')
        plt.ylabel('F1 Score')
        plt.title('Feature Selection Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(selection_df['Threshold'], selection_df['F1_Mean'], 'go-')
        plt.xlabel('Importance Threshold')
        plt.ylabel('F1 Score')
        plt.title('Importance Threshold vs. Performance')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.bar(range(len(importance_df)), importance_df['Importance'])
        plt.axhline(y=best_threshold, color='r', linestyle='--', 
                   label=f'Selected threshold ({best_threshold})')
        plt.xlabel('Features (sorted by importance)')
        plt.ylabel('Importance')
        plt.title('Feature Importance Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        # Show top selected features
        selected_features = importance_df[importance_df['Importance'] >= best_threshold].head(10)
        plt.barh(range(len(selected_features)), selected_features['Importance'])
        plt.yticks(range(len(selected_features)), selected_features['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {len(selected_features)} Selected Features')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig("plots/feature_selection.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return selection_df, importance_df
    
    def example_partial_dependence_with_importance(self):
        """Example: Combining importance with partial dependence analysis."""
        print("\n" + "="*60)
        print("EXAMPLE: IMPORTANCE WITH PARTIAL DEPENDENCE")
        print("="*60)
        
        # Create synthetic data with clear interactions
        X, y = make_classification(
            n_samples=1000, n_features=8, n_informative=4,
            n_clusters_per_class=1, random_state=self.random_state
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Get top 3 features for partial dependence
        top_features = importance_df.head(3)['Feature'].tolist()
        
        print("Top 3 Features for Partial Dependence Analysis:")
        print(importance_df.head(3).to_string(index=False))
        
        # Calculate partial dependence manually (simplified version)
        from sklearn.inspection import partial_dependence
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, feature in enumerate(top_features):
            feature_idx = X.columns.get_loc(feature)
            
            # Calculate partial dependence
            pd_results = partial_dependence(rf, X, [feature_idx])
            pd_values = pd_results[0]
            feature_values = pd_results[1][0]
            
            # Plot
            axes[i].plot(feature_values, pd_values)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Partial Dependence')
            axes[i].set_title(f'{feature} (Importance: {importance_df[importance_df["Feature"] == feature]["Importance"].iloc[0]:.3f})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/partial_dependence.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df, top_features
    
    def example_importance_across_models(self):
        """Example: Comparing importance across different model types."""
        print("\n" + "="*60)
        print("EXAMPLE: IMPORTANCE ACROSS MODEL TYPES")
        print("="*60)
        
        # Use wine dataset (multiclass)
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        
        print(f"Dataset: {X.shape} (Wine - Multiclass)")
        
        # Train different models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state)
        }
        
        importance_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X, y)
            
            # Get importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                continue
            
            importance_results[name] = importance
            
            # Evaluate model
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(importance_results, index=X.columns)
        
        # Calculate correlations between models
        model_names = list(importance_results.keys())
        corr_matrix = comparison_df[model_names].corr()
        
        print("\nImportance Correlation Between Models:")
        print(corr_matrix.round(3))
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top features comparison
        top_n = 8
        for i, model_name in enumerate(model_names):
            top_features = comparison_df[model_name].nlargest(top_n)
            axes[0, 0].plot(range(top_n), top_features.values, 
                             marker='o', label=model_name, linewidth=2)
        
        axes[0, 0].set_xlabel('Rank')
        axes[0, 0].set_ylabel('Importance')
        axes[0, 0].set_title(f'Top {top_n} Features Across Models')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
                   cmap="coolwarm", center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Importance Correlation Between Models')
        
        # Plot 3: Feature rankings comparison
        rankings = comparison_df.rank(ascending=False)
        ranking_std = rankings.std(axis=1)
        
        axes[1, 0].bar(range(len(ranking_std)), ranking_std)
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Ranking Std Dev')
        axes[1, 0].set_title('Ranking Consistency (Lower = More Consistent)')
        
        # Plot 4: Individual feature importance
        feature_idx = 0  # Show first feature
        model_importances = [comparison_df.iloc[feature_idx][model] for model in model_names]
        
        axes[1, 1].bar(model_names, model_importances)
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_title(f'Importance for {X.columns[feature_idx]}')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/importance_across_models.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df, corr_matrix
    
    def example_time_based_importance(self):
        """Example: Importance analysis over time/concept drift."""
        print("\n" + "="*60)
        print("EXAMPLE: TEMPORAL IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Simulate temporal data
        np.random.seed(self.random_state)
        n_periods = 4
        samples_per_period = 500
        
        results = []
        
        for period in range(n_periods):
            print(f"\nAnalyzing period {period + 1}...")
            
            # Create data with changing patterns
            X, y = make_classification(
                n_samples=samples_per_period,
                n_features=10,
                n_informative=6,
                weights=[0.7 + period * 0.05, 0.3 - period * 0.05],  # Changing class balance
                random_state=self.random_state + period
            )
            
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
            
            # Train model
            rf = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Store importance
            period_results = {'period': period + 1}
            for i, feature in enumerate(feature_names):
                period_results[feature] = rf.feature_importances_[i]
            
            # Calculate performance
            cv_scores = cross_val_score(
                rf, X, y,
                cv=StratifiedKFold(3, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            period_results['cv_accuracy'] = cv_scores.mean()
            
            results.append(period_results)
        
        # Convert to DataFrame
        temporal_df = pd.DataFrame(results)
        
        # Analyze importance changes
        feature_cols = [col for col in temporal_df.columns if col.startswith('feature_')]
        
        # Calculate importance volatility
        importance_volatility = temporal_df[feature_cols].std()
        
        volatility_df = pd.DataFrame({
            'Feature': feature_cols,
            'Volatility': importance_volatility,
            'Mean_Importance': temporal_df[feature_cols].mean()
        }).sort_values('Volatility', ascending=False)
        
        print("\nFeature Importance Volatility Across Periods:")
        print(volatility_df.round(4).to_string(index=False))
        
        # Visualize temporal changes
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Performance over time
        axes[0, 0].plot(temporal_df['period'], temporal_df['cv_accuracy'], 'bo-')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('CV Accuracy')
        axes[0, 0].set_title('Model Performance Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Top features over time
        top_features = volatility_df.head(3)['Feature'].tolist()
        for feature in top_features:
            axes[0, 1].plot(temporal_df['period'], temporal_df[feature], 
                             marker='o', label=feature, linewidth=2)
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Importance')
        axes[0, 1].set_title('Importance Evolution (Top 3 Most Volatile)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility vs. Mean importance
        axes[1, 0].scatter(volatility_df['Mean_Importance'], volatility_df['Volatility'], alpha=0.6)
        axes[1, 0].set_xlabel('Mean Importance')
        axes[1, 0].set_ylabel('Importance Volatility')
        axes[1, 0].set_title('Importance Stability Analysis')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of importance over time
        heatmap_data = temporal_df[feature_cols].T
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", 
                   cmap="YlOrRd", ax=axes[1, 1])
        axes[1, 1].set_title('Importance Heatmap Over Periods')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig("plots/temporal_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return temporal_df, volatility_df
    
    def create_importance_dashboard(self, importance_data: Dict, model_name: str = "Model"):
        """
        Create a comprehensive dashboard for feature importance analysis.
        
        Args:
            importance_data: Dictionary with various importance metrics
            model_name: Name of the model for display
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Main importance plot
        ax1 = fig.add_subplot(gs[0, :2])
        if 'mdi_importance' in importance_data:
            mdi_df = importance_data['mdi_importance']
            ax1.barh(mdi_df['Feature'][:10], mdi_df['MDI_Importance'][:10])
            ax1.set_title('Top 10 Features - MDI Importance')
            ax1.set_xlabel('Importance')
        
        # 2. Permutation importance
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'permutation_importance' in importance_data:
            perm_df = importance_data['permutation_importance']
            ax2.barh(perm_df['Feature'][:10], perm_df['Permutation_Importance'][:10], color='orange')
            ax2.set_title('Top 10 Features - Permutation Importance')
            ax2.set_xlabel('Importance')
        
        # 3. Correlation heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        if 'correlation_matrix' in importance_data:
            corr_matrix = importance_data['correlation_matrix']
            # Take top 10 features for readability
            top_features = corr_matrix.abs().sum().nlargest(10).index
            corr_subset = corr_matrix.loc[top_features, top_features]
            sns.heatmap(corr_subset, annot=True, fmt=".2f", 
                       cmap="coolwarm", center=0, ax=ax3)
            ax3.set_title('Feature Correlations (Top 10)')
        
        # 4. Bias detection summary
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'biases' in importance_data:
            biases = importance_data['biases']
            bias_types = []
            bias_detected = []
            
            for bias_type, bias_info in biases.items():
                if isinstance(bias_info, dict) and 'detected' in bias_info:
                    bias_types.append(bias_type.replace('_bias', '').title())
                    bias_detected.append(1 if bias_info['detected'] else 0)
            
            colors = ['red' if detected else 'green' for detected in bias_detected]
            ax4.bar(bias_types, bias_detected, color=colors, alpha=0.7)
            ax4.set_title('Bias Detection Results')
            ax4.set_ylabel('Detected (1=Yes, 0=No)')
            ax4.set_ylim(0, 1.2)
        
        # 5. Model performance
        ax5 = fig.add_subplot(gs[2, :2])
        if 'model_performance' in importance_data:
            perf = importance_data['model_performance']
            metrics = list(perf.keys())
            values = list(perf.values())
            ax5.bar(metrics, values, color='steelblue', alpha=0.7)
            ax5.set_title('Model Performance Metrics')
            ax5.set_ylabel('Score')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Recommendations
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        recommendations = [
            "• Validate high-importance features with domain experts",
            "• Check for data leakage in top predictors",
            "• Test feature removal by retraining",
            "• Monitor importance stability over time",
            "• Use permutation importance for critical decisions"
        ]
        
        y_pos = 0.9
        for rec in recommendations:
            ax6.text(0.05, y_pos, rec, fontsize=10, transform=ax6.transAxes)
            y_pos -= 0.15
        
        ax6.set_title('Key Recommendations')
        
        plt.suptitle(f'{model_name} - Feature Importance Dashboard', fontsize=16, y=0.98)
        plt.savefig("plots/importance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()


def run_all_examples():
    """Run all feature importance examples."""
    print("""
🎯 FEATURE IMPORTANCE EXAMPLES
=============================

Additional practical examples for feature importance interpretation.
    """)
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    examples = FeatureImportanceExamples(random_state=42)
    
    # Run all examples
    example_funcs = [
        ("Importance Stability", examples.example_importance_stability),
        ("Feature Selection", examples.example_feature_selection_with_importance),
        ("Partial Dependence", examples.example_partial_dependence_with_importance),
        ("Cross-Model Comparison", examples.example_importance_across_models),
        ("Temporal Analysis", examples.example_time_based_importance)
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
