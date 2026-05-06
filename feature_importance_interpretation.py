"""
Feature Importance Interpretation for Tree-Based Models

Comprehensive implementation of feature importance interpretation techniques
for Decision Trees, Random Forests, and Gradient Boosting models.

This module covers:
- MDI (Mean Decrease in Impurity) importance extraction and visualization
- Permutation importance as a more reliable alternative
- Bias detection in impurity-based importance
- Correlation analysis for feature groups
- Practical interpretation framework
- Common mistakes and how to avoid them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any, Optional
import logging

# ML imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FeatureImportanceInterpreter:
    """
    Comprehensive feature importance interpretation system for tree-based models.
    
    This class provides methods to extract, visualize, and interpret feature importance
    from tree-based models using multiple techniques and validation approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the interpreter.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.importance_results = {}
        
    def create_sample_data(self, n_samples: int = 1000, n_features: int = 10, 
                        problem_type: str = "classification", 
                        add_correlated_features: bool = True,
                        add_high_cardinality: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create sample data with various challenges for importance interpretation.
        
        Args:
            n_samples: Number of samples
            n_features: Number of base features
            problem_type: 'classification' or 'regression'
            add_correlated_features: Whether to add correlated feature pairs
            add_high_cardinality: Whether to add high-cardinality features
            
        Returns:
            Tuple of (X, y) as pandas DataFrame and Series
        """
        if problem_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(n_features // 2, 2),
                n_redundant=max(n_features // 4, 1),
                n_clusters_per_class=1,
                random_state=self.random_state
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(n_features // 2, 2),
                noise=0.1,
                random_state=self.random_state
            )
        
        # Convert to DataFrame with meaningful names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")
        
        # Add correlated features
        if add_correlated_features:
            # Create correlated pairs
            for i in range(min(2, n_features // 3)):
                base_feature = f"feature_{i}"
                correlated_feature = f"correlated_with_{i}"
                # Add noise to create correlation
                X[correlated_feature] = X[base_feature] + np.random.normal(0, 0.1, n_samples)
        
        # Add high-cardinality feature
        if add_high_cardinality:
            # Create a feature with many unique values
            X["high_cardinality_id"] = np.random.randint(0, min(100, n_samples // 10), n_samples)
            
            # Add a categorical feature with moderate cardinality
            categories = [f"cat_{i}" for i in range(20)]
            X["moderate_cardinality_cat"] = np.random.choice(categories, n_samples)
        
        logger.info(f"Created {problem_type} dataset: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
        return X, y
    
    def train_decision_tree(self, X: pd.DataFrame, y: pd.Series, 
                        max_depth: int = 4, **kwargs) -> DecisionTreeClassifier:
        """
        Train a single Decision Tree for importance analysis.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            max_depth: Maximum depth of the tree
            **kwargs: Additional tree parameters
            
        Returns:
            Trained DecisionTreeClassifier
        """
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        
        tree.fit(X, y)
        self.models['decision_tree'] = tree
        
        logger.info(f"Trained Decision Tree with max_depth={max_depth}")
        return tree
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                        n_estimators: int = 100, **kwargs) -> RandomForestClassifier:
        """
        Train a Random Forest for importance analysis.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_estimators: Number of trees in the forest
            **kwargs: Additional forest parameters
            
        Returns:
            Trained RandomForestClassifier
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        
        rf.fit(X, y)
        self.models['random_forest'] = rf
        
        logger.info(f"Trained Random Forest with {n_estimators} trees")
        return rf
    
    def extract_mdi_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract Mean Decrease in Impurity (MDI) importance.
        
        Args:
            model: Trained tree-based model
            feature_names: List of feature names
            
        Returns:
            DataFrame with MDI importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "MDI_Importance": model.feature_importances_
        }).sort_values("MDI_Importance", ascending=False)
        
        # Add cumulative importance
        importance_df["Cumulative_Importance"] = importance_df["MDI_Importance"].cumsum()
        
        logger.info("Extracted MDI importance scores")
        return importance_df
    
    def compute_permutation_importance(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                  n_repeats: int = 10, scoring: str = "accuracy") -> pd.DataFrame:
        """
        Compute permutation importance using scikit-learn's implementation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            n_repeats: Number of permutation repeats
            scoring: Scoring metric
            
        Returns:
            DataFrame with permutation importance scores
        """
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring,
            n_jobs=-1
        )
        
        perm_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Permutation_Importance": result.importances_mean,
            "Permutation_Std": result.importances_std
        }).sort_values("Permutation_Importance", ascending=False)
        
        logger.info(f"Computed permutation importance with {n_repeats} repeats")
        return perm_df
    
    def analyze_feature_correlations(self, X: pd.DataFrame, threshold: float = 0.8) -> Dict:
        """
        Analyze correlations between features to identify potential importance bias.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for flagging high correlation
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Compute correlation matrix for numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        corr_matrix = X[numerical_features].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Calculate average correlation per feature
        avg_correlations = corr_matrix.abs().mean().sort_values(ascending=False)
        
        results = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'avg_correlations': avg_correlations,
            'threshold': threshold
        }
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (threshold={threshold})")
        return results
    
    def visualize_importance_comparison(self, mdi_df: pd.DataFrame, perm_df: pd.DataFrame,
                                 title: str = "Feature Importance Comparison"):
        """
        Create comprehensive visualization comparing MDI and permutation importance.
        
        Args:
            mdi_df: DataFrame with MDI importance
            perm_df: DataFrame with permutation importance
            title: Plot title
        """
        # Merge dataframes for comparison
        comparison_df = mdi_df.merge(perm_df, on='Feature', how='outer')
        comparison_df = comparison_df.fillna(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: MDI Importance (horizontal bar)
        mdi_sorted = comparison_df.sort_values('MDI_Importance', ascending=True)
        axes[0, 0].barh(mdi_sorted['Feature'], mdi_sorted['MDI_Importance'], 
                          color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('MDI Importance')
        axes[0, 0].set_title('Mean Decrease in Impurity (MDI)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Permutation Importance (horizontal bar)
        perm_sorted = comparison_df.sort_values('Permutation_Importance', ascending=True)
        axes[0, 1].barh(perm_sorted['Feature'], perm_sorted['Permutation_Importance'],
                          color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('Permutation Importance')
        axes[0, 1].set_title('Permutation Importance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot comparison
        valid_mask = (comparison_df['MDI_Importance'] > 0) | (comparison_df['Permutation_Importance'] > 0)
        valid_data = comparison_df[valid_mask]
        
        axes[1, 0].scatter(valid_data['MDI_Importance'], valid_data['Permutation_Importance'],
                          alpha=0.6, s=60)
        
        # Add diagonal line
        max_val = max(valid_data['MDI_Importance'].max(), valid_data['Permutation_Importance'].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x (perfect agreement)')
        
        axes[1, 0].set_xlabel('MDI Importance')
        axes[1, 0].set_ylabel('Permutation Importance')
        axes[1, 0].set_title('MDI vs. Permutation Importance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Difference analysis
        comparison_df['Importance_Difference'] = comparison_df['MDI_Importance'] - comparison_df['Permutation_Importance']
        diff_sorted = comparison_df.sort_values('Importance_Difference', ascending=True)
        
        colors = ['red' if x > 0 else 'blue' for x in diff_sorted['Importance_Difference']]
        axes[1, 1].barh(diff_sorted['Feature'], diff_sorted['Importance_Difference'],
                          color=colors, alpha=0.7)
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('MDI - Permutation Importance')
        axes[1, 1].set_title('Importance Difference (MDI > Perm: Red, MDI < Perm: Blue)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_correlation_matrix(self, corr_matrix: pd.DataFrame, title: str = "Feature Correlation Matrix"):
        """
        Visualize the correlation matrix as a heatmap.
        
        Args:
            corr_matrix: Correlation matrix
            title: Plot title
        """
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt=".2f", 
                    cmap="coolwarm", 
                    center=0,
                    square=True,
                    cbar_kws={"shrink": 0.8})
        
        plt.title(title)
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_importance_biases(self, mdi_df: pd.DataFrame, perm_df: pd.DataFrame,
                            corr_analysis: Dict, X: pd.DataFrame) -> Dict:
        """
        Detect potential biases in importance scores.
        
        Args:
            mdi_df: MDI importance DataFrame
            perm_df: Permutation importance DataFrame
            corr_analysis: Correlation analysis results
            X: Feature DataFrame
            
        Returns:
            Dictionary with bias detection results
        """
        biases = {}
        
        # Merge for analysis
        comparison_df = mdi_df.merge(perm_df, on='Feature', how='outer').fillna(0)
        
        # 1. High-cardinality bias detection
        cardinalities = {col: X[col].nunique() for col in X.columns}
        comparison_df['Cardinality'] = comparison_df['Feature'].map(cardinalities)
        
        # Check if high-cardinality features have disproportionately high MDI importance
        high_card_threshold = np.percentile(list(cardinalities.values()), 75)
        high_card_features = comparison_df[comparison_df['Cardinality'] >= high_card_threshold]
        
        if len(high_card_features) > 0:
            high_card_mdi = high_card_features['MDI_Importance'].mean()
            overall_mdi = comparison_df['MDI_Importance'].mean()
            cardinality_bias = high_card_mdi / overall_mdi if overall_mdi > 0 else 0
            
            biases['cardinality_bias'] = {
                'detected': cardinality_bias > 1.5,
                'bias_ratio': cardinality_bias,
                'high_cardinality_features': high_card_features['Feature'].tolist()
            }
        
        # 2. Correlation bias detection
        high_corr_pairs = corr_analysis['high_correlation_pairs']
        biased_pairs = []
        
        for pair in high_corr_pairs:
            feat1, feat2 = pair['feature1'], pair['feature2']
            feat1_importance = comparison_df[comparison_df['Feature'] == feat1]['MDI_Importance'].iloc[0]
            feat2_importance = comparison_df[comparison_df['Feature'] == feat2]['MDI_Importance'].iloc[0]
            
            # Check if importance is heavily skewed
            importance_ratio = max(feat1_importance, feat2_importance) / (min(feat1_importance, feat2_importance) + 1e-8)
            
            if importance_ratio > 5:  # One feature has 5x more importance
                biased_pairs.append({
                    'features': [feat1, feat2],
                    'correlation': pair['correlation'],
                    'importance_ratio': importance_ratio,
                    'dominant_feature': feat1 if feat1_importance > feat2_importance else feat2
                })
        
        biases['correlation_bias'] = {
            'detected': len(biased_pairs) > 0,
            'biased_pairs': biased_pairs
        }
        
        # 3. MDI vs Permutation disagreement detection
        comparison_df['Importance_Difference'] = abs(comparison_df['MDI_Importance'] - comparison_df['Permutation_Importance'])
        significant_disagreements = comparison_df[comparison_df['Importance_Difference'] > 0.1]
        
        biases['method_disagreement'] = {
            'detected': len(significant_disagreements) > 0,
            'disagreements': significant_disagreements[['Feature', 'MDI_Importance', 'Permutation_Importance', 'Importance_Difference']].to_dict('records')
        }
        
        logger.info(f"Detected biases: Cardinality={biases['cardinality_bias']['detected']}, "
                   f"Correlation={biases['correlation_bias']['detected']}, "
                   f"Method Disagreement={biases['method_disagreement']['detected']}")
        
        return biases
    
    def generate_interpretation_report(self, mdi_df: pd.DataFrame, perm_df: pd.DataFrame,
                                  corr_analysis: Dict, biases: Dict, model_performance: Dict) -> str:
        """
        Generate a comprehensive interpretation report.
        
        Args:
            mdi_df: MDI importance DataFrame
            perm_df: Permutation importance DataFrame
            corr_analysis: Correlation analysis results
            biases: Bias detection results
            model_performance: Model performance metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("FEATURE IMPORTANCE INTERPRETATION REPORT")
        report.append("=" * 80)
        
        # Model Performance Summary
        report.append("\n📊 MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        for metric, value in model_performance.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Top Features (MDI)
        report.append("\n🎯 TOP FEATURES (MDI IMPORTANCE)")
        report.append("-" * 40)
        top_mdi = mdi_df.head(5)
        for _, row in top_mdi.iterrows():
            report.append(f"{row['Feature']:<25} {row['MDI_Importance']:.4f}")
        
        # Top Features (Permutation)
        report.append("\n🎯 TOP FEATURES (PERMUTATION IMPORTANCE)")
        report.append("-" * 40)
        top_perm = perm_df.head(5)
        for _, row in top_perm.iterrows():
            report.append(f"{row['Feature']:<25} {row['Permutation_Importance']:.4f}")
        
        # Bias Detection Results
        report.append("\n⚠️  BIAS DETECTION RESULTS")
        report.append("-" * 40)
        
        # Cardinality bias
        card_bias = biases['cardinality_bias']
        if card_bias['detected']:
            report.append(f"🔴 HIGH-CARDINALITY BIAS DETECTED")
            report.append(f"   Bias ratio: {card_bias['bias_ratio']:.2f}")
            report.append(f"   Affected features: {card_bias['high_cardinality_features']}")
        else:
            report.append("✅ No significant cardinality bias detected")
        
        # Correlation bias
        corr_bias = biases['correlation_bias']
        if corr_bias['detected']:
            report.append(f"🔴 CORRELATION BIAS DETECTED")
            report.append(f"   {len(corr_bias['biased_pairs'])} biased feature pairs found")
            for pair in corr_bias['biased_pairs'][:3]:  # Show top 3
                report.append(f"   {pair['features'][0]} ↔ {pair['features'][1]}: "
                           f"corr={pair['correlation']:.3f}, ratio={pair['importance_ratio']:.1f}")
        else:
            report.append("✅ No significant correlation bias detected")
        
        # Method disagreement
        method_disag = biases['method_disagreement']
        if method_disag['detected']:
            report.append(f"🔴 MDI vs PERMUTATION DISAGREEMENTS")
            report.append(f"   {len(method_disag['disagreements'])} features with significant disagreements")
            for dis in method_disag['disagreements'][:3]:  # Show top 3
                report.append(f"   {dis['Feature']}: MDI={dis['MDI_Importance']:.3f}, "
                           f"Perm={dis['Permutation_Importance']:.3f}")
        else:
            report.append("✅ MDI and permutation importance agree well")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if card_bias['detected']:
            report.append("• Consider reducing cardinality of high-cardinality features")
            report.append("• Validate high-cardinality feature rankings with permutation importance")
        
        if corr_bias['detected']:
            report.append("• Investigate correlated feature groups before dropping features")
            report.append("• Consider feature engineering to combine correlated features")
        
        if method_disag['detected']:
            report.append("• Trust permutation importance over MDI for critical decisions")
            report.append("• Investigate features with large method disagreements")
        
        # General recommendations
        report.append("• Always validate importance findings with domain expertise")
        report.append("• Test feature removal by retraining and comparing performance")
        report.append("• Use importance for hypothesis generation, not final decisions")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def demonstrate_single_tree_importance(self):
        """Demonstrate feature importance with a single Decision Tree."""
        print("\n" + "="*70)
        print("DEMONSTRATION 1: SINGLE DECISION TREE IMPORTANCE")
        print("="*70)
        
        print("""
📖 SCENARIO: Understanding basic feature importance with a single Decision Tree

🎯 GOAL: Learn how MDI importance is calculated and interpreted
📊 DATASET: Classification with clear feature patterns
🔧 APPROACH: Train shallow tree and extract importance scores
        """)
        
        # Create simple data
        X, y = self.create_sample_data(n_samples=800, n_features=6, 
                                     add_correlated_features=False,
                                     add_high_cardinality=False)
        
        # Train a shallow tree for interpretability
        tree = self.train_decision_tree(X, y, max_depth=3)
        
        # Extract importance
        mdi_importance = self.extract_mdi_importance(tree, X.columns)
        
        print("\n📊 MDI IMPORTANCE SCORES:")
        print(mdi_importance.to_string(index=False))
        
        # Visualize the tree structure
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, 
                 class_names=['Class 0', 'Class 1'])
        plt.title("Decision Tree Structure (max_depth=3)")
        plt.savefig("plots/decision_tree_structure.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Visualize importance
        plt.figure(figsize=(10, 6))
        plt.barh(mdi_importance['Feature'], mdi_importance['MDI_Importance'],
                 color='steelblue', alpha=0.7)
        plt.xlabel('MDI Importance (fraction of total impurity reduction)')
        plt.title('Feature Importance - Single Decision Tree')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("plots/single_tree_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results
        self.importance_results['single_tree'] = {
            'mdi_importance': mdi_importance,
            'model': tree
        }
        
        return mdi_importance
    
    def demonstrate_random_forest_importance(self):
        """Demonstrate feature importance with Random Forest."""
        print("\n" + "="*70)
        print("DEMONSTRATION 2: RANDOM FOREST IMPORTANCE")
        print("="*70)
        
        print("""
📖 SCENARIO: Comparing single tree vs. forest importance stability

🎯 GOAL: Show how Random Forest provides more reliable importance estimates
📊 DATASET: Classification with correlated features and noise
🔧 APPROACH: Train Random Forest and compare with single tree results
        """)
        
        # Create more complex data
        X, y = self.create_sample_data(n_samples=1200, n_features=8,
                                     add_correlated_features=True,
                                     add_high_cardinality=True)
        
        # Train Random Forest
        rf = self.train_random_forest(X, y, n_estimators=100)
        
        # Extract MDI importance
        mdi_importance = self.extract_mdi_importance(rf, X.columns)
        
        print("\n📊 RANDOM FOREST MDI IMPORTANCE SCORES:")
        print(mdi_importance.to_string(index=False))
        
        # Compute permutation importance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Retrain on training set only
        rf_train = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf_train.fit(X_train, y_train)
        
        perm_importance = self.compute_permutation_importance(rf_train, X_test, y_test)
        
        print("\n📊 PERMUTATION IMPORTANCE SCORES:")
        print(perm_importance.to_string(index=False))
        
        # Visualize comparison
        self.visualize_importance_comparison(mdi_importance, perm_importance, 
                                      "Random Forest: MDI vs. Permutation Importance")
        
        # Store results
        self.importance_results['random_forest'] = {
            'mdi_importance': mdi_importance,
            'permutation_importance': perm_importance,
            'model': rf_train
        }
        
        return mdi_importance, perm_importance
    
    def demonstrate_bias_detection(self):
        """Demonstrate bias detection in feature importance."""
        print("\n" + "="*70)
        print("DEMONSTRATION 3: BIAS DETECTION IN IMPORTANCE SCORES")
        print("="*70)
        
        print("""
📖 SCENARIO: Detecting and analyzing biases in importance scores

🎯 GOAL: Identify cardinality bias, correlation bias, and method disagreements
📊 DATASET: Classification with high-cardinality and correlated features
🔧 APPROACH: Systematic bias detection with multiple validation methods
        """)
        
        # Create challenging data
        X, y = self.create_sample_data(n_samples=1000, n_features=10,
                                     add_correlated_features=True,
                                     add_high_cardinality=True)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Extract both importance types
        mdi_importance = self.extract_mdi_importance(rf, X.columns)
        perm_importance = self.compute_permutation_importance(rf, X_test, y_test)
        
        # Analyze correlations
        corr_analysis = self.analyze_feature_correlations(X)
        
        # Detect biases
        biases = self.detect_importance_biases(mdi_importance, perm_importance, 
                                         corr_analysis, X)
        
        # Visualize correlation matrix
        self.visualize_correlation_matrix(corr_analysis['correlation_matrix'])
        
        # Model performance
        y_pred = rf.predict(X_test)
        model_performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Generate comprehensive report
        report = self.generate_interpretation_report(
            mdi_importance, perm_importance, corr_analysis, biases, model_performance
        )
        print(report)
        
        # Store results
        self.importance_results['bias_detection'] = {
            'mdi_importance': mdi_importance,
            'permutation_importance': perm_importance,
            'correlation_analysis': corr_analysis,
            'biases': biases,
            'model_performance': model_performance
        }
        
        return mdi_importance, perm_importance, biases
    
    def demonstrate_practical_interpretation(self):
        """Demonstrate practical interpretation framework."""
        print("\n" + "="*70)
        print("DEMONSTRATION 4: PRACTICAL INTERPRETATION FRAMEWORK")
        print("="*70)
        
        print("""
📖 SCENARIO: Complete practical interpretation workflow

🎯 GOAL: Demonstrate end-to-end interpretation with business context
📊 DATASET: Customer churn prediction scenario
🔧 APPROACH: Complete workflow from training to business recommendations
        """)
        
        # Create realistic churn-like data
        X, y = self.create_sample_data(n_samples=2000, n_features=12,
                                     problem_type="classification",
                                     add_correlated_features=True,
                                     add_high_cardinality=True)
        
        # Rename features to be more realistic
        feature_mapping = {
            'feature_0': 'tenure',
            'feature_1': 'monthly_charges',
            'feature_2': 'contract_type',
            'feature_3': 'payment_method',
            'feature_4': 'internet_service',
            'feature_5': 'tech_support',
            'correlated_with_0': 'total_charges',  # Correlated with tenure
            'correlated_with_1': 'avg_daily_charges',  # Correlated with monthly_charges
            'high_cardinality_id': 'customer_id',
            'moderate_cardinality_cat': 'region'
        }
        
        # Rename columns we have, add missing ones if needed
        available_features = list(X.columns)
        for old_name, new_name in feature_mapping.items():
            if old_name in available_features:
                X = X.rename(columns={old_name: new_name})
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Extract importance
        mdi_importance = self.extract_mdi_importance(rf, X.columns)
        perm_importance = self.compute_permutation_importance(rf, X_test, y_test, n_repeats=15)
        
        # Analyze correlations
        corr_analysis = self.analyze_feature_correlations(X)
        
        # Detect biases
        biases = self.detect_importance_biases(mdi_importance, perm_importance, 
                                         corr_analysis, X)
        
        # Model performance
        y_pred = rf.predict(X_test)
        model_performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Generate interpretation report
        report = self.generate_interpretation_report(
            mdi_importance, perm_importance, corr_analysis, biases, model_performance
        )
        print(report)
        
        # Business recommendations based on importance
        self._generate_business_recommendations(mdi_importance, perm_importance, biases)
        
        # Store results
        self.importance_results['practical_interpretation'] = {
            'mdi_importance': mdi_importance,
            'permutation_importance': perm_importance,
            'correlation_analysis': corr_analysis,
            'biases': biases,
            'model_performance': model_performance
        }
        
        return mdi_importance, perm_importance
    
    def _generate_business_recommendations(self, mdi_df: pd.DataFrame, 
                                      perm_df: pd.DataFrame, biases: Dict):
        """Generate business recommendations based on importance analysis."""
        print("\n🏢 BUSINESS RECOMMENDATIONS")
        print("-" * 50)
        
        # Get top features from both methods
        top_mdi = mdi_df.head(3)['Feature'].tolist()
        top_perm = perm_df.head(3)['Feature'].tolist()
        
        # Consensus top features
        consensus_features = list(set(top_mdi) & set(top_perm))
        
        print("🎯 KEY PREDICTORS OF CHURN:")
        for feature in consensus_features:
            print(f"  • {feature}: Strong predictor in both MDI and permutation importance")
        
        # Features to investigate
        mdi_only = list(set(top_mdi) - set(top_perm))
        perm_only = list(set(top_perm) - set(top_mdi))
        
        if mdi_only:
            print(f"\n🔍 FEATURES TO INVESTIGATE (High MDI, Low Permutation):")
            for feature in mdi_only:
                print(f"  • {feature}: May have cardinality or correlation bias")
        
        if perm_only:
            print(f"\n🔍 FEATURES TO INVESTIGATE (High Permutation, Low MDI):")
            for feature in perm_only:
                print(f"  • {feature}: May be undervalued by MDI due to correlations")
        
        # Data collection recommendations
        print(f"\n📊 DATA COLLECTION RECOMMENDATIONS:")
        
        if biases['cardinality_bias']['detected']:
            print("  • Review high-cardinality features for encoding improvements")
            print("  • Consider grouping or binning high-cardinality categorical variables")
        
        if biases['correlation_bias']['detected']:
            print("  • Evaluate whether to combine correlated features")
            print("  • Consider feature engineering to create interaction terms")
        
        # Model improvement recommendations
        print(f"\n🔧 MODEL IMPROVEMENT RECOMMENDATIONS:")
        print("  • Test removing low-importance features and retrain")
        print("  • Consider feature selection based on permutation importance")
        print("  • Validate model with business domain experts")
    
    def run_complete_tutorial(self):
        """Run the complete feature importance interpretation tutorial."""
        print("""
🎯 FEATURE IMPORTANCE INTERPRETATION TUTORIAL
============================================

This tutorial provides comprehensive coverage of feature importance
interpretation for tree-based models, covering both theory and practice.

📚 TUTORIAL STRUCTURE:
1. Single Decision Tree Importance
2. Random Forest Importance & Comparison
3. Bias Detection and Analysis
4. Practical Interpretation Framework

⏱️  EXPECTED DURATION: 45-60 minutes
        """)
        
        # Create output directory
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Run all demonstrations
        demonstrations = [
            ("Single Decision Tree", self.demonstrate_single_tree_importance),
            ("Random Forest Comparison", self.demonstrate_random_forest_importance),
            ("Bias Detection", self.demonstrate_bias_detection),
            ("Practical Interpretation", self.demonstrate_practical_interpretation)
        ]
        
        for i, (title, demo_func) in enumerate(demonstrations, 1):
            print(f"\n{'='*20} {i}. {title} {'='*20}")
            
            try:
                result = demo_func()
                
                if i < len(demonstrations):
                    input("\nPress Enter to continue to next demonstration...")
                    
            except Exception as e:
                print(f"Error in {title}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*70)
        print("TUTORIAL COMPLETED!")
        print("="*70)
        
        print(f"\n📁 Files created: plots/")
        print("\n🎯 Key Takeaways:")
        takeaways = [
            "• MDI importance is fast but can be biased by cardinality and correlation",
            "• Permutation importance is more reliable but requires held-out data",
            "• Always compare both methods before making important decisions",
            "• Check for correlated features that may split importance unfairly",
            "• High-cardinality features can appear artificially important in MDI",
            "• Use importance for hypothesis generation, not final business decisions",
            "• Always validate findings with domain expertise",
            "• Test feature removal by retraining and comparing performance"
        ]
        
        for takeaway in takeaways:
            print(f"  {takeaway}")
        
        return True


def main():
    """Main function to run the feature importance interpretation tutorial."""
    interpreter = FeatureImportanceInterpreter(random_state=42)
    return interpreter.run_complete_tutorial()


if __name__ == "__main__":
    main()
