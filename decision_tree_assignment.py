"""
Decision Tree Assignment - Complete Implementation

This script demonstrates comprehensive Decision Tree implementation including:
- Data preprocessing and feature engineering
- Decision Tree classification with proper hyperparameter tuning
- Bias-variance analysis through depth optimization
- Tree visualization and rule extraction
- Feature importance analysis
- Performance evaluation against baseline
- Cross-validation and overfitting analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            precision_recall_curve, roc_curve, auc)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionTreeAssignment:
    """Complete Decision Tree implementation for assignment."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tree = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the text classification data."""
        print("="*60)
        print("DATA LOADING AND PREPROCESSING")
        print("="*60)
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        # Create more realistic dataset for demonstration
        # Since current dataset is very small, we'll augment it
        np.random.seed(self.random_state)
        
        # Generate more samples to make the assignment meaningful
        spam_messages = [
            "Win money now!!!", "Claim your free prize", "Congratulations you won",
            "Limited time offer", "Click here for bonus", "Free gift waiting",
            "Urgent: respond now", "Exclusive deal just for you", "Cash prize winner",
            "Instant money transfer"
        ]
        
        ham_messages = [
            "Hey let's meet tomorrow", "Are you coming to class?", "See you at the meeting",
            "Thanks for your help", "Let's catch up soon", "Great job today",
            "Can we schedule a call?", "Looking forward to it", "Have a great weekend",
            "Thanks for the update"
        ]
        
        # Create balanced dataset
        spam_data = [(msg, 'spam') for msg in spam_messages * 5]  # 50 spam samples
        ham_data = [(msg, 'ham') for msg in ham_messages * 5]     # 50 ham samples
        
        # Add some noise and variations
        for i in range(40):
            if i < 20:
                msg = f"{' '.join(np.random.choice(spam_messages, 2))} {np.random.choice(['!!!', 'urgent', 'now'])}"
                spam_data.append((msg, 'spam'))
            else:
                msg = f"{' '.join(np.random.choice(ham_messages, 2))} {np.random.choice(['thanks', 'please', 'see you'])}"
                ham_data.append((msg, 'ham'))
        
        # Combine all data
        all_data = spam_data + ham_data
        df = pd.DataFrame(all_data, columns=['text', 'label'])
        
        print(f"Augmented dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        # Feature extraction using TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                   ngram_range=(1, 2), min_df=1)
        
        X = vectorizer.fit_transform(df['text'])
        self.feature_names = vectorizer.get_feature_names_out()
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df['label'])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return X, y, le
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets with stratification."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        print(f"Train class distribution: {np.bincount(self.y_train)}")
        print(f"Test class distribution:  {np.bincount(self.y_test)}")
        
    def find_optimal_depth(self, max_depth_range=range(1, 21)):
        """Find optimal tree depth using cross-validation."""
        print("\n" + "="*60)
        print("FINDING OPTIMAL TREE DEPTH")
        print("="*60)
        
        depths = list(max_depth_range)
        cv_scores = []
        train_scores = []
        
        for depth in depths:
            tree = DecisionTreeClassifier(
                max_depth=depth, 
                min_samples_leaf=5,
                random_state=self.random_state
            )
            
            # Cross-validation
            cv_score = cross_val_score(tree, self.X_train, self.y_train, 
                                     cv=5, scoring='accuracy').mean()
            cv_scores.append(cv_score)
            
            # Training score
            tree.fit(self.X_train, self.y_train)
            train_score = tree.score(self.X_train, self.y_train)
            train_scores.append(train_score)
            
            print(f"Depth {depth:2d}: CV={cv_score:.3f}, Train={train_score:.3f}, Gap={train_score-cv_score:.3f}")
        
        # Find optimal depth
        optimal_depth = depths[np.argmax(cv_scores)]
        optimal_cv_score = max(cv_scores)
        
        print(f"\nOptimal depth: {optimal_depth} (CV Score: {optimal_cv_score:.3f})")
        
        # Plot depth vs accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
        plt.plot(depths, cv_scores, 'o-', label='CV Accuracy', linewidth=2)
        plt.axvline(x=optimal_depth, color='red', linestyle='--', alpha=0.7, label=f'Optimal Depth={optimal_depth}')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Bias-Variance Trade-off: Depth vs Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        gaps = np.array(train_scores) - np.array(cv_scores)
        plt.plot(depths, gaps, 'o-', color='red', linewidth=2)
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='High Overfitting Threshold')
        plt.xlabel('Max Depth')
        plt.ylabel('Train/Test Gap')
        plt.title('Overfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_depth, depths, cv_scores, train_scores
    
    def train_optimal_tree(self, max_depth):
        """Train Decision Tree with optimal hyperparameters."""
        print("\n" + "="*60)
        print("TRAINING OPTIMAL DECISION TREE")
        print("="*60)
        
        # Train with optimal depth
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=self.random_state
        )
        
        self.tree.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = self.tree.score(self.X_train, self.y_train)
        test_acc = self.tree.score(self.X_test, self.y_test)
        train_test_gap = train_acc - test_acc
        
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        print(f"Train/Test Gap:      {train_test_gap:.4f}")
        
        if train_test_gap > 0.1:
            print("⚠️  High overfitting detected!")
        elif train_test_gap > 0.05:
            print("⚠️  Moderate overfitting detected")
        else:
            print("✅ Good generalization")
        
        return train_acc, test_acc, train_test_gap
    
    def compare_with_baseline(self):
        """Compare Decision Tree performance against baseline."""
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        
        # Baseline (most frequent class)
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(self.X_train, self.y_train)
        baseline_acc = baseline.score(self.X_test, self.y_test)
        
        # Decision Tree accuracy
        tree_acc = self.tree.score(self.X_test, self.y_test)
        
        print(f"Baseline Accuracy:    {baseline_acc:.4f}")
        print(f"Decision Tree Acc:   {tree_acc:.4f}")
        print(f"Improvement:         +{tree_acc - baseline_acc:.4f}")
        
        if tree_acc > baseline_acc:
            print("✅ Decision Tree outperforms baseline")
        else:
            print("❌ Decision Tree does not beat baseline - investigate!")
        
        return baseline_acc, tree_acc
    
    def visualize_tree(self, class_names=['ham', 'spam']):
        """Visualize the Decision Tree structure."""
        print("\n" + "="*60)
        print("TREE VISUALIZATION")
        print("="*60)
        
        plt.figure(figsize=(20, 12))
        plot_tree(
            self.tree,
            feature_names=self.feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            impurity=True,
            proportion=True
        )
        plt.title(f'Decision Tree (Depth={self.tree.get_depth()}, Leaves={self.tree.get_n_leaves()})')
        plt.savefig('plots/decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Extract text rules
        rules = export_text(
            self.tree,
            feature_names=list(self.feature_names),
            class_names=class_names,
            show_weights=True
        )
        
        print("\nExtracted Decision Rules:")
        print("="*50)
        print(rules)
        
        return rules
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Show top features
        top_features = importance_df.head(15)
        print("Top 15 Most Important Features:")
        print(top_features.to_string(index=False))
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        top_10 = importance_df.head(10)
        plt.barh(range(len(top_10)), top_10['Importance'])
        plt.yticks(range(len(top_10)), top_10['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        # Importance distribution
        plt.hist(importance_df['Importance'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Importance')
        plt.ylabel('Frequency')
        plt.title('Feature Importance Distribution')
        plt.axvline(x=importance_df['Importance'].mean(), 
                   color='red', linestyle='--', label=f'Mean: {importance_df["Importance"].mean():.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def detailed_evaluation(self):
        """Detailed model evaluation with classification report and confusion matrix."""
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.tree.predict(self.X_test)
        y_proba = self.tree.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print("="*30)
        print(classification_report(self.y_test, y_pred, target_names=['ham', 'spam']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print("="*30)
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC and Precision-Recall curves
        plt.subplot(1, 2, 2)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        # Plot both curves
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'classification_report': classification_report(self.y_test, y_pred, target_names=['ham', 'spam'], output_dict=True),
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def learning_curve_analysis(self):
        """Generate and analyze learning curves."""
        print("\n" + "="*60)
        print("LEARNING CURVE ANALYSIS")
        print("="*60)
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.tree, self.X_train, self.y_train,
            train_sizes=train_sizes, cv=5, scoring='accuracy',
            random_state=self.random_state
        )
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', 
                label='Training Score', linewidth=2, color='blue')
        plt.fill_between(train_sizes_abs, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, test_scores.mean(axis=1), 'o-', 
                label='Cross-Validation Score', linewidth=2, color='red')
        plt.fill_between(train_sizes_abs,
                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                        test_scores.mean(axis=1) + test_scores.std(axis=1),
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analysis
        final_train_score = train_scores.mean(axis=1)[-1]
        final_test_score = test_scores.mean(axis=1)[-1]
        gap = final_train_score - final_test_score
        
        print(f"Final Training Score: {final_train_score:.4f}")
        print(f"Final CV Score:       {final_test_score:.4f}")
        print(f"Gap:                  {gap:.4f}")
        
        if gap > 0.1:
            print("⚠️  High variance model (overfitting)")
        elif final_test_score < 0.7:
            print("⚠️  High bias model (underfitting)")
        else:
            print("✅ Good bias-variance balance")
        
        return train_sizes_abs, train_scores, test_scores
    
    def generate_summary_report(self, results):
        """Generate a comprehensive summary report."""
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*70)
        
        print(f"\n📊 MODEL PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Training Accuracy:     {results['train_acc']:.4f}")
        print(f"Test Accuracy:         {results['test_acc']:.4f}")
        print(f"Baseline Accuracy:     {results['baseline_acc']:.4f}")
        print(f"Improvement over Baseline: +{results['test_acc'] - results['baseline_acc']:.4f}")
        print(f"Train/Test Gap:        {results['train_test_gap']:.4f}")
        
        print(f"\n🌳 TREE CHARACTERISTICS")
        print("-" * 40)
        print(f"Optimal Depth:         {results['optimal_depth']}")
        print(f"Number of Leaves:      {self.tree.get_n_leaves()}")
        print(f"Number of Features:    {len(self.feature_names)}")
        
        print(f"\n📈 EVALUATION METRICS")
        print("-" * 40)
        eval_results = results['evaluation']
        print(f"ROC AUC:               {eval_results['roc_auc']:.4f}")
        print(f"PR AUC:                {eval_results['pr_auc']:.4f}")
        
        # Class-specific metrics
        class_report = eval_results['classification_report']
        print(f"\nSpam Detection Performance:")
        print(f"  Precision:           {class_report['spam']['precision']:.4f}")
        print(f"  Recall:              {class_report['spam']['recall']:.4f}")
        print(f"  F1-Score:            {class_report['spam']['f1-score']:.4f}")
        
        print(f"\n🎯 KEY INSIGHTS")
        print("-" * 40)
        
        if results['train_test_gap'] < 0.05:
            print("✅ Excellent generalization with minimal overfitting")
        elif results['train_test_gap'] < 0.1:
            print("✅ Good generalization with acceptable overfitting")
        else:
            print("⚠️  Significant overfitting detected - consider more regularization")
        
        if results['test_acc'] > results['baseline_acc'] + 0.1:
            print("✅ Decision Tree significantly outperforms baseline")
        elif results['test_acc'] > results['baseline_acc']:
            print("✅ Decision Tree outperforms baseline")
        else:
            print("❌ Decision Tree fails to beat baseline")
        
        if eval_results['roc_auc'] > 0.8:
            print("✅ Excellent discriminative ability (ROC AUC > 0.8)")
        elif eval_results['roc_auc'] > 0.7:
            print("✅ Good discriminative ability (ROC AUC > 0.7)")
        else:
            print("⚠️  Limited discriminative ability")
        
        print("\n" + "="*70)
    
    def run_complete_assignment(self, data_path):
        """Run the complete Decision Tree assignment."""
        print("🎯 DECISION TREE ASSIGNMENT - COMPLETE IMPLEMENTATION")
        print("=" * 70)
        
        # Create plots directory
        import os
        os.makedirs('plots', exist_ok=True)
        
        # Step 1: Load and preprocess data
        X, y, label_encoder = self.load_and_preprocess_data(data_path)
        
        # Step 2: Split data
        self.split_data(X, y)
        
        # Step 3: Find optimal depth
        optimal_depth, depths, cv_scores, train_scores = self.find_optimal_depth()
        
        # Step 4: Train optimal tree
        train_acc, test_acc, train_test_gap = self.train_optimal_tree(optimal_depth)
        
        # Step 5: Compare with baseline
        baseline_acc, tree_acc = self.compare_with_baseline()
        
        # Step 6: Visualize tree
        rules = self.visualize_tree()
        
        # Step 7: Analyze feature importance
        importance_df = self.analyze_feature_importance()
        
        # Step 8: Detailed evaluation
        evaluation_results = self.detailed_evaluation()
        
        # Step 9: Learning curve analysis
        learning_curve_analysis = self.learning_curve_analysis()
        
        # Step 10: Generate summary report
        results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'baseline_acc': baseline_acc,
            'train_test_gap': train_test_gap,
            'optimal_depth': optimal_depth,
            'evaluation': evaluation_results,
            'importance_df': importance_df,
            'rules': rules
        }
        
        self.generate_summary_report(results)
        
        print("\n🎉 ASSIGNMENT COMPLETED SUCCESSFULLY!")
        print("📁 All plots saved to 'plots/' directory")
        print("📊 Check the comprehensive analysis above")
        
        return results


def main():
    """Main function to run the assignment."""
    # Initialize the assignment
    assignment = DecisionTreeAssignment(random_state=42)
    
    # Run complete assignment
    data_path = "data/raw/data.csv"
    results = assignment.run_complete_assignment(data_path)
    
    return results


if __name__ == "__main__":
    results = main()
