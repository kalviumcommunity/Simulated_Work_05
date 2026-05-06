"""
Data Leakage Demonstration: Target Leakage in Spam Email Detection

This script demonstrates target leakage by including target-derived features
in the training data, leading to artificially strong model performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

def generate_spam_dataset():
    """
    Generate a synthetic spam email dataset.
    Returns features X and target y.
    """
    print("🔧 Generating synthetic spam email dataset...")
    
    # Generate base classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=RANDOM_STATE,
        flip_y=0.1
    )
    
    # Create feature names for interpretability
    feature_names = [
        'word_freq_free', 'word_freq_offer', 'word_freq_win', 'word_freq_money',
        'word_freq_click', 'word_freq_business', 'word_freq_email', 'word_freq_internet',
        'word_freq_order', 'word_freq_credit', 'char_freq_exclamation', 'char_freq_dollar',
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
        'email_length', 'subject_length', 'has_html', 'has_attachments', 'sender_reputation'
    ]
    
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='is_spam')
    
    print(f"✅ Dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Class distribution: {y.value_counts().to_dict()}")
    
    return X, y

def create_leaky_features(X, y):
    """
    ❌ INCORRECT APPROACH: Create features that leak target information.
    
    This demonstrates TARGET LEAKAGE by:
    1. Adding target directly as a feature
    2. Creating features derived from target
    3. Using target to create "obvious" patterns
    
    WHY THIS IS LEAKAGE:
    - These features won't be available at prediction time
    - Model learns target patterns instead of email patterns
    - Performance is artificially inflated
    """
    print("\n❌ CREATING LEAKY FEATURES (Target Leakage)")
    print("⚠️  This approach is INCORRECT and demonstrates data leakage!")
    
    X_leaky = X.copy()
    
    # 1. Direct target leakage - include target as feature
    X_leaky['target_direct'] = y
    
    # 2. Target-derived features
    X_leaky['target_squared'] = y ** 2
    X_leaky['target_log'] = np.log1p(np.abs(y))
    
    # 3. Obvious patterns based on target
    X_leaky['obvious_spam_indicator'] = (y > 0.5).astype(int)
    X_leaky['spam_probability_leak'] = y  # Another direct leak
    
    # 4. Target-based interactions
    X_leaky['target_times_word_freq_money'] = y * X['word_freq_money']
    X_leaky['target_plus_capital_run'] = y + X['capital_run_length_average']
    
    print(f"🔍 Added {X_leaky.shape[1] - X.shape[1]} leaky features")
    print("📝 Leaky features created:")
    leaky_features = [col for col in X_leaky.columns if col not in X.columns]
    for feature in leaky_features:
        print(f"   - {feature}")
    
    return X_leaky

def create_clean_features(X):
    """
    ✅ CORRECT APPROACH: Use only legitimate features.
    
    This approach is VALID because:
    1. No target information in features
    2. Features available at prediction time
    3. Model learns actual email patterns
    """
    print("\n✅ CREATING CLEAN FEATURES (No Leakage)")
    print("✨ This approach is CORRECT and follows ML best practices!")
    
    X_clean = X.copy()
    
    # Only use legitimate email features
    # No target information included
    # All features available at prediction time
    
    print(f"🔍 Using {X_clean.shape[1]} clean features")
    print("📝 Clean features:")
    for feature in X_clean.columns[:5]:  # Show first 5 for brevity
        print(f"   - {feature}")
    print(f"   ... and {X_clean.shape[1] - 5} more")
    
    return X_clean

def train_and_evaluate(X, y, version_name):
    """
    Train model and evaluate performance.
    """
    print(f"\n🚀 Training {version_name} model...")
    
    # Split data (IMPORTANT: Split before any fitting!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 {version_name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    # Feature importance (top 5)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'feature_importance': feature_importance,
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def compare_performance(leaky_results, clean_results):
    """
    Compare performance between leaky and clean approaches.
    """
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Metric':<15} {'Leaky Model':<15} {'Clean Model':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    metric_names = ['Accuracy', 'F1-Score', 'ROC AUC']
    
    for metric, name in zip(metrics, metric_names):
        leaky_val = leaky_results[metric]
        clean_val = clean_results[metric]
        diff = leaky_val - clean_val
        print(f"{name:<15} {leaky_val:<15.4f} {clean_val:<15.4f} {diff:+15.4f}")
    
    # Performance impact analysis
    print(f"\n💡 PERFORMANCE IMPACT ANALYSIS:")
    accuracy_drop = (leaky_results['accuracy'] - clean_results['accuracy']) * 100
    print(f"   Accuracy dropped by {accuracy_drop:.1f} percentage points after fixing leakage")
    
    if leaky_results['accuracy'] > 0.95:
        print("   ⚠️  Leaky model shows suspiciously high performance (>95%)")
        print("   ⚠️  This is a red flag for data leakage!")
    
    print(f"   Clean model performance ({clean_results['accuracy']:.1%}) is more realistic")
    print(f"   Clean model will generalize better to new data")

def main():
    """
    Main demonstration function showing both incorrect and correct approaches.
    """
    print("🎯 DATA LEAKAGE DEMONSTRATION: Target Leakage in Spam Detection")
    print("=" * 80)
    
    # Generate dataset
    X, y = generate_spam_dataset()
    
    # ❌ INCORRECT APPROACH: With Target Leakage
    print("\n" + "="*40)
    print("❌ INCORRECT APPROACH: WITH TARGET LEAKAGE")
    print("="*40)
    
    X_leaky = create_leaky_features(X, y)
    leaky_results = train_and_evaluate(X_leaky, y, "Leaky (Target Leakage)")
    
    # ✅ CORRECT APPROACH: Without Target Leakage
    print("\n" + "="*40)
    print("✅ CORRECT APPROACH: WITHOUT TARGET LEAKAGE")
    print("="*40)
    
    X_clean = create_clean_features(X)
    clean_results = train_and_evaluate(X_clean, y, "Clean (No Leakage)")
    
    # Compare performance
    compare_performance(leaky_results, clean_results)
    
    # Final explanation
    print("\n" + "="*80)
    print("📚 KEY LEARNINGS")
    print("="*80)
    print("❌ WHY FIRST VERSION WAS INVALID:")
    print("   • Included target information in features")
    print("   • Created features derived from target values")
    print("   • These features won't be available at prediction time")
    print("   • Model learned target patterns, not email patterns")
    print("   • Artificially inflated performance metrics")
    
    print("\n✅ WHY SECOND VERSION IS VALID:")
    print("   • Only uses legitimate email features")
    print("   • All features available at prediction time")
    print("   • Model learns actual spam patterns")
    print("   • Realistic performance metrics")
    print("   • Will generalize to new data")
    
    print("\n🛡️  PREVENTION DISCIPLINE:")
    print("   • ALWAYS split data before any preprocessing")
    print("   • NEVER include target in feature engineering")
    print("   • Ensure features are available at prediction time")
    print("   • Validate feature availability in production")
    print("   • Use domain knowledge, not target information")

if __name__ == "__main__":
    main()
