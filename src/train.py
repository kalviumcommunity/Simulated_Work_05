from src.data_loader import load_data
from src.data_preprocessing import split_data, build_pipeline
from sklearn.linear_model import LogisticRegression
import joblib


def train():
    # 📥 Load dataset
    df = load_data("data/raw/data.csv")

    # 🔒 Split BEFORE any preprocessing (prevents leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # 🧠 Build TF-IDF vectorizer
    vectorizer = build_pipeline()

    # ✅ Fit ONLY on training data
    X_train_transformed = vectorizer.fit_transform(X_train)

    # ✅ Transform test data (no fitting!)
    X_test_transformed = vectorizer.transform(X_test)

    # 🤖 Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_transformed, y_train)

    # 💾 Save artifacts
    joblib.dump(vectorizer, "models/preprocessing.pkl")
    joblib.dump(model, "models/model.pkl")

def main():
    """
    Main training pipeline function.
    """
    logger.info("Starting training pipeline...")
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        X, y = load_data(synthetic=True)
        X_clean, y_clean = clean_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        preprocessor, X_train_transformed = fit_preprocessor(X_train.values, y_train.values)
        X_test_transformed = preprocessor.transform(X_test.values)
        
        # Step 3: Train model
        logger.info("Step 3: Training model...")
        model, training_info = train_model(
            X_train_transformed, 
            y_train.values, 
            model_type=MODEL_TYPE,
            **MODEL_PARAMS
        )
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model...")
        test_metrics = evaluate_training_model(model, X_test_transformed, y_test.values)
        
        # Step 5: Save artifacts
        logger.info("Step 5: Saving artifacts...")
        save_model(model)
        save_preprocessor(preprocessor)
        
        # Save feature names if feature selection was used
        if 'feature_selector' in preprocessor.named_steps:
            selector = preprocessor.named_steps['feature_selector']
            selected_indices = selector.get_support(indices=True)
            save_feature_names(FEATURE_NAMES, selected_indices)
        else:
            save_feature_names(FEATURE_NAMES)
        
        # Step 6: Generate training summary
        logger.info("Step 6: Generating training summary...")
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            if 'feature_selector' in preprocessor.named_steps:
                selector = preprocessor.named_steps['feature_selector']
                selected_indices = selector.get_support(indices=True)
                scores = selector.scores_
                # Create importance dictionary for selected features
                for idx in selected_indices:
                    if idx < len(FEATURE_NAMES):
                        feature_importance[FEATURE_NAMES[idx]] = float(scores[idx])
        
        # Combine all information
        training_summary = {
            'training_info': training_info,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'data_info': {
                'total_samples': len(X_clean),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'original_features': X_clean.shape[1],
                'transformed_features': X_train_transformed.shape[1]
            }
        }
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1-score: {test_metrics['test_f1']:.4f}")
        
        if 'test_roc_auc' in test_metrics:
            logger.info(f"Test ROC AUC: {test_metrics['test_roc_auc']:.4f}")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
    print("\n✅ Training complete and model saved!")
    print("✅ Test set remained untouched during training")


if __name__ == "__main__":
    train()
