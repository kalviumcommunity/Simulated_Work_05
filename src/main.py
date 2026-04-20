import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import load_data, get_data_info
from preprocessing import preprocess_data
from model import train_model, save_model, get_model_summary
from evaluate import evaluate_model, print_evaluation_results, calculate_business_metrics


def main():
    """
    Main function to orchestrate the complete ML workflow.
    """
    print("Starting Spam Email Detection ML Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Load Data
        print("Step 1: Loading Data...")
        X, y = load_data(synthetic=True)
        
        # Get data information
        data_info = get_data_info(X, y)
        print(f"Dataset loaded successfully!")
        print(f"Samples: {data_info['n_samples']}, Features: {data_info['n_features']}")
        print(f"Class distribution: {data_info['class_distribution']}")
        print()
        
        # Step 2: Preprocess Data
        print("Step 2: Preprocessing Data...")
        X_train, X_test, y_train, y_test, preprocess_info = preprocess_data(
            X, y, test_size=0.2, random_state=42, scale_features=True
        )
        
        print(f"Data preprocessed successfully!")
        print(f"Training set: {preprocess_info['train_shape']}")
        print(f"Test set: {preprocess_info['test_shape']}")
        print(f"Features scaled: {preprocess_info['features_scaled']}")
        print()
        
        # Step 3: Train Model
        print("Step 3: Training Model...")
        model, training_info = train_model(
            X_train, y_train, 
            model_type='random_forest',
            n_estimators=100,
            max_depth=10
        )
        
        print(f"Model trained successfully!")
        print(f"Model type: {training_info['model_type']}")
        print(f"Training samples: {training_info['training_samples']}")
        print()
        
        # Step 4: Evaluate Model
        print("Step 4: Evaluating Model...")
        evaluation_results = evaluate_model(model, X_test, y_test)
        
        # Print evaluation results
        print_evaluation_results(evaluation_results)
        
        # Calculate and print business metrics
        business_metrics = calculate_business_metrics(evaluation_results)
        print("\nBusiness Metrics:")
        print(f"False Positive Rate: {business_metrics['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {business_metrics['false_negative_rate']:.4f}")
        print(f"Total Cost: ${business_metrics['total_cost']:.2f}")
        print(f"Spam Caught Rate: {business_metrics['spam_caught_rate']:.4f}")
        print(f"Legitimate Preserved Rate: {business_metrics['legitimate_preserved_rate']:.4f}")
        print()
        
        # Step 5: Save Model
        print("Step 5: Saving Model...")
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "spam_classifier.pkl"
        
        save_model(model, str(model_path))
        print(f"Model saved to: {model_path}")
        
        # Get model summary
        model_summary = get_model_summary(model, training_info)
        print(f"\nModel Summary:")
        print(f"Model type: {model_summary['model_type']}")
        if 'feature_importances' in model_summary:
            top_features = sorted(
                zip(model_summary['feature_importances'], preprocess_info['feature_names']),
                reverse=True
            )[:5]
            print("Top 5 Important Features:")
            for importance, feature in top_features:
                print(f"  {feature}: {importance:.4f}")
        
        print("\n" + "=" * 60)
        print("ML Pipeline completed successfully!")
        print("=" * 60)
        
        return {
            'data_info': data_info,
            'preprocess_info': preprocess_info,
            'training_info': training_info,
            'evaluation_results': evaluation_results,
            'business_metrics': business_metrics,
            'model_summary': model_summary
        }
        
    except Exception as e:
        print(f"Error in ML pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
