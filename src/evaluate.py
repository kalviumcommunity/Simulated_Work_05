# src/evaluate.py

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model"""
    
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds)
    }