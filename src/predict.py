import joblib
import pandas as pd


def load_artifacts():
    """
    Load the saved preprocessing pipeline and trained model
    """
    try:
        pipeline = joblib.load("models/preprocessing.pkl")
        model = joblib.load("models/spam_classifier.pkl")  # ✅ fixed name
        return pipeline, model
    except Exception as e:
        print("Error loading model or pipeline:", e)
        exit()


def predict(input_path):
    """
    Load input data, transform it, and generate predictions
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print("Error reading input file:", e)
        return

    # ✅ Check if required column exists
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    # Load pipeline and model
    pipeline, model = load_artifacts()

    # Extract feature
    X = df["text"]

    # ✅ IMPORTANT: Only transform (no fit)
    X_transformed = pipeline.transform(X)

    # Make predictions
    predictions = model.predict(X_transformed)

    # Add predictions to dataframe
    df["prediction"] = predictions

    # ✅ Convert numeric prediction to readable label
    df["label"] = df["prediction"].map({0: "Not Spam", 1: "Spam"})

    # ✅ (Optional but good) Add confidence score
    try:
        probs = model.predict_proba(X_transformed)
        df["confidence"] = probs.max(axis=1)
    except:
        df["confidence"] = "N/A"

    # Print results
    print("\n=== Predictions ===")
    print(df[["text", "label", "confidence"]])


if __name__ == "__main__":
    predict("data/raw/data.csv")