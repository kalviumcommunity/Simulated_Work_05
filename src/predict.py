import joblib
import pandas as pd

def load_artifacts():
    pipeline = joblib.load("models/preprocessing.pkl")
    model = joblib.load("models/model.pkl")
    return pipeline, model

def predict(input_path):
    df = pd.read_csv(input_path)

    pipeline, model = load_artifacts()

    X = df["text"]

    X_transformed = pipeline.transform(X)  # ✅ NOT fit_transform

    predictions = model.predict(X_transformed)

    df["prediction"] = predictions
    print(df[["text", "prediction"]])

if __name__ == "__main__":
    predict("data/raw/sample_input.csv")
