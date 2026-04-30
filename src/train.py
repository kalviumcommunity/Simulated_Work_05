from data_loader import load_data
from preprocessing import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train():
    df = load_data("data/raw/data.csv")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_transformed, y_train)

    # Save artifacts
    joblib.dump(pipeline, "models/preprocessing.pkl")
    joblib.dump(model, "models/model.pkl")

    print("Training complete and model saved!")

if __name__ == "__main__":
    train()
