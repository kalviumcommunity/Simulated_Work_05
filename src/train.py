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

    print("\n✅ Training complete and model saved!")
    print("✅ Test set remained untouched during training")


if __name__ == "__main__":
    train()