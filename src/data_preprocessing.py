from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(df):
    """
    Splits dataset into train and test sets with stratification.
    Ensures no data leakage by splitting before any preprocessing.
    """

    # 🔁 Adjust column names if needed
    X = df["text"]     # feature column (email content)
    y = df["label"]    # target column (spam/ham)

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ✅ Verification prints (required for assignment)
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    print("\nTrain distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

    print("\n✅ No preprocessing applied before splitting (No Data Leakage)")

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """
    Returns TF-IDF vectorizer.
    IMPORTANT: Fit ONLY on training data.
    """
    return TfidfVectorizer()
