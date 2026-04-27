import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import *
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_pipeline
from src.train import train_model
from src.evaluate import evaluate_model


# Step 1: Load
df = load_data(DATA_PATH)

# Step 2: Clean
df = clean_data(df)

# Step 3: Split
X_train, X_test, y_train, y_test = split_data(
    df, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
)

# 🔥 IMPORTANT FIX: Use only TEXT column
TEXT_COLUMN = "text"

X_train = X_train[TEXT_COLUMN]
X_test = X_test[TEXT_COLUMN]

# Step 4: Pipeline (TF-IDF)
pipeline = build_pipeline()

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Step 5: Train
model = train_model(X_train, y_train, RANDOM_STATE)

# Step 6: Evaluate
metrics = evaluate_model(model, X_test, y_test)

print(metrics)