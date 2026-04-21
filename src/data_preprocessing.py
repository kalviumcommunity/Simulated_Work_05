# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV"""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning (handle missing values)"""
    return df.dropna()


def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    """Split into train and test"""
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)