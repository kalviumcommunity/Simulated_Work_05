# src/feature_engineering.py

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_pipeline():
    """Create preprocessing pipeline"""
    
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    return pipeline