from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline():
    """Create preprocessing pipeline for text data"""
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer())
    ])

    return pipeline