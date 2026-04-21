"""
Configuration module for spam email detection project.
Centralizes shared constants and configuration parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "random_forest"
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "max_depth": 10
}

# Data configuration
SYNTHETIC_DATA = True
N_SAMPLES = 1000
N_FEATURES = 20
N_INFORMATIVE = 15
N_REDUNDANT = 5

# Feature names for synthetic data
FEATURE_NAMES = [
    'word_freq_free', 'word_freq_offer', 'word_freq_win', 'word_freq_money',
    'word_freq_click', 'word_freq_business', 'word_freq_email', 'word_freq_internet',
    'word_freq_order', 'word_freq_credit', 'char_freq_exclamation', 'char_freq_dollar',
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
    'email_length', 'subject_length', 'has_html', 'has_attachments', 'sender_reputation'
]

# File paths
MODEL_FILE = MODELS_DIR / "spam_classifier.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
