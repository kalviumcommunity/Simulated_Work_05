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

# Feature type definitions with professional reasoning
TARGET_COLUMN = "is_spam"

# NUMERICAL_FEATURES: Continuous numerical variables that can be measured
# Reasoning: These features represent frequencies, counts, and measurements
# Scaling: Will apply StandardScaler for normalization
NUMERICAL_FEATURES = [
    'word_freq_free',          # Word frequency - continuous count
    'word_freq_offer',          # Word frequency - continuous count  
    'word_freq_win',            # Word frequency - continuous count
    'word_freq_money',           # Word frequency - continuous count
    'word_freq_click',           # Word frequency - continuous count
    'word_freq_business',         # Word frequency - continuous count
    'word_freq_email',           # Word frequency - continuous count
    'word_freq_internet',         # Word frequency - continuous count
    'word_freq_order',           # Word frequency - continuous count
    'word_freq_credit',           # Word frequency - continuous count
    'char_freq_exclamation',      # Character frequency - continuous count
    'char_freq_dollar',           # Character frequency - continuous count
    'capital_run_length_average',  # Statistical measure - continuous
    'capital_run_length_longest',  # Statistical measure - continuous
    'capital_run_length_total',    # Statistical measure - continuous
    'email_length',              # Text measurement - continuous
    'subject_length',              # Text measurement - continuous
    'sender_reputation'           # Numerical score - continuous
]

# CATEGORICAL_FEATURES: Discrete categories with finite values
# Reasoning: Binary indicators represent distinct categories
# Encoding: Will use one-hot encoding if needed
CATEGORICAL_FEATURES = [
    'has_html',                 # Binary: HTML presence (0/1)
    'has_attachments'            # Binary: Attachment presence (0/1)
]

# EXCLUDED_COLUMNS: Features explicitly excluded from modeling
# Reasoning: These columns should not be used for predictions
EXCLUDED_COLUMNS = [
    'email_id',                 # Identifier column - no predictive value
    'timestamp',                # Temporal leakage risk - future information
    'sender_domain',             # High cardinality - too many unique values
    'recipient_count'            # Data collection artifact - not inherent to email
]

# Complete feature set for modeling
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Validation assertions
assert TARGET_COLUMN not in ALL_FEATURES, "Target column cannot be in feature set"
assert len(set(ALL_FEATURES)) == len(ALL_FEATURES), "Duplicate features detected"
assert len(NUMERICAL_FEATURES) > 0, "Must have numerical features"
assert len(CATEGORICAL_FEATURES) >= 0, "Must have categorical features (can be empty)"

# Feature metadata for documentation
FEATURE_METADATA = {
    'target_variable': {
        'name': TARGET_COLUMN,
        'type': 'binary',
        'description': 'Spam classification (1=spam, 0=not_spam)',
        'business_meaning': 'Identifies unsolicited promotional emails'
    },
    'numerical_features': {
        'count': len(NUMERICAL_FEATURES),
        'scaling_required': True,
        'description': 'Continuous variables requiring normalization'
    },
    'categorical_features': {
        'count': len(CATEGORICAL_FEATURES),
        'encoding_strategy': 'one_hot',
        'description': 'Binary indicators for categorical presence'
    },
    'excluded_columns': {
        'count': len(EXCLUDED_COLUMNS),
        'reasons': ['identifiers', 'temporal_leakage', 'high_cardinality', 'data_collection_artifacts']
    }
}

# File paths
MODEL_FILE = MODELS_DIR / "spam_classifier.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"
SCALER_FILE = MODELS_DIR / "minmax_scaler.pkl"  # MinMaxScaler for numerical features

# Scaling configuration - MinMaxScaler Implementation
SCALING_CONFIG = {
    "scaler_type": "MinMaxScaler",
    "scaler_range": [0, 1],
    "target_features": "NUMERICAL_FEATURES only",
    "fit_on": "training data only",
    "transform_on": "both train and test data",
    "categorical_scaling": False,
    "target_scaling": False,
    "outlier_strategy": "minmax_sensitive_documented",
    "verification_required": True
}

# Outlier configuration for MinMaxScaler
OUTLIER_CONFIG = {
    "outlier_presence": "checked_in_analysis",
    "minmax_sensitivity": "high_extreme_values_affect_bounds",
    "mitigation_strategy": "documented_in_readme",
    "capping_considered": False,
    "rationale": "MinMaxScaler chosen for bounded [0,1] output suitable for neural networks"
}

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
