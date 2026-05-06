# Target
TARGET_COLUMN = "label"

# Numerical features
NUMERICAL_FEATURES = [
    # none for now
]

# Categorical features
CATEGORICAL_FEATURES = [
    # none for now
]

# Text features
TEXT_FEATURES = [
    "text"
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
    'subject_length'              # Text measurement - continuous
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
SCALER_FILE = MODELS_DIR / "standard_scaler.pkl"  # StandardScaler for numerical features

# Scaling configuration
SCALING_CONFIG = {
    "scaler_type": "StandardScaler",
    "target_features": "NUMERICAL_FEATURES only",
    "fit_on": "training data only",
    "transform_on": "both train and test data",
    "categorical_scaling": False,
    "target_scaling": False
}
# Excluded
EXCLUDED_COLUMNS = [
    # add if dataset has ID column
]

# Combined
ALL_FEATURES = TEXT_FEATURES
