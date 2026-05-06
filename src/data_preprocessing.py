from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import logging

from config import (
    SYNTHETIC_DATA, N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_REDUNDANT,
    RANDOM_STATE, TEST_SIZE, FEATURE_NAMES, DATA_DIR,
    TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, 
    EXCLUDED_COLUMNS, ALL_FEATURES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_features(df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and separate features from target with professional assertions.
    
    Args:
        df (pd.DataFrame): Complete dataset with all columns
        target (pd.Series): Target variable
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Validated features (X) and target (y)
        
    Raises:
        AssertionError: If validation fails
    """
    logger.info("🔍 Validating feature types and separation...")
    
    # Validate target column exists in the dataframe
    if TARGET_COLUMN not in df.columns:
        raise AssertionError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    assert TARGET_COLUMN not in ALL_FEATURES, "Target column cannot be in feature set"
    
    # Validate excluded columns are not in features
    for excluded_col in EXCLUDED_COLUMNS:
        assert excluded_col not in ALL_FEATURES, f"Excluded column '{excluded_col}' found in feature set"
        if excluded_col in df.columns:
            assert excluded_col not in ALL_FEATURES, f"Excluded column '{excluded_col}' found in feature set"
    
    # Separate features and target
    features = df[ALL_FEATURES].copy()
    target_clean = target.copy()
    
    # Validate no target leakage in features
    for col in features.columns:
        assert col != TARGET_COLUMN, f"Target column '{TARGET_COLUMN}' found in features"
    
    # Print feature type summary
    print(f"\n📊 FEATURE TYPE VALIDATION RESULTS:")
    print(f"   Target column: {TARGET_COLUMN}")
    print(f"   Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"   Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"   Excluded columns: {len(EXCLUDED_COLUMNS)}")
    print(f"   Total features for modeling: {len(ALL_FEATURES)}")
    print(f"   Available columns in dataset: {list(df.columns)}")
    
    # Validate feature presence
    missing_features = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            missing_features.append(feature)
    
    # Additional validation: check for any column name issues
    available_columns = list(df.columns)
    for col in available_columns:
        if col not in ALL_FEATURES and col not in EXCLUDED_COLUMNS and col != TARGET_COLUMN:
            logger.warning(f"Unexpected column found: {col}")
    
    if missing_features:
        logger.warning(f"Missing expected features: {missing_features}")
    else:
        logger.info("✅ All expected features found in dataset")
    
    return features, target_clean

def load_data(data_path: Optional[str] = None, synthetic: bool = SYNTHETIC_DATA) -> Tuple[pd.DataFrame, pd.Series]:
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(df):
    """
    Splits dataset into train and test sets with stratification.
    Ensures no data leakage by splitting before any preprocessing.
    """
    logger.info("Loading data...")
    
    if synthetic:
        logger.info(f"Generating synthetic data with {N_SAMPLES} samples and {N_FEATURES} features")
        
        # Generate synthetic spam email data
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=N_INFORMATIVE,
            n_redundant=N_REDUNDANT,
            n_classes=2,
            random_state=RANDOM_STATE
        )
        
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        y_series = pd.Series(y, name=TARGET_COLUMN)
        
        logger.info(f"Data loaded successfully. Shape: {X_df.shape}")
        logger.info(f"Class distribution: {y_series.value_counts().to_dict()}")
        
        # Apply feature validation
        complete_df = pd.concat([X_df, y_series], axis=1)
        return validate_features(complete_df, y_series)
    
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when synthetic=False")
            
        try:
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            # Apply feature validation
            target = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else data.iloc[:, -1]
            features = data.drop(columns=[TARGET_COLUMN] + EXCLUDED_COLUMNS, errors='ignore')
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            logger.info(f"Class distribution: {target.value_counts().to_dict()}")
            
            return validate_features(features, target)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

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
