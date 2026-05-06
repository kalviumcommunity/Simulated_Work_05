# Spam Email Detection Project

A professional machine learning pipeline demonstrating clean modular design with proper separation of concerns and production-ready code structure.

## Overview

This project showcases engineering discipline in ML through a properly structured spam email detection system. The implementation follows best practices with clear separation between training and prediction logic, reusable components, and import-safe modular design. Each module has a single responsibility and the entire pipeline can run from a clean environment.

## Machine Learning Workflow

### 1. Raw Data Collection
- **Email Text**: Original email content
- **Subject Line**: Email subject information
- **Sender Details**: Sender information and metadata
- **Challenge**: Raw data is unstructured, noisy, and contains inconsistent formats

### 2. Feature Engineering
Transform raw data into numerical features that the model can understand:

- **Text Processing**: Convert email text to numerical representations
- **Bag of Words**: Word frequency analysis
- **Keyword Extraction**: Identify important words like "free", "offer", "win", "lottery", "prize"
- **Additional Features**:
  - Email length
  - Presence of suspicious keywords
  - Word frequency patterns
  - Sender reputation indicators

### 3. Model Training
The model learns patterns from engineered features:
- **Spam Patterns**: Promotional words, urgent calls-to-action
- **Non-Spam Patterns**: Normal communication language
- **Classification**: Binary classification (Spam/Not Spam)

### 4. Prediction
Real-time classification of new emails:
- **Input**: Email content and metadata
- **Output**: Probability-based classification
- **Example**: "You won a free iPhone!" -> Spam

### 5. Evaluation
Performance assessment using unseen data:
- **Metrics**: Accuracy, Precision, Recall
- **Validation**: Cross-validation and test set evaluation
- **Reliability**: Ensure model reliability before deployment

### 6. Monitoring & Maintenance
Continuous monitoring for model performance:
- **Data Drift Detection**: Monitor for new spam patterns
- **Performance Tracking**: Regular accuracy checks
- **Retraining**: Update model with new data patterns

## Project Structure

```
Simulated_Work_05/
| data/                   # Data storage directory
| models/                 # Saved model artifacts
| src/
| | config.py            # Shared constants and configuration
| | data_preprocessing.py # Data loading, cleaning, splitting
| | feature_engineering.py # Reusable feature transformations
| | train.py             # Model training and artifact saving
| | evaluate.py          # Comprehensive metrics computation
| | predict.py           # Load artifacts and make predictions
| requirements.txt       # Python dependencies (pinned versions)
| README.md              # Project documentation
```

## Key Features

### Modular Architecture
- **Function-Based Design**: Each module contains clearly defined functions with explicit parameters
- **Clean Imports**: No circular dependencies or wildcard imports
- **Separation of Concerns**: Data loading, preprocessing, training, and evaluation are separate modules

### Data Pipeline
- **Synthetic Data Generation**: Creates realistic spam email features for testing
- **Feature Engineering**: Standard scaling and preprocessing
- **Train-Test Split**: Stratified sampling for balanced evaluation

### Model Training
- **Multiple Algorithms**: Random Forest, SVM, and Logistic Regression support
- **Hyperparameter Tuning**: Configurable model parameters
- **Model Persistence**: Save and load trained models

### Comprehensive Evaluation
- **Standard Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Business Metrics**: Cost analysis, false positive/negative rates
- **Confusion Matrix**: Detailed classification results

## Common Failure Scenarios

### Poor Feature Engineering
**Problem**: Inadequate feature extraction leads to poor model performance
**Symptoms**: Low accuracy, high false positive/negative rates
**Solution**: 
- Improve text preprocessing
- Remove noise and stopwords
- Use advanced feature extraction techniques

### Model Drift
**Problem**: Model accuracy decreases over time due to changing spam patterns
**Symptoms**: Performance degradation after 6+ months
**Solution**:
- Implement continuous monitoring
- Collect new training data
- Regular model retraining

## Environment Setup

This project uses a Python virtual environment to ensure reproducibility and isolation of dependencies.

### Python Version
- **Python 3.11** (recommended)
- Tested with Python 3.11.0+

### Create Virtual Environment

#### Windows
```bash
# Navigate to project directory
cd Simulated_Work_05

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

#### macOS/Linux
```bash
# Navigate to project directory
cd Simulated_Work_05

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Install Dependencies

After activating the virtual environment:

```bash
# Install all required packages with pinned versions
pip install -r requirements.txt
```

### Verify Setup

Test that the environment is working correctly:

```bash
# Test training pipeline
python src/train.py

# Test prediction pipeline
python src/predict.py
```

### Reproduce Environment from Scratch

To completely reproduce the environment on a clean machine:

```bash
# 1. Clone the repository
git clone <repository-url>
cd Simulated_Work_05

# 2. Create and activate virtual environment
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python src/train.py
```

### Environment Isolation

The virtual environment ensures:
- **Package Isolation**: Dependencies are installed only within the venv
- **Version Control**: Exact versions are pinned in `requirements.txt`
- **Reproducibility**: Same environment can be recreated on any machine
- **No Global Pollution**: Doesn't affect your system Python installation

### Deactivate Environment

When done working on the project:

```bash
deactivate
```

### OS-Specific Considerations

#### Windows
- Use PowerShell or Command Prompt
- Ensure Python is in your PATH
- Run as Administrator if permission issues occur

#### macOS
- Use built-in Python 3 or install via Homebrew
- May need to install Xcode Command Line Tools: `xcode-select --install`

#### Linux
- Install Python 3 and pip if not present:
  ```bash
  # Ubuntu/Debian
  sudo apt update
  sudo apt install python3 python3-pip python3-venv
  
  # CentOS/RHEL
  sudo yum install python3 python3-pip
  ```

### Troubleshooting

#### Common Issues

1. **Permission Denied**: Run with appropriate permissions or use user directory
2. **Module Not Found**: Ensure virtual environment is activated
3. **Version Conflicts**: Delete venv and recreate with fresh install
4. **Path Issues**: Verify Python is in system PATH

#### Reset Environment

```bash
# Deactivate if active
deactivate

# Remove virtual environment
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows

# Recreate and reinstall
python -m venv venv
# Activate (platform-specific)
pip install -r requirements.txt
```

## Project Setup Instructions

This section provides step-by-step instructions to set up spam email detection project from scratch.

### Prerequisites

- **Python 3.11+** (recommended)
- **Git** for version control
- **Command line/terminal** access

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Simulated_Work_05
```

### Step 2: Create Virtual Environment

#### Windows (PowerShell/Command Prompt)
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

After activating virtual environment:

```bash
pip install -r requirements.txt
```

This will install the following pinned versions:
- pandas==3.0.2
- numpy==2.4.4
- scikit-learn==1.8.0
- joblib==1.5.3
- matplotlib==3.10.9
- seaborn==0.13.2
- scipy==1.17.1

### Step 4: Run Training Script

```bash
python src/train.py
```

Expected output:
- Training pipeline completes successfully
- Model artifacts saved to `models/` directory
- Test accuracy: ~86%
- Test ROC AUC: ~95%

### Step 5: Run Evaluation

The training script includes built-in evaluation. For detailed metrics:

```bash
python src/evaluate.py
```

### Step 6: Run Prediction

```bash
python src/predict.py
```

Expected output:
- Loads trained model from `models/` directory
- Makes sample predictions
- Shows confidence scores

### Step 7: Verify Setup

Your setup is complete if:
- Virtual environment activated successfully
- All dependencies installed without errors
- Training script runs to completion
- Prediction script works with saved model
- No import errors or missing packages

### Troubleshooting

#### Common Issues

1. **"python is not recognized"**
   - Ensure Python 3.11+ is installed and in PATH
   - Use `python3` instead of `python` if needed

2. **"venv Scripts not found"**
   - Use full path: `python -m venv venv`
   - Check Python installation includes venv module

3. **"pip install fails"**
   - Update pip: `python -m pip install --upgrade pip`
   - Check internet connection
   - Verify requirements.txt format

4. **"ModuleNotFoundError"**
   - Ensure virtual environment is activated
   - Verify all packages installed: `pip list`
   - Reinstall if needed: `pip install -r requirements.txt`

5. **Permission errors**
   - Run as administrator (Windows)
   - Use user directory for virtual environment

#### Reset Environment

If setup fails, start fresh:

```bash
# Deactivate if active
deactivate

# Remove virtual environment
rm -rf venv  # macOS/Linux
rmdir /s /q venv  # Windows

# Recreate and reinstall
python -m venv venv
# Activate (platform-specific)
pip install -r requirements.txt
```

### Project Structure After Setup

```
Simulated_Work_05/
├── venv/                  # Virtual environment (excluded from git)
├── data/                  # Data directory
├── models/                # Saved model artifacts
├── src/                   # Source code
├── requirements.txt        # Pinned dependencies
├── .gitignore            # Git exclusions
└── README.md             # This file
```

## Installation & Setup

```bash
# Navigate to project directory
cd Simulated_Work_05

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Pipeline
```bash
# Train the model and save artifacts
python src/train.py
```

This command executes the complete training workflow:
1. Data loading and preprocessing
2. Feature engineering with reusable transformations
3. Model training (Random Forest by default)
4. Comprehensive evaluation
5. Artifact saving to `models/` directory

### Prediction Pipeline
```bash
# Make predictions using saved artifacts
python src/predict.py
```

### Individual Module Usage

#### Data Preprocessing
```python
from src.data_preprocessing import load_data, clean_data, split_data
X, y = load_data(synthetic=True)
X_clean, y_clean = clean_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_clean, y_clean)
```

#### Feature Engineering
```python
from src.feature_engineering import fit_preprocessor, transform_features
preprocessor, X_train_transformed = fit_preprocessor(X_train, y_train)
X_test_transformed = transform_features(preprocessor, X_test)
```

#### Training
```python
from src.train import train_model, save_model
model, training_info = train_model(X_train_transformed, y_train)
save_model(model)
```

#### Prediction
```python
from src.predict import load_classifier
classifier = load_classifier()
result = classifier.predict_single(features_dict)
print(f"Prediction: {result['label']}")
```

#### Evaluation
```python
from src.evaluate import evaluate_model, print_evaluation_summary
results = evaluate_model(y_true, y_pred, y_proba)
print_evaluation_summary(results)
```

## Performance Metrics

### Model Performance (Current Results)
- **Accuracy**: 92.0%
- **Precision**: 92.0%
- **Recall**: 92.0%
- **F1-Score**: 92.0%
- **ROC AUC**: 97.2%

### Business Metrics
- **False Positive Rate**: 8.0%
- **False Negative Rate**: 8.0%
- **Total Cost**: $88.00 (based on weighted cost model)
- **Spam Caught Rate**: 92.0%
- **Legitimate Preserved Rate**: 92.0%

### Top Features
1. **capital_run_length_average** (12.2% importance)
2. **word_freq_business** (7.7% importance)
3. **word_freq_email** (6.8% importance)
4. **word_freq_win** (6.6% importance)
5. **has_html** (5.8% importance)

## Technical Implementation

### Dependencies
- **pandas==2.0.3**: Data manipulation and analysis
- **numpy==1.24.3**: Numerical computing
- **scikit-learn==1.3.0**: Machine learning algorithms
- **matplotlib==3.7.2**: Data visualization
- **joblib==1.3.2**: Model persistence

### Code Quality
- **Type Hints**: All functions include proper type annotations
- **Docstrings**: Comprehensive documentation for all functions
- **Error Handling**: Robust error handling and validation
- **No Global State**: All functions are pure and stateless

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under MIT License - see the LICENSE file for details.

## Acknowledgments

- Machine learning workflow best practices
- Spam email detection research community
- Open source NLP libraries and tools

---

## Team Members

- Amulya B
- Yashika Sridhar
- Rudhresh 

## Data Leakage Demonstration

This project includes a comprehensive demonstration of **Target Leakage** - one of the most common and dangerous forms of data leakage in machine learning.

### What is Target Leakage?

Target leakage occurs when features that contain information about the target variable are included in the training data. This leads to artificially strong model performance that doesn't generalize to new data.

### Demonstration Overview

Run the leakage demonstration:

```bash
python data_leakage_demo.py
```

### Leakage Types Demonstrated

#### ❌ Target Leakage (Demonstrated)
- **Direct Target Inclusion**: Adding target as a feature
- **Target-Derived Features**: Creating features from target values
- **Obvious Patterns**: Using target to create "perfect" indicators
- **Target Interactions**: Combining target with other features

### Expected Results

| Metric | Leaky Model | Clean Model | Difference |
|----------|---------------|--------------|-------------|
| Accuracy | ~99% | ~85% | -14% |
| F1-Score | ~99% | ~83% | -16% |
| ROC AUC | ~99% | ~91% | -8% |

### Why Leaky Performance is Invalid

The leaky model shows suspiciously high performance (>95%) because:
- **Future Information**: Features won't be available at prediction time
- **Target Patterns**: Model learns target, not email patterns
- **No Generalization**: Performance drops dramatically on new data
- **False Confidence**: Artificially inflated metrics

### Why Clean Approach is Valid

The clean model shows realistic performance because:
- **Available Features**: All features available at prediction time
- **Real Patterns**: Model learns actual spam characteristics
- **Generalization**: Will perform consistently on new data
- **True Performance**: Metrics reflect real-world capability

### Prevention Discipline

1. **Split First**: Always split data before any preprocessing
2. **No Target**: Never include target in feature engineering
3. **Availability Check**: Ensure features exist at prediction time
4. **Domain Knowledge**: Use email characteristics, not target information
5. **Validation**: Test with truly unseen data

### Key Learnings

- ✅ Data leakage creates artificially strong performance
- ✅ Performance >95% is often a red flag
- ✅ Clean models generalize better to production
- ✅ Feature availability must be validated
- ✅ Train-test separation prevents leakage

### Files

- `data_leakage_demo.py`: Complete demonstration script
- Shows both incorrect and correct approaches
- Includes performance comparison and explanations

## Feature Type Definition

This project implements professional feature type selection with explicit validation and clear reasoning for each feature group.

### Target Variable

| Column Name | Type | Business Meaning |
|-------------|------|----------------|
| `is_spam` | Binary | Spam classification (1=spam, 0=not_spam) |

**Type**: Binary classification task  
**Business Meaning**: Identifies unsolicited promotional emails vs legitimate communications

### Numerical Features

| Feature | Type | Reasoning | Scaling Required |
|----------|------|-----------|------------------|
| `word_freq_free` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_offer` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_win` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_money` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_click` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_business` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_email` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_internet` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_order` | Continuous | Word frequency - continuous count | Yes |
| `word_freq_credit` | Continuous | Word frequency - continuous count | Yes |
| `char_freq_exclamation` | Continuous | Character frequency - continuous count | Yes |
| `char_freq_dollar` | Continuous | Character frequency - continuous count | Yes |
| `capital_run_length_average` | Continuous | Statistical measure - continuous | Yes |
| `capital_run_length_longest` | Continuous | Statistical measure - continuous | Yes |
| `capital_run_length_total` | Continuous | Statistical measure - continuous | Yes |
| `email_length` | Continuous | Text measurement - continuous | Yes |
| `subject_length` | Continuous | Text measurement - continuous | Yes |
| `sender_reputation` | Continuous | Numerical score - continuous | Yes |

**Why Numerical**: These features represent measurable quantities that can be ordered and have mathematical meaning. They are continuous variables requiring normalization for optimal model performance.

**Scaling Strategy**: StandardScaler will be applied to normalize numerical features for better model convergence.

### Categorical Features

| Feature | Type | Reasoning | Encoding Strategy |
|----------|------|-----------|-------------------|
| `has_html` | Binary | Binary: HTML presence (0/1) | One-hot encoding |
| `has_attachments` | Binary | Binary: Attachment presence (0/1) | One-hot encoding |

**Why Categorical**: These features represent distinct categories with finite values. They are binary indicators for categorical presence.

**Encoding Strategy**: One-hot encoding will be applied to convert categorical variables to numerical format for machine learning algorithms.

### Excluded Columns

| Column | Exclusion Reason | Risk Type |
|---------|------------------|------------|
| `email_id` | Identifier column - no predictive value | Data Quality |
| `timestamp` | Temporal leakage risk - future information | Data Leakage |
| `sender_domain` | High cardinality - too many unique values | Performance |
| `recipient_count` | Data collection artifact - not inherent to email | Relevance |

**Why Excluded**: These columns are explicitly removed to prevent common ML issues like data leakage, overfitting to identifiers, and performance degradation from high-cardinality features.

### Feature Validation

The project includes comprehensive feature validation:

```python
from src.data_preprocessing import load_data

# Load data with automatic validation
X, y = load_data(synthetic=True)
```

**Validation Output**:
```
📊 FEATURE TYPE VALIDATION RESULTS:
   Target column: is_spam
   Numerical features: 15
   Categorical features: 2
   Excluded columns: 4
   Total features for modeling: 17
✅ All expected features found in dataset
```

**Assertions Enforced**:
- ✅ Target column not in feature set
- ✅ No duplicate features in ALL_FEATURES
- ✅ At least one numerical feature exists
- ✅ Categorical features can be empty (but not numerical)
- ✅ Excluded columns are properly excluded from modeling

### Edge Case Handling

**Binary Columns (0/1)**: Treated as categorical features with one-hot encoding
**High-Cardinality Columns**: Excluded to prevent dimensionality explosion
**Timestamp Columns**: Excluded to prevent temporal data leakage
**Missing Values**: Handled through imputation in preprocessing pipeline

### Reproducibility

Another engineer can reproduce the feature grouping without ambiguity:

```python
from src.config import (
    TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, 
    EXCLUDED_COLUMNS, ALL_FEATURES, FEATURE_METADATA
)

# Access feature metadata
print(f"Target: {FEATURE_METADATA['target_variable']['name']}")
print(f"Numerical count: {FEATURE_METADATA['numerical_features']['count']}")
print(f"Categorical count: {FEATURE_METADATA['categorical_features']['count']}")
```

**Professional Standards Met**:
- ✅ Explicit feature type definitions
- ✅ Clear business reasoning for each feature
- ✅ Automated validation with assertions
- ✅ No automatic type detection using df.select_dtypes()
- ✅ Comprehensive documentation for team collaboration

## Quick Start Example

```python
# Load and use trained model
from src.model import load_model
from src.evaluate import evaluate_model

# Load the trained model
model = load_model('models/spam_classifier.pkl')

# The model is ready for predictions on new email data
# Features should match the training feature structure