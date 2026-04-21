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

## Installation & Setup

```bash
# Navigate to project directory
cd Simulated_Work_05

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

## Quick Start Example

```python
# Load and use the trained model
from src.model import load_model
from src.evaluate import evaluate_model

# Load the trained model
model = load_model('models/spam_classifier.pkl')

# The model is ready for predictions on new email data
# Features should match the training feature structure