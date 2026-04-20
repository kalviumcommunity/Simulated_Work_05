# Spam Email Detection Project

A modular machine learning pipeline for detecting spam emails with clean function-based design and professional ML engineering practices.

## Overview

This project implements a complete, production-ready ML workflow for spam email classification. The system uses a structured approach with separate modules for data loading, preprocessing, model training, and evaluation. It demonstrates best practices in ML engineering including clean imports, function-based design, and reproducible workflows.

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
├── src/
│   ├── data_loader.py      # Data loading functionality
│   ├── preprocessing.py    # Data preprocessing and feature engineering
│   ├── model.py           # Model training and management
│   ├── evaluate.py        # Model evaluation and metrics
│   └── main.py            # Workflow orchestration
├── models/                # Trained models saved here
├── requirements.txt       # Python dependencies (pinned versions)
└── README.md              # Project documentation
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

### Run Complete ML Pipeline
```bash
python src/main.py
```

This single command executes the entire workflow:
1. Data loading (synthetic spam email features)
2. Data preprocessing and feature engineering
3. Model training (Random Forest by default)
4. Model evaluation with comprehensive metrics
5. Model saving to `models/spam_classifier.pkl`

### Individual Module Usage

#### Load Data
```python
from src.data_loader import load_data
X, y = load_data(synthetic=True)
```

#### Preprocess Data
```python
from src.preprocessing import preprocess_data
X_train, X_test, y_train, y_test, info = preprocess_data(X, y)
```

#### Train Model
```python
from src.model import train_model
model, training_info = train_model(X_train, y_train, model_type='random_forest')
```

#### Evaluate Model
```python
from src.evaluate import evaluate_model
results = evaluate_model(model, X_test, y_test)
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

## Quick Start Example

```python
# Load and use the trained model
from src.model import load_model
from src.evaluate import evaluate_model

# Load the trained model
model = load_model('models/spam_classifier.pkl')

# The model is ready for predictions on new email data
# Features should match the training feature structure