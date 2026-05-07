# Professional Model Comparison Framework

## Overview

This framework implements systematic, fair comparison of multiple machine learning models following professional ML practices. It ensures that model selection is based on rigorous evidence rather than familiarity or chance.

## Why This Matters

Training a single model and reporting its accuracy is not professional ML practice. Real model selection requires:

- **Multiple candidates**: Different models make different assumptions about data
- **Fair comparison**: Identical preprocessing, metrics, and evaluation conditions
- **Uncertainty quantification**: Mean and standard deviation from cross-validation
- **Bias-variance diagnosis**: Train/CV gap analysis reveals model behavior
- **Statistical rigor**: Significance testing for meaningful differences
- **Holistic selection**: Consider interpretability, latency, and business constraints

## Key Features

### 1. Consistent Preprocessing
- Shared preprocessing pipeline for all models
- Automatic feature type detection
- Proper handling of missing values and categorical variables
- Prevents preprocessing artifacts from biasing comparison

### 2. Fair Hyperparameter Tuning
- Equal tuning budget (n_iter) for all models
- Appropriate parameter distributions for each model type
- Same cross-validation folds for all models
- Prevents tuning effort from biasing results

### 3. Comprehensive Evaluation
- Cross-validation with mean ± standard deviation
- Train/CV gap analysis for bias-variance diagnosis
- Multiple metrics for imbalanced problems
- Statistical significance testing
- Training and inference time measurement

### 4. Visualization & Reporting
- Performance comparison plots with error bars
- Bias-variance analysis visualizations
- Comprehensive text reports
- Model selection recommendations

## Quick Start

```python
from src.model_comparison import ModelComparisonFramework, ComparisonConfig
import pandas as pd

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').squeeze()

# Setup framework
config = ComparisonConfig(
    cv_folds=5,
    scoring_metric="f1",
    n_iter_tuning=30,
    verbose=True
)

framework = ModelComparisonFramework(config)
framework.setup_data(X, y)

# Compare models
results = framework.compare_models(tune_hyperparameters=True)

# Generate visualizations
framework.plot_model_comparison()
framework.plot_bias_variance_analysis()

# Get recommendations
report = framework.generate_report()
print(report)
```

## Core Components

### ModelComparisonFramework

The main class that orchestrates the entire comparison workflow.

#### Key Methods

- `setup_data(X, y, numeric_features, categorical_features)`: Prepare data and identify feature types
- `compare_models(model_names, tune_hyperparameters)`: Run systematic comparison
- `analyze_bias_variance()`: Diagnose bias-variance characteristics
- `statistical_significance_test(model1, model2)`: Test if differences are significant
- `multi_metric_evaluation(model_name)`: Evaluate on multiple metrics
- `generate_report()`: Create comprehensive comparison report

### ComparisonConfig

Configuration class for customizing the comparison process.

#### Parameters

- `test_size`: Train/test split ratio (default: 0.2)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `random_state`: Random seed for reproducibility (default: 42)
- `scoring_metric`: Primary evaluation metric (default: "f1")
- `n_iter_tuning`: Hyperparameter search iterations (default: 30)
- `n_jobs`: Parallel computation cores (default: -1)
- `verbose`: Print progress information (default: True)

### ModelComparisonResult

Data class storing results for each model.

#### Fields

- `model_name`: Name of the model
- `cv_mean`: Cross-validation mean score
- `cv_std`: Cross-validation standard deviation
- `train_mean`: Training mean score
- `train_std`: Training standard deviation
- `gap`: Train - CV score difference
- `best_params`: Best hyperparameters found
- `training_time`: Time taken for training/tuning
- `inference_time`: Time for prediction

## Supported Models

The framework includes these models by default:

1. **Logistic Regression**: Linear, interpretable, fast
2. **Ridge Classifier**: Linear with L2 regularization
3. **Decision Tree**: Nonlinear, interpretable, high variance
4. **Random Forest**: Ensemble, reduced variance
5. **Gradient Boosting**: Sequential error correction, high accuracy
6. **AdaBoost**: Adaptive boosting
7. **KNN**: Instance-based, local patterns
8. **SVM**: Maximum margin classifier
9. **Naive Bayes**: Probabilistic, fast training

## Model Selection Guidelines

### Reading the Results Table

| Model | CV Mean | CV Std | Gap | Interpretation |
|-------|---------|---------|-----|----------------|
| High, Low | < 0.03 | Well-fitted, ideal |
| High, High | > 0.08 | High potential but unstable |
| Low, Low | < 0.03 | Underfitting, needs complexity |
| Low, High | > 0.08 | Unreliable and weak |

### Bias-Variance Diagnosis

- **Gap < 0.03**: Well-fitted (target behavior)
- **Gap 0.03-0.08**: Mild overfitting, monitor
- **Gap > 0.08**: Significant overfitting, needs regularization

### Statistical Significance

If difference < max(std_A, std_B), treat models as equivalent and prefer the simpler one.

### Selection Criteria

- **Performance**: Highest CV mean score
- **Stability**: Lowest CV standard deviation  
- **Efficiency**: Best performance/time ratio

## Evaluation Metrics

### Classification
- **Accuracy**: Overall correctness (balanced classes only)
- **Precision**: Minimize false positives
- **Recall**: Minimize false negatives  
- **F1-score**: Balance precision and recall
- **ROC-AUC**: Threshold-independent performance

### Choosing the Right Metric

**Balanced Classification**: Use F1-score or accuracy
**Imbalanced Classification**: 
- High FN cost (disease, fraud) → Use Recall
- High FP cost (spam, alerts) → Use Precision
- Balanced tradeoff needed → Use F1-score
- Extreme imbalance → Use PR-AUC

## Fair Comparison Requirements

1. ✅ **Same train/test split** - All models see identical data
2. ✅ **Same preprocessing pipeline** - Performance reflects model, not preprocessing
3. ✅ **Same evaluation metric** - Scores on same scale
4. ✅ **Same CV folds** - Fold variation shared, not model-specific
5. ✅ **Comparable hyperparameter tuning** - Same search budget
6. ✅ **No test set reuse** - Final evaluation only once

## Common Mistakes to Avoid

❌ **Different preprocessing** - Creates artifacts, invalid comparison
❌ **Different metrics** - Incomparable scores
❌ **Test set tuning** - Inflates final performance
❌ **Ignoring standard deviation** - High variance models unreliable
❌ **Tuning only complex models** - Effort bias, not model bias
❌ **Single metric focus** - Misses important tradeoffs
❌ **No stratified CV** - Unreliable estimates for imbalanced data

## Advanced Usage

### Custom Model Library

```python
def get_custom_models():
    return {
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }

framework.model_library = get_custom_models()
```

### Custom Hyperparameter Distributions

```python
def get_custom_params():
    return {
        "XGBoost": {
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(1e-3, 1e-1),
            "model__max_depth": randint(3, 10)
        }
    }

framework.hyperparameter_distributions = get_custom_params()
```

### Multi-Metric Optimization

```python
# For imbalanced problems
config.scoring_metric = "recall"  # Focus on minority class

# Then evaluate multiple metrics
for model in top_models:
    metrics = framework.multi_metric_evaluation(model)
    print(f"{model}: {metrics}")
```

## Interpretation Guide

### What the Comparison Tells You

**Linear models perform well** → Relationship approximately linear
**Tree models dominate** → Important nonlinear structure
**KNN wins** → Local structure matters more than global
**All models similar** → Problem may be inherently difficult

### Model Characteristics

| Model | Bias | Variance | Interpretability | Speed |
|-------|------|----------|------------------|-------|
| Logistic Regression | High | Low | High | Fast |
| Decision Tree | Low | High | High | Fast |
| Random Forest | Low | Medium | Low | Medium |
| Gradient Boosting | Low | Low-Medium | Low | Slow |
| KNN | Low | High | Medium | Slow inference |

### Production Considerations

**Real-time requirements** (< 2ms): Logistic Regression, Naive Bayes
**Interpretability required**: Logistic Regression, Decision Tree
**Maximum accuracy**: Gradient Boosting, Random Forest
**Limited training time**: Naive Bayes, Logistic Regression
**Regulatory compliance**: Logistic Regression (coefficients)

## Statistical Analysis

### When Differences Matter

- **Overlap in (mean ± std) ranges**: Difference likely not meaningful
- **Difference < max(std_A, std_B)**: Treat as equivalent
- **P-value > 0.05**: Not statistically significant

### Effect Size Interpretation

- **Cohen's d < 0.2**: Negligible effect
- **Cohen's d 0.2-0.5**: Small effect
- **Cohen's d 0.5-0.8**: Medium effect  
- **Cohen's d > 0.8**: Large effect

## Best Practices Checklist

Before reporting results:

- [ ] Same train/test split used for all models
- [ ] Preprocessing inside shared pipeline
- [ ] Evaluation metric chosen before training
- [ ] Stratified CV for classification
- [ ] Both CV mean and std reported
- [ ] Train/CV gap analyzed
- [ ] Comparable hyperparameter tuning
- [ ] Test set evaluated exactly once
- [ ] Multiple metrics for imbalanced problems
- [ ] Selection rationale documented

## Example Output

```
COMPREHENSIVE MODEL COMPARISON REPORT
================================================================================

1. PERFORMANCE RANKINGS
----------------------------------------
             Model  CV Mean  CV Std   Gap
 Gradient Boosting    0.891   0.028 0.042
     Random Forest    0.882   0.047 0.058
Logistic Regression    0.843   0.019 0.021
               KNN    0.821   0.012 0.035

2. BIAS-VARIANCE DIAGNOSIS
----------------------------------------
Gradient Boosting     : Mild overfitting: Good performance with moderate variance
Random Forest        : Mild overfitting: Good performance with moderate variance
Logistic Regression  : Well-fitted: High performance with low variance
KNN                  : Well-fitted: Low performance, low variance - increase complexity

3. RECOMMENDATIONS
----------------------------------------
Best Performance: Gradient Boosting (CV 0.891 ± 0.028)
Most Stable: Logistic Regression (CV 0.843 ± 0.019)

4. SELECTION CONSIDERATIONS
----------------------------------------
Gradient Boosting     : High performance
Logistic Regression  : Very stable, Well-fitted, Fast training, Fast inference
```

## Troubleshooting

### Common Issues

**Memory errors**: Reduce `n_iter_tuning` or use fewer models
**Slow execution**: Reduce `cv_folds` or set `n_jobs=1`
**Convergence warnings**: Increase `max_iter` for linear models
**Poor performance**: Check feature quality and preprocessing

### Performance Tips

- Use `n_jobs=-1` for parallel computation
- Reduce `n_iter_tuning` for faster results
- Use `cv_folds=5` as balance between speed and reliability
- Enable `verbose=False` to reduce output

## Integration with Existing Workflows

### With Scikit-learn Pipelines

```python
# The framework creates proper scikit-learn pipelines
pipeline = framework.make_pipeline(LogisticRegression())
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### With MLflow

```python
import mlflow

with mlflow.start_run():
    results = framework.compare_models()
    for result in framework.results:
        mlflow.log_metric(f"{result.model_name}_cv_mean", result.cv_mean)
        mlflow.log_metric(f"{result.model_name}_cv_std", result.cv_std)
```

### With Pandas

```python
# Results are returned as pandas DataFrames
results_df = framework.compare_models()
results_df.to_csv('model_comparison_results.csv', index=False)

# Easy filtering and analysis
best_models = results_df[results_df['CV Mean'] > 0.8]
stable_models = results_df[results_df['CV Std'] < 0.02]
```

## Contributing

To add new models:

1. Add to `get_model_library()` method
2. Add hyperparameter distributions to `get_hyperparameter_distributions()`
3. Test with `compare_models()` method
4. Update documentation

## License

This framework follows the same license as the project.

## References

- Scikit-learn Model Selection Guide
- "No Free Lunch Theorem" - Wolpert (1996)
- "Model Selection and Inference" - Burnham & Anderson
- "Pattern Recognition and Machine Learning" - Bishop
