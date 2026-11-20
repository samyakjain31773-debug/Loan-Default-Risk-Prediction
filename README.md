### Loan Default Risk Prediction

An end-to-end Machine Learning project that predicts the likelihood of serious loan delinquency using customer financial and credit-history information.

## Project Overview

The objective is to identify customers who may experience serious delinquency within two years. The project focuses on imbalanced classification, model comparison, threshold optimization, and business cost analysis.

**Problem Type:** Binary Classification
**Best Model:** HistGradientBoostingClassifier
**Dataset Size:** 104,805 records
**Target Variable:** `SeriousDlqin2yrs`

- `0`: No serious delinquency
- `1`: Serious delinquency

## Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Google Colab
- Joblib

## Project Workflow

1. Data loading and exploratory data analysis
2. Missing-value and outlier investigation
3. Duplicate and feature analysis
4. Group-aware train-test splitting
5. Data preprocessing and imputation
6. Handling class imbalance using class weights
7. Model training and comparison
8. Cross-validation and out-of-fold predictions
9. Classification threshold analysis
10. Business cost analysis
11. Model evaluation and prediction pipeline
12. Model serialization using Joblib

## Models Compared

- Logistic Regression
- Random Forest Classifier
- HistGradientBoosting Classifier

HistGradientBoosting achieved the strongest overall cross-validation ROC-AUC and PR-AUC among the evaluated models. It was selected for final evaluation.

## Final Model Performance

Evaluated on a held-out test set of 20,961 records.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 91.35% |
| Precision | 39.78% |
| Recall    | 51.92% |
| F1 Score  | 45.04% |
| ROC-AUC   | 86.46% |
| PR-AUC    | 40.53% |

### Confusion Matrix

|                   | Predicted No Default | Predicted Default |
| ----------------- | -------------------: | ----------------: |
| Actual No Default |               18,405 |             1,125 |
| Actual Default    |                  688 |               743 |

## Business Cost Analysis

A hypothetical cost framework was used:

- False Negative cost: ₹10,000
- False Positive cost: ₹1,000

Training out-of-fold analysis identified 0.58 as the lowest-cost threshold among the tested candidates.

This is an illustrative business assumption, not an actual bank loss estimate. The threshold requires validation against realistic business costs before deployment.

## Model Prediction Example

The reusable prediction function accepts customer financial information and returns:

- Default probability
- Binary prediction
- Risk classification

Example output:

```python
{
    "Default_Probability": 0.9192,
    "Prediction": 1,
    "Risk": "High Risk"
}
```

## Model Saving

The trained model was saved using Joblib:

```python
import joblib

joblib.dump(final_model, "loan_default_model.pkl")
```

## Key Learnings

- Handling imbalanced classification datasets
- Understanding precision-recall trade-offs
- Avoiding data leakage through proper preprocessing
- Group-aware train-test splitting
- Cross-validation and out-of-fold predictions
- Classification threshold optimization
- Business-oriented model evaluation
- Building reusable prediction functions

## Limitations

- The business cost values are hypothetical.
- Model probabilities should not be interpreted as guaranteed default outcomes.
- Further probability calibration and external validation are required.
- The model is a learning project, not a production-ready credit approval system.
- Fairness, explainability, and regulatory requirements must be evaluated before real-world use.

## Future Improvements

- SHAP-based model explainability
- Probability calibration
- Hyperparameter optimization using training-only cross-validation
- Streamlit prediction dashboard
- API deployment using FastAPI
- Model monitoring and drift detection

## Author

Samyak Jain
samyaklundia@gmail.com
MCA – Artificial Intelligence & Machine Learning

**Project:** Loan Default Risk Prediction
