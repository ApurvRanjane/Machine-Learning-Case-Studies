# Breast Cancer Classification using Machine Learning and Ensemble Learning

## Overview

This project explores different Machine Learning classification algorithms and Ensemble Learning techniques for breast cancer classification.

Seven individual Python implementations are developed using the same dataset to understand and compare different classification approaches.

The project includes individual classifiers as well as ensemble techniques such as Bagging, Boosting, Random Forest, and Voting Classifiers.

## Machine Learning Algorithms Implemented

### 1. Individual Machine Learning Algorithms

| No. | Algorithm | Python File |
|---|---|---|
| 1 | Decision Tree Classifier | `Individual_Breast_Cancer_DT.py` |
| 2 | Logistic Regression | `Individual_Breast_Cancer_Logistic.py` |

### 2. Ensemble Machine Learning Algorithms

| No. | Algorithm | Python File |
|---|---|---|
| 1 | Bagging Classifier | `Ensemble_Breast_Cancer_Bagging.py` |
| 2 | AdaBoost Classifier | `Ensemble_Breast_Cancer_Boosting.py` |
| 3 | Random Forest Classifier | `Ensemble_Breast_Cancer_RandomForest.py` |
| 4 | Voting Classifier – Hard Voting | `Ensemble_Breast_Cancer_Voting_Hard.py` |
| 5 | Voting Classifier – Soft Voting | `Ensemble_Breast_Cancer_Voting_Soft.py` |

## Dataset Information

**Dataset Name:** `breast_cancer.csv`

The dataset contains 569 records and 31 columns.

### Dataset Summary

| Property | Details |
|---|---|
| Total Records | 569 |
| Total Columns | 31 |
| Input Features | 30 |
| Target Variable | `target` |
| Missing Values | 0 |
| Target Class 0 | 212 records |
| Target Class 1 | 357 records |

### Input Features

The dataset contains 30 numerical features related to breast cancer measurements.

These features include:

- Mean measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- Error measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- Worst measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

### Target Variable

`target`

The target variable contains two classification labels:

- `0`
- `1`

## Project Workflow

The individual Python implementations follow these general steps:

1. Load the breast cancer dataset using Pandas.
2. Display dataset shape and initial records.
3. Separate independent features (X) and dependent labels (Y).
4. Divide the dataset into training and testing subsets.
5. Apply feature scaling using StandardScaler.
6. Create the respective Machine Learning classification model.
7. Train the model using the training dataset.
8. Predict results using the testing dataset.
9. Evaluate model performance using accuracy score and confusion matrix.

## Ensemble Learning Techniques

### Bagging Classifier

Uses a Decision Tree as the base estimator and creates an ensemble with 10 estimators.

### AdaBoost Classifier

Uses the AdaBoost algorithm with:

- Number of Estimators: 50
- Learning Rate: 1.0
- Random State: 42

### Random Forest Classifier

Uses multiple decision trees for classification.

- Number of Estimators: 10
- Random State: 42

### Voting Classifier

Combines predictions from three individual models:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors

Two voting strategies are implemented:

- **Hard Voting:** Predicts based on majority voting.
- **Soft Voting:** Predicts based on the average predicted class probabilities.

## Model Evaluation

The implementations use the following evaluation metric:

- Accuracy Score
- Confusion Matrix

The performance results are displayed in the console when each Python file is executed.

## Technologies and Libraries

- Python
- Pandas
- Scikit-learn

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Project Setup

Keep the dataset and all Python files in the same directory.

### 3. Run Individual Implementations

Execute each Python file separately.

For example:

```bash
python Individual_Breast_Cancer_DT.py
```

```bash
python Individual_Breast_Cancer_Logistic.py
```

```bash
python Ensemble_Breast_Cancer_Bagging.py
```

Similarly, execute the remaining Python files to explore the other algorithms.

## Project Structure

```text
Breast_Cancer_CaseStudy/
│
├── breast_cancer.csv
│
├── Individual_Breast_Cancer_DT.py
├── Individual_Breast_Cancer_Logistic.py
│
├── Ensemble_Breast_Cancer_Bagging.py
├── Ensemble_Breast_Cancer_Boosting.py
├── Ensemble_Breast_Cancer_RandomForest.py
├── Ensemble_Breast_Cancer_Voting_Hard.py
├── Ensemble_Breast_Cancer_Voting_Soft.py
│
├── README.md
└── requirements.txt
```

## Author

**Apurv Ranjane**