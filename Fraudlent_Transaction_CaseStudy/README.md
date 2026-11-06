# Fraudulent Transaction Detection using Ensemble Learning

## Overview

This project focuses on detecting potentially fraudulent financial transactions using different machine learning ensemble algorithms.

The objective is to build and compare five classification models and evaluate their performance using accuracy, precision, recall, F1 score, and confusion matrix.

## Algorithms Used

1. Decision Tree Classifier
2. Bagging Classifier
3. Random Forest Classifier
4. AdaBoost Classifier
5. Voting Classifier

## Technologies and Dependencies

- Python
- Pandas
- Scikit-learn

## Dataset Information

**Dataset Name:** `Fraudulent_Transaction_Detection.csv`

The dataset contains 130 records and 8 columns, including 7 input features and 1 target variable.

### Input Features

- TransactionAmount
- TransactionHour
- AccountAgeMonths
- PreviousTransactions
- LocationDifferenceKm
- DeviceType
- FailedLoginAttempts

### Target Variable

`Fraud`

- `0` – Normal Transaction
- `1` – Fraudulent Transaction

### Dataset Summary

- Total Records: 130
- Total Columns: 8
- Input Features: 7
- Target Variable: Fraud
- Missing Values: 0
- Normal Transactions: 100
- Fraudulent Transactions: 30

## Project Workflow

1. Load the dataset using Pandas.
2. Inspect the dataset shape, records, and missing values.
3. Separate independent features and the target variable.
4. Split the dataset into training and testing sets.
5. Scale the input features.
6. Create five classification models.
7. Train each model using the training dataset.
8. Generate predictions using the testing dataset.
9. Evaluate each model using accuracy, precision, recall, F1 score, and confusion matrix.
10. Compare the performance of all five algorithms.

## Model Evaluation

Each model is evaluated using the following performance metrics:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many transactions predicted as fraudulent are actually fraudulent.
- **Recall:** Measures how many actual fraudulent transactions are correctly detected.
- **F1 Score:** Provides a balance between precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for both classes.

## Final Model Comparison

The project compares all five algorithms using the following metrics:

| Algorithm | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Decision Tree | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| Bagging | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| Random Forest | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| AdaBoost | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| Voting | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |

The most suitable model should be selected after comparing the actual evaluation results. For fraud detection, recall and F1 score are particularly important because failing to identify fraudulent transactions can be costly.

## How to Run the Project

1. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Keep the dataset and Python file in the same project directory.

3. Run the Python file.

   ```bash
   python FraudulentTransactionDetection.py
   ```

4. Review the evaluation metrics and confusion matrix printed for each model.

## Author

**Apurv Ranjane**