# Loan Default Prediction using MLP Classifier

## Overview

This project predicts whether a loan applicant is likely to default on a loan using a Multi-Layer Perceptron (MLP) Classifier.

The model learns patterns from historical loan applicant records and classifies applicants into two categories:

- **0 – Low Default Risk (No Default)**
- **1 – High Default Risk (Default)**

The project also explores different MLP hyperparameters, including activation functions, hidden layer architectures, and learning rates.

## Algorithm Used

- Multi-Layer Perceptron (MLP) Classifier

## Technologies and Dependencies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset Information

**Dataset Name:** `Loan_Default.csv`

The dataset contains 1,200 records and 11 columns.

### Input Features

1. Age
2. Income
3. LoanAmount
4. CreditScore
5. EmploymentYears
6. ExistingLoans
7. MonthlyDebt
8. LoanTerm
9. PreviousDefault
10. HomeOwnership

### Target Variable

`Default`

- `0` – No Default
- `1` – Default

### Dataset Summary

- Total Records: 1,200
- Total Columns: 11
- Input Features: 10
- Target Variable: Default
- Missing Values: 0
- No Default Records: 879 (73.25%)
- Default Records: 321 (26.75%)

### Categorical Features

- `PreviousDefault`: No, Yes
- `HomeOwnership`: Own, Rent, Mortgage

These categorical features are converted into numerical values using LabelEncoder.

## Project Workflow

1. Load the dataset using Pandas.
2. Perform Exploratory Data Analysis (EDA).
3. Display dataset shape, column names, and first five records.
4. Check for missing values.
5. Analyze the target variable distribution.
6. Convert categorical features into numerical values.
7. Separate independent and dependent variables.
8. Split the dataset into training and testing sets using an 80:20 ratio.
9. Apply stratified splitting to preserve class distribution.
10. Scale features using StandardScaler.
11. Create an MLP Classifier with hidden layers (32, 16).
12. Train the model using training data.
13. Evaluate testing accuracy.
14. Generate a confusion matrix and classification report.
15. Calculate precision, recall, and F1 score.
16. Plot the training loss curve.
17. Predict default risk for a new loan applicant.

## Hyperparameter Experiments

### Experiment 1: Activation Functions

Compare the following activation functions:

- Identity
- Logistic
- Tanh
- ReLU

### Experiment 2: Hidden Layer Architectures

Compare different hidden layer configurations:

- (10,)
- (20, 10)
- (50, 25)
- (100, 50, 25)

### Experiment 3: Learning Rates

Compare the following learning rates:

- 0.001
- 0.01
- 0.1

The accuracy of each configuration is displayed to compare model performance.

## Model Evaluation

The model is evaluated using:

- **Accuracy:** Measures the percentage of correct predictions.
- **Confusion Matrix:** Displays correct and incorrect classifications.
- **Classification Report:** Provides precision, recall, and F1 score.
- **Training Loss Curve:** Helps observe the model's training convergence.

## How to Run the Project

1. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Keep the dataset and Python file in the same project directory.

3. Run the Python file.

   ```bash
   python LoanDefaultPrediction.py
   ```

4. Review the evaluation results, training loss curve, new applicant prediction, and hyperparameter experiment outputs.

## Author

**Apurv Ranjane**