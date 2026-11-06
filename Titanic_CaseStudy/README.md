# Titanic Survival Prediction using Logistic Regression

## Overview

This project implements a Machine Learning classification model to predict passenger survival using the Titanic dataset.

The application uses Logistic Regression to classify whether a passenger survived or not. It follows a modular approach with separate functions for data loading, preprocessing, dataset splitting, model training, evaluation, and model preservation.

## Algorithm Used

- Logistic Regression

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Dataset Information

**Dataset Name:** `MarvellousTitanicDataset.csv`

The dataset contains 1,309 records and 10 columns.

### Dataset Columns

- `Passengerid`
- `Age`
- `Fare`
- `Sex`
- `sibsp`
- `Parch`
- `zero`
- `Pclass`
- `Embarked`
- `Survived`

### Target Variable

`Survived`

- `1` – Passenger survived
- `0` – Passenger did not survive

## Project Workflow

### Step 1: Load Data

- Load the Titanic dataset using Pandas.
- Display the first five records.

### Step 2: Data Preprocessing

- Remove unnecessary columns: `Passengerid`, `zero`, and `name` when present.
- Handle missing values in Age and Fare using median values.
- Handle missing values in Embarked using its mode.
- Convert the Embarked categorical column into numerical dummy variables using `pd.get_dummies()`.

### Step 3: Dataset Splitting

- Separate independent features (X) and dependent variable (Y).
- Split the dataset into training and testing sets.
- Test size: 20%.
- Random state: 42.

### Step 4: Model Training

- Create a Logistic Regression classifier.
- Train the model using the training dataset with the `fit()` method.

### Step 5: Model Evaluation

Evaluate the trained model using:

- Accuracy Score
- Confusion Matrix

### Step 6: Model Preservation

- Save the trained Logistic Regression model using Joblib.
- Generated model filename: `MarvellousTitanic.pkl`.

## Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Test Size | 20% |
| Random State | 42 |
| Model Saving | Joblib |

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Keep the Files Together

Keep the dataset and Python file in the same directory.

### 3. Execute the Python File

```bash
python YourPythonFile.py
```

### 4. Generated Output

After successful execution:

- Model accuracy is displayed.
- Confusion matrix is displayed.
- Trained model is saved as `MarvellousTitanic.pkl`.

## Author

**Apurv Ranjane**