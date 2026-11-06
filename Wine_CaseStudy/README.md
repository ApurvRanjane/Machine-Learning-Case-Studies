# Wine Classification using K-Nearest Neighbors (KNN)

## Overview

This project implements a Machine Learning classification model using the K-Nearest Neighbors (KNN) algorithm to classify wine samples into different categories based on their chemical properties.

The application performs data loading, preprocessing, feature selection, dataset splitting, feature scaling, hyperparameter tuning, graphical representation, and model evaluation.

## Algorithm Used

- K-Nearest Neighbors (KNN) Classifier

## Technologies and Libraries

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset Information

**Dataset Name:** `WinePredictor.csv`

The dataset contains 178 records and 14 columns.

- Total Records: 178
- Total Columns: 14
- Input Features: 13
- Target Variable: `Class`
- Missing Values: 0

### Target Variable

The `Class` column represents three wine categories:

| Wine Class | Records |
|---|---:|
| Class 1 | 59 |
| Class 2 | 71 |
| Class 3 | 48 |

## Project Workflow

### Step 1: Load Dataset

- Load the CSV dataset using Pandas.
- Display the first five records.

### Step 2: Clean the Dataset

- Remove rows containing missing values using `dropna()`.
- Display dataset shape, total records, and total columns.

### Step 3: Separate Independent and Dependent Variables

- Select all columns except `Class` as independent features (X).
- Select `Class` as the dependent variable (Y).

### Step 4: Split Dataset

- Divide the dataset into training and testing subsets.
- Test Size: 50%
- Random State: 42
- Stratified splitting is used to preserve class proportions.

### Step 5: Feature Scaling

- Apply StandardScaler to standardize the input features.
- Fit the scaler on training data.
- Transform both training and testing data.

### Step 6: Hyperparameter Tuning

- Train KNN models using different K values.
- Evaluate K values from 1 to 20.
- Calculate testing accuracy for each model.
- Identify the K value that provides the highest accuracy.

### Step 7: Graphical Representation

- Plot K values against their corresponding accuracy scores.
- Display the graph using Matplotlib.

### Step 8: Model Evaluation

- Display the best K value.
- Display the best accuracy percentage.
- Generate and display the confusion matrix for the best-performing model.

## Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | K-Nearest Neighbors |
| Feature Scaling | StandardScaler |
| Test Size | 50% |
| Random State | 42 |
| K Values | 1 to 20 |
| Evaluation Metric | Accuracy Score |

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Keep the Files Together

Keep `WinePredictor.csv` and `WinePredictor.py` in the same directory.

### 3. Run the Application

```bash
python WinePredictor.py
```

### 4. Generated Output

The application displays:

- Dataset information
- Training and testing data shapes
- Best K value
- Best accuracy percentage
- K versus accuracy graph
- Confusion matrix

## Author

**Apurv Ranjane**