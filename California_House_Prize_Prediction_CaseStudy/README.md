# California Housing Price Prediction using Machine Learning

## Overview
This case study focuses on predicting California housing prices using machine learning regression algorithms.

The project implements three regression approaches:
- Individual Decision Tree Regression
- Ensemble Bagging Regression
- Ensemble Gradient Boosting Regression

All three models use the same dataset and follow a common training and evaluation workflow.

## Dataset Information

**Dataset Name:** `california_housing.csv`

- Total Records: 20,640
- Total Columns: 9
- Input Features: 8
- Target Column: `target`
- Missing Values: 0

### Features

| Feature | Description |
|---|---|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average rooms |
| AveBedrms | Average bedrooms |
| Population | Population |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| target | Median house value (target variable) |

## Algorithms Implemented

### 1. Decision Tree Regression
- Uses `DecisionTreeRegressor`.
- Builds a tree-based regression model.
- Predicts housing values using the input features.
- Uses `random_state=42`.

### 2. Bagging Regression
- Uses `BaggingRegressor`.
- Uses `DecisionTreeRegressor` as the base estimator.
- Creates 10 estimators.
- Combines predictions from multiple models to improve prediction stability.

### 3. Gradient Boosting Regression
- Uses `GradientBoostingRegressor`.
- Builds models sequentially, with each model learning from the errors of previous models.
- Parameters:
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=3`
  - `random_state=42`

## Project Workflow

1. Load the California housing dataset using Pandas.
2. Separate input features and target variable.
3. Split the dataset into training and testing sets.
4. Train the selected regression algorithm.
5. Predict housing values using test data.
6. Evaluate model performance.

### Train-Test Split
- Training Data: 80%
- Testing Data: 20%
- Random State: 42

## Model Evaluation

The models are evaluated using:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Lower values indicate better performance.
- **R² Score:** Measures how well the model explains variations in the target values. A higher score generally indicates better performance.

The scripts print the MSE and R² score for each model.

## Technologies Used

- Python
- Pandas
- Scikit-learn

## How to Run

1. Install the required dependencies:

   `pip install -r requirements.txt`

2. Keep the dataset and Python files in the same folder.
3. Run any of the following scripts:

   - `Individual_California_DT.py`
   - `Ensembel_California_Bagging.py`
   - `Ensembel_California_Boosting.py`

Example:

`python Individual_California_DT.py`

## Folder Structure

```text
California_Housing/
│
├── california_housing.csv
├── Individual_California_DT.py
├── Ensembel_California_Bagging.py
├── Ensembel_California_Boosting.py
├── README.md
└── requirements.txt
```

## Author

Apurv Ranjane