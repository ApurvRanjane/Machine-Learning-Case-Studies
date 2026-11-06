# Advertising Sales Prediction using Linear Regression

This project demonstrates a Machine Learning approach for predicting product sales based on advertising expenditure across different marketing channels using the Advertising dataset.

It uses **Linear Regression** to understand the relationship between advertising investments and sales.

It follows a structured Machine Learning workflow by:

- Loading and analyzing the Advertising dataset using Pandas.
- Removing the unnecessary `Unnamed: 0` column.
- Checking missing values and generating statistical summaries.
- Analyzing correlations between features.
- Splitting data into training and testing sets.
- Using Linear Regression for sales prediction.
- Evaluating model performance using MSE, RMSE and R2 Score.
- Analyzing model coefficients and intercept.

## Dependencies

Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

## Dataset Information

**Dataset:** Advertising.csv

**Dataset Shape:** 200 rows and 5 columns (including the index column).

**Features (3):**

1. TV
2. radio
3. newspaper

**Target:**

- sales – Product sales generated through advertising expenditure.

**Dataset Details:**

- Total Records: 200
- Total Columns: 5
- Missing Values: 0
- Unwanted Column: `Unnamed: 0` (removed during preprocessing).

## Workflow

### Data Preparation

- Load the Advertising dataset using Pandas.
- Remove the unwanted `Unnamed: 0` column if present.
- Check missing values.
- Generate statistical summaries using `describe()`.
- Calculate feature correlations using `corr()`.

### Train-Test Split

- Split the dataset into 80% training and 20% testing data.
- Use `random_state=42` for reproducibility.

### Model Construction

- **Algorithm:** Linear Regression
- Create a Linear Regression model using Scikit-learn.
- Train the model using advertising features and sales values.

### Model Training & Evaluation

The model is trained using the training dataset and makes predictions on the testing dataset.

**Evaluation Metrics:**

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the original target scale.
- **R2 Score:** Measures how well the model explains variations in sales.

### Coefficient Analysis

- Display TV advertising coefficient.
- Display Radio advertising coefficient.
- Display Newspaper advertising coefficient.
- Display the model intercept.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute the Python File

Keep `Advertising.csv` in the same folder as the Python file, then run:

```bash
python MarvellousAdvertising.py
```

### Expected Output

The program displays:

- First five dataset records.
- Missing value information.
- Statistical summary.
- Correlation matrix.
- Training and testing dataset shapes.
- Model training confirmation.
- First three expected and predicted sales values.
- MSE, RMSE and R2 Score.
- Regression coefficients and intercept.

*Actual numerical results are generated when the program is executed.*

## Model Evaluation

The model is evaluated using:

- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures prediction error in the original target scale.
- **R2 Score:** Measures how well the model explains variations in sales.

## Sample Prediction

The program generates predictions on the testing dataset:

```python
Y_pred = model.predict(X_test)

print("Expected Answers: ")
print(Y_test[:3])

print("Predicted answers: ")
print(Y_pred[:3])
```

## Author

**Apurv Ranjane**

**Project:** Advertising Sales Prediction using Linear Regression