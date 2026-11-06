# Iris Flower Classification using Decision Tree

## Overview

This project uses a Decision Tree Classifier to classify Iris flowers into three species based on their sepal and petal measurements.

The project includes Exploratory Data Analysis (EDA), data visualization, training and testing, and model performance evaluation.

## Algorithm Used

- Decision Tree Classifier

## Technologies and Dependencies

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset Information

**Dataset Name:** `iris.csv`

The dataset contains 150 records and 5 columns.

### Input Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Target Variable

`species`

The target contains three Iris flower species:

- Setosa
- Versicolor
- Virginica

### Dataset Summary

- Total Records: 150
- Total Columns: 5
- Input Features: 4
- Target Variable: species
- Missing Values: 0
- Setosa: 50 records
- Versicolor: 50 records
- Virginica: 50 records

## Project Workflow

1. Load the Iris dataset using Pandas.
2. Perform Exploratory Data Analysis (EDA).
3. Check dataset shape, column names, missing values, and class distribution.
4. Generate a statistical summary of the dataset.
5. Select independent and dependent variables.
6. Visualize flower measurements using a scatter plot.
7. Split the dataset into training and testing sets (50:50).
8. Build a Decision Tree Classifier with `max_depth=5`.
9. Train the model using training data.
10. Predict flower species using testing data.
11. Evaluate the model using accuracy, confusion matrix, and classification report.

## Model Evaluation

The model is evaluated using:

- **Accuracy:** Measures the percentage of correct predictions.
- **Confusion Matrix:** Displays correct and incorrect classifications for each species.
- **Classification Report:** Provides precision, recall, and F1 score for each species.

## How to Run the Project

1. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Keep the dataset and Python file in the same project directory.

3. Run the Python file.

   ```bash
   python IrisDecisionTree.py
   ```

4. Review the scatter plot and model evaluation results.

## Author

**Apurv Ranjane**