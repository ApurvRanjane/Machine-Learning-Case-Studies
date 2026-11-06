# Play Prediction using K-Nearest Neighbors (KNN)

## Overview

This project implements the K-Nearest Neighbors (KNN) classification algorithm to predict whether an activity can be played based on weather conditions and temperature.

The application trains a KNN classifier, accepts user input, predicts the result, and evaluates model accuracy using different values of K.

## Algorithm Used

- K-Nearest Neighbors (KNN) Classifier

## Technologies and Dependencies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset Information

**Dataset Name:** `PlayPredictor.csv`

The dataset contains 40 records and 4 original columns.

### Input Features

1. `Wether` – Weather conditions (Sunny, Rainy, Overcast)
2. `Temperature` – Temperature category (Hot, Mild, Cool)

### Target Variable

`Play`

- `Yes` – Activity can be played
- `No` – Activity cannot be played

### Dataset Summary

- Total Records: 40
- Original Columns: 4
- Input Features: 2
- Target Variable: Play
- Missing Values: 0
- Yes Records: 26
- No Records: 14

The `Unnamed: 0` column is removed during preprocessing.

## Project Workflow

1. Load the dataset using Pandas.
2. Display the initial records.
3. Remove missing values and the unnecessary index column.
4. Encode categorical features using LabelEncoder.
5. Separate independent and dependent variables.
6. Create a KNN Classifier with `n_neighbors=3`.
7. Train the classifier using the complete dataset.
8. Accept weather and temperature from the user.
9. Predict whether the activity can be played.
10. Split the dataset into training and testing sets using a 50:50 ratio.
11. Apply stratified splitting to preserve the target distribution.
12. Evaluate KNN accuracy for K values from 1 to 15.

## Model Training

The initial KNN model uses:

- Algorithm: K-Nearest Neighbors
- Number of Neighbors: 3
- Training Data: Complete dataset

For accuracy evaluation, the dataset is divided into:

- Training Data: 50%
- Testing Data: 50%
- Random State: 42
- Stratified Splitting: Enabled

## Sample Prediction

The application accepts weather and temperature values.

Example:

```text
Enter Weather : Sunny
Enter Temperature : Hot
```

The trained classifier displays a predicted result of `Yes` or `No`.

## Model Evaluation

The `AccuracyChecker(K)` function trains and evaluates a KNN classifier for a given value of K.

The application evaluates:

- K = 1
- K = 2
- K = 3
- ...
- K = 15

The accuracy percentage for each value is printed to the console.

## How to Run the Project

1. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Keep the dataset and Python file in the same directory.

3. Run the application.

   ```bash
   python PlayPredictor.py
   ```

4. Enter valid weather and temperature values when prompted.

5. Review the prediction and accuracy results.

## Author

**Apurv Ranjane**