# Student Performance Prediction using Decision Tree Classifier

## Overview

This project uses a Decision Tree Classification algorithm to predict student performance based on study hours, attendance, previous scores, assignments completed, and sleep hours.

The application trains a classification model, predicts results for testing data, calculates accuracy, generates a confusion matrix, compares different tree depths, and predicts the result for a new student.

## Algorithm Used

- Decision Tree Classifier

## Technologies and Dependencies

- Python
- Pandas
- Scikit-learn

## Dataset Information

**Dataset Name:** `student_performance_ml.csv`

The dataset contains 30 records and 6 columns.

### Input Features

1. `StudyHours` – Hours spent studying.
2. `Attendance` – Student attendance.
3. `PreviousScore` – Previous academic score.
4. `AssignmentsCompleted` – Number of completed assignments.
5. `SleepHours` – Hours of sleep.

### Target Variable

`FinalResult`

- `1` – Pass
- `0` – Fail

### Dataset Summary

- Total Records: 30
- Total Columns: 6
- Input Features: 5
- Target Variable: FinalResult
- Missing Values: 0
- Pass Records: 18
- Fail Records: 12

## Project Workflow

1. Load the student performance dataset using Pandas.
2. Select independent and dependent variables.
3. Split the dataset into training and testing sets.
4. Create a Decision Tree Classifier with `max_depth=5`.
5. Train the model using the `fit()` method.
6. Predict results for testing data.
7. Calculate model accuracy.
8. Generate a confusion matrix.
9. Calculate training and testing accuracy.
10. Compare Decision Tree models with different maximum depths.
11. Predict the result for a new student.

## Model Configuration

Initial model:

- Algorithm: Decision Tree Classifier
- Maximum Depth: 5
- Test Size: 20%
- Random State: 42

## Model Experiments

The project compares three Decision Tree configurations:

| Experiment | Maximum Depth |
|---|---|
| Model 1 | 1 |
| Model 2 | 3 |
| Model 3 | None |

The testing accuracy is calculated for each configuration.

## New Student Prediction

The application predicts the result for a student with the following details:

| Feature | Value |
|---|---:|
| StudyHours | 6 |
| Attendance | 85 |
| PreviousScore | 66 |
| AssignmentsCompleted | 7 |
| SleepHours | 7 |

The model displays whether the student is predicted to pass or fail.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Keep the dataset and Python file in the same directory.

Run the application:

```bash
python StudentPerformance.py
```

## Author

**Apurv Ranjane**