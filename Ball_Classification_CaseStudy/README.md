# Ball Classification using Decision Tree Classifier

This project demonstrates a basic Machine Learning approach for classifying balls into different categories using a Decision Tree Classification algorithm.

It uses manually created training data to understand how a Decision Tree learns patterns from input features and predicts the class of new data.

It follows a basic Machine Learning workflow by:

- Creating independent and dependent variables manually.
- Understanding input features and target labels.
- Creating a Decision Tree Classifier using Scikit-learn.
- Training the model using the provided dataset.
- Predicting classifications for new input samples.

## Dependencies

Install the required Python package before running the project:

```bash
pip install -r requirements.txt
```

## Dataset Information

**Dataset:** Manually created dataset (no external CSV file required).

**Independent Variables:**

- Weight
- Texture (numerically encoded)

**Dependent Variable:**

- Ball Classification (Class 1 and Class 2)

**Dataset Details:**

- Total Training Samples: 13
- Total Input Features: 2
- Total Target Classes: 2

**Testing Features:**

- [35, 1]
- [95, 0]

## Workflow

### Data Preparation

- Create independent variables manually.
- Define dependent variables as classification labels.
- Prepare input features and corresponding output classes.

### Model Construction

- **Algorithm:** Decision Tree Classifier
- Import the Decision Tree module from Scikit-learn.
- Create the classification model using `DecisionTreeClassifier()`.

### Model Training

- Train the Decision Tree model using `fit()`.
- The model learns classification patterns from the training samples.

### Model Prediction

- Predict the class of new input samples using `predict()`.
- Test the model using `[35,1]` and `[95,0]`.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute the Python File

```bash
python BallClassification.py
```

### Expected Output

```text
Ball Classification Case Study
Predicted result of model is: [1 2]
```

## Model Evaluation

This project focuses on understanding the basic working of Decision Tree classification.

It demonstrates:

- Training a classification model.
- Learning patterns from manually provided data.
- Predicting output classes for new input samples.

No separate accuracy calculation or testing dataset evaluation is implemented in the current code.

## Sample Prediction

The model predicts classifications for two new input samples:

```python
Result = model.predict([[35,1],[95,0]])

print("Predicted result of model is:",Result)
```

## Author

**Apurv Ranjane**

**Project:** Ball Classification using Decision Tree Classifier