# Employee Attrition Prediction using MLPClassifier

This project demonstrates a Machine Learning approach for predicting whether an employee is likely to stay in or leave an organization using historical employee data.

It uses a Multi-Layer Perceptron (MLP) Classifier, a neural network-based classification algorithm, to identify employee attrition patterns.

It follows a structured Machine Learning workflow by:

- Loading and analyzing the Employee Attrition dataset using Pandas.
- Checking missing values and examining dataset information.
- Converting categorical features into numerical representation.
- Separating independent and dependent variables.
- Splitting the dataset into training and testing data.
- Applying feature scaling using StandardScaler.
- Designing an MLP neural network with two hidden layers.
- Training the model using MLPClassifier.
- Calculating training and testing accuracy.
- Generating a confusion matrix.
- Plotting the model training loss curve.
- Predicting attrition for new employee records.
- Examining possible overfitting or underfitting.

## Dependencies

Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

## Dataset Information

**Dataset:** Employee_Attrition.csv

**Dataset Shape:** 1,000 rows and 11 columns.

**Features (10):**

1. Age
2. MonthlyIncome
3. YearsAtCompany
4. TotalWorkingYears
5. DistanceFromHome
6. JobSatisfaction
7. WorkLifeBalance
8. OverTime
9. NumCompaniesWorked
10. TrainingTimesLastYear

**Target:**

- Attrition – Indicates whether an employee leaves the organization.

**Target Values:**

- 0 – Employee stays.
- 1 – Employee leaves.

**Dataset Details:**

- Total Records: 1,000
- Total Input Features: 10
- Total Target Classes: 2
- Missing Values: 0
- Employees Staying: 792
- Employees Leaving: 208

## Workflow

### Data Preparation

- Load the Employee Attrition dataset using Pandas.
- Display dataset shape, column names and first five records.
- Check missing values using `isnull().sum()`.
- Convert OverTime and Attrition categorical columns into numerical values using LabelEncoder.

### Train-Test Split

- Split the dataset into 80% training and 20% testing data.
- Use `random_state=42` for reproducibility.

### Feature Scaling

- Apply StandardScaler to standardize input features.
- Fit the scaler using training data.
- Transform training and testing datasets using the fitted scaler.

### Model Construction

- **Algorithm:** Multi-Layer Perceptron Classifier (MLPClassifier)
- Hidden Layers: (16, 12)
- Activation Function: ReLU
- Solver: Adam
- Maximum Iterations: 3000
- Random State: 42

The neural network contains two hidden layers with 16 and 12 neurons respectively.

### Model Training & Evaluation

- Train the MLP model using scaled training data.
- Display the number of iterations required for training.
- Calculate training accuracy.
- Calculate testing accuracy.
- Generate a confusion matrix to evaluate classification performance.

### Loss Curve Visualization

- Plot the training loss curve using Matplotlib.
- Display the change in training loss across iterations.
- Observe the learning behavior of the neural network.

### Employee Attrition Prediction

Create a prediction function named `predictAttrition(employee_data)`.

The function:

- Accepts employee information as input.
- Applies the previously fitted StandardScaler.
- Predicts the employee's attrition status.
- Displays whether the employee is likely to stay or leave.

The system is tested using five new employee records.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute the Python File

Keep `Employee_Attrition.csv` in the same folder as the Python file, then run:

```bash
python EmployeeAttrition.py
```

### Expected Output

The program displays:

- Dataset shape and column names.
- First five dataset records.
- Missing value information.
- Independent and dependent variables.
- Training and testing dataset shapes.
- Scaled training data.
- Model training confirmation.
- Number of iterations required.
- Training accuracy.
- Testing accuracy.
- Confusion matrix.
- MLP training loss curve.
- Attrition predictions for employee records.

*Actual numerical results depend on model training and dataset execution.*

## Model Evaluation

The model is evaluated using:

- **Training Accuracy:** Measures the percentage of correctly classified training records.
- **Testing Accuracy:** Measures the percentage of correctly classified testing records.
- **Confusion Matrix:** Shows correct and incorrect classification results for both classes.
- **Loss Curve:** Shows how the model's training loss changes during training.

Training and testing accuracy can also be compared to understand whether the model may be suffering from overfitting or underfitting.

## Sample Prediction

The project predicts employee attrition using the following function:

```python
def predictAttrition(employee_data):

    employee_data_scaled = sclar.transform([employee_data])

    prediction = model.predict(employee_data_scaled)

    if prediction[0] == 1:
        print("Employee is likely to leave the company.")
    else:
        print("Employee is likely to stay in the company.")
```

## Author

**Apurv Ranjane**

**Project:** Employee Attrition Prediction using MLPClassifier