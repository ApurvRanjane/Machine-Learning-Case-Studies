# Customer Loan Approval using Voting Classifier

This project demonstrates a Machine Learning approach for predicting whether a customer's loan application should be approved or rejected using a Voting Classification algorithm.

It combines three different Machine Learning algorithms to make predictions through Hard Voting and Soft Voting techniques.

It follows a structured Machine Learning workflow by:

- Loading and analyzing the Customer Loan Approval dataset.
- Checking missing values in the dataset.
- Separating independent and dependent variables.
- Splitting the dataset into training and testing data.
- Scaling features using StandardScaler.
- Training Logistic Regression, Decision Tree and K-Nearest Neighbors models.
- Evaluating individual model performance using Accuracy Score.
- Creating Hard Voting and Soft Voting Classifiers.
- Comparing the accuracy of individual and ensemble models.

## Dependencies

Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

## Dataset Information

**Dataset:** Customer_Loan_Approval.csv

**Dataset Shape:** 50 rows and 7 columns.

**Features (6):**

1. Age
2. Income
3. CreditScore
4. ExistingLoan
5. EmploymentExperience
6. LoanAmount

**Target:**

- LoanApproved – Indicates whether the customer loan is approved or rejected.

**Target Values:**

- 0 – Loan Rejected
- 1 – Loan Approved

**Dataset Details:**

- Total Records: 50
- Total Input Features: 6
- Total Target Classes: 2
- Missing Values: 0

## Workflow

### Data Preparation

- Load the Customer Loan Approval dataset using Pandas.
- Check dataset dimensions using `shape`.
- Display the first five records using `head()`.
- Check missing values using `isnull().sum()`.
- Separate independent variables and dependent variables.

### Train-Test Split

- Split the dataset into 80% training and 20% testing data.
- Use `random_state=42` for reproducibility.

### Feature Scaling

- Apply StandardScaler to normalize the input features.
- Fit the scaler using training data.
- Transform both training and testing data.

### Model Construction

Three individual classification algorithms are used:

**1. Logistic Regression**

- Algorithm: LogisticRegression
- Maximum iterations: 1000

**2. Decision Tree**

- Algorithm: DecisionTreeClassifier
- Random State: 42

**3. K-Nearest Neighbors**

- Algorithm: KNeighborsClassifier
- Number of Neighbors: 5

### Model Training & Evaluation

- Train all three individual classification models.
- Generate predictions using testing data.
- Calculate individual model accuracy using Accuracy Score.

### Voting Classifier

Two ensemble learning techniques are implemented:

**1. Hard Voting**

- Combines predictions from multiple classifiers.
- Selects the class receiving the majority of votes.

**2. Soft Voting**

- Combines predicted class probabilities from multiple classifiers.
- Selects the class with the highest average predicted probability.

### Model Comparison

The accuracy of all five models is calculated and displayed:

| Model | Evaluation |
|---|---|
| Logistic Regression | Accuracy Score |
| Decision Tree | Accuracy Score |
| KNN | Accuracy Score |
| Hard Voting | Accuracy Score |
| Soft Voting | Accuracy Score |

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute the Python File

Keep `Customer_Loan_Approval.csv` in the same folder as the Python file, then run:

```bash
python CustomerLoanApproval.py
```

### Expected Output

The program displays:

- Dataset shape.
- First five dataset records.
- Missing value information.
- Training and testing dataset shapes.
- Logistic Regression accuracy.
- Decision Tree accuracy.
- KNN accuracy.
- Hard Voting accuracy.
- Soft Voting accuracy.

*Actual numerical results depend on the dataset and model predictions.*

## Model Evaluation

The models are evaluated using Accuracy Score.

Accuracy measures the percentage of correctly classified loan applications out of the total predictions.

The individual classification models and both voting techniques are compared to understand their predictive performance.

## Author

**Apurv Ranjane**

**Project:** Customer Loan Approval using Voting Classifier