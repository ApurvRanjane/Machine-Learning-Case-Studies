# Mall Customer Segmentation using K-Means Clustering

## Overview

This project performs customer segmentation using the K-Means Clustering algorithm.

The objective is to group mall customers based on their annual income and spending score. The Elbow Method is used to analyze the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters.

## Algorithm Used

- K-Means Clustering

## Technologies and Dependencies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset Information

**Dataset Name:** `Mall_Customers.csv`

The dataset contains 100 records and 4 columns.

### Dataset Columns

1. CustomerID
2. Age
3. AnnualIncome
4. SpendingScore

### Selected Features

- AnnualIncome
- SpendingScore

### Dataset Summary

- Total Records: 100
- Total Columns: 4
- Selected Features: 2
- Missing Values: 0

## Project Workflow

1. Load the Mall Customers dataset using Pandas.
2. Display the first five records.
3. Check for missing values.
4. Select AnnualIncome and SpendingScore as input features.
5. Scale the selected features using StandardScaler.
6. Apply the Elbow Method.
7. Calculate WCSS for cluster values from 1 to 10.
8. Display WCSS values.
9. Visualize the Elbow Method using Matplotlib.
10. Build the final K-Means model.
11. Assign cluster labels to customers.
12. Display the dataset with cluster assignments.

## Elbow Method

The Elbow Method helps determine a suitable number of clusters by analyzing WCSS values.

- WCSS stands for Within-Cluster Sum of Squares.
- Lower WCSS indicates that data points are closer to their assigned cluster centers.
- The elbow point helps identify a suitable cluster count.

## Model Output

The project generates:

- WCSS values for different cluster counts.
- An Elbow Method graph.
- Customer records with their assigned cluster labels.

## How to Run the Project

1. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Keep the dataset and Python file in the same project directory.

3. Run the Python file.

   ```bash
   python MallCustomerSegmentation.py
   ```

4. Review the Elbow Method graph and cluster assignments.

## Author

**Apurv Ranjane**