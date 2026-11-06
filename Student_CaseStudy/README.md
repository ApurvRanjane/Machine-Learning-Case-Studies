# Student Marks Analysis using Pandas and Matplotlib

## Overview

This project demonstrates basic data analysis and visualization operations using Python libraries. It creates a student marks dataset and performs normalization, categorical encoding, grouping, calculations, CSV export, and graphical visualization.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset Information

The dataset is created directly in the Python program.

| Name | Math | Science | English |
|---|---:|---:|---:|
| Amit | 85 | 92 | 75 |
| Sagar | 90 | 88 | 85 |
| Pooja | 78 | 80 | 82 |

A Gender column is also added during execution.

## Operations Performed

1. Create a student dataset using Pandas.
2. Normalize Mathematics marks using MinMaxScaler.
3. Encode the Gender column using `pd.get_dummies()`.
4. Group students by gender and calculate average subject marks.
5. Display Sagar's subject marks using a pie chart.
6. Calculate total marks and determine Pass/Fail status.
7. Count the number of passed students.
8. Export student data into `Student_Data.csv`.
9. Plot a histogram of Mathematics marks.
10. Rename the Math column to Mathematics.
11. Plot a boxplot of English marks to examine distribution and potential outliers.

## Visualizations

- Pie Chart: Subject-wise marks of Sagar.
- Histogram: Distribution of Mathematics marks.
- Boxplot: Distribution and potential outliers in English marks.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Python program:

```bash
python StudentMarksAnalysis.py
```

The program displays the analysis results and visualizations. It also generates `Student_Data.csv`.

## Author

**Apurv Ranjane**