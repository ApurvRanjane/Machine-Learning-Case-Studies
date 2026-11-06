
# Required Python Packages

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ########################################################
# Function : main()
# Description : Perform Mall Customer Segmentation using K-Means
# Input : Mall_Customers.csv
# Output : Customer clusters and Elbow Method visualization
# Author : Apurv Ranjane
# ########################################################

def main():

    # ########################################################
    # Step 1 : Load the data
    # ########################################################

    df = pd.read_csv("Mall_Customers.csv")

    print("dataset loaded with values")
    print(df.head())

    print("Misiing values : ")
    print(df.isnull().sum())


    # ########################################################
    # Step 2 : Feature Selection
    # ########################################################

    X = df[["AnnualIncome", "SpendingScore"]]

    print("Selected features : ")
    print(X.head())


    # ########################################################
    # Step 3 : Scale the Data
    # ########################################################

    sclar = StandardScaler()

    X_scaled = sclar.fit_transform(X)

    print("Scaled Data : ")
    print(X_scaled[:5])


    # ########################################################
    # Step 4 : Elbow Method
    # ########################################################

    WCSS = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        WCSS.append(model.inertia_)

    print("Values of WCSS: ")

    for i in range(len(WCSS)):
        print(f"{i+1} : {WCSS[i]}")


    # ########################################################
    # Step 5 : Visualization
    # ########################################################

    plt.plot(range(1, 11), WCSS, marker="o")

    plt.xlabel("Number of clustures: k")
    plt.ylabel("WCSS")
    plt.title("Marvellous Elbow method")

    plt.grid(True)
    plt.show()


    # ########################################################
    # Step 6 : Final Model
    # ########################################################

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clustures = model.fit_predict(X_scaled)

    df["Clusture"] = clustures

    print("Dataset with clustures : ")
    print(df.head(20))


# ########################################################
# Application Starter
# ########################################################

if __name__ == "__main__":
    main()
