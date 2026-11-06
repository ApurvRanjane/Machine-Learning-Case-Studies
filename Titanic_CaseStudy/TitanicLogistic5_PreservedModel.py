import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#Step 1: Load Data

#-------------------------------------------------------------------------
#  Function Name : LoadData
#  Description : Load the data from CSV
#  Input : Name of CSV file
#  Output : Data Frame
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset loads successfully")
    print(df.head())

    return df

#Step 2: Data Preprocessing

#-------------------------------------------------------------------------
#  Function Name : PreProcessData
#  Description : It performs Data Analysis
#  Input : Data Frame
#  Output : Updated Data Frame
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def PreProcessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],errors="ignore"
    )

    #Handles missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    #Conver Categorical to numeric data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )
    
    print(df.head())
    print("Data Pre Processing Completed...")
    return df

#step 3:Splitting data

#-------------------------------------------------------------------------
#  Function Name : SplitData
#  Description : It performs splitting activity
#  Input : Data Frame
#  Output : 4 subsets for training and testing
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def SplitData(df):
    X = df.drop("Survived",axis=1)
    Y = df["Survived"]

    X_train,X_test,Y_train,Y_test = train_test_split(X,
    Y,
    test_size=0.2,
    random_state=42)

    print("Dataset Splitting Completed Successfully..")
    return X_train,X_test,Y_train,Y_test

#step 4: Train Model
#-------------------------------------------------------------------------
#  Function Name : TrainModel
#  Description : It performs model training
#  Input : Training features and labels
#  Output : Trained model
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def TrainModel(X_train,Y_train):
    model = LogisticRegression()

    model = model.fit(X_train,Y_train)

    print("MOdel trained successfully..")

    return model

#step 5: Evaluate Model
#-------------------------------------------------------------------------
#  Function Name : EvaluateModel
#  Description : It performs model testing
#  Input : model,testing data(features,labels)
#  Output : none
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def EvaluateModel(model,X_test,Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy is: ",accuracy)
    print(confusion_matrix(Y_test,Y_pred))

#step 6: Preserve Model
#-------------------------------------------------------------------------
#  Function Name : PreserveModel
#  Description : It performs model preservation into .pkl file
#  Input : model
#  Output : none
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model Preserved with name: ",filename)

#-------------------------------------------------------------------------
#  Function Name : main
#  Description : Entry point function
#  Input : None
#  Output : None
#  Author : Apurv Ranjane
#  Date : 16-8-2026
#-------------------------------------------------------------------------

def main():
    #step1
    df = LoadData("MarvellousTitanicDataset.csv")

    #step2
    df =PreProcessData(df)

    #step3
    X_train,X_test,Y_train,Y_test = SplitData(df)

    #step4
    model = TrainModel(X_train,Y_train)

    #step5
    EvaluateModel(model,X_test,Y_test)

    #step6
    PreserveModel(model,"MarvellousTitanic.pkl")


if __name__=="__main__":
    main()