
# ########################################################
# Required Python Packages
# ########################################################

from sklearn import tree


# ########################################################
# Dataset Information
# ########################################################

# Independent Variables:
# Weight and Texture

# Dependent Variable:
# Ball Classification

# Testing Features:
# [35,1], [95,0]

# Testing Labels:
# 1, 2


# ########################################################
# Function name : main
# Description   : Performs Ball Classification using Decision Tree
# Input         : None
# Output        : Predicted classification results
# Author        : Apurv Ranjane
# ########################################################

def main():
    print("Ball Classification Case Study")

    Independent = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,1],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]]
    #Testing features = [35,1],[95,0]

    Dependent = [1,1,2,1,2,1,2,1,1,1,2,1,2]
    #Testing labels = 1,2

    model = tree.DecisionTreeClassifier()

    model = model.fit(Independent,Dependent)

    Result = model.predict([[35,1],[95,0]])
    print("Predicted result of model is:",Result)


# ########################################################
# Application Starter
# ########################################################

if __name__=="__main__":
    main()
