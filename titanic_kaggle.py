# Training Titanic dataset from kaggle to predict Survival accuracy

#author: Rishu Shrivastava
#last updated: 12 June 2017

# Help from: https://www.kaggle.com/omarelgabry/a-journey-through-titanic

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

titanic = pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/train.csv')
titanic_test= pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/test.csv')

#preview the dataframe
print(titanic.head())

#get the info about the data
print(titanic.info())
print("--------------------------------------------")
print(titanic_test.info())


# Drop unnecessary columns that are not required for titanic dataset
titanic = titanic.drop(['PassengerId','Name','Ticket','Embarked'], axis=1)
titanic_test = titanic_test.drop(['Name','Ticket','Embarked'], axis=1)

