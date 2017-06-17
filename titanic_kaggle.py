# Training Titanic dataset from kaggle to predict Survival accuracy

#author: Rishu Shrivastava
#last updated: 18 June 2017

# Help from: https://www.kaggle.com/omarelgabry/a-journey-through-titanic

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

#read the data files
titanic = pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/train.csv')
titanic_test= pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/test.csv')

#preview the dataframe
#print(titanic.head())

#get the info about the data
#print(titanic.info())
#print("--------------------------------------------")
#print(titanic_test.info())


# Drop unnecessary columns that are not required for titanic dataset
titanic = titanic.drop(['PassengerId','Name','Ticket'], axis=1)
titanic_test = titanic_test.drop(['Name','Ticket'], axis=1)

# analysing EMBARKED columns w.r.t Survived

# find the most occuring values
#print(titanic['Embarked'].value_counts())
titanic['Embarked'] = titanic['Embarked'].fillna("S")  #fill missing values with S (southhampton embarkment) as it is the most occuring

def show_embarked_plot():
    #sns.factorplot('Embarked','Survived', data=titanic,size=4,aspect=3)  #plotting factorplot from seaborn
    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    sns.countplot(x='Embarked', data=titanic, ax=axis1)
    sns.countplot(x='Survived', hue="Embarked", data=titanic, order=[1,0], ax=axis2)
    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = titanic[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S','C','Q'], ax=axis3)
    sns.plt.show()

#show_embarked_plot() #calling plot function to display EMBARKED relationship


# analysing FARE column w.r.t Survived

fare_survived = titanic['Fare'][titanic['Survived']==1]
fare_unsurvived = titanic['Fare'][titanic['Survived']==0]

titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)

def show_fare_plot():
    titanic['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))
    plt.show()

#show_fare_plot()
