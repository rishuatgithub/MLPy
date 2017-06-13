# Training Titanic dataset from kaggle to predict Survival accuracy

#author: Rishu Shrivastava
#last updated: 12 June 2017

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/train.csv')
titanic_test= pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/test.csv')

#print("Data set type: ",type(titanic))
#print("feature list: ", titanic.keys())

X_train = titanic.ix[:,2:12]
y_train = titanic['Survived']


#print(X.shape)

	
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

s = knn.score(X_test,y_test)

print("Score :",s)
