#Building to Model to predict fruits data with colors using SKlearn and KNN

# Author: Rishu Shrivastava (rishu.shrivastava@gmail.com)
# Date : June 4, 2017

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.neighbors import KNeighborsClassifier

#reading data
fruits = pd.read_table('./data/fruit_data_with_colors.txt')

print("Displaying sample rows of Flower data set")
print(fruits.head())

#create a mapping from fruit label value to fruit name to make results easier to interpret
print("Lookup fruit names to make it easier to interpret the prediction")
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

#plotting scatter matrix
X = fruits[['height', 'width', 'mass']]
y = fruits['fruit_label']

#creating a train and test data set. Split it in 75%/25%
print("Generating train and test dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
mp.show()

# Training the Dataset using KNN algorithm | neighbours=5
print("Training KNNeighbour Classifier")
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

print("The ACCURACY score = ",knn.score(X_test,y_test))

# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print("PREDICTING fruit with mass 20g, width 4.3 cm, height 5.5 cm : ",lookup_fruit_name[fruit_prediction[0]])
