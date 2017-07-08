# Linear Regression with diabetics dataset (in Scikit learn)
# Author: Rishu Shrivastava
# Date: 08.07.2017
# Practice example

#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

print(diabetes.keys())

# use only one feature
# np.newaxis is for basic slicing of the datbaset
diabetes_X=diabetes.data[:,np.newaxis,2]
print(len(diabetes_X))

#split the dataset into train and test
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

print(len(diabetes_X_train), len(diabetes_X_test))

# split the target dataset into train and test
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

print(len(diabetes_y_train), len(diabetes_y_test))

# Run the Liner model against the train dataset
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
print("Prediction: %.2f",regr.predict(diabetes_X_test))
# The mean squared error
print('Mean squared error: %.2f',np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f',regr.score(diabetes_X_test, diabetes_y_test) * 100)


# plotting the dataset
plt.scatter(diabetes_X_test,diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)
#to remove the axes
#plt.xticks(())
#plt.yticks(())
plt.show()
