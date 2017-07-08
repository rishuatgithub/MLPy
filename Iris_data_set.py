#Building to Model the Iris dataset using Skitlearn

#importing libs
import numpy as np
import matplotlib.pyplot as mplt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#load iris data set - pre built
iris = datasets.load_iris()

print("Iris sample data set - 5 Rows printing")
print(iris.keys())
print(iris.feature_names[:])
print(iris.data[:5])
print(iris.target[:5])

X = iris.data[:]
y = iris.target[:]

#spliting the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train using SGD Classifier
sgd_clf = linear_model.SGDClassifier()
sgd_clf.fit(X_train, y_train)

# train using KNNeighbour
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=8)
knn_clf.fit(X_train,y_train)

print('Score (SGD): %.2f',sgd_clf.score(X_test,y_test) * 100)
print('Score (SGD): %.2f',knn_clf.score(X_test,y_test) * 100)
