## predict iris dataset

## imports
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import neptune
import os
from dotenv import load_dotenv
load_dotenv()

## setup neptune account
NEPTUNE_API_KEY=os.getenv('NEPTUNE_API_TOKEN')
neptune.init(project_qualified_name='rishushrivastava/sandbox', api_token=NEPTUNE_API_KEY)

## create an neptune experiment
neptune.create_experiment()

## load the data set
iris = datasets.load_iris()

## pre-processing and train/test split
X = iris.data[:]
y = iris.target[:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## train and fit the model - KNN
knn_clf = KNeighborsClassifier(n_neighbors=8)
knn_clf.fit(X_train,y_train)

neptune.log_metric('Training Score :',knn_clf.score(X_test,y_test)*100)

## stop the execution
neptune.stop()


