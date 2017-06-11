# Using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.

# Author: Rishu Shrivastava (rishu.shrivastava@gmail.com)
# Date : June 4, 2017
# last updated : June 12, 2017

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print("Data set type: ",type(cancer))

#print(cancer.items())
#print(cancer.data)
#print(type(cancer['feature_names']))
#print(cancer['feature_names'][1])
#print("adding element to numpy ndarray", np.append(cancer['feature_names'],[['target']]))


def print_keys():
	# return keys
	return cancer.keys()

def len_dataset():
	# return length of dataset
	return len(cancer['feature_names'])
	
def create_dataframe():
	# create a dataframe using the dataset
	dframe = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
	dframe['target'] = pd.Series(data=cancer.target, index=dframe.index)
	
	print("shape of dataframe :",dframe.shape)
	return dframe

	
def cancer_instances():
	#return a series with the total number of malignant and benign instances
	cancerdf = create_dataframe()
	cancer_count = cancerdf['target'].value_counts()

	#print("Malignant : Benign count = ", cancer_count[0],":", cancer_count[1])
	
	dict= {'malignant': cancer_count[0], 'benign':cancer_count[1]}
	
	s = pd.Series(dict, index=['malignant', 'benign'])
	
	return s
	
	
def split_data():
	# split the data into tuples X and y 
	cancerdf = create_dataframe()
	
	X = cancerdf.ix[:,:30]
	y = cancerdf['target']
	
	return X, y

from sklearn.model_selection import train_test_split
def generate_test_train():
	# generate test train data set using the data frame
    X, y = split_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

	
from sklearn.neighbors import KNeighborsClassifier
def fit_knn_classifier():
    X_train, X_test, y_train, y_test = generate_test_train()
    
    # for KNN neighbors = 1
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    return knn


def accuracy_score():
	# calculate prediction score
    X_train, X_test, y_train, y_test = generate_test_train()
    knn = fit_knn_classifier()
    s = knn.score(X_test,y_test)
    return s

def calculate_mean_feature():
    cancerdf = create_dataframe()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
	knn = fit_knn_classifier()
	  
    pr = knn.predict(means)
    
    return pr


print(print_keys())
print(len_dataset())
#print(create_dataframe())
print(cancer_instances())
#print(split_data())
#print(generate_test_train())
print(fit_knn_classifier())
print(accuracy_score())

print(calculate_mean_feature())