# Using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.

# Author: Rishu Shrivastava (rishu.shrivastava@gmail.com)
# Date : June 4, 2017

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

	#print("Malignant : Benign count = ", cancer_count[1],":", cancer_count[0])
	
	dict= {'malignant': cancer_count[1], 'benign':cancer_count[0]}
	
	s = pd.Series(dict, index=['malignant', 'benign'])
	
	return s
	
	
def split_data():
	# split the data into tuples X and y 
	cancerdf = create_dataframe()
	
	X = cancerdf[1::30]
	y = cancerdf['target']
	
	return X, y


	
print(print_keys())
print(len_dataset())
#print(create_dataframe())
print(cancer_instances())

#print(split_data())