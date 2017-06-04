# Using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.

# Author: Rishu Shrivastava (rishu.shrivastava@gmail.com)
# Date : June 4, 2017

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print("Data set type: ",type(cancer))

#print(cancer.items())
print(cancer['feature_names'].add('target'))


def print_keys():
	# return keys
	return cancer.keys()

def len_dataset():
	# return length of dataset
	return len(cancer['feature_names'])
	
def create_dataframe():
	# create a dataframe using the dataset
	dframe = pd.DataFrame(data = np.c_[cancer['data'],cancer['target']],
                         index = cancer.data[0:,0:],
                         columns=np.append(cancer.feature_names,'target',1))
	return print(pd.DataFrame(dframe))

	
print(print_keys())
print(len_dataset())
#print(cancer.data[0:5,:])

#print(create_dataframe())
