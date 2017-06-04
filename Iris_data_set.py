#Building to Model the Iris dataset using Skitlearn

#importing libs
import numpy as np
import matplotlib.pyplot as mplt
import pandas as pd
from sklearn import datasets

#load iris data set - pre built
iris = datasets.load_iris()

print("Iris sample data set - 5 Rows printing")
print(iris.data[0:5,:])