# Working on realtime flowering dataset using linear Regression
# Author: Rishu Shrivastava
# Date: 08.07.2017
# Copyright protected. Don't use this without permission

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm as cm
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# import dataset
flowering = pd.read_csv('../data/flowering_data/floweringtime_dataset.csv', sep=',')

flowering_df=flowering[['Year','Elevation','temparature','flowering_time']]

# printing the flowering head sample
print(">>> Printing the head of the dataframe")
print(flowering_df.head())

# finding out if the flowering time is null or not
print(">>> If the flowering dataframe is null or not")
print(flowering_df.isnull().sum())

flowering_df['temparature'].dropna().astype(int);
mean_temparature_train=flowering['temparature'].mean();
flowering_df['temparature'][np.isnan(flowering_df['temparature'])] = mean_temparature_train

print(mean_temparature_train)
print(flowering_df.head())

def correlation_df(df):
    c = df.corr()
    sns.plt.title('Flowering time')
    sns.heatmap(c)
    plt.yticks(rotation=0)
    sns.plt.show()

correlation_df(flowering_df)

#plotting a column map to vizualize the factor
def visualize_data():
    fig = plt.figure(figsize=(10,60))
    fig.subplots_adjust(left=0.06, bottom=0.07, right=0.95, top=0.95, wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(311)
    ax1.set_title('Year vs Flowering time')
    ax1.plot(flowering_df['Year'], flowering_df['temparature'], color='black')

    ax2 = fig.add_subplot(312)
    ax2.set_title('Elevation vs Flowering time')
    ax2.plot(flowering_df['Elevation'], flowering_df['temparature'], color='yellow')

    ax3 = fig.add_subplot(313)
    ax3.set_title('Temperature vs Flowering time')
    ax3.plot(flowering_df['temparature'], flowering_df['temparature'], color='red')
    plt.show()

visualize_data()


# defining data and targets
flowering_X = flowering_df[['Year']]
flowering_y = flowering_df['flowering_time']

#split the dataframe into test and train
flowering_X_train, flowering_X_test, flowering_y_train, flowering_y_test = train_test_split(flowering_X, flowering_y, random_state=0)

#print(len(flowering_X_train)," ",len(flowering_y_train)," ",len(flowering_X_test)," ",len(flowering_y_test))

# training Linear regression to the train data set
regr = linear_model.LinearRegression()
regr.fit(flowering_X_train, flowering_y_train)

#printing coefficients
print('Coefficients: \n', regr.coef_)
#print("Prediction: %.2f",regr.predict(flowering_X_test))
# mean squared error
print('Mean squared error : %.2f',np.mean((regr.predict(flowering_X_test) - flowering_y_test) ** 2))
print("Score : %.2f",regr.score(flowering_X_test, flowering_y_test) * 100)

# plotting the dataset
plt.scatter(flowering_X_test,flowering_y_test, color='black')
plt.plot(flowering_X_test, regr.predict(flowering_X_test), color='blue',linewidth=3)
plt.show()
