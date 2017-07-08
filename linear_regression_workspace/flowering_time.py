# Working on realtime flowering dataset using linear Regression
# Author: Rishu Shrivastava
# Date: 08.07.2017
# Copyright protected. Don't use this without permission

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# import dataset
flowering = pd.read_csv('../data/flowering_data/floweringtime_dataset.csv', sep=',')

#print(flowering.head())

flowering_X_train=flowering[['Year','Elevation','temparature']]
flowering_y_train=flowering['flowering_time']

print(flowering_X_train.head())

#plotting a column map to vizualize the factor
fig = plt.figure(figsize=(10,60))
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.95, top=0.95, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(311)
ax1.set_title('Year vs Flowering time')
ax1.plot(flowering_X_train['Year'], flowering_y_train, color='black')

ax2 = fig.add_subplot(312)
ax2.set_title('Elevation vs Flowering time')
ax2.plot(flowering_X_train['Elevation'], flowering_y_train, color='yellow')

ax3 = fig.add_subplot(313)
ax3.set_title('Temperature vs Flowering time')
ax3.plot(flowering_X_train['temparature'], flowering_y_train, color='red')

plt.show()

#split the data into test and train
