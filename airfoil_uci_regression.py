# analysis of uci ml data set for regression - airfoil_uci_regression
# author: rishu shrivastava
# date: 23.07.2017

#imports
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
from sklearn import linear_model

sns.set(style='white')


airfoil = pd.read_csv('./data/airfoil_self_noise.csv')

#printing the first head of the airfoil dataset
print(airfoil.head())

# check if any missing or NaN values in the dataset
print(airfoil.isnull().sum())

#finding correlation between data set
print(airfoil.corr())

#plotting correlation matrix between dataset
def correlation_df(df):
    fig = plt.figure(figsize=(10,60))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Airfoil UCI Regression Correlation Chart')
    labels=['Frquency(Hz)','Angle_of_Attack','Chord_Length','Free_stream_velocity','Displacement','Sound_pressure_level']
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1.0,0,.75,.8,.85,.90,.95,1])
    plt.show()

#correlation_df(airfoil)
#print(airfoil.columns)

def correlation_df_seabrn(df):
    c = df.corr()
    sns.plt.title('Airfoil UCI Regression Correlation Chart - Heatmap')
    sns.heatmap(c)
    plt.yticks(rotation=0)
    sns.plt.show()

correlation_df_seabrn(airfoil)
