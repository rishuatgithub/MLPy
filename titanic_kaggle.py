# Training Titanic dataset from kaggle to predict Survival accuracy

#author: Rishu Shrivastava
#last updated: 18 June 2017

# Help from: https://www.kaggle.com/omarelgabry/a-journey-through-titanic

# >>> Import packages and libs
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# >>> read the data files
titanic = pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/train.csv')
titanic_test= pd.read_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/test.csv')

# >>> preview the dataframe
#print(titanic.head())

# >>> get the info about the data
#print(titanic.info())
#print("---------------------")
#print(titanic_test.info())


# >>> Drop unnecessary columns that are not required for titanic dataset
titanic = titanic.drop(['PassengerId','Name','Ticket'], axis=1)
titanic_test = titanic_test.drop(['Name','Ticket'], axis=1)

# >>> analysing EMBARKED columns w.r.t Survived

# find the most occuring values
#print(titanic['Embarked'].value_counts())

def rebuild_embarked():
    titanic['Embarked'] = titanic['Embarked'].fillna("S")  #fill missing values with S (southhampton embarkment) as it is the most occuring
    titanic_test['Embarked'] = titanic_test['Embarked'].fillna("S")

def show_embarked_plot():
    #sns.factorplot('Embarked','Survived', data=titanic,size=4,aspect=3)  #plotting factorplot from seaborn
    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    sns.countplot(x='Embarked', data=titanic, ax=axis1)
    sns.countplot(x='Survived', hue="Embarked", data=titanic, order=[1,0], ax=axis2)
    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = titanic[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S','C','Q'], ax=axis3)
    sns.plt.show()

rebuild_embarked()
#show_embarked_plot() #calling plot function to display EMBARKED relationship


# >>> analysing FARE column

#fare_survived = titanic['Fare'][titanic['Survived']==1]
#fare_unsurvived = titanic['Fare'][titanic['Survived']==0]

#print(titanic["Fare"].median())
#print(titanic["Fare"].mean())

def rebuild_fare():
    titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
    titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)

def show_fare_plot():
    titanic['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))
    plt.show()

#show_fare_plot()
rebuild_fare()

# >>>> analysing AGE column

# find all the NULL age values
#print(titanic["Age"].isnull().sum()) #177 of null values

def show_age_plot():
    #titanic["Age"].plot(kind='hist', figsize=(15,3), bins=100, xlim=(0,50))
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
    axis1.set_title('Original Age values')
    axis2.set_title('Modified Age values')

    titanic['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

    mean_age = titanic["Age"].mean()
    stdev_age = titanic["Age"].std()
    null_age_fields = titanic["Age"].isnull().sum()
    generate_randm_age = np.random.randint(mean_age - stdev_age,mean_age + stdev_age, size = null_age_fields)
    titanic["Age"][np.isnan(titanic["Age"])] = generate_randm_age

    titanic['Age'].hist(bins=70, ax=axis2)
    plt.show()

# show_age_plot()
def rebuild_age():
    # generate random age between (mean - age) and (mean + age) for all the null age rows
    titanic['Age'].dropna().astype(int)
    mean_age = titanic["Age"].mean()
    stdev_age = titanic["Age"].std()
    null_age_fields = titanic["Age"].isnull().sum()
    generate_randm_age = np.random.randint(mean_age - stdev_age,mean_age + stdev_age, size = null_age_fields)
    titanic["Age"][np.isnan(titanic["Age"])] = generate_randm_age

    titanic_test['Age'].dropna().astype(int)
    mean_age_test = titanic_test["Age"].mean()
    stdev_age_test = titanic_test["Age"].std()
    null_age_fields_test = titanic_test["Age"].isnull().sum()
    generate_randm_age_test = np.random.randint(mean_age_test - stdev_age_test,mean_age_test + stdev_age_test, size = null_age_fields_test)
    titanic_test["Age"][np.isnan(titanic_test["Age"])] = generate_randm_age_test

rebuild_age()


# >>> analysing CABIN
# Dropping cabin values since most of the values are NULL. Not expecting impact
titanic = titanic.drop(['Cabin'], axis=1)
titanic_test = titanic_test.drop(['Cabin'], axis=1)


# >>> preprocessing of data to handle String values
def pre_process_data(data):
    le = preprocessing.LabelEncoder()
    data["Embarked"]=le.fit_transform(data.Embarked)
    data["Sex"]=le.fit_transform(data.Sex)
    data=data.fillna(-999)
    return data


# >>> split the data set to create cross validation set
from sklearn.model_selection import train_test_split
X = pre_process_data(titanic.drop(['Survived'], axis=1))
y = titanic["Survived"]
X_train, y_train, X_crossval, y_crossval = train_test_split(X, y, random_state=0)

print(X_crossval.info())
# >>> Train Data set
# generate the X and y values for the Train data set
#X_train = pre_process_data(titanic.drop(['Survived'], axis=1))
#y_train = titanic["Survived"]
X_test  = pre_process_data(titanic_test.drop(["PassengerId"], axis=1))

#def fit_knn_classifier():
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accuracy_score_knn = knn.score(X_crossval, y_crossval)
    print("Accuracy score (KNN) :", accuracy_score_knn)


#fit_knn_classifier()
def generate_csv():
    titanic.to_csv("C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/Sample.csv")
generate_csv()

def create_submission():
    submission = pd.DataFrame({
            "PassengerId": titanic_test["PassengerId"],
            "Survived": y_predict
            })
    submission.to_csv('C:/Users/Rishu/Documents/GitHub/MLPy/data/titanic/titanic_output.csv', index=False)

create_submission()
