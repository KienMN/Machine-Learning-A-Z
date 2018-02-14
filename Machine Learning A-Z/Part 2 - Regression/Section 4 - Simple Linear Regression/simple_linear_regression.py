# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regresstion to Training set
from sklearn.linear_model import LinearRegression
regresstor = LinearRegression()
regresstor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regresstor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regresstor.predict(X_train), color = 'blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Training set)')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Test set)')
plt.show()