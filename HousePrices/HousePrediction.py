# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 22:05:47 2018

@author: Ariel

Practice of data preparation and backward elimination
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer

train_df = pd.read_csv("train.csv")
y = train_df.SalePrice

# dropping rows where there is no target
train_df.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
train_df = train_df.drop(['SalePrice'], axis = 1)

test_df = pd.read_csv("test.csv")
ID = test_df.Id

combine = [train_df, test_df]

"""
print(train_df.columns.values)
print(train_df.info())
print('-'*40)
print(test_df.info())
"""

# selecting numerical features
numeric_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']]

train_df = train_df[numeric_cols]
test_df = test_df[numeric_cols]
combine = [train_df, test_df]

#prints columns with nan values
train_df.columns[train_df.isnull().any()].tolist()

#print(train_df.columns)

train_names =  train_df.columns
test_names = test_df.columns

# Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
train_df = imputer.fit_transform(train_df)
test_df = imputer.transform(test_df)

train_df = pd.DataFrame(train_df, columns = train_names)
test_df = pd.DataFrame(test_df, columns = test_names)

train_df.shape[0]

import statsmodels.formula.api as sm
def backwardElimination(x, test, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    test = np.delete(test, j, 1)
    regressor_OLS.summary()
    return x, test
 
SL = 0.05

train_df = np.append(arr = np.ones((train_df.shape[0], 1)).astype(int), values = train_df, axis = 1)
test_df = np.append(arr = np.ones((test_df.shape[0], 1)).astype(int), values = test_df, axis = 1)

X_opt = train_df

train_df, test_df = backwardElimination(X_opt, test_df, SL)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_df, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = regressor.predict(test_df)

submission = pd.DataFrame({
        "Id": ID,
        "SalePrice": y_pred
    })
submission.to_csv('submission.csv', index=False)






