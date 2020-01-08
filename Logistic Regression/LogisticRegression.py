# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:49:09 2020

@author: John Lang
"""
import pandas as pd

LoanData = pd.read_csv("01Exercise1.csv")

LoanPrep = LoanData.copy()

LoanPrep.isnull().sum(axis=0)

LoanPrep = LoanPrep.dropna()

# Now Loaded data with no missing values

LoanPrep = LoanPrep.drop(['gender'], axis=1)

LoanPrep.dtypes

#Get Dummies
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)

#Normalise the data
from sklearn.preprocessing import StandardScaler
scaler_=StandardScaler()

LoanPrep['income'] = scaler_.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scaler_.fit_transform(LoanPrep[['loanamt']])

Y = LoanPrep[['status_Y']]
X = LoanPrep.drop(['status_Y'], axis=1)

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_predict)

lr.score(X_test, Y_test)

