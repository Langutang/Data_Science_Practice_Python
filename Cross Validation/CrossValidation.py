# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:14:18 2020

@author: John Lang
"""

import os

os.chdir("C:\\Users\\John Lang\\Desktop\\007 - Classification")

import pandas as pd

data = pd.read_csv("04 - decisiontreeAdultIncome.csv")

data_prep = pd.get_dummies(data, drop_first=True)

X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma = 0.5)

from sklearn.model_selection import cross_validate
cv_result_dtc = cross_validate(dtc, X, Y, cv = 10, return_train_score = True)
cv_result_rfc = cross_validate(rfc, X, Y, cv = 10, return_train_score = True)
cv_result_svc = cross_validate(svc, X, Y, cv = 10, return_train_score = True)

# Get average of results
import numpy as np

dtc_test_average = np.average(cv_result_dtc['test_score'])
rfc_test_average = np.average(cv_result_rfc['test_score'])
svc_test_average = np.average(cv_result_svc['test_score'])

dtc_train_average = np.average(cv_result_dtc['train_score'])
rfc_train_average = np.average(cv_result_rfc['train_score'])
svc_train_average = np.average(cv_result_svc['train_score'])

# Analyze results

print(" ")
print(" ")
print(" ")

print("                 ","Decision Tree      ", " Random Forest      ", "Support Vector")
print("                 -------------    ------------     -------------")

print('Test   :  ',
      round(dtc_test_average, 4), '           ',
      round(rfc_test_average, 4), '           ',
      round(svc_test_average, 4))

print('Train   :  ',
      round(dtc_train_average, 4), '           ',
      round(rfc_train_average, 4), '           ',
      round(svc_train_average, 4))

# Generalize predictions in SVC since variance is smaller, and more accurate in test










