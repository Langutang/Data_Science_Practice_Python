# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:33:58 2020

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

from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=1234)




#Model Selection
from sklearn.model_selection import GridSearchCV
rfc_param = {'n_estimators':[10, 15, 20],
             'min_samples_split' : [8,16],
             'min_samples_leaf': [1, 2, 3, 4, 5],
             }

# 3 estimators by 5 splites, by 2 leafs = 30 combinations

rfc_grid = GridSearchCV(estimator=rfc, 
                        param_grid = rfc_param,
                        scoring = 'accuracy',
                        cv = 10,
                        return_train_score=True)

#How many jobs will be executed? - 300 jobs. 30 combinations x 10 folds

rfc_grid_fit = rfc_grid.fit(X, Y)

cv_results_rfc = pd.DataFrame.from_dict(rfc_grid_fit.cv_results_)



# LOGISTIC REGRESSION GridSearchCV
lrc_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10],
             'penalty' : ['l2'],
             'solver': ['liblinear', 'lbfgs', 'saga'],
             }

# 3 estimators by 5 splites, by 2 leafs = 30 combinations

lrc_grid = GridSearchCV(estimator=lrc, 
                        param_grid = lrc_param,
                        scoring = 'accuracy',
                        cv = 10,
                        return_train_score=True)

#How many jobs will be executed? - 300 jobs. 30 combinations x 10 folds

lrc_grid_fit = lrc_grid.fit(X, Y)

cv_results_lrc = pd.DataFrame.from_dict(lrc_grid_fit.cv_results_)


# SVC GridSearchCV
svc_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10],
             'kernel' : ['rbf', 'linear'],
             'gamma': [0.1, 0.25, 0.5, 1, 5],
             }

# 3 estimators by 5 splites, by 2 leafs = 30 combinations

svc_grid = GridSearchCV(estimator=svc, 
                        param_grid = svc_param,
                        scoring = 'accuracy',
                        cv = 10,
                        return_train_score=True)

#How many jobs will be executed? - 300 jobs. 30 combinations x 10 folds

svc_grid_fit = svc_grid.fit(X, Y)

cv_results_svc = pd.DataFrame.from_dict(svc_grid_fit.cv_results_)




#Picking the best
rfc_top_rank = cv_results_rfc[cv_results_rfc['rank_test_score']==1]
lrc_top_rank = cv_results_lrc[cv_results_lrc['rank_test_score']==1]
svc_top_rank = cv_results_svc[cv_results_svc['rank_test_score']==1]


print('\n\n')

print('              ',
      '  Random Forest ',
      '  Logistic Regression', 
      '  Support Vector  ')

print('                    ',
      ' ------------------ ',
      ' -----------------------',
      '--------------------')

print( ' Mean Test Score   : ',
      str('%.4f' %rfc_top_rank['mean_test_score']),
      str('%.4f' %lrc_top_rank['mean_test_score']),
      str('%.4f' %svc_top_rank['mean_test_score']))

print( ' Mean Train Score   : ',
      str('%.4f' %rfc_top_rank['mean_train_score']),
      str('%.4f' %lrc_top_rank['mean_train_score']),
      str('%.4f' %svc_top_rank['mean_train_score']))

print(rfc_grid_fit.best_param_)

from sklearn.model_selection import cross_validate
cv_result_dtc = cross_validate(dtc, X, Y, cv = 10, return_train_score = True)
cv_result_rfc = cross_validate(rfc, X, Y, cv = 10, return_train_score = True)
cv_result_svc = cross_validate(svc, X, Y, cv = 10, return_train_score = True)
cv_result_lrc = cross_validate(lrc, X, Y, cv = 10, return_train_score = True)

# Get average of results
import numpy as np

dtc_test_average = np.average(cv_result_dtc['test_score'])
rfc_test_average = np.average(cv_result_rfc['test_score'])
svc_test_average = np.average(cv_result_svc['test_score'])
lrc_test_average = np.average(cv_result_lrc['test_score'])

dtc_train_average = np.average(cv_result_dtc['train_score'])
rfc_train_average = np.average(cv_result_rfc['train_score'])
svc_train_average = np.average(cv_result_svc['train_score'])
lrc_train_average = np.average(cv_result_lrc['train_score'])

# Analyze results

print(" ")
print(" ")
print(" ")

print("                 ","Decision Tree      ", " Random Forest      ", "Support Vector", "Logistic Regression")
print("                 -------------",    "------------",     "--------------", "-----------------")

print('Test   :  ',
      round(dtc_test_average, 4), '           ',
      round(rfc_test_average, 4), '           ',
      round(svc_test_average, 4), '           ',
      round(lrc_test_average, 4))

print('Train   :  ',
      round(dtc_train_average, 4), '           ',
      round(rfc_train_average, 4), '           ',
      round(svc_train_average, 4), '           ',
      round(lrc_train_average,4))

# Generalize predictions in SVC since variance is smaller, and more accurate in test
