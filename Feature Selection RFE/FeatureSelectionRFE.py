# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:26:47 2020

@author: John Lang
"""

import os

os.chdir("C:\\Users\\John Lang\\Desktop\\Features")

import pandas as pd
f=pd.read_csv('bank.csv')

f = f.drop("duration", axis=1)

#Features
x = f.iloc[:,:-1]
y = f.iloc[:,-1]

#get dummies
x = pd.get_dummies(x, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3,
                                                    random_state = 1234,
                                                    stratify=y)

from sklearn.ensemble import RandomForestClassifier

rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(X_train, Y_train)
Y_predict1 = rfc1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, Y_predict1)

score1 = rfc1.score(X_test, Y_test)

#RFE 
from sklearn.feature_selection import RFE

rfc2 = RandomForestClassifier(random_state=1234)

rfe = RFE(estimator=rfc2, n_features_to_select=30, step=1)

rfe.fit(x, y)

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

rfc2.fit(X_train_rfe, Y_train)
Y_predict = rfc2.predict(X_test_rfe)

cm_rfe = confusion_matrix(Y_test, Y_predict)

score_rfe = rfc2.score(X_test_rfe, Y_test)

# Get the Selected Features
columns = list(x.columns)

#get rankings of features
ranking = rfe.ranking_
# Selected have a 1

#Gefeature_importances_
feature_importance = rfc1.feature_importances_

#Concatenate lists
rfe_selected = pd.DataFrame()

rfe_selected = pd.concat([pd.DataFrame(columns), 
                          pd.DataFrame(ranking),
                          pd.DataFrame(feature_importance)], axis=1)


