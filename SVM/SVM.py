# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:24:52 2020

@author: John Lang
"""


####################
# Fitting the Iris Dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)


# Train SVC as a new kernel with Gramma = 0.1

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)

cm_rbf01 = confusion_matrix(Y_test, Y_predict) 

# Increase Gamma to 10

svc10 = SVC(kernel='rbf', gamma=10)
svc10.fit(X_train, Y_train)
Y_predict = svc10.predict(X_test)

cm_rbf10 = confusion_matrix(Y_test, Y_predict) 
# Accuracy Down

#Now SVC for linearl and polynomial
svclin = SVC(kernel='linear')
svclin.fit(X_train, Y_train)
Y_predict = svclin.predict(X_test)
cm_lin = confusion_matrix(Y_test, Y_predict)

#Polynomial
svcpoly = SVC(kernel='poly')
svcpoly.fit(X_train, Y_train)
Y_predict = svcpoly.predict(X_test)
cm_poly = confusion_matrix(Y_test, Y_predict)

#Sigmoid
svcsigmoid = SVC(kernel='sigmoid')
svcsigmoid.fit(X_train, Y_train)
Y_predict = svcsigmoid.predict(X_test)
cm_sigmoid = confusion_matrix(Y_test, Y_predict)