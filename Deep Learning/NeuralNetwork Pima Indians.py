# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:56:33 2020

@author: John Lang
"""

# neural net with keras

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import os

os.chdir("C:\\Users\\John Lang\\Desktop\\009 - Deep Learning")
diabetes = pd.read_csv('diabetes.csv')

#Check for nulls
diabetes.isnull().sum(axis=1)

X = diabetes.iloc[:, :-1]
Y = diabetes.iloc[:, -1]

# Split the rows
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=1234,
                                                    stratify=Y)

#Defire keras model
model = Sequential()

model.add(Dense(24, 
                input_shape=(8,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(12, 
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(1, 
                activation = 'sigmoid',))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Run the model
model.fit(X_train, Y_train, epochs=160, batch_size=10)

accuracy_test = model.evaluate(X_test, Y_test)

#predict
Y_predict = model.predict_classes(X_test)
Y_predict_prob = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)






