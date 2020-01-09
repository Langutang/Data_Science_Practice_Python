# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:04:09 2020

@author: John Lang
"""

import pandas as pd

f = pd.read_csv('Students2.csv')

x = f.iloc[:, :-1]
y = f.iloc[:,-1]

#Select Transform Feature Selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, GenericUnivariateSelect, f_regression

selectorK = SelectKBest(score_func=f_regression, k=3)
x_k = selectorK.fit_transform(x,y)

#f-score and p-values
f_score = selectorK.scores_
p_values = selectorK.pvalues_

columns = list(x.columns)

print(" ")
print(" ")
print(" ")

print("                 Feature      ", "  F-Score      ", "P-Values")
print("                 -------------    ------------     -------------")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("           ", columns[i].ljust(12), f1.rjust(8), "    ", p1.rjust(8))

cols=selectorK.get_support(indices=True)
selectedcols = x.columns[cols].tolist()

print(selectedcols)

###########
#Implement SelectPercentile for Feature selection
selectorP = SelectPercentile()

x_p = selectorP(x,y)

f_score = selectorP.scores_
p_values = selectorP.pvalues_

columns = list(x.columns)

print(" ")
print(" ")
print(" ")

print("                 Feature      ", "  F-Score      ", "P-Values")
print("                 -------------    ------------     -------------")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("           ", columns[i].ljust(12), f1.rjust(8), "    ", p1.rjust(8))

cols=selectorP.get_support(indices=True)
selectedcols = x.columns[cols].tolist()

print(selectedcols)


#############
# Generic Univariate Select

selectorG1 = GenericUnivariateSelect(score_func=f_regression,
                                     mode='k_best',
                                     param=3)

x_g1 = selectorG1.fit_transform(x,y)


selectorG2 = GenericUnivariateSelect(score_func=f_regression,
                                     mode='percentile',
                                     param=50)

x_g2 = selectorG2.fit_transform(x,y)











