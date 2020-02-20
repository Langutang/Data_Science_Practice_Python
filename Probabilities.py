# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:44:31 2019

@author: jlang
"""


import matplotlib.pyplot as plt
from IPython.display import Math, Latex
from IPython.core.display import Image
import seaborn as sns
from scipy.stats import uniform

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})
#Uniform Distribution
n = 10000
start = 10 
width = 20

data_uniform = uniform.rvs(size = n, loc = start, scale = width)

ax = sns.distplot(data_uniform,
                  bins = 100,
                  kde = True,
                  color='skyblue',
                  hist_kws={"linewidth":15, 'alpha':1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')

#Normal Distribution / Gaussian Distribution
#68% of data in 1 std, 95% in 2 std
#mean 0, std 1 is a standard normal distribution
#Maintain reproducibilty include random_state argument
from scipy.stats import norm
#Generate random numbers from N(0,1)
data_normal = norm.rvs(size=10000, loc=0, scale=1)
ax1 = sns.distplot(data_normal,
                   bins=100,
                   kde=True,
                   color='red',
                   hist_kws={"linewidth": 15, 'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
#Gamma - rarely used in normal form, but is in chi, exponential, erlang.Erlang when alpha = integer, alpha = 1 is exponential
 
#uses two parameters
#shape/ - alpha=k, inverse scale parameter - beta=1/theta, rate parameter
#Gamma(n) is gamma function defined as (n-1)
from scipy.stats import gamma
data_gamma = gamma.rvs(a=5, size=10000)
ax2 = sns.distplot(data_gamma, 
                   kde=True,
                   bins=100,
                   color='green',
                   hist_kws={"linewidth": 15, 'alpha':1})
ax2.set(xlabel='Gamma Distribution', ylabel='Frequency')
##Exponential describes the time between events in Poisson point process - events
#Occur continuously and independently at a constant average rate.
from scipy.stats import expon
data_expon = expon.rvs(scale=1, loc=0, size=1000)
ax3=sns.distplot(data_expon,
                 kde=True,
                 bins=100,
                 color='yellow',
                 hist_kws={"linewidth":15, 'alpha':1})
ax3.set(xlabel='Exponential Distribution', ylabel='Frequency')
#Poisson Dist - # of times something happens in a time interval
#Number of users visited on a website in an interbal can be thought of a poisson process
#Poisson dist is the rate (u) at which events happen
from scipy.stats import poisson
data_poisson = poisson.rvs(mu = 3, size=10000)
ax4 = sns.distplot(data_poisson,
                   bins=30,
                   kde=False,
                   color='purple',
                   hist_kws={"linewidth":15, 'alpha':1})
ax4.set(xlabel='Poisson Distribution', ylabel='Frequency')
#Binomial - only two outcomes possible - Success and failure are the same but 
#outcomes need not be equally likely, and it is independent. 
#n = number trials, p= probability of success
from scipy.stats import binom
data_binom = binom.rvs(n=10, p=0.8, size=10000)
ax5 = sns.displot(data_binom,
                  kds=False,
                  color= 'orange',
                  hist_kws={"linewidth":15, 'alpha':1})
ax5.set(xlabel = 'Binomial Distributions', ylabel='Frequency')
#Bernoulli Dist - Success, failure, for a single trial
from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000, p=0.6)

ax6 = sns.distplot(data_bern,
                 kd=False,
                 color="magenta",
                 hist_kws={"linewidth":15, 'alpha':1})
ax.set(xlabel='Bernoulli Dist', ylabel='Frequency')


print(x)