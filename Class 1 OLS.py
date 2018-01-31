# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request

data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/costsalary.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=";",skiprows=1)
# my = np.loadtxt(open(r'D:\france life\2017 T2\metric\costsalary.csv',"rb"),delimiter=";",skiprows=1)
x, y = zip(*my)
X = sm.add_constant(x)
results = sm.OLS(y,X).fit()
print(results.summary())
#with plt.xkcd():    #XKCD-style sketch plots ;-)
plt.plot(x,y,'bo',x,results.fittedvalues,'r')

A=(np.mat(X).T*np.mat(X)).I*np.mat(X).T*np.mat(y).T   #do it manually for alpha and beta
a,b=A[0,0],A[1,0]
