# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request

data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/costsalary.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=";",skiprows=1)

x, y = zip(*my)
X = sm.add_constant(x)
results = sm.OLS(y,X).fit()
print(results.summary())
#with plt.xkcd():    #XKCD-style sketch plots ;-)
plt.plot(x,y,'bo',x,results.fittedvalues,'r')

#method 2:do it manually for alpha and beta θ=(X.T*X).I*X.T*y
A=(np.mat(X).T*np.mat(X)).I*np.mat(X).T*np.mat(y).T   
a,b=A[0,0],A[1,0]

#method 3:beta=Cov/Var
beta= np.cov(x,y)[1][0]/np.cov(x,y)[0][0]
alpha=np.mean(y)-beta*np.mean(x)
