# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request

#load datas:
data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/costsalary.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=";",skiprows=1)
##if from local path:
#my = np.loadtxt(open(r'C:\Users\Downloads\costsalary.csv','r'),delimiter=";",skiprows=1)
x, y = zip(*my)
X = sm.add_constant(x)

##### Method 1:
#statsmodels.api.OLS
results = sm.OLS(y,X).fit()
print(results.summary())
#with plt.xkcd():    #XKCD-style sketch plots ;-)
plt.plot(x,y,'bo',x,results.fittedvalues,'r')
print(*results.params)

##### Method 2:
#normal equation: alpha and beta θ=(X.T*X).I*X.T*y
A=(np.mat(X).T*np.mat(X)).I*np.mat(X).T*np.mat(y).T   
a,b=A[0,0],A[1,0]
print(a,b)

##### Method 3:
#beta=Cov/Var
beta= np.cov(x,y)[1][0]/np.cov(x,y)[0][0]
alpha=np.mean(y)-beta*np.mean(x)
print(alpha,beta)

##### Method 4:
#gradient descent
theta=np.mat(np.array([[0],[0]]))
step=0.1
m=len(y)
for _ in range(500):
    theta=theta-step/m*(np.mat(X).T*(np.mat(X)*theta-np.mat(y).T))
print(theta[0,0],theta[1,0])

