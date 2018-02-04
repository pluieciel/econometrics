# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request
import scipy

data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/CAPM%20Exercise.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=",",skiprows=1)

##if from local path:
# my = np.loadtxt(open(r'D:\Users\JVKR\Dropbox\VRP\BD_VRP\HighFreqData\AAPL1min_NASDAQ.csv',"rb"),delimiter=";",skiprows=1)

y, x = zip(*my) #x:sp500, y:AMZN
X = sm.add_constant(x) #X is a matrix with first col of '1' & second col of x

regression = sm.OLS(y,X).fit()
print(regression.summary())
plt.plot(x,y,'.',x,regression.fittedvalues,'r')

Z=(regression.params[1]-1)/regression.bse[1] #Z<-(summary(regression)$coefficients[2,1]-1)/summary(regression)$coefficients[2,2]
print("Z =",Z)
print("possibility of (upper&lower) :",2*(1-scipy.stats.norm.cdf(Z)) ) #pnorm(Z)
