# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request
import scipy

data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/CAPM%20Exercise.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=",",skiprows=1)

y, x = zip(*my) #x:sp500, y:AMZN
X = sm.add_constant(x) #X is a matrix with first col of '1' & second col of x
plt.plot(x,y,'.')
regression = sm.OLS(y,X).fit()
print(regression.summary())

Z=(regression.params[1]-1)/regression.bse[1]
scipy.stats.norm.cdf(Z)
