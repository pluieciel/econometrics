# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request

data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/costsalary.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=";",skiprows=1)
#If from file: my = np.loadtxt(open(r'D:\france life\2017 T2\metric\costsalary.csv',"rb"),delimiter=";",skiprows=1)
x, y = zip(*my) #x:Costs, y:Salary
X = sm.add_constant(x) #X is a matrix with first col of '1' & second col of x
model1 = sm.OLS(y,X).fit()
print(model1.summary())

#alphe&beta: model1.params
#std err: model1.bse
########################################

np.mean(model1.resid) #mean(model1$residuals) 
plt.plot(x,model1.resid,'ro') #plot(model1$residuals)

sigma2hat=sum(model1.resid**2)/(len(x)-len(model1.params))  #sigma2hat=sum(model1$residuals^2)/(length(x)-length(model1$coefficients))
np.sqrt(sigma2hat) #sqrt(sigma2hat)
varbetahat=sigma2hat/(np.var(x,ddof=1)*(len(x)-1)) #varbetahat=sigma2hat/(var(x)*(length(x)-1))
np.sqrt(varbetahat) #sqrt(varbetahat)
