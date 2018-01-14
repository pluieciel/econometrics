# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

my = np.loadtxt(open(r'D:\france life\2017 T2\metric\costsalary.csv',"rb"),delimiter=";",skiprows=1)
x, y = zip(*my)
X = sm.add_constant(x)
results = sm.OLS(y,X).fit()
print(results.summary())
with plt.xkcd():    #XKCD-style sketch plots ;-)
    plt.plot(x,y,'bo',x,results.fittedvalues,'r')
