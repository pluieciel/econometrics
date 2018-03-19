# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols,glm
from io import StringIO  
import urllib.request,statsmodels.stats,scipy.stats,seaborn as sns
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf
###load data:
##from web:
#data = StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/France%20GDP.csv').read().decode('ascii', 'ignore'))
#AER_App_data=pd.read_table(data,sep=',')
##from local:
France_GDP=pd.read_table(r'D:\france life\2017 T2\econometrics\France GDP.csv',sep=',')

######
gdp=France_GDP["Value"]
plt.plot(gdp,'.')
plt.plot(gdp)

rng = pd.period_range('1961Q1','2017Q4', freq='Q')
ts=pd.Series(np.array(gdp),index=rng)
ts.plot()

plot_acf(gdp)
