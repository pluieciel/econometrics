# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols
from io import StringIO  
import urllib.request,statsmodels.stats,scipy.stats,seaborn as sns
from scipy.optimize import minimize
###load data:
##from web:
#data = StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/AER_App_data.csv').read().decode('ascii', 'ignore'))
#AER_App_data=pd.read_table(data,sep=',')
##from local:
AER_App_data=pd.read_table(r'D:\france life\2017 T2\econometrics\AER_App_data.csv',sep=',')

game=AER_App_data['cat5']


def bernlik(prob):
    vlog=game*np.log(prob)+(1-game)*np.log(1-prob)
    loglik=sum(vlog)
    return loglik

vgrid=np.arange(0.01,1,0.01)
vloglik=np.zeros(len(vgrid))
for j in range(len(vgrid)):
    vloglik[j]=bernlik(vgrid[j])

plt.plot(vgrid,vloglik)
plt.show()
mloglik=np.column_stack([vgrid,vloglik])
mlogliksort=sorted(mloglik,key=lambda x:x[1])
print(mlogliksort[-1],'\n'*3) #maximizer
print(pd.Series(game).describe(),'\n'*3)

def bernlik(prob):
    vlog=game*np.log(prob)+(1-game)*np.log(1-prob)
    loglik=sum(-vlog)
    return loglik

#activate library "stats4"
print(minimize(bernlik,0.55,bounds=((0.01,0.99),)),'\n'*3)
