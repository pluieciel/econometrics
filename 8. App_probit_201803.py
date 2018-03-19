# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols,glm
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


# class 8 ################################

AER_App_data_gameonly=AER_App_data[game==1]
  
killer=AER_App_data_gameonly["killerappgros"]
score=AER_App_data_gameonly["scoreapp"]
print(killer.describe(),'\n'*3)


def probitlik(x):
    beta0,beta1=x[0],x[1]
    vlog=killer*np.log(scipy.stats.norm.cdf(beta0+beta1*score))+(1-killer)*np.log(1-scipy.stats.norm.cdf(beta0+beta1*score))
    return sum(-vlog)

x0=np.asarray((0.1,0.1))
probitmodel=minimize(probitlik,x0,method="BFGS")
print(probitmodel,'\n'*3)
llikunrestr=probitmodel.fun
print("llikunrestr = ",llikunrestr,'\n'*3)


def probitlik_restrict(beta0):
    vlog=killer*np.log(scipy.stats.norm.cdf(beta0))+(1-killer)*np.log(1-scipy.stats.norm.cdf(beta0))
    return sum(-vlog)

probitmodel_restrict=minimize(probitlik_restrict,0.1,method="BFGS")
print(probitmodel_restrict,'\n'*3)
llikrestr=probitmodel_restrict.fun
print("llikrestr = ",llikrestr,'\n'*3)

lrtest=-2*(llikunrestr-llikrestr)
scipy.stats.chi2.ppf(0.95,1)
print("lrtest= ",lrtest)
print("chisq0.95,1=",scipy.stats.chi2.ppf(0.95,1),'\n'*3)

df=pd.DataFrame({
        "killer":killer,
        "score":score
        })
probitmodel2=glm("killer ~ score",df,family = sm.families.Binomial(link=statsmodels.genmod.families.links.probit)).fit()
print(probitmodel2.summary())


