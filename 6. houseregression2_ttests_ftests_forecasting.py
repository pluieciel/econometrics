# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols
from io import StringIO  
import urllib.request,statsmodels.stats,scipy.stats,seaborn as sns

###load data:
##from web:
data = StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/housing.dat').read().decode('ascii', 'ignore'))
housing=pd.read_table(data,sep='\s+')
##from local:
#housing=pd.read_table(r'D:\france life\2017 T2\econometrics\R from prof\housing.dat',sep='\s+')

y=np.reshape(np.mat(np.log(housing['price'])),(-1,1))
lnlotsize=np.log(housing['lotsize'])
bedrms=housing['bedrooms']
bathrms=housing['bathrms']
airco=housing['airco']
df=pd.DataFrame({
        'lnlotsize':lnlotsize,
        'bedrms':bedrms,
        'bathrms':bathrms,
        'airco':airco,
        'lnprice':np.log(housing['price'])
        })
x=np.column_stack([lnlotsize,bedrms,bathrms,airco])
X=np.mat(sm.add_constant(x))

##calculation: (X.T*X).I*X.T*y
XprimeX=X.T*X
XprimeXinv=XprimeX.I
XprimeY=X.T*y
betahat=XprimeXinv*XprimeY

print(betahat,'\n'*3)

housereg=ols('lnprice~lnlotsize+bedrms+bathrms+airco',df).fit()
print(housereg.summary(),'\n'*3)


epsilonhat=y-X*betahat
sigma2hat=(epsilonhat.T*epsilonhat)/(len(y)-X.shape[1])
varbetahat=sigma2hat[0,0]*XprimeXinv
sebetahat=np.sqrt(np.diag(varbetahat))


zbetadiff=(betahat[3]-betahat[2])/np.sqrt(varbetahat[3,3]+varbetahat[2,2]-2*varbetahat[2,3])
print('zbetadiff =', zbetadiff[0,0],'\n'*3)

bedbath=bedrms+bathrms
x1=np.column_stack([lnlotsize,bedbath,bathrms,airco])
X1=np.mat(sm.add_constant(x1))
housereg_test=sm.OLS(y,X1).fit()
print(housereg_test.summary(),'\n'*3)

###################################################
#Ftest
ssru=sum(housereg.resid**2)
print('ssru=',ssru,'\n'*3)

#restricted model

houserestrict=ols('lnprice~airco',df).fit()
ssrr=sum(houserestrict.resid**2)
print('ssrr=',ssrr,'\n'*3)
nrestr=3

Ftest=((ssrr-ssru)/nrestr)/(ssru/(len(y)-len(housereg.params)))
print('Ftest=',Ftest,'\n'*3)
#critical value
print(scipy.stats.f.ppf(0.95,nrestr,len(y)-len(housereg.params)),'\n'*3)

print('statsmodels.stats.anova.anova_lm(housereg,houserestrict):\n')
print(statsmodels.stats.anova.anova_lm(housereg,houserestrict))
print('\033[1;31;43m','Fvalue not right, this method use ssrr/544 in denominator of F','\033[0m','\n')
print('statsmodels.stats.anova.anova_lm(houserestrict,housereg):\n')
print(statsmodels.stats.anova.anova_lm(houserestrict,housereg))
print('\033[1;31;43m','the input order impact the F value. input Restricted Model first','\033[0m','\n'*3)


#forecasting
lotf=np.log(5000)
bedf=2
bathf=2
aircof=1

bhouse=housereg.params

hforecast=bhouse[0]+bhouse[1]*lotf+bhouse[2]*bedf+bhouse[3]*bathf+bhouse[4]*aircof

lnlotsizef=lnlotsize-lotf
bedrmsf=bedrms-bedf
bathrmsf=bathrms-bathf
aircoff=airco-aircof

x3=np.column_stack([lnlotsizef,bedrmsf,bathrmsf,aircoff])
X3=np.mat(sm.add_constant(x3))
houseregf=sm.OLS(y,X3).fit()
print(houseregf.summary())




