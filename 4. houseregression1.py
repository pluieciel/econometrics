# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from io import StringIO  
import urllib.request,scipy.stats,seaborn as sns

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
x=np.column_stack([lnlotsize,bedrms,bathrms,airco])
X=np.mat(sm.add_constant(x))

##calculation: (X.T*X).I*X.T*y
XprimeX=X.T*X
XprimeXinv=XprimeX.I
XprimeY=X.T*y
betahat=XprimeXinv*XprimeY

print(betahat)

housereg=sm.OLS(y,X).fit()
print(housereg.summary())
