# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols,glm
from io import StringIO  
import urllib.request,statsmodels.stats,scipy.stats,seaborn as sns
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
###load data:
##from web:
data = StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/France%20GDP.csv').read().decode('ascii', 'ignore'))
France_GDP=pd.read_table(data,sep=',')
##from local:
#France_GDP=pd.read_table(r'D:\france life\2017 T2\econometrics\France GDP.csv',sep=',')

######
gdp=France_GDP["Value"]
plt.plot(gdp,'.')
plt.show()
plt.plot(gdp)
plt.show()
rng = pd.period_range('1961Q1','2017Q4', freq='Q')
ts=pd.Series(np.array(gdp),index=rng)
ts.plot()
plot_acf(gdp)

##Box-Ljung test for gdp########
Xsq=acorr_ljungbox(gdp, lags=5)[0][-1]
pv=acorr_ljungbox(gdp, lags=5)[1][-1]
print('X-squared =',Xsq)
print('p-value =',pv)
print('\n'*3)

##AR model#########
df=pd.DataFrame({
        'y':np.array(gdp[1:]),
        'x':np.array(gdp[:-1])
        })
ar1model=ols('y~x',df).fit()
print(ar1model.summary())
print('\n'*3)

##Box-Ljung test for residuals########
Xsq1=acorr_ljungbox(ar1model.resid, lags=5)[0][-1]
pv1=acorr_ljungbox(ar1model.resid, lags=5)[1][-1]
print('X-squared =',Xsq1)
print('p-value =',pv1)
print('\n'*3)



#########################
##functions#############
def AR1_forecast(vy,nfor,hor):  
  T=len(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  mforecast=np.zeros((Tout,hor));
  mparam=np.zeros((Tout,2)); #2 parameters in AR1 model
  
  for t in range(1,Tout+1):
    vyestim=np.array(vy[t-1:Tin+t-1]);
    Testim=len(vyestim);
    df1=pd.DataFrame({
        'y':np.array(vyestim[1:Testim]),
        'x':np.array(vyestim[0:Testim-1])
        })
    AR1_estim=ols('y~x',df1).fit();
    mparam[t-1,]=AR1_estim.params;
    vyfor=np.array([1,vyestim[Testim-1]]);
    
    for h in range(1,hor+1):
      mforecast[t-1,h-1]=np.dot(vyfor,mparam[t-1,]);
      vyfor=np.array([1,mforecast[t-1,h-1]]);      
  
  plt.plot(mparam[:,0],'.r'); 
  plt.show()
  plt.plot(mparam[:,1],'.b');
  plt.show() 
  return mforecast

def forecast_performance(vy,nfor,mforecasts):

  T=len(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  vyestim=vy[0:Tin];
  vyout=np.array(vy[Tin:T]);
  hor=len(mforecasts[0,]);
  
  vrmse=np.zeros((hor,1));
  vmae=np.zeros((hor,1));
  
  for h in range(1,hor+1):
    Teval=Tout-h+1;
    verror=np.zeros((Teval,1));
    verror_sq=np.zeros((Teval,1));
    
    for t in range(1,Teval+1):
      verror[t-1,0]=mforecasts[t-1,h-1]-vyout[t+h-2];      
      verror_sq[t-1,0]=(mforecasts[t-1,h-1]-vyout[t+h-2])**2;
    
    vrmse[h-1,0]=np.sqrt(np.mean(verror_sq));
    vmae[h-1,0]=np.mean(verror);
  
  print("Forecast horizons: ");
  print(hor)
  print("Mean Error: ");
  print(vmae);
  print("Root Mean Squared Error: ");
  print(vrmse)
#############################
  
##forecast#########
mforecast=AR1_forecast(gdp,0.2,4)
forecast_performance(gdp,0.2,mforecast)