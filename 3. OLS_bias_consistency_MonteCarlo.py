# -*- coding: utf-8 -*-
#R-Code adapted from XIE-YE-HU
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from io import StringIO  
import urllib.request,scipy.stats,seaborn as sns

##OLS_bias
def mystats(x,na_omit=False):
  if(na_omit):
    x=x[~np.isnan(x)]
  m=np.mean(x)
  n=np.size(x)
  s=np.std(x)
  skew=sum((x-m)**3/s**3)/n
  kurt=sum((x-m)**4/s**4)/n-3
  return({'n':n,'mean':m,'stdev':s,'skewness':skew,'kurtosis':kurt})


def OLS_bias(M,n,a,b):   
    np.random.seed(611)
    x=np.reshape(np.mat(np.random.normal(size=n)),(n,1))
    X = np.mat(sm.add_constant(x))
    e=np.mat(np.empty([n,M]))
    y=np.mat(np.empty([n,M]))
    ahat=np.empty([M,1])
    bhat=np.empty([M,1])
    vara=np.empty([M,1])
    varb=np.empty([M,1])
    
    for i in range(M):
      e[:,i]=np.reshape(np.mat(np.random.normal(size=n)),(n,1))
      y[:,i]=a+b*x+e[:,i]
      model = sm.OLS(y[:,i],X).fit()
      ahat[i],bhat[i]=model.params
      vara[i],varb[i]=model.bse
      
    datafour=pd.DataFrame(np.column_stack([ahat,bhat,vara,varb]),columns=['ahat','bhat','vara','varb'])
    statistics=datafour.apply(mystats)
    statistics=pd.DataFrame({'ahat':statistics[0],'bhat':statistics[1],'vara':statistics[2],'varb':statistics[3]},index=['n','mean','stdev','skewness','kurtosis'])
    print(statistics)
    
    bias_a=datafour['ahat']#-a
    bias_b=datafour['bhat']#-b
    biasab=pd.DataFrame(np.column_stack([bias_a,bias_b]),columns=["bias_a","bias_b"])
    
    fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows=2,ncols=2,figsize=(9,6))
    ax0.hist(datafour['ahat'],100,normed=1,histtype='bar',facecolor="r",alpha=0.75)
    ax0.set_title('Histogram of estimates a')
    ax0.set_xlabel('estimates of a')
    ax1.hist(datafour['bhat'],100,normed=1,histtype='bar',facecolor="b",alpha=0.75)
    ax1.set_title('Histogram of estimates b')
    ax1.set_xlabel('estimates of b')
    
    #par(mfrow=c(2,2))
    #kd_a = scipy.stats.gaussian_kde(datafour['ahat'])
    #kd_b = scipy.stats.gaussian_kde(datafour['bhat'])
    #kd_a.covariance_factor = lambda : .25
    #kd_a._compute_covariance()
    #kd_b.covariance_factor = lambda : .25
    #kd_b._compute_covariance()
    xs = np.linspace(0,8,200)
    #ax2.plot(xs,kd_a(xs),'r')
    ax2.set_xlim(ax0.get_xlim())
    ax2.set_ylim(ax0.get_ylim())
    plt.subplot(223)
    sns.kdeplot(datafour['ahat'],color='red')
    ax2.set_title('Kernel density of a estimates')
    #ax3.plot(xs,kd_b(xs),'b')
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim(ax1.get_ylim())
    plt.subplot(224)
    sns.kdeplot(datafour['bhat'],color='blue')
    ax3.set_title('Kernel density of b estimates')
    fig.tight_layout(h_pad=2)
    plt.show()
  
    return(biasab)
    #return(statistics)

mout=OLS_bias(1000,100,1.5,0.7)

##OLS_consistent
def OLS_consistent(M,n,a,b,I):
    def OLS_ebias(M,n,a,b):
      bias=OLS_bias(M,n,a,b)
      a=np.var(bias['bias_a'])
      b=np.var(bias['bias_b'])
      return pd.DataFrame(np.column_stack([a,b]))
    
    ee=np.mat(np.empty([I,2]))
    for t in range(I):
      Ebias=OLS_ebias(M,n,a,b)
      ee[t,0]=Ebias.iloc[0,0]
      ee[t,1]=Ebias.iloc[0,1]
      n=n+300
    fig,(ax0,ax1) = plt.subplots(ncols=2,figsize=(9,3))
    ax0.plot(ee[:,0],'ro')
    ax0.set_title('loops of adding 300 observations')
    ax0.set_xlabel('Trend of variance ahat')
    ax1.plot(ee[:,1],'bo')
    ax1.set_title('loops of adding 300 observations')
    ax1.set_xlabel('Trend of variance bhat')
    print(ee)
    #return ee

OLS_consistent(500,300,1.5,0.7,10)
