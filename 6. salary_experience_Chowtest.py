# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt,numpy as np,statsmodels.api as sm,pandas as pd
from statsmodels.formula.api import ols
from io import StringIO  
import urllib.request,statsmodels.stats,scipy.stats,seaborn as sns

###load data:
##from web:
data = StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/salary_experience.csv').read().decode('ascii', 'ignore'))
salary_experience=pd.read_table(data,sep=';')
##from local:
#salary_experience=pd.read_table(r'D:\france life\2017 T2\econometrics\salary_experience.csv',sep=';')

salary=salary_experience['vy']
experience=salary_experience['vx']
education=salary_experience['vxz']

df=pd.DataFrame({
        'salary':salary,
        'experience':experience,
        'education':education
        })

#regression:
model1=ols('salary~experience',df).fit()
print(model1.summary())
#plot:
plt.plot(experience,salary,'k.')
plt.plot(experience,model1.fittedvalues,'b')
plt.show()

###subsets:
#points
plt.plot(experience[education==1],salary[education==1],'g.')
plt.plot(experience[education==0],salary[education==0],'r.')

eduexp=education*experience
df['eduexp']=eduexp  #update DataFrame
model2=ols('salary~experience+education+eduexp',df).fit()

#regression lines
a,b,c,d=model2.params  #beta0,beta1,beta2,beta3
plt.plot(experience,a+b*experience,'r')
plt.plot(experience,(a+c)+(b+d)*experience,'g')
plt.show()
print(model2.summary(),'\n'*3)

#Ftest
ssru=sum(model2.resid**2)
print('ssru=',ssru,'\n'*3)

#restricted model
ssrr=sum(model1.resid**2)
print('ssrr=',ssrr,'\n'*3)
nrestr=2

Ftest=((ssrr-ssru)/nrestr)/(ssru/(len(salary)-len(model2.params)))
print('Ftest=',Ftest,'\n'*3)

#critical value
#R: qf(0.95,nrestr,length(salary)-length(model2$coefficients))
print(scipy.stats.f.ppf(0.95,nrestr,len(salary)-len(model2.params)),'\n'*3)
#R: 1-pf(Ftest,nrestr,length(salary)-length(model2$coefficients))
print(1-scipy.stats.f.cdf(Ftest,nrestr,len(salary)-len(model2.params)),'\n'*3)
#R: anova(model2,model1)
print(statsmodels.stats.anova.anova_lm(model1,model2))
#Attention:
print('\033[1;31;43m','Attention: the input order impacts the F value. input Restricted Model first','\033[0m','\n'*3)

mlow=salary_experience[salary_experience['vxz']==0]
mhigh=salary_experience[salary_experience['vxz']==1]
df2=pd.DataFrame({
        'mlowvy':mlow['vy'],
        'mlowvx':mlow['vx'],
        'mhighvy':mhigh['vy'],
        'mhighvx':mhigh['vx']
        })
mregrlow=ols('mlowvy~mlowvx',df2).fit()
mregrhigh=ols('mhighvy~mhighvx',df2).fit()

ssru1=sum(mregrlow.resid**2)
ssru2=sum(mregrhigh.resid**2)
totalssru=ssru1+ssru2
print('totalssru=',totalssru)
