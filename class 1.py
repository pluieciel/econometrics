# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import csv
with open(r'D:\france life\2017 T2\metric\costsalary.csv','r') as csvf:
    a=csv.reader(csvf)
    x=[row[0].split(';') for row in a][1:]
    
x,y=zip(*x)
x,y=list(map(int,x)),list(map(int,y))
X = sm.add_constant(x)
results = sm.OLS(y, X).fit()
print(results.summary())
plt.plot(x,y,'bo',x,results.fittedvalues,'r')


'''
R:

a = read.csv(costsalary, header = TRUE, sep=";")
aa=unlist(strsplit(x,split=";"))
aa=unlist(lapply(xx,as.numeric))
aaa=matrix(c(xx),nrow=2)

x=aaa[1,]
y=aaa[2,]

r=lm(y~x)
summary(r)
'''
