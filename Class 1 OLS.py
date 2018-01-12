# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import statsmodels.api as sm
import csv

with open(r'D:\france life\2017 T2\metric\costsalary.csv','r') as csvf:
    x=[row[0].split(';') for row in csv.reader(csvf)][1:]
x,y=zip(*x)
x,y=list(map(int,x)),list(map(int,y))
X = sm.add_constant(x)
results = sm.OLS(y, X).fit()
print(results.summary())
plt.plot(x,y,'bo',x,results.fittedvalues,'r')

'''
R:
a = read.csv("D:/france life/2017 T2/metric/costsalary.csv", header = TRUE, sep=";")

x=a$Costs
y=a$Salary

r=lm(y~x)
summary(r)
plot(x,y,'o')
'''
