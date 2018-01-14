a = read.csv("D:/france life/2017 T2/metric/costsalary.csv", header = TRUE, sep=";")
x=a$Costs
y=a$Salary
r=lm(y~x)
summary(r)
plot(x,y)
abline(r)
