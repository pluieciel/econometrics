a = read.csv("https://raw.githubusercontent.com/pluieciel/econometrics/master/costsalary.csv", header = TRUE, sep=";")
x=a$Costs
y=a$Salary
r=lm(y~x)
summary(r)
plot(x,y)
abline(r)
