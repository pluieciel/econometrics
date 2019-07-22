# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO  
import urllib.request

#load datas:
data = urllib.request.urlopen('https://raw.githubusercontent.com/pluieciel/econometrics/master/data/costsalary.csv').read().decode('ascii', 'ignore')
my = np.loadtxt(StringIO(data),delimiter=";",skiprows=1)
##if from local path:
#my = np.loadtxt(open(r'C:\Users\Downloads\costsalary.csv','r'),delimiter=";",skiprows=1)
x, y = zip(*my)
X = sm.add_constant(x)

##### Method 1:
#statsmodels.api.OLS
results = sm.OLS(y,X).fit()
print(results.summary())
#with plt.xkcd():    #XKCD-style sketch plots ;-)
plt.plot(x,y,'bo',x,results.fittedvalues,'r')
print(*results.params)

##### Method 2:
#normal equation: alpha and beta θ=(X.T*X).I*X.T*y
A=(np.mat(X).T*np.mat(X)).I*np.mat(X).T*np.mat(y).T   
a,b=A[0,0],A[1,0]
print(a,b)

##### Method 3:
#beta=Cov/Var
beta= np.cov(x,y)[1][0]/np.cov(x,y)[0][0]
alpha=np.mean(y)-beta*np.mean(x)
print(alpha,beta)

##### Method 4:
#gradient descent 
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
theta=np.mat([[0],[1]])
step=0.1
m=len(y)
xx=np.array(x).reshape(-1,1)/10000;yy=np.array(y).reshape(-1,1)/10000
XX=np.concatenate((np.ones(m).reshape(-1,1),xx),axis=1)
for _ in range(5000):
  theta=theta-step/m*(np.mat(XX).T*(np.mat(XX)*theta-np.mat(yy)))
plt.plot(xx,yy,'bo')  
abline(theta[1,0],theta[0,0])
plt.show()
print(theta[0,0],theta[1,0])

#with PyTorch:
import torch
import torch.nn as nn

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 6000
learning_rate = 0.01

# Toy dataset
x_train = np.array(xx, dtype=np.float32)
y_train = np.array(yy, dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'bo', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
for i in model.parameters():print(i)

