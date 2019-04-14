import numpy as np
from numpy.linalg import  inv
from numpy import  dot
import  pandas as pd


dataSet = pd.read_csv()

x = np.mat([1, 2, 3]).reshape(3, 1)
y = 2*x

# theta =dot( np.dot(inv(np.dot(x.T, x)), x.T),y)
theta = 1.
alpha = 0.1
for i in range(100):
    theta = theta + np.sum(alpha*(y-dot(x,theta))*x.reshape(1,3))/3.
print(theta)