import numpy as np
import matplotlib
import matplotlib.pyplot as plt

colors = ['red', 'blue']

data = np.loadtxt('toyexample.txt')
x = data[:,0]
y = data[:,1]
targets = np.zeros(len(x))
targets[19:66] = 1

plt.scatter(x, y, c=targets, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

print(data.shape)
