"""
Algoritmo deformação de placas
Modified: 25/07/2022
By: @lorenzoppx
"""

# Libraries
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
width= 50
lenght = 50
number_of_points= width*lenght

# Create matrix
Matrix_zeros = np.matrix(np.zeros(number_of_points,dtype=float).reshape((width,lenght)));
Matrix_ones = np.matrix(np.ones(number_of_points,dtype=float).reshape((width,lenght)));

# Create points inside each unit vector
x = np.linspace(-2,2,num=1000)
y = np.linspace(-2,2,num=1000)

# Create meshgrid with the points
X,Y = np.meshgrid(x,y)

# Create function to plot
FUNC = X**2 + Y**2

# Create zeros matrix with same shape of X
T = np.zeros_like(X)

# Create 3D print
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, FUNC, 100, cmap='binary')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('FUNC');
plt.show()

# Create 2d print
plt.contourf(X,Y,FUNC,colormap='binary')
plt.colorbar()
plt.show()

"""
print('x')
print(X)
print('y')
print(Y)
print('t')
print(T)
print('\n')
"""

# Some print stuff
Matrix = Matrix_ones
# Print matrix format
print(Matrix)
print('\n')
# Print array format
print(Matrix.getA())

