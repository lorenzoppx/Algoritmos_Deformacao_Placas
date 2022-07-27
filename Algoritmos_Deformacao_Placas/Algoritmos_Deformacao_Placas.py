"""
Algoritmo deformação de placas
Modified: 25/07/2022
By: @lorenzoppx
"""

# Libraries
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
width= 10
lenght = 10
number_of_points= width*lenght

# Create matrix
Matrix_zeros = np.matrix(np.zeros(number_of_points,dtype=float).reshape((width,lenght)));
Matrix_ones = np.matrix(np.ones(number_of_points,dtype=float).reshape((width,lenght)));

# Condições de contorno
# Temperatura fixa lado direito
Matrix_ones[-1,:]=10
# Temperatura fixa do lado esquerdo
Matrix_ones[0,:]=10
# Temperatura fixa de baixo
Matrix_ones[:,-1]=10
# Temperatura fixa de cima
Matrix_ones[:,0]=10

#Stencil size (estrela)
"""
        1
     1 -4  1
        1
"""
       

# Create points inside each unit vector
x = np.linspace(-2,2,num=1000)
y = np.linspace(-2,2,num=1000)

# Create meshgrid with the points
X,Y = np.meshgrid(x,y)

# Create function to plot
FUNC = X**2 + Y**2

# Condições de contorno
# Temperatura fixa lado direito
FUNC[-1,:]=10
# Temperatura fixa do lado esquerdo
FUNC[0,:]=10
# Temperatura fixa de baixo
FUNC[:,-1]=10
# Temperatura fixa de cima
FUNC[:,0]=10

# Create zeros matrix with same shape of X
T = np.zeros_like(Matrix_ones)
M = np.array([0])
# Print
shape = np.shape(Matrix_ones)

for i in range(1,shape[0]-1):
    for j in range(1,shape[1]-1):
        T[i,j]=-4
        T[i-1,j]=1
        T[i+1,j]=1
        T[i,j+1]=1
        T[i,j-1]=1
        T = np.reshape(T,(1,number_of_points))
        M = np.append(M,T) 
        print(T)
        T = np.zeros_like(Matrix_ones)
print(np.shape(M))



## OUTPUT EXCEL

## convert your array into a dataframe
df = pd.DataFrame (M)

## save to xlsx file

filepath = 'my_excel_file.xlsx'

df.to_excel(filepath, index=False)

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

