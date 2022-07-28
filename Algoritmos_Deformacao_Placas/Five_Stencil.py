"""
Algoritmo deformação de placas Finite Difference Method com Five-stencil
-> Five Stencil size
    -----------
    |    1    |
    | 1 -4  1 |
    |    1    |
    -----------
-> Borda com isolante térmico, ou seja, derivada nula. 
-> Stencil adaptado nas bordas, '*1*': Ghost point.
        *1*     
    -----------
    | 1 -5  1 |   ... (and others)
    |    1    |
    -----------     

         *1*     
        --------
    *1* |-6  1 |  ... (and others)
        | 1    |
        --------

Modified: 27/07/2022
By: @lorenzoppx
"""

# Libraries
import time
import pandas as pd
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
width= 6
lenght = 6
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


       

# Create points inside each unit vecto
Nx=50
Ny=50
x = np.linspace(1,6,Nx+1)
y = np.linspace(1,6,Ny+1)

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
T = np.zeros_like(X)

# Print
shape = np.shape(X)
number_of_points = shape[0]*shape[1] 
print(shape)

# Lado esquerdo e cima
T[0,0]=-6
T[1,0]=1
T[0,1]=1
T = np.reshape(T,(1,number_of_points))
M = T
print(T)
T = np.zeros_like(X)
# Lado de cima
i=0
for j in range(1,shape[1]-1):
        T[i,j]=-5
        T[i+1,j]=1
        T[i,j+1]=1
        T[i,j-1]=1
        T = np.reshape(T,(1,number_of_points))
        M = np.row_stack((M,T))
        print(T)
        T = np.zeros_like(X)
# Lado direito e cima
T[0,shape[1]-1]=-6
T[1,shape[1]-1]=1
T[0,shape[1]-2]=1
T = np.reshape(T,(1,number_of_points))
M = np.row_stack((M,T))
print(T)
T = np.zeros_like(X)





# Pontos da borda (um do lado de fora)
# Lado esquerdo
"""
j=0
for i in range(1,shape[0]-1):
        T[i,j]=-5
        T[i-1,j]=1
        T[i+1,j]=1
        T[i,j+1]=1
        T = np.reshape(T,(1,number_of_points))
        M = np.row_stack((M,T)) 
        print(T)
        T = np.zeros_like(X)
# Lado direito
j=shape[1]-1
for i in range(1,shape[0]-1):
        T[i,j]=-5
        T[i-1,j]=1
        T[i+1,j]=1
        T[i,j-1]=1
        T = np.reshape(T,(1,number_of_points))
        M = np.row_stack((M,T))
        print(T)
        T = np.zeros_like(X)

# Lado de baixo
l=shape[0]-1
for j in range(1,shape[1]-1):
        T[i,j]=-5
        T[i-1,j]=1
        T[i,j+1]=1
        T[i,j-1]=1
        T = np.reshape(T,(1,number_of_points))
        M=np.append(M,T,axis=0) 
        print(T)
        T = np.zeros_like(X)
print("Pontos exteriores:")
print(np.shape(M))
"""
# Pontos interiores
for i in range(1,shape[0]-1):
    for j in range(0,shape[1]):
        if j==0:
            T[i,j]=-5
            T[i-1,j]=1
            T[i+1,j]=1
            T[i,j+1]=1
            T = np.reshape(T,(1,number_of_points))
            M = np.row_stack((M,T)) 
            print(T)
            T = np.zeros_like(X)
        elif j==shape[1]-1:
            T[i,j]=-5
            T[i-1,j]=1
            T[i+1,j]=1
            T[i,j-1]=1
            T = np.reshape(T,(1,number_of_points))
            M = np.row_stack((M,T))
            print(T)
            T = np.zeros_like(X)
        else:
            T[i,j]=-4
            T[i-1,j]=1
            T[i+1,j]=1
            T[i,j+1]=1
            T[i,j-1]=1
            T = np.reshape(T,(1,number_of_points))
            M = np.row_stack((M,T))
            print(T)
            T = np.zeros_like(X)
print(np.shape(M))

# Lado esquerdo e baixo
T[shape[0]-1,0]=-6
T[shape[0]-1,1]=1
T[shape[0]-2,0]=1
T = np.reshape(T,(1,number_of_points))
M = np.row_stack((M,T)) 
print(T)
T = np.zeros_like(X)
# Lado de baixo
l=shape[0]-1
for j in range(1,shape[1]-1):
        T[i,j]=-5
        T[i-1,j]=1
        T[i,j+1]=1
        T[i,j-1]=1
        T = np.reshape(T,(1,number_of_points))
        M=np.append(M,T,axis=0) 
        print(T)
        T = np.zeros_like(X)

# Lado direito e baixo
T[shape[0]-1,shape[1]-1]=-6
T[shape[0]-2,shape[1]-1]=1
T[shape[0]-1,shape[1]-2]=1
T = np.reshape(T,(1,number_of_points))
M = np.row_stack((M,T))
print(T)
T = np.zeros_like(X)

print(np.shape(M))

b = np.zeros(shape[0]*shape[0]) #RHS
b[0:Nx+1] = -100
shape_b = np.shape(b)
b[-Nx-1:shape_b[0]] = 100

tic=time.time()
theta2=scipy.linalg.solve(np.asarray(M),b)
toc=time.time()
print('linalg solver time:',toc-tic)

## OUTPUT EXCEL

## convert your array into a dataframe
df = pd.DataFrame(theta2)

## save to xlsx file

filepath = 'my_excel_file_b.xlsx'

df.to_excel(filepath, index=False)

# Create 3D print
fig = plt.figure()
ax = plt.axes(projection='3d')
theta2 = np.reshape(theta2,(shape[0],shape[1]))
ax.contour3D(X, Y, theta2, 800, cmap='jet')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature');
plt.show()

# Create 2d print
plt.contourf(X,Y,FUNC,colormap='jet')
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
print(np.shape(FUNC))
print('\n')
# Print array format
print(Matrix.getA())

