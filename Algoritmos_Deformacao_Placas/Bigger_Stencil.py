"""
Algoritmo deformação de placas Finite Difference Method com Five-stencil
-> Five Stencil size
    -------------------
    |        1        |
    |    2  -8   2    |
    | 1 -8  20  -8  1 |
    |    2  -8   2    |
    |        1        |
    -------------------
-> Borda com isolante térmico, ou seja, derivada nula. 
-> Stencil adaptado nas bordas, '*1*': Ghost point.
            *1*     
    ------------------
    |    2  -8  2    |
    | 1 -8  19 -8  1 | ...(and others)
    |    2  -8  2    |
    |        1       |
    ------------------

            *1*
         0   0   0
    -------------------
    | 1 -8  19  -8  1 | ...(and others)
    |    2  -8   2    |
    |        1        |
    -------------------

             *1*
          0   0   0
        ----------------
    *1* |-8  18  -8  1 | ...(and others)
        | 2  -8   2    |
        |     1        |
        ----------------

             *1*
        0   0   0
          -------------
    *1* 0 | 18  -8  1 | ...(and others)
        0 | -8   2    |
          |     1     |
          -------------

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

# Create points inside each unit vecto
Nx=50
Ny=50
x = np.linspace(1,6,Nx+1)
y = np.linspace(1,6,Ny+1)

# Create meshgrid with the points
X,Y = np.meshgrid(x,y)

# Create zeros matrix with same shape of X
s = np.shape(X)
print(s)
T = np.zeros((s[0]+4, s[1]+4))
"""
T[0,:] = -1
T[-1,:] = -1
T[:,0] = -1
T[:,-1] = -1
"""
print(T)

# Print
shape = np.shape(X)
number_of_points = shape[0]*shape[1] 
print(shape)

# Pontos interiores
for i in range(2,shape[0]+2):
    for j in range(2,shape[1]+2):
        T[i,j]=20
        T[i-1,j]=-8
        T[i+1,j]=-8
        T[i,j+1]=-8
        T[i,j-1]=-8
        T[i+1,j+1]=2
        T[i-1,j+1]=2
        T[i+1,j-1]=2
        T[i-1,j-1]=2
        T[i,j+2]=1
        T[i,j-2]=1            
        T[i+2,j]=1
        T[i-2,j]=1
        # Ajustes no central
        if(i==2 or j==2 or i==shape[0]+2 or j==shape[1]+2):
            if( (i==2 and j==2) or (i==2 and j==shape[1]+2) or (i==shape[0]+2 and j==2) or (i==shape[0]+2 and j==shape[1]+2)) :
                T[i,j]=T[i,j]-2
            else:
                T[i,j]=T[i,j]-1
        T = np.delete(T,[0,1,-2,-1],0)
        T = np.delete(T,[0,1,-2,-1],1)
        T = np.reshape(T,(1,number_of_points))
        if(i==2 and j==2):
            M = T
        else:
            M = np.row_stack((M,T))
        print(T)
        T = np.zeros((s[0]+4, s[1]+4))
print(np.shape(M))


b = np.zeros(shape[0]*shape[0]) #RHS
b[0:Nx+1] = -100
shape_b = np.shape(b)
b[-Nx-1:shape_b[0]] = 100

tic=time.time()
theta2=scipy.linalg.solve(np.asarray(M),b)
toc=time.time()
print('linalg solver time:',toc-tic)

"""
## OUTPUT EXCEL
## convert your array into a dataframe
df = pd.DataFrame(M)
## save to xlsx file
filepath = 'my_excel_file_b.xlsx'
df.to_excel(filepath, index=False)
"""

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
plt.contourf(X,Y,theta2,colormap='jet')
plt.colorbar()
plt.show()

