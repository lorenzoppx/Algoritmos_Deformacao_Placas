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
width= 5
lenght = 5
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

# Create zeros matrix with same shape of X
T = np.zeros_like(Matrix_ones)
T = np.reshape(T,(1,number_of_points))
M = T

print("M-shape:",np.shape(M))
print(M)
print("T-shape:",np.shape(T))
print(T)
M=np.row_stack((M,T)) 
print("M-shape:",np.shape(M))
print(M)

T = np.zeros_like(Matrix_ones)



