import numpy as np
list =np.array([-2,-1,0,1, 2, 3])
shape=np.shape(list)
list[-3:shape[0]] = 5 
# Set the last element
print(list)
