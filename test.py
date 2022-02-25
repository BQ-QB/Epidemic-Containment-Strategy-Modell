import numpy as np

x = np.array([1,2,3,4])
R = np.array([x+ [1,1,1,7],x+[2,2,2,2],x+ [3,3,3,3],x+[4,4,4,4]])


print([R[i][0] for i in range(4)])