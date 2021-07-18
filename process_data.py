import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import full
from numpy.linalg.linalg import multi_dot


a=[[1,1],[1,2]]
b=[2,3]
print(np.linalg.solve(a,b))


c = np.arange(9).reshape(3,3) 
U, s, V = np.linalg.svd(c, full_matrices=True)
S = np.zeros((3,3))
for i in range(len(s)):
    S[i][i] = s[i]
print(multi_dot([U,S,V]))


eigenvalue, eigenvector = np.linalg.eig(np.array([[1,2],[-2,3]]))
print(eigenvalue)
print(eigenvector)

print(c.transpose())
