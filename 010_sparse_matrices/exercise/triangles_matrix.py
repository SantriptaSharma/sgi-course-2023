import numpy as np, scipy as sp

def triangles_matrix(n):
    """This function constructs an n x n sparse matrix with a triangular pattern
    of ones, as by the following pattern:
    0 1 1 1 1 1 1 1 1 1 1
    1 0 1 0 0 0 0 0 0 0 1
    1 1 0 1 0 0 0 0 0 0 1
    1 0 1 0 1 0 0 0 0 0 1
    1 0 0 1 0 1 0 0 0 0 1
    1 0 0 0 1 0 1 0 0 0 1
    1 0 0 0 0 1 0 1 0 0 1
    1 0 0 0 0 0 1 0 1 0 1
    1 0 0 0 0 0 0 1 0 1 1
    1 0 0 0 0 0 0 0 1 0 1
    1 1 1 1 1 1 1 1 1 1 0
    """

    i = [1 + i for i in range(n-1)] + [n-1 for i in range(n-1)] + [1 + i for i in range(n-1)] + [0 for i in range(n - 1)] + [i for i in range(n-1)] + [i for i in range(n-1)]
    j = [0 for i in range(n-1)] + [i for i in range(n-1)] + [i for i in range(n-1)] + [i + 1 for i in range(n-1)] + [n-1 for i in range(n-1)] + [i + 1 for i in range(n-1)]
    k = [1 for _ in j]
    A = sp.sparse.csr_matrix((k,(i,j)), shape=(n,n))

    return A

A = triangles_matrix(30)

import matplotlib.pyplot as plt

plt.spy(A)
plt.show()