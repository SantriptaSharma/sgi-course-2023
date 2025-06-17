import numpy as np, scipy as sp
def four_corners(m,n):
    """This function constructs an m x n sparse matrix with ones in each of its
    four corners.
    """

    i = [0, 0, m-1, m-1]
    j = [0, n-1, 0, n-1]
    k = [1, 1, 1, 1]
    A = sp.sparse.csr_matrix((k,(i,j)), shape=(m,n))

    return A

print(four_corners(5, 5))

import matplotlib.pyplot as plt

plt.spy(four_corners(500, 500))
plt.show()