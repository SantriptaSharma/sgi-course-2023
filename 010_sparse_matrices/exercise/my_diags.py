import numpy as np
import scipy.sparse as sp

def my_diags(v, diagonals, shape):
    """This functin constructs a diagonal or banded sparse matrix, just
    like SciPy's sparse.diag
    """

    rows = []
    cols = []
    vals = []

    for diag, off in zip(diagonals, v):
        i = 0
        j = 0
        k = 0

        if off < 0:
            i -= off
        elif off > 0:
            j += off

        while i < shape[0] and j < shape[1]:
            rows.append(i)
            cols.append(j)
            vals.append(diag[k])
            
            i += 1
            j += 1
            k += 1

    return sp.csr_matrix((vals, (rows, cols)), shape)

mat = my_diags([0, 5, 12, -4, -9], [np.arange(100), np.arange(50, 200), np.arange(20, 99, 2), np.arange(500, 600), np.arange(400, 800, 4)], (30, 30))

import matplotlib.pyplot as plt

plt.spy(mat)
plt.show()

print(mat.toarray())