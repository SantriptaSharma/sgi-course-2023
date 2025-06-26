import numpy as np
import scipy
# from sksparse.cholmod import cholesky

def mqwf_dense(A, B, known, known_val):
    """
    This function solves the following problem 

    minimize_x 0.5 * x' * A * x - x' * B 
    such that x[known] = known_val

    Inputs:
        A: n x n np array 
        B: n x dim np array
        known: 1D np array of indices of constrained vertices
        known_val: constrained values at "known"

    Outputs:
        u: n x dim of output solution
    """

    if known_val.ndim == 1:
        known_val = known_val.reshape(-1, 1)
        B = B.reshape(-1, 1)

    dim = known_val.shape[-1]

    #### Fill in the missing part #####
    unknown = np.setdiff1d(np.arange(A.shape[0]), known)

    An = A[unknown, :][:, unknown]
    bn = A[unknown, :][:, known] @ known_val + (A[known, :][:, unknown]).T @ known_val + B[unknown]

    # adding eq constraints just requires tacking on the Aeq and beq rows
    # [ 2*An ] x = [ -bn ]
    # [ Aeq  ]     [ beq ]

    unknowns = np.linalg.solve(2 * An, -bn)

    ###################################

    u = np.zeros((A.shape[0], dim))
    u[known] = known_val
    u[unknown] = unknowns

    return u