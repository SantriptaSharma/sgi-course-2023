import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg

def mqwf(A, B, known, known_val):
    """
    This function solves the following problem 

    minimize_x 0.5 * x' * A * x - x' * B 
    such that x[known] = known_val

    Inputs
        A: n x n scipy sparse array 
        B: n x dim np array
        known: 1D np array of indices of constrained vertices
        known_val: constrained values at "known"

    Outputs
        u: n x dim of output solution
    """

    #### Fill in the missing part #####
    unknown = np.setdiff1d(np.arange(A.shape[0]), known)
    An = A[unknown, :][:, unknown]

    bn = A[unknown, :][:, known] * known_val + (A[known, :][:, unknown]).T * known_val + B[unknown]

    # adding eq constraints just requires tacking on the Aeq and beq rows
    # [ 2*An ] x = [ -bn ]
    # [ Aeq  ]     [ beq ]

    unknowns = scipy.sparse.linalg.spsolve(2 * An, -bn)

    ###################################

    dim = known_val.shape[-1]

    u = np.zeros((A.shape[0], dim))
    u[known] = known_val
    u[unknown] = unknowns

    return u