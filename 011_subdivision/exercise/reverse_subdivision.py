import gpytoolbox as gpy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def reverse_subdivision(V, F, uu, k):
    """Given a function uu on a mesh which has been subdivided k times from the
    coarse V,F, reconstruct a function u on the coarse meth V,F.
    """

    _, _, S = gpy.subdivide(V, F, "loop", iters=k, return_matrix=True)

    u = scipy.sparse.linalg.lsqr(S, uu)
    
    return u
