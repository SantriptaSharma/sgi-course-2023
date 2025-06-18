import gpytoolbox as gpy
import numpy as np
import scipy.sparse

def cotangent_laplacian_from_grad(V,F):
    """
    Compute the cotangent Laplacian (Laplacian matrix) for a mesh using the
    gradient of a function as well as triangle areas
    """

    G = gpy.grad(V, F)
    A = gpy.doublearea(V, F) / 2
    A = scipy.sparse.diags(np.tile(A, 3))

    L = G.T * A * G

    return L
    

V, F = gpy.read_mesh("data/bunny.obj")
print(V.shape[0], F.shape[0])

cot_lap_my = cotangent_laplacian_from_grad(V, F)
cot_lap_gt = gpy.cotangent_laplacian(V, F)

diff = np.abs(cot_lap_gt - cot_lap_my).toarray()
print(np.all(diff < 1e-5))