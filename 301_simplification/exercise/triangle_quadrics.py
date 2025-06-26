import numpy as np

def triangle_quadrics(V,F):
    """
    Compute triangle quadrics for all the triangle faces in F

    Inputs:
        V: |V|x3 vertex list
        F: |F|x3 face list

    Outputs:
        Qf: (nF,4,4) triangle quadrics
    """
    
    #### Fill in the missing part #####
    ab = V[F[:, 1]] - V[F[:, 0]]
    ac = V[F[:, 2]] - V[F[:, 0]]

    # calculate plane normals
    normals = np.cross(ab, ac)
    normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    d = -np.sum(normals * V[F[:, 0]], axis=-1)

    p = np.c_[normals, d]

    # outer product
    return np.einsum("fi,fj->fij", p, p)

import gpytoolbox as gpy

V, F = gpy.read_mesh("data/spot.obj")
print(V.shape[0], F.shape[0])

quadrics = triangle_quadrics(V, F)

print(quadrics.shape)