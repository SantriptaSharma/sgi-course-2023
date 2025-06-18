import gpytoolbox as gpy, numpy as np

def critical_points(V,F,u,tol):
    """
    This function computes the critical points of the function u on the mesh
    V,F by finding the indices of the faces where the gradient of the function
    u is smaller than tol.
    """

    G = gpy.grad(V,F)

    gu = np.c_[np.split(G * u, 3)]
    print(gu.shape)
    
    pts = np.nonzero(np.linalg.norm(gu, axis=-1) < tol)

    return pts

V, F = gpy.read_mesh("data/mug.obj")

print(F.shape[0], V.shape[0])

u = np.random.random(V.shape[0])
print(critical_points(V, F, u, 1))