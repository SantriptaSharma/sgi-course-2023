import numpy as np, gpytoolbox as gpy
from timeit import default_timer as dt
import polyscope as ps
import scipy
import scipy.sparse

def my_grad(V,F):
    """This function computes the finite element gradient matrix for the mesh
    V,F.
    """

    m = F.shape[0]
    n = V.shape[0]

    he = gpy.halfedges(F)

    # three vertices: alpha, beta, gamma. corr. opposite halfedges: ab, bc, ca
    alp_idxs = he[:, 1, 1]
    bet_idxs = he[:, 2, 1]
    gam_idxs = he[:, 0, 1]

    # get edges as vectors, we'll project them to find altitudes
    he_ab = V[gam_idxs] - V[bet_idxs]
    he_bc = V[alp_idxs] - V[gam_idxs]
    he_ca = V[bet_idxs] - V[alp_idxs]

    dir_ab = he_ab / np.linalg.norm(he_ab, axis=-1)[:, np.newaxis]
    dir_bc = he_bc / np.linalg.norm(he_bc, axis=-1)[:, np.newaxis]
    dir_ca = he_ca / np.linalg.norm(he_ca, axis=-1)[:, np.newaxis]
    
    # projection using dot products, find altitudes
    # first find foot of altitude (parenthesis) and then find the altitude vector
    # we want the altitude to point towards the source vertex
    # since that's obviously where the hat function is increasing towards
    alt_alp = V[alp_idxs] - (V[gam_idxs] + np.sum(he_bc * dir_ab, axis=-1)[:, np.newaxis] * dir_ab)
    alt_bet = V[bet_idxs] - (V[alp_idxs] + np.sum(he_ca * dir_bc, axis=-1)[:, np.newaxis] * dir_bc)
    alt_gam = V[gam_idxs] - (V[bet_idxs] + np.sum(he_ab * dir_ca, axis=-1)[:, np.newaxis] * dir_ca)

    print(alt_alp.shape)

    # divide by squared norms since the length of the gradient has to be 1/len(altitude)
    alt_alp /= np.sum(alt_alp**2, axis=-1)[:, np.newaxis]
    alt_bet /= np.sum(alt_bet**2, axis=-1)[:, np.newaxis]
    alt_gam /= np.sum(alt_gam**2, axis=-1)[:, np.newaxis]

    i = np.concatenate([
        # face indices for x entries, once for each vertex 
        np.arange(m),
        np.arange(m),
        np.arange(m),

        # face indices for y entries, once for each vertex 
        np.arange(m, 2*m),
        np.arange(m, 2*m),
        np.arange(m, 2*m),

        # face indices for z entries, once for each vertex 
        np.arange(2*m, 3*m),
        np.arange(2*m, 3*m),
        np.arange(2*m, 3*m),

    ])

    j = np.concatenate([
        # vertex indices for alpha, beta, gamma, repeated for each axis
        alp_idxs,
        bet_idxs,
        gam_idxs,

        alp_idxs,
        bet_idxs,
        gam_idxs,

        alp_idxs,
        bet_idxs,
        gam_idxs
    ])

    k = np.concatenate([
        alt_alp[:, 0],
        alt_bet[:, 0],
        alt_gam[:, 0],

        alt_alp[:, 1],
        alt_bet[:, 1],
        alt_gam[:, 1],

        alt_alp[:, 2],
        alt_bet[:, 2],
        alt_gam[:, 2],
    ])

    return scipy.sparse.csr_matrix((k, (i, j)), (3 * m, n))

V, F = gpy.read_mesh("data/mug.obj")

print(F.shape[0], V.shape[0])

k = 3

start = dt()
G_my = my_grad(V, F)
end = dt()

print(end - start)

start = dt()
G_gt = gpy.grad(V, F)
end = dt()

print(end - start)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.spy(G_my)
ax2.spy(G_gt)

ax1.set_title("mine")
ax2.set_title("ground truth")

plt.show()

F_my = G_my.toarray()
F_gt = G_gt.toarray()

print(np.all(np.abs(F_my - F_gt) < 1e-3))

xg, yg, zg = np.split(F_gt, 3, 0)
xm, ym, zm = np.split(F_my, 3, 0)

print(xg.shape)

face_index = 285

ps.init()
mug = ps.register_surface_mesh("mug", V, F)
mug.add_vector_quantity("grad gt", np.c_[xg[face_index], yg[face_index], zg[face_index]], 'vertices')
mug.add_vector_quantity("grad mine", np.c_[xm[face_index], ym[face_index], zm[face_index]], 'vertices')
ps.show()