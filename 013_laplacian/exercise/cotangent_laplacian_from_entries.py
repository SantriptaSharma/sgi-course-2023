import gpytoolbox as gpy
import numpy as np
import scipy.sparse

from timeit import default_timer as dt

def cotangent_laplacian_from_entries(V,F):
    """
    Compute the cotangent Laplacian (Laplacian matrix) for a mesh using the
    cotangent formula directly.
    """

    n = V.shape[0]

    he, E, he_to_E, E_to_he = gpy.halfedge_edge_map(F)
    tip_angles = gpy.tip_angles(V, F)

    int_e_mask = E_to_he[:, 1, 0] != -1
    int_e_count = np.count_nonzero(int_e_mask)
    int_he_idxs = np.r_[E_to_he[int_e_mask, 0], E_to_he[int_e_mask, 1]]

    # alpha and beta are complementary for corr. half-edges
    cot_alphs = 1/np.tan(tip_angles[int_he_idxs[:, 0], int_he_idxs[:, 1]])
    cot_betas = np.r_[
        cot_alphs[int_e_count:],
        cot_alphs[:int_e_count]
    ]

    off_diag_entries = -0.5 * (cot_alphs + cot_betas)
    
    # first, construct the off-diagonal cotangent laplacian
    i = he[int_he_idxs[:, 0], int_he_idxs[:, 1], 0]
    j = he[int_he_idxs[:, 0], int_he_idxs[:, 1], 1]
    k = off_diag_entries

    L_od = scipy.sparse.csr_matrix((k, (i, j)), shape=(n, n))

    diagonal_entries = np.array(-np.sum(L_od, axis=0))[0]
    L_diag = scipy.sparse.diags(diagonal_entries)

    return L_od + L_diag
    
    
V, F = gpy.read_mesh("data/bunny.obj")
print(V.shape[0], F.shape[0])

start = dt()
cot_lap_my = cotangent_laplacian_from_entries(V, F)
end = dt()

print(end - start)

start = dt()
cot_lap_gt = gpy.cotangent_laplacian(V, F)
end = dt()

print(end - start)

diff = np.abs(cot_lap_gt - cot_lap_my).toarray()
print(np.all(diff < 1e-5))