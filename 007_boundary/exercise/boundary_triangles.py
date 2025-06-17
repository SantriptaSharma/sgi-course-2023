import gpytoolbox as gpy, numpy as np

def boundary_triangles(F):
    """Return a list of boundary triangle indices for an input triangulation F.
    """

    # Compute boundary edges (only want the indices here), copied over from my_boundary_edges (could be an optional return)

    # convert faces to edges, edge_idx % #faces = face_idx
    edges = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]

    # sort can probably be optimised by just performing a transposition required check per row
    sorted_idxs = np.argsort(edges, axis=1)

    _, idxs, counts = np.unique(np.take_along_axis(edges, sorted_idxs, axis=1), return_index=True, return_counts=True, axis=0)

    # get triangle indices
    return idxs[counts == 1] % F.shape[0]

def btri(F):
    """Return a list of boundary triangle indices for an input triangulation F.
    """

    # Compute boundary edges.
    bdry_edges = gpy.boundary_edges(F)

    # Find all triangles that contain both vertices of a boundary edge.
    # HINT: Look at the documentation of the `where` or `nonzero` function in
    # NumPy.
    bdry_tri_list = []
    for ei in range(bdry_edges.shape[0]):
        e = bdry_edges[ei,:]
        for fi in range(F.shape[0]):
            f = F[fi,:]
            if np.nonzero(e[0]==f)[0].shape[0]>0 and np.nonzero(e[1]==f)[0].shape[0]>0:
                bdry_tri_list.append(fi)

    return np.array(bdry_tri_list)

# F = np.array([[0, 1, 2], [1, 2, 3]])
# print(boundary_triangles(F))

from timeit import default_timer

V, F = gpy.read_mesh("data/goathead.obj")

start = default_timer()
mine = boundary_triangles(F)
end = default_timer()

print(end - start)

start = default_timer()
gt = btri(F)
end = default_timer()

print(end - start)

print(np.all(mine == gt))