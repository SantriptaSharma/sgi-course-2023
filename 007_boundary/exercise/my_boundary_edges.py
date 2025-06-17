from collections import Counter
import numpy as np

def my_boundary_edges(F):
    """Given a triangle mesh with face indices F, returns all unique oriented
    boundary edges as indices into the vertex array.
    Works only on manifold meshes.

    Parameters
    ----------
    F : (m,3) numpy int array.
        face index list of a triangle mesh

    Returns
    -------
    bE : (be,2) numpy int array.
        indices of boundary edges into the vertex array
    """

    edges = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]

    # sort can probably be optimised by just performing a transposition required check per row
    sorted_idxs = np.argsort(edges, axis=1)

    unique_edges, idxs, counts = np.unique(np.take_along_axis(edges, sorted_idxs, axis=1), return_index=True, return_counts=True, axis=0)

    # the original rows (in edges) from which these boundary edges are coming from
    orig_rows = idxs[counts==1]
    return edges[orig_rows, :]

import gpytoolbox as gpy
import numpy as np
from timeit import default_timer

F = np.array([[0, 1, 2], [1, 2, 3]])

print(f"Number of faces: {F.shape[0]}")

start = default_timer()
mine = my_boundary_edges(F)
end = default_timer()
print(f"time (mine): {end - start}s")

start = default_timer()
gt = gpy.boundary_edges(F)
end = default_timer()
print(f"time (gt): {end - start}s")

print(mine.shape, gt.shape)
print(np.all(np.sort(mine, axis=0) == np.sort(gt, axis=0)))

V, F = gpy.read_mesh("data/goathead.obj")
print(f"Number of faces: {F.shape[0]}")

start = default_timer()
mine = my_boundary_edges(F)
end = default_timer()
print(f"time (mine): {end - start}s")

start = default_timer()
gt = gpy.boundary_edges(F)
end = default_timer()
print(f"time (gt): {end - start}s")

print(mine.shape, gt.shape)
print(np.all(np.sort(mine, axis=0) == np.sort(gt, axis=0)))