import gpytoolbox as gpy, polyscope as ps, numpy as np
from timeit import default_timer

def flipped_normals(V,F):
	"""
	Compute the flipped per-face normals of a triangle mesh.
	"""
	edge_idxs_a = F[:, 0:2]
	edge_idxs_b = F[:, 1:3]

	edges_a = V[edge_idxs_a[:, 1]] - V[edge_idxs_a[:, 0]]
	edges_b = V[edge_idxs_b[:, 1]] - V[edge_idxs_b[:, 0]]

	normals = np.cross(edges_b, edges_a)
	normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]

	return normals

V, F = gpy.read_mesh("data/spot_low_resolution.obj")

start = default_timer()
normals = flipped_normals(V, F)
end = default_timer()

print(end - start)

start = default_timer()
gt = -gpy.per_face_normals(V, F)
end = default_timer()

print(end - start)

print(np.all(np.abs(normals - gt) < 1e-3))

ps.init()
spot = ps.register_surface_mesh("spot", V, F)
spot.add_vector_quantity("normals", normals, defined_on="faces")
spot.add_vector_quantity("gt normals", normals, defined_on="faces")
ps.show()