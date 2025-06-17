import gpytoolbox as gpy, polyscope as ps, numpy as np
from timeit import default_timer

def my_per_face_normals(V,F, unit_norm=True):
	"""Vector perpedicular to all faces on a mesh
	
		Computes per face (optionally unit) normal vectors for a triangle mesh.
	
		Parameters
		----------
		V : (n,d) numpy array
			vertex list of a triangle mesh
		F : (m,d) numpy int array
			face index list of a triangle mesh
		unit_norm : bool, optional (default True)
			Whether to normalize each face's normal before outputting
	
		Returns
		-------
		N : (n,d) numpy double array
			Matrix of per-face normals
  	"""

	edge_idxs_a = F[:, 0:2]
	edge_idxs_b = F[:, 1:3]

	edges_a = V[edge_idxs_a[:, 1]] - V[edge_idxs_a[:, 0]]
	edges_b = V[edge_idxs_b[:, 1]] - V[edge_idxs_b[:, 0]]

	normals = np.cross(edges_a, edges_b)

	if unit_norm:
		normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]

	return normals

V, F = gpy.read_mesh("data/spot_low_resolution.obj")

start = default_timer()
normals = my_per_face_normals(V, F)
end = default_timer()

print(end - start)

start = default_timer()
gt = gpy.per_face_normals(V, F)
end = default_timer()

print(end - start)

print(np.all(np.abs(normals - gt) < 1e-3))

ps.init()
spot = ps.register_surface_mesh("spot", V, F)
spot.add_vector_quantity("normals", normals, defined_on="faces")
spot.add_vector_quantity("gt normals", normals, defined_on="faces")
ps.show()