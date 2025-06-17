import gpytoolbox as gpy, polyscope as ps, numpy as np
from timeit import default_timer

def my_per_vertex_normals(V,F):
	"""Normal vectors to all vertices on a mesh

		Computes area-weighted per-vertex unit normal vectors for a triangle mesh.

		Parameters
		----------
		V : (n,d) numpy array
			vertex list of a triangle mesh
		F : (m,d) numpy int array
			face index list of a triangle mesh

		Returns
		-------
		N : (m,d) numpy double array
			Matrix of per-vertex normals
	"""

	face_normals = gpy.per_face_normals(V, F)
	face_areas = gpy.doublearea(V, F)

	vert_normals = np.zeros((V.shape[0], 3))	

	# i feel like this search can be vectorised but with a really big tiled array storing face indices (for vert lookup)
	for vi in range(V.shape[0]):
		faces = np.nonzero(vi == F)[0]
				
		vert_normals[vi] = np.mean(face_areas[faces, np.newaxis] * face_normals[faces], axis=0)

	return vert_normals / np.linalg.norm(vert_normals, axis=-1)[:, np.newaxis]

V, F = gpy.read_mesh("data/spot_low_resolution.obj")

start = default_timer()
normals = my_per_vertex_normals(V, F)
end = default_timer()

print(end - start)

start = default_timer()
gt = gpy.per_vertex_normals(V, F)
end = default_timer()

print(end - start)

print(np.all(np.abs(normals - gt) < 1e-3))

face_areas = gpy.doublearea(V, F)

indices = np.array([413, 415, 460, 462])
face_normals = gpy.per_face_normals(V, F)

ps.init()
spot = ps.register_surface_mesh("spot", V, F)
spot.add_vector_quantity("normals", normals)
spot.add_vector_quantity("gt normals", gt)
spot.add_vector_quantity("face normals", face_normals, 'faces')
ps.show()