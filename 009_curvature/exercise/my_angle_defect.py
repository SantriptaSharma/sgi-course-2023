import gpytoolbox as gpy, numpy as np
import polyscope as ps
from timeit import default_timer as dt

# not sure why this doesn't work, will have to check my work later
# def tip_angle(V, F):
# 	hes = gpy.halfedges(F)
# 	he_vecs = V[hes[:, :, 1]] - V[hes[:, :, 0]]
# 	he_vecs /= np.linalg.norm(he_vecs, axis=-1)[:, np.newaxis]

# 	assert he_vecs.shape[1:] == (3, 3)

# 	tip_angles = np.zeros((F.shape[0], 3))

# 	tip_angles[:, 0] = np.acos(np.clip(np.sum(-he_vecs[:, 1] * he_vecs[:, 2], axis=-1), -1, 1))
# 	tip_angles[:, 1] = np.acos(np.clip(np.sum(-he_vecs[:, 2] * he_vecs[:, 0], axis=-1), -1, 1))
# 	tip_angles[:, 2] = np.acos(np.clip(np.sum(-he_vecs[:, 0] * he_vecs[:, 1], axis=-1), -1, 1))

def my_angle_defect(V,F):
	"""
	Compute the angle defect per vertex on the mesh V,F
	"""

	tip_angles = gpy.tip_angles(V, F)

	total_angles = np.bincount(np.ravel(F), weights=np.ravel(tip_angles))

	return (2 * np.pi) - total_angles
	
V, F = gpy.read_mesh("data/armadillo.obj")

start = dt()
my_defect = my_angle_defect(V, F)
end = dt()

print(end - start)

start = dt()
gt_defect = gpy.angle_defect(V, F)
end = dt()

print(end - start)

print(np.all(np.abs(my_defect - gt_defect) < 1e-3))

ps.init()
arma = ps.register_surface_mesh("armadillo", V, F)
arma.add_scalar_quantity("defect", gt_defect)
arma.add_scalar_quantity("my defect", my_defect)
ps.show()