import numpy as np, gpytoolbox as gpy
from timeit import default_timer as dt
import polyscope as ps

def my_upsample(V,F,k):
	"""This function performs k iterations of upsampling on the mesh V,F
	"""

	def upsample_once(V, F):
		hes, Es, he_to_E, E_to_he = gpy.halfedge_edge_map(F)
		n = V.shape[0]

		# each unique edge in E gets ownership of a new vert via the formula n + E_idx
		new_verts = (V[Es[:, 0]] + V[Es[:, 1]]) / 2
		V_new = np.r_[V, new_verts]

		# each face gets replaced by faces (face_idx*4...face_idx*4+3)
		F_new = np.block([
			[F[:,0], n+he_to_E[:,2], n+he_to_E[:,1], n+he_to_E[:,1]],
			[n+he_to_E[:,2], F[:,1], n+he_to_E[:,0], n+he_to_E[:,2]],
			[n+he_to_E[:,1], n+he_to_E[:,0], F[:,2], n+he_to_E[:,0]]
		]).transpose()

		return V_new, F_new


	V_new = np.copy(V)
	F_new = np.copy(F)

	for i in range(k):
		V_new, F_new = upsample_once(V_new, F_new)

	return V_new, F_new

V, F = gpy.read_mesh("data/mug.obj")

k = 3

start = dt()
V_my, F_my = my_upsample(V, F, k)
end = dt()

print(end - start)

start = dt()
V_gt, F_gt = gpy.subdivide(V, F, "upsample", k)
end = dt()

print(end - start)

ps.init()
arma = ps.register_surface_mesh("unupped", V, F)
arma = ps.register_surface_mesh("upped_gt", V_gt, F_gt)
arma = ps.register_surface_mesh("upped_mine", V_my, F_my)
ps.show()