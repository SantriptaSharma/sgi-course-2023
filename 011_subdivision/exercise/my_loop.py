import numpy as np, gpytoolbox as gpy
from timeit import default_timer as dt
import polyscope as ps
import scipy
import scipy.sparse

def my_loop(V,F,k):
	"""This function performs k iterations of upsampling on the mesh V,F.
	This function returns a sparse matrix that allows the mapping of functions
	from the coarse to the fine mesh.
	"""

	def upsample_once(V, F):
		hes, Es, he_to_E, E_to_he = gpy.halfedge_edge_map(F)
		e = Es.shape[0]
		n = V.shape[0]

		# if only one he in E_to_he, we have a boundary edge
		# lack of index repr'd by -1
		boundary_mask = np.any(E_to_he[:, 1] == -1, axis=1)
		boundary_hedge_idxs = E_to_he[boundary_mask, 0]
		interior_hedge_idxs = np.r_[E_to_he[~boundary_mask, 0], E_to_he[~boundary_mask, 1]]

		# for moving the existing interior vertices, we want to operate on interior edges where the tail (first el of the edge) is an interior vertex
		interior_verts_mask = np.full(n, True)
		interior_verts_mask[Es[boundary_hedge_idxs].ravel()] = False

		# now find all halfedges with interior tails
		interior_tail_hes = np.stack(np.nonzero(interior_verts_mask[hes[:, :, 0]]), axis=-1)

		# the number of times a vertex index occurs in the unique edge list is its degree
		neighbour_counts = np.bincount(Es.ravel())
		betas = np.where(neighbour_counts > 3, 3/8/neighbour_counts, 3/16)

		# each face gets replaced by faces (face_idx*4...face_idx*4+3)
		F_new = np.block([
			[F[:,0], n+he_to_E[:,2], n+he_to_E[:,1], n+he_to_E[:,1]],
			[n+he_to_E[:,2], F[:,1], n+he_to_E[:,0], n+he_to_E[:,2]],
			[n+he_to_E[:,1], n+he_to_E[:,0], F[:,2], n+he_to_E[:,0]]
		]).transpose()

		# construct S
		# V_new = S * V
		# S is (n + e) x n, so that V_new is (n + e) x 3

		# constructed using corr. (i, j, k) pairs
		# i determines which vertex (0...n+e-1) gets the result
		# j determines which vertex the position vector comes from
		# k determines the weight for this summand 

		i = np.concat([
			# new boundary vertices
			n + he_to_E[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1]],
			n + he_to_E[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1]],

			# new interior vertices
			n + he_to_E[interior_hedge_idxs[:, 0], interior_hedge_idxs[:, 1]],
			n + he_to_E[interior_hedge_idxs[:, 0], interior_hedge_idxs[:, 1]],

			# old boundary vertices
			# i find it helpful to somewhat think of this as a multi party computation
			# going around the boundary, say (a, b, c, d), we first see edge (a, b) -> (b, c) -> (c, d) -> (d, a)
			# note after (a, b), b never 'sees' a again (since unique halfedge for boundaries)
			# so, the only chance we have to 'give' b 1/8th of a's position vector is when we're processing the (a, b) edge. at the same time, we will 'take' 3/4th of a's own position. finally, (d, a) will come around and 'give' a the remaining 1/8th of d's vert position, completing the computation
			# so the recipients for (a, b) are (a, a, b)
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 0],
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 0],
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 1],
		
			# old interior vertices
			# similar idea, consider a star-like topology with vertex a at its center
			# (a, b) -> (a, c) -> ... -> (a, _)
			# give a (1/n) * (1 - n*beta) of a's position (since it will get this n times)
			# give a beta of the connected vertices position
			hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0],
			hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0]
		])

		j = np.concat([
			# new boundary vertices, centered at boundary edge
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 0],
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 1],

			# new interior vertices
			# left wing (other end is considered by opposing half-edge) 
			hes[interior_hedge_idxs[:, 0], interior_hedge_idxs[:, 1], 0],
			# head verts (other end is considered by opposing half-edge)
			# works because this half-edge is by definition opposing the vertex 
			# it shares its index with
			F[interior_hedge_idxs[:, 0], interior_hedge_idxs[:, 1]],
			
			# old boundary verts
			# the donors for (a, b) are (a, b, a) as discussed above
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 0],
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 1],
			hes[boundary_hedge_idxs[:, 0], boundary_hedge_idxs[:, 1], 0],

			# old interior verts
			hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0],
			hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 1]
		])

		k = np.concat([
			# new boundary vertices, midpoint
			np.full(boundary_hedge_idxs.shape[0] * 2, 0.5),

			# new interior vertices
			np.full(interior_hedge_idxs.shape[0], 0.375),
			np.full(interior_hedge_idxs.shape[0], 0.125),

			# old boundary verts
			# the weights for (a, b) are (3/4, 1/8, 1/8)
			np.full(boundary_hedge_idxs.shape[0], 0.75),
			np.full(boundary_hedge_idxs.shape[0], 0.125),
			np.full(boundary_hedge_idxs.shape[0], 0.125),

			# old interior verts
			(1/neighbour_counts[hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0]])-(betas[hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0]]),
			betas[hes[interior_tail_hes[:, 0], interior_tail_hes[:, 1], 0]]
		])

		S = scipy.sparse.csr_matrix((k, (i, j)), shape=(n + e, n))
		
		V_new = S * V

		return V_new, F_new, S


	S = scipy.sparse.eye(V.shape[0])

	V_new = np.copy(V)
	F_new = np.copy(F)

	for i in range(k):
		V_new, F_new, S_new = upsample_once(V_new, F_new)
		S = S_new * S

	return V_new, F_new

V, F = gpy.read_mesh("data/mug.obj")

k = 3

start = dt()
V_my, F_my = my_loop(V, F, k)
end = dt()

print(end - start)

start = dt()
V_gt, F_gt = gpy.subdivide(V, F, "loop", k)
end = dt()

print(end - start)

ps.init()
arma = ps.register_surface_mesh("unupped", V, F)
arma = ps.register_surface_mesh("upped_gt", V_gt, F_gt)
arma = ps.register_surface_mesh("upped_mine", V_my, F_my)
ps.show()