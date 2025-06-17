import gpytoolbox as gpy, polyscope as ps, numpy as np
from timeit import default_timer

def tangents(V,F):
    """
    Computes two orthogonal, oriented tangent vectors for each face in a
    triangle mesh.
    """

    # Extract the first edge of each face and normalize it.
    edges_a = V[F[:, 1]] - V[F[:, 0]]
    edges_a /= np.linalg.norm(edges_a, axis=-1)[:, np.newaxis]

    # Extract the second edges and project onto the orthogonal complement of E1.
    edges_b = V[F[:, 2]] - V[F[:, 1]]
    edges_b -= np.sum(edges_a * edges_b, axis=-1)[:, np.newaxis] * edges_a

    # Normalize to get unit vectors
    edges_b /= np.linalg.norm(edges_b, axis=-1)[:, np.newaxis]

    return edges_a, edges_b

V, F = gpy.read_mesh("data/spot_low_resolution.obj")

start = default_timer()
tangent, bitangent = tangents(V, F)
end = default_timer()

print(end - start)

ps.init()
spot = ps.register_surface_mesh("spot", V, F)
spot.add_vector_quantity("tangent", tangent, defined_on="faces")
spot.add_vector_quantity("bitangent", bitangent, defined_on="faces")
ps.show()