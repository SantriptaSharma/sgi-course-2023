import numpy as np

def face_areas(V, F):
    """
    Compute triangle area per face

    Inputs:
        V: |V|x3 numpy array of vertex positions
        F: |F|x3 numpy array of face indices
    Outputs:
        FA: |F| numpy array of face areas
    """

    #### Fill in the missing part #####

    ab = V[F[:, 1]] - V[F[:, 0]]
    ac = V[F[:, 2]] - V[F[:, 0]]

    areas = np.linalg.norm(np.cross(ab, ac), axis=-1) / 2

    ###################################
    
    return areas

import gpytoolbox as gpy

V, F = gpy.read_mesh("data/spot.obj")
print(V.shape[0], F.shape[0])

areas_gt = gpy.doublearea(V, F) / 2
areas_mine = face_areas(V, F)

diff = np.abs(areas_gt - areas_mine)
print(np.all(diff < 1e-5))