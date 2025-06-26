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