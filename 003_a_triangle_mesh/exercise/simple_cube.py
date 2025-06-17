import numpy as np

def simple_cube():
    """Construct a triangle mesh for a single cube.

    This function returns two variables, the vertex-list V and the face-list
    F describing a triangle mesh of a single cube.
    """

    V = np.array([[0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]])

    F = np.array([
        [0, 1, 2],
        [2, 3, 0],
        
        # bottom
        [0, 1, 5],
        [5, 4, 0]
    ])

    return V, F
