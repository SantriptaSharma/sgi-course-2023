import numpy as np

def optimal_location_and_cost(Qe):
    """
    Compute the optimal vertex location after a single edge removal and evaluate its cost (quadric error)

    Inputs:
        Qe: (4,4) edge quadric

    Outputs:
        v_optimal: (3,1) optimal vertex position
        quadric_error: scalar quadric error
    """

    #### Fill in the missing part #####
    A = Qe[:3, :3]
    b = -Qe[:3, 3]
    c = Qe[3, 3]

    v_opt = np.linalg.solve(A, b)
    quadric_error = np.dot((v_opt @ A), v_opt) - 2 * np.dot(b, v_opt) + c

    ###################################

    return v_opt, quadric_error