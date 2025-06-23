import numpy as np

def cubic_hermite(P0, P1, M0, M1, t):
    """
    Given a pair of points and a pair of vectors, compute the cubic Hermite
    polynomial they define and evaluate it at time t.
    """
    
    if type(t) is np.ndarray:
        t = t[:, np.newaxis]

    a = M1 + M0 - 2*P1 + 2*P0
    b = 3*P1 - 3*P0 - 2*M0 - M1
    c = M0
    d = P0

    return a * t**3 + b * t**2 + c * t + d