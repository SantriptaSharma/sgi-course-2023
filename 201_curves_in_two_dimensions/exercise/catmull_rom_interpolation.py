from cubic_hermite import cubic_hermite
from estimate_derivatives_catmull_rom import estimate_derivatives_catmull_rom
import numpy as np

def catmull_rom_interpolation(P, t):
    """
    P is a set of points, which we assume are corresponded to equally spaced
    times between 0 and 1. t is a scalar value between 0 and 1. Return the
    value of the catmull-rom interpolation of P at time t. You may use
    cubic_hermite.
    """
    
    n = P.shape[0]
    
    M = estimate_derivatives_catmull_rom(P)

    full_t = t * n
    edge_idx = int(np.ceil(full_t) - 1)
    next_idx = int((edge_idx + 1) % n)

    edge_t = full_t - edge_idx

    return cubic_hermite(P[edge_idx], P[next_idx], M[edge_idx], M[next_idx], edge_t)