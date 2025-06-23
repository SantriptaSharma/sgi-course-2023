import numpy as np

def estimate_derivatives_catmull_rom(P):
    """
    Assuming P are the points of a closed curve, use the Catmull-Rom
    technique to estimate the derivative at each point of P by looking at its
    neighbors.
    """
    
    derivatives = 0.5 * (np.roll(P, -1, 0) - np.roll(P, 1, 0))
    return derivatives
