import numpy as np

def estimate_derivatives_catmull_rom(P):
    """
    Assuming P are the points of a closed curve, use the Catmull-Rom
    technique to estimate the derivative at each point of P by looking at its
    neighbors.
    """
    
    derivatives = 0.5 * (np.roll(P, -1, 0) - np.roll(P, 1, 0))
    return derivatives

import numpy as np
import matplotlib.pyplot as plt

th = np.linspace(0, 2*np.pi, 20)
th = th[:-1]
V = np.column_stack((np.cos(th), np.sin(th)))

M = estimate_derivatives_catmull_rom(V)

plt.quiver(V[:,0], V[:,1], M[:,0], M[:,1])
plt.axis('equal')
plt.show()
