import numpy as np
from catmull_rom_interpolation import catmull_rom_interpolation

def upsample_spline(V, n):
    """
    Sample n equally spaced points on the Catmull-Rom spline defined by V.
    """
    
    ts = np.linspace(0, 1, n+1)

    pts = []

    for t in ts:
        pts.append(catmull_rom_interpolation(V, t))

    return np.array(pts)

import numpy as np
import matplotlib.pyplot as plt

th = np.linspace(0, 2*np.pi, 8)
th = th[:-1]
V = np.column_stack((np.cos(th), np.sin(th)))
plt.plot(V[:, 0], V[:, 1], linewidth=3)
plt.axis('equal')
plt.axis('off')
plt.show()

U = upsample_spline(V, 100)
plt.plot(U[:, 0], U[:, 1], linewidth=3)
plt.axis('equal')
plt.axis('off')
plt.show()
