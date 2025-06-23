import numpy as np
from catmull_rom_interpolation import catmull_rom_interpolation

def upsample_spline(V, n):
    """
    Sample n equally spaced points on the Catmull-Rom spline defined by V.
    """
    
    ts = np.linspace(0, 1, n+1)

    pts = catmull_rom_interpolation(V, ts)

    return pts
