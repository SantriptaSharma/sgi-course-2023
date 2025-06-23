from closed_catmull_rom_sdf import distance_to_cubic_segment
import numpy as np
import matplotlib.pyplot as plt

p0 = np.array([0, 0])
p1 = np.array([1, 0])
m0 = np.array([0, 1])
m1 = np.array([0, 1])

x = np.linspace(-0.3, 1.3, 500)
y = np.linspace(-0.3, 0.5, 500)

X, Y = np.meshgrid(x, y)
points = np.array([X.flatten(), Y.flatten()]).T

n = points.shape[0]
dists = distance_to_cubic_segment(points, p0[np.newaxis, :], p1[np.newaxis, :], m0[np.newaxis, :], m1[np.newaxis, :])

print(dists)
dists = dists.reshape(X.shape)

plt.pcolormesh(X, Y, dists, cmap="RdBu", vmin=0.0, vmax=2.0)
plt.colorbar()
plt.contour(X, Y, dists, levels=[0.002, 0.15], colors=["black", "yellow"], linewidths=5)
plt.show()