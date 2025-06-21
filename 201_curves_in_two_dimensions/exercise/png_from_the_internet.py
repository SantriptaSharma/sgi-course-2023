#  Choose a png file from the internet that represents something from your
#  home country, your life or just something you generally like, and plot
#  its polyline silhouette.

import gpytoolbox as gpy

polylines = gpy.png2poly("data/delhi_metro.png")

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
cols = list(np.random.uniform(0, 1, (len(polylines), 3)))

for (V, C) in zip(polylines, cols):
	plt.plot(V[:, 0], V[:, 1], "o-", color=C, lw=0.5)

plt.show()

print(len(polylines))