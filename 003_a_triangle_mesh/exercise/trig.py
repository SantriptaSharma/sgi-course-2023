import numpy as np

V = np.array([
	[1, 0, 0],
	[3, 0, 1],
	[1.5, 2, 4]
])

F = np.array([
	[0, 1, 2]
])

import polyscope as ps

ps.init()
ps.register_surface_mesh("triangle", V, F)
ps.show()