import gpytoolbox as gpy, polyscope as ps, numpy as np

def plot_z_coord(V,F):
    """This method plots the z-cordinate on the input mesh V,F
    """

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    mesh.add_scalar_quantity("z-coord", V[:, 2], enabled=True)
    ps.show()

V, F = gpy.read_mesh("data/spot.obj")
plot_z_coord(V, F)