import numpy as np

def closed_cylinder(nx,nz):
    """Returns a **closed** cylinder mesh.

    Parameters
    ----------
    nx : int
         number of vertices along the equator of the cylinder (at least 3)
    nz : int
         number of vertices on the z-axis of the cylinder (at least 2)

    Returns
    -------
    V : (n,3) numpy array
        vertex positions of the cylinder
    F : (m,3) numpy array
        face indices of the cylinder
    """

    vert_pos = np.zeros((nz, nx, 3))

    layer_heights = np.linspace(0, 1, nz)

    angles = np.linspace(0, 2*np.pi, nx+1)[:nx]
    vert_pos[:, :, 0] = np.cos(angles)[np.newaxis, :] * 0.5
    vert_pos[:, :, 1] = np.sin(angles)[np.newaxis, :] * 0.5
    vert_pos[:, :, 2] = layer_heights[:, np.newaxis]

    vert_idxs = np.arange(nx * (nz-1)).reshape(-1, nx)
    layer_start = (vert_idxs // nx) * nx
    vert_idxs = vert_idxs[np.newaxis, 0]

    faces = np.zeros((nz - 1, nx*2, 3))

    """ for nx = 4, nz = 2, we stitch two layers as follows:
    7      6   
    . ---- .
    |      |
    . ---- .
    4      5

    ||||||||
    vvvvvvvv

    3      2   
    . ---- .
    |      |
    . ---- .
    0      1
    """

    # lower triangles, 015, 126, 237, 304
    faces[:, ::2, 0] = layer_start + vert_idxs
    faces[:, ::2, 1] = layer_start + (vert_idxs + 1) % nx 
    faces[:, ::2, 2] = faces[:, ::2, 1] + nx

    # upper triangles, 054, 165, 276, 347
    faces[:, 1::2, 0] = layer_start + vert_idxs
    faces[:, 1::2, 1] = faces[:, ::2, 2]
    faces[:, 1::2, 2] = layer_start + vert_idxs + nx

    verts = vert_pos.reshape(-1, 3)
    faces = faces.reshape(-1, 3)

    verts = np.r_[verts, np.zeros(3)[np.newaxis, :], np.array([0, 0, 1])[np.newaxis, :]]
    n_verts = verts.shape[0]

    # if only i knew how to use block well enough, too tired to look up
    faces = np.r_[
        faces,
        np.c_[vert_idxs[0], (vert_idxs[0] + 1) % nx, np.repeat(n_verts-2, nx)],
        np.c_[(nz - 1) * nx + vert_idxs[0], (nz - 1) * nx + (vert_idxs[0] + 1) % nx, np.repeat(n_verts-1, nx)]
    ]

    return verts, faces

V, F = closed_cylinder(18, 24)

import polyscope as ps

import gpytoolbox as gpy

ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.register_surface_mesh("cylinder", V, F)
ps.show()