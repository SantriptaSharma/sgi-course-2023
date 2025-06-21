import numpy as np

def closed_cone(nx,nz):
    """Returns a **closed** cone mesh.

    Parameters
    ----------
    nx : int
         number of vertices along the base of the cone (at least 3)
    nz : int
         number of vertices on the z-axis of the cone (at least 2)

    Returns
    -------
    V : (n,3) numpy array
        vertex positions of the cone
    F : (m,3) numpy array
        face positions of the cone
    """

    vert_pos = np.zeros((nz - 1, nx, 3))

    layer_heights = np.linspace(0, 1, nz)[:-1]

    angles = np.linspace(0, 2*np.pi, nx+1)[:nx]
    radii = np.linspace(0.5, 0, nz)[:-1]
    vert_pos[:, :, 0] = np.cos(angles)[np.newaxis, :] * radii[:, np.newaxis]
    vert_pos[:, :, 1] = np.sin(angles)[np.newaxis, :] * radii[:, np.newaxis]
    vert_pos[:, :, 2] = layer_heights[:, np.newaxis]

    vert_idxs = np.arange(nx * (nz-2)).reshape(-1, nx)
    layer_start = (vert_idxs // nx) * nx
    vert_idxs = vert_idxs[np.newaxis, 0]

    faces = np.zeros((nz - 2, nx*2, 3))

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
        np.c_[(nz - 2) * nx + vert_idxs[0], (nz - 2) * nx + (vert_idxs[0] + 1) % nx, np.repeat(n_verts-1, nx)]
    ]

    return verts, faces

V, F = closed_cone(18, 24)

import polyscope as ps

import gpytoolbox as gpy

ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.register_surface_mesh("cylinder", V, F)
ps.show()