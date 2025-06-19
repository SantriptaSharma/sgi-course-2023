import torch

def compute_barycentric_coords(triangles, points):
    """ Compute the barycentric coordinates for a collection of points with respect to a
    set of triangles.

    Args:
        triangles (torch.tensor): F x 3 x 2 array of faces in UV space
        points (torch.tensor): N x 2 array of texel coordinates
    
    Returns:
        barycentric_coords (torch.tensor): N x F x 3 array of barycentric coordinates
    """
    
    # F x 2
    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]

    # (N x 1 x 2) - (1 x F x 2) -> (N x F x 2)
    # so for each point, for each triangle, distance from local origin 
    v2 = points[:, torch.newaxis] - triangles[torch.newaxis, :, 0]

    d00 = torch.sum(v0 * v0, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)

    d20 = torch.sum(v2 * v0[torch.newaxis, :], dim=2)
    d21 = torch.sum(v2 * v1[torch.newaxis, :], dim=2)

    D = d00 * d11 - d01 * d01
    
    betas = (d20 * d11 - d01 * d21) / D
    gammas = (d00 * d21 - d01 * d20) / D
    alphas = 1 - (betas + gammas)

    return torch.stack([alphas, betas, gammas], dim=-1)