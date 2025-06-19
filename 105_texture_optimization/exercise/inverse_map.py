import torch
from .compute_barycentric_coords import compute_barycentric_coords

def inverse_map(vertices, faces, uv_triangles, texels, tolerance=1e-6):
   """ Compute the inverse map from texels to surface points.

   Recommended approach:
   1. Scale triangles to the scale of the texels (0-1 --> 0-texture_image_size)
   2. Determine which texels are covered by which triangles
      (can do this using barycentric coordinates)
   3. Compute the barycentric coordinates for each covered texel with respect to the triangle that covers it
   4. For each covered texel, compute its 3D coordinate on the surface by using
      barycentric interpolation (use the barycentric coordinates of the triangle that covers it and the 3D vertices of that triangle)

   Args:
      vertices (torch.tensor): V x 3 array of vertex coordinates
      faces (torch.tensor): F x 3 array of triangle vertex indices
      uv_triangles (torch.tensor): F x 3 x 2 array of triangle coordinates in UV space
      texels (torch.tensor): N x 2 array of texel coordinates
      tolerance (float): Tolerance for barycentric coordinates validity. Use this when checking if a texel is covered by a triangle.

   Returns:
      surface_points (torch.tensor): N x 3 array of surface points
      texel_indices (torch.tensor): N x 1 array of texel indices
   """
   # scale triangles to texel coordinates
   tex_max = torch.max(texels) + 1
   trigs = uv_triangles * tex_max

   # Compute the barycentric coordinates for each texel with respect to each triangle
   tex_bary = compute_barycentric_coords(trigs, texels)

   # For each texel, for each triangle, determine if the texel lies inside the triangle
   in_trig_mask = (tex_bary[:, :, 1] + tex_bary[:, :, 2] <= 1) & (tex_bary[:, :, 1] >= 0) & (tex_bary[:, :, 2] >= 0)

   # Get indices of valid texels and corresponding triangles
   covered_tex_idxs = torch.nonzero(in_trig_mask)
   # this is Z x 2 (where Z is the number of covered texels)

   # Select valid barycentric coordinates
   covered_bary = tex_bary[covered_tex_idxs]
   # Z x 3 (alpha, beta, gamma)

   # Select corresponding triangle vertices for each valid point
   covered_verts = vertices[faces[covered_tex_idxs[:, 1]]]
   # comes out to (in my account): Z x 3 x 3 (three verts per face, three coords each)

   # Compute 3D coordinates for valid points
   # Hint: Use the barycentric coordinates of the texel and the vertices of the triangle
   surf_pts = torch.sum(covered_bary[:, :, torch.newaxis] * covered_verts, axis=1)

   # Return the surface points and texel indices
   return surf_pts, covered_tex_idxs
