import numpy as np
import scipy as sp
import gpytoolbox as gpy

def solve_cubic_hermite(P0, P1, M0, M1):
    """
    Given a pair of points and a pair of vectors, compute the cubic Hermite
    polynomial they define, returning the vector coefficients
    """
    
    a = M1 + M0 - 2*P1 + 2*P0
    b = 3*P1 - 3*P0 - 2*M0 - M1
    c = M0
    d = P0

    return a, b, c, d

def distance_to_cubic_segment(query_points, P0, P1, M0, M1, opt_inits=3, max_iter=50):
    a, b, c, d = solve_cubic_hermite(P0, P1, M0, M1)

    aa = np.sum(a * a, axis=-1)
    bb = np.sum(b * b, axis=-1)
    cc = np.sum(c * c, axis=-1)

    ab = np.sum(a * b, axis=-1)
    ac = np.sum(a * c, axis=-1)
    ad = np.sum(a * d, axis=-1)
    ap = np.sum(a * query_points, axis=-1)

    bc = np.sum(b * c, axis=-1)
    bd = np.sum(b * d, axis=-1)
    bp = np.sum(b * query_points, axis=-1)

    cd = np.sum(c * d, axis=-1)
    cp = np.sum(c * query_points, axis=-1)

    deriv = lambda t: 6*t**5*aa + 10*t**4*ab + 4*t**3*(bb + 2*ac) + 6*t**2*(ad + bc - ap) + 2*t*(cc + 2*bd - 2*bp) + 2*(cd - cp)

    ts = np.array([sp.optimize.newton(deriv, np.repeat(x, query_points.shape[0]), maxiter=max_iter)[:, np.newaxis] for x in np.linspace(0, 1, opt_inits)])

    ts = np.clip(ts, 0.0, 1.0)

    ps = a[np.newaxis, :, :]*ts**3 + b[np.newaxis, :, :]*ts**2 + c[np.newaxis, :, :]*ts + d[np.newaxis, :, :]

    dists = np.linalg.norm(ps - query_points[np.newaxis, :], axis = -1)
    
    dists = np.min(dists, axis=0)

    # check against endpoints
    dists = np.c_[dists, np.linalg.norm(d - query_points, axis=-1), np.linalg.norm((a+b+c+d) - query_points, axis=-1)]

    dists = np.min(dists, axis=1)

    return dists

from upsample_spline import upsample_spline
from estimate_derivatives_catmull_rom import estimate_derivatives_catmull_rom

def closed_catmull_rom_sdf(query_point, spline_points, sample_rate_mult=2, t_inits=3, max_iter=50):
    """
    Given a cubic Catmull-Rom spline defined by the points in spline_points,
    return the signed distance from the query_point to the spline. The spline
    is closed, so the first point in spline_points is connected to the last
    point in spline_points.
    """

    # shannon save me
    splineified_points = upsample_spline(spline_points, spline_points.shape[0] * sample_rate_mult)    
    sgn_dist, edge_idx, _ = gpy.signed_distance(query_point, splineified_points)
    
    # sgn_dist, edge_idx, _ = gpy.signed_distance(query_point, spline_points)

    edge_idx = np.floor(edge_idx / sample_rate_mult).astype(np.int64)

    P0 = spline_points
    P1 = np.roll(P0, -1)
    M0 = estimate_derivatives_catmull_rom(spline_points)
    M1 = np.roll(M0, -1)

    print("calling into cubic seg")
    
    return distance_to_cubic_segment(query_point, P0[edge_idx], P1[edge_idx], M0[edge_idx], M1[edge_idx], opt_inits=t_inits, max_iter=max_iter) * np.sign(sgn_dist)

V = max(gpy.png2poly("data/switzerland.png"), key=len)

print(V.shape)
V = gpy.normalize_points(V)

x = np.linspace(-0.75, 0.75, 200)
y = np.linspace(-0.75, 0.75, 200)
X, Y = np.meshgrid(x, y)
qp = np.array([X.ravel(), Y.ravel()]).T

dists = closed_catmull_rom_sdf(qp, V, sample_rate_mult=1, t_inits=5, max_iter=50)
dists = dists.reshape(X.shape)
print(np.min(dists), np.max(dists))

import matplotlib.pyplot as plt

_ = plt.pcolormesh(X, Y, dists, cmap = 'RdBu', vmin = np.min(dists), vmax = np.max(dists))
# Add polyline
_ = plt.plot(V[:, 0], V[:, 1], '-k', linewidth=5)
_ = plt.colorbar()
_ = plt.contour(X, Y, dists, levels=[0.003, 0.1], colors=["green", "yellow"], linewidths=2)
_ = plt.axis('equal')

plt.show()