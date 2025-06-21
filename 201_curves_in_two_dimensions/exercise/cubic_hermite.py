def cubic_hermite(P0, P1, M0, M1, t):
    """
    Given a pair of points and a pair of vectors, compute the cubic Hermite
    polynomial they define and evaluate it at time t.
    """
    
    a = M1 + M0 - 2*P1 + 2*P0
    b = 3*P1 - 3*P0 - 2*M0 - M1
    c = M0
    d = P0

    return a * t**3 + b * t**2 + c * t + d

import numpy as np
import matplotlib.pyplot as plt

p0 = np.array([0, 0])
p1 = np.array([1, 0])
m0 = np.array([0, 1])
m1 = np.array([0, -1])
t = 0.5
Pt = cubic_hermite(p0, p1, m0, m1, t)
print(Pt)

curve = []  # Initialize your curve to nothing
t = np.linspace(0, 1, 100)  # t is a vector of 100 values between 0 and 1

for ti in t:
    curve.append(cubic_hermite(p0, p1, m0, m1, ti))  # add new point to the curve

curve = np.array(curve)
plt.plot(curve[:, 0], curve[:, 1])
plt.axis('equal')
plt.show()
