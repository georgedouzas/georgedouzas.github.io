# Exercise exr-sr-velocity-triangle: angles of a triangle of velocities on the hyperboloid, its defect, and the rotation
# produced by the closed loop of pure boosts u0 -> u1 -> u2 -> u0.
import numpy as np
chi = 1.0
u0 = np.array([1, 0, 0, 0.]); u1 = np.array([np.cosh(chi), np.sinh(chi), 0, 0]); u2 = np.array([np.cosh(chi), 0, np.sinh(chi), 0])
eta = np.diag([1., -1, -1, -1]); dot = lambda p, q: p @ eta @ q
a = np.arccosh(dot(u0, u1)); b = np.arccosh(dot(u0, u2)); c = np.arccosh(dot(u1, u2))
ang = lambda opp, s1, s2: np.arccos((np.cosh(s1)*np.cosh(s2) - np.cosh(opp))/(np.sinh(s1)*np.sinh(s2)))
A0, A1, A2 = ang(c, a, b), ang(b, a, c), ang(a, b, c)
print("sides: %.4f %.4f %.4f   angles: %.4f %.4f %.4f   sum %.4f   defect %.4f" % (a, b, c, A0, A1, A2, A0 + A1 + A2, np.pi - A0 - A1 - A2))
def boost_between(u, w):     # the pure boost in the plane of u and w that maps u to w
    return np.eye(4) - np.outer(u + w, eta @ (u + w))/(1 + dot(u, w)) + 2*np.outer(w, eta @ u)
L = boost_between(u2, u0) @ boost_between(u1, u2) @ boost_between(u0, u1)
print("loop maps u0 to itself:", np.allclose(L @ u0, u0))
R = L[1:, 1:]
print("rotation angle of the loop: %.4f rad about the axis %s" % (np.arccos((np.trace(R) - 1)/2), np.round([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], 4)))
