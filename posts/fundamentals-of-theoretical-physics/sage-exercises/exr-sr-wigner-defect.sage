# Exercise exr-sr-wigner-defect: the Wigner rotation of B(0.7 y) B(0.6 x) as the holonomy of the triangle e0, B(v2) e0,
# B(v2) B(v1) e0 on the hyperboloid of velocities, compared with the angular defect.
import numpy as np
eta = np.diag([1., -1, -1, -1])
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def angle_axis(R):
    R3 = R[1:, 1:]; ang = np.arccos(np.clip((np.trace(R3) - 1)/2, -1, 1))
    ax = np.array([R3[2, 1] - R3[1, 2], R3[0, 2] - R3[2, 0], R3[1, 0] - R3[0, 1]])
    return ang, ax/np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-14 else ax

dot = lambda p, q: p @ eta @ q
def P(a, b): return np.eye(4) - np.outer(a + b, eta @ (a + b))/(1 + dot(a, b)) + 2*np.outer(b, eta @ a)
e0 = np.array([1., 0, 0, 0]); B1, B2 = boost([0.6, 0, 0]), boost([0, 0.7, 0]); w = B2 @ e0; u = B2 @ B1 @ e0
print("B2 B1 = P(w -> u) P(e0 -> w):", np.allclose(B2 @ B1, P(w, u) @ P(e0, w)))
W = P(u, e0) @ P(w, u) @ P(e0, w)
side = lambda p, q: np.arccosh(dot(p, q))
a, b, c = side(e0, w), side(w, u), side(e0, u)
ang = lambda opp, s1, s2: np.arccos((np.cosh(s1)*np.cosh(s2) - np.cosh(opp))/(np.sinh(s1)*np.sinh(s2)))
defect = np.pi - ang(b, a, c) - ang(c, a, b) - ang(a, b, c)
print("rotation angle of the loop %.4f, defect of the triangle %.4f" % (angle_axis(W)[0], defect))
u1 = B1 @ e0; a, b, c = side(e0, u1), side(u1, u), side(e0, u)
print("defect of the other triangle e0, B1 e0, u: %.4f" % (np.pi - ang(b, a, c) - ang(c, a, b) - ang(a, b, c)))
