# Exercise exr-sr-wigner-random: the Wigner rotation of random pairs of boosts against the half-angle formula, the cosine
# formula and the axis n1 x n2.
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

rng = np.random.default_rng(int(9)); worst = [0, 0, 0]
for _ in range(100):
    v1 = rng.normal(size=3); v1 *= rng.uniform(0.01, 0.99)/np.linalg.norm(v1)
    v2 = rng.normal(size=3); v2 *= rng.uniform(0.01, 0.99)/np.linalg.norm(v2)
    U = boost(v2) @ boost(v1) @ np.array([1., 0, 0, 0]); W = np.linalg.inv(boost(U[1:]/U[0])) @ boost(v2) @ boost(v1)
    Om, ax = angle_axis(W)
    x1, x2 = np.arctanh(np.linalg.norm(v1)), np.arctanh(np.linalg.norm(v2)); n1, n2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    th = np.arccos(n1 @ n2)
    tan_f = 2*np.arctan(np.sin(th)/(1/np.tanh(x1/2)/np.tanh(x2/2) + np.cos(th)))
    g1, g2 = np.cosh(x1), np.cosh(x2); g = g1*g2*(1 + v1 @ v2)
    cos_f = np.arccos((1 + g1 + g2 + g)**2/((1 + g1)*(1 + g2)*(1 + g)) - 1)
    axis = np.cross(n1, n2)/np.linalg.norm(np.cross(n1, n2))
    worst = [max(worst[0], abs(Om - tan_f)), max(worst[1], abs(Om - cos_f)), max(worst[2], np.abs(ax - axis).max())]
print("max |Omega - tan formula| = %.1e, max |Omega - cos formula| = %.1e, max |axis - n1 x n2| = %.1e" % tuple(worst))
