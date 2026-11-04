# Exercise exr-sr-composite-velocity: Einstein's addition of velocities in arbitrary directions and its Lorentz factor,
# against the product of boost matrices.
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

rng = np.random.default_rng(int(4)); worst = 0
for _ in range(100):
    v1 = rng.normal(size=3); v1 *= rng.uniform(0, 0.99)/np.linalg.norm(v1)
    v2 = rng.normal(size=3); v2 *= rng.uniform(0, 0.99)/np.linalg.norm(v2)
    g2 = 1/np.sqrt(1 - v2 @ v2); n2 = v2/np.linalg.norm(v2)
    u = (v2 + v1/g2 + (1 - 1/g2)*(n2 @ v1)*n2)/(1 + v1 @ v2)
    U = boost(v2) @ boost(v1) @ np.array([1., 0, 0, 0])
    g1 = 1/np.sqrt(1 - v1 @ v1)
    worst = max(worst, np.abs(U[1:]/U[0] - u).max(), abs(U[0] - g1*g2*(1 + v1 @ v2)))
print("largest deviation over 100 random pairs: %.2e" % worst)
