# Exercise exr-sr-wigner-reversed: the two composite velocities of 0.6 x and 0.7 y and the angle between them, compared
# with the angle of the Wigner rotation.
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

v1, v2 = np.array([0.6, 0, 0]), np.array([0, 0.7, 0])
U12 = boost(v2) @ boost(v1) @ np.array([1., 0, 0, 0]); U21 = boost(v1) @ boost(v2) @ np.array([1., 0, 0, 0])
u, up = U12[1:]/U12[0], U21[1:]/U21[0]
print("v1 + v2 =", np.round(u, 4), "  v2 + v1 =", np.round(up, 4), "  lengths %.4f %.4f" % (np.linalg.norm(u), np.linalg.norm(up)))
print("angle between them: %.4f rad" % np.arccos(u @ up/np.linalg.norm(u)/np.linalg.norm(up)))
W = np.linalg.inv(boost(u)) @ boost(v2) @ boost(v1)
print("Wigner rotation: angle %.4f about" % angle_axis(W)[0], np.round(angle_axis(W)[1], 6))
