# Exercise exr-sr-polar-compute: the polar decomposition of B(0.7 y) B(0.6 x) by the square root of Lambda Lambda^T and
# by the boost determined by Lambda e0; axis and angle of the rotation part.
import numpy as np
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v; B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
Lam = boost([0, 0.7, 0]) @ boost([0.6, 0, 0])
w, V = np.linalg.eigh(Lam @ Lam.T)
B1 = V @ np.diag(np.sqrt(w)) @ V.T
u = Lam[:, 0]; B2 = boost(u[1:]/u[0])
print("velocity of the boost part:", np.round(u[1:]/u[0], 4), " |v| = %.4f" % np.linalg.norm(u[1:]/u[0]))
print("sqrt(Lambda Lambda^T) equals B(Lambda e0):", np.allclose(B1, B2))
R = np.linalg.inv(B2) @ Lam
print("rotation part:"); print(np.round(R, 6))
R3 = R[1:, 1:]; ang = np.arccos((np.trace(R3) - 1)/2)
axis = np.array([R3[2, 1] - R3[1, 2], R3[0, 2] - R3[2, 0], R3[1, 0] - R3[0, 1]])/(2*np.sin(ang))
print("angle = %.4f rad, axis =" % ang, np.round(axis, 6))
