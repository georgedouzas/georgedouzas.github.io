# Exercise exr-sr-induced-wigner: the little-group element W = B_p^{-1} B(0.8 y) B_q for a massive particle with
# p = (cosh 1, sinh 1, 0, 0), compared with the Wigner angle of the pair (v_q, 0.8 y).
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

p = np.array([np.cosh(1), np.sinh(1), 0, 0]); Bw = boost([0, 0.8, 0]); q = np.linalg.inv(Bw) @ p
vp, vq = p[1:]/p[0], q[1:]/q[0]
W = np.linalg.inv(boost(vp)) @ Bw @ boost(vq)
Om, ax = angle_axis(W)
x1, x2 = np.arctanh(np.linalg.norm(vq)), np.arctanh(0.8); n1, n2 = vq/np.linalg.norm(vq), np.array([0, 1., 0])
th = np.arccos(n1 @ n2)
print("W is a rotation:", np.allclose(W[:, 0], [1, 0, 0, 0]), " angle %.6f about" % Om, np.round(ax, 6))
print("Wigner formula: %.6f" % (2*np.arctan(np.sin(th)/(1/np.tanh(x1/2)/np.tanh(x2/2) + np.cos(th)))))
