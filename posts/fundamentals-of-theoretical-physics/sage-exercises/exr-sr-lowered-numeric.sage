# Exercise exr-sr-lowered-numeric: the identity Lambda_mu^nu = (Lambda^-1)^nu_mu and the invariance of the trace and
# determinant of a (1,1) tensor, for a boost times a rotation.
import numpy as np
eta = np.diag([1., -1, -1, -1])
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v; B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def rotation(axis, th):
    n = np.array(axis, float)/np.linalg.norm(axis); K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = np.eye(4); R[1:, 1:] = np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K
    return R
Lam = boost([0.3, -0.4, 0.2]) @ rotation([1, 1, 1], 1.0)
print("Lambda in O(1,3):", np.allclose(Lam.T @ eta @ Lam, eta))
print("max |eta Lambda eta^-1 - (Lambda^-1)^T| =", np.abs(eta @ Lam @ np.linalg.inv(eta) - np.linalg.inv(Lam).T).max())
rng = np.random.default_rng(int(3)); Tt = rng.normal(size=(4, 4))
Tp = Lam @ Tt @ np.linalg.inv(Lam)
print("trace: %.12f %.12f   det: %.12f %.12f" % (np.trace(Tt), np.trace(Tp), np.linalg.det(Tt), np.linalg.det(Tp)))
