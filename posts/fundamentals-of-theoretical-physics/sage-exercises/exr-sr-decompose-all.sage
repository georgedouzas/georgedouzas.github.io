# Exercise exr-sr-decompose-all: component, reduction to SO+(1,3), polar decomposition and spinor lift of
# B(0.5, 0.2, 0.1) R_n(0.7) P with n = (0, 1, 1)/sqrt 2.
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

from scipy.linalg import expm
n = np.array([0, 1, 1])/np.sqrt(2); Kn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
R = np.eye(4); R[1:, 1:] = expm(0.7*Kn); P = np.diag([1., -1, -1, -1])
Lam = boost([0.5, 0.2, 0.1]) @ R @ P
print("in O(1,3):", np.allclose(Lam.T @ eta @ Lam, eta), " det = %.0f, Lambda00 = %.4f" % (np.linalg.det(Lam), Lam[0, 0]))
Lp = Lam @ P
u = Lp[:, 0]; v = u[1:]/u[0]; Rr = np.linalg.inv(boost(v)) @ Lp; ang, ax = angle_axis(Rr)
print("boost velocity:", np.round(v, 6), "  rotation angle %.6f about" % ang, np.round(ax, 6))
s = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
xi = np.arctanh(np.linalg.norm(v)); vh = v/np.linalg.norm(v)
A = expm(xi/2*sum(vh[i]*s[i + 1] for i in range(3))) @ expm(-0.5j*ang*sum(ax[i]*s[i + 1] for i in range(3)))
LamA = np.array([[0.5*np.trace(s[m] @ A @ s[k] @ A.conj().T).real for k in range(4)] for m in range(4)])
print("Lambda(A) equals Lambda P:", np.allclose(LamA, Lp), "  det A = %.6f" % abs(np.linalg.det(A)))
