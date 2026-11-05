# Figure: the polar decomposition along the one-parameter subgroup exp(s (K1 + J3)): rapidity and rotation angle of the
# factors, and the direction of the boost part. Output: figures/sr-polar.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-polar.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
K1 = np.zeros((4, 4)); K1[0, 1] = K1[1, 0] = 1
J3 = np.zeros((4, 4)); J3[1, 2] = -1; J3[2, 1] = 1
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
ss = np.linspace(0, 6, 300); rap, ang, dirs = [], [], []
for s in ss:
    L = expm(s*(K1 + J3)); u = L[:, 0]; v = u[1:]/u[0]
    R = np.linalg.inv(boost(v)) @ L
    rap.append(np.arccosh(L[0, 0])); ang.append(np.arctan2(R[2, 1], R[1, 1]))
    dirs.append(v/np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else np.array([1., 0, 0]))
dirs = np.array(dirs)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ss, rap, color="royalblue", lw=2, label="rapidity of the boost part")
ax[0].plot(ss, np.unwrap(ang), color="darkorange", lw=2, label="angle of the rotation part (about $z$)")
ax[0].set_xlabel("$s$"); ax[0].legend(fontsize=8)
ax[1].plot(dirs[:, 0], dirs[:, 1], color="seagreen", lw=2); ax[1].plot(dirs[0, 0], dirs[0, 1], "o", color="seagreen")
ax[1].set_aspect("equal"); ax[1].set_xlim(-1.1, 1.1); ax[1].set_ylim(-1.1, 1.1); ax[1].set_xlabel("$\\hat v_x$"); ax[1].set_ylabel("$\\hat v_y$")
ax[1].set_title("direction of the boost part", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
L = expm(2.0*(K1 + J3)); u = L[:, 0]; B = boost(u[1:]/u[0]); w, V = np.linalg.eigh(L @ L.T)
print("[PASS] B(Lambda e0) is the positive square root of Lambda Lambda^T at s = 2:", np.allclose(V @ np.diag(np.sqrt(w)) @ V.T, B))
R = np.linalg.inv(B) @ L
print("[PASS] the rotation part fixes e0 and is orthogonal:", np.allclose(R[:, 0], [1, 0, 0, 0]) and np.allclose(R[1:, 1:] @ R[1:, 1:].T, np.eye(3)))
