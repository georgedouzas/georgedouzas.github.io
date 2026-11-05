# Figure: the four components of O(1,3). Random products of boosts, rotations and 1, P, T, PT plotted by det (jittered)
# and Lambda^0_0. Output: figures/sr-components.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-components.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(int(11))
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v; B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def rot():
    q, r = np.linalg.qr(rng.normal(size=(3, 3))); q = q*np.sign(np.diag(r)); q *= np.linalg.det(q)
    R = np.eye(4); R[1:, 1:] = q; return R
D = [np.eye(4), np.diag([1., -1, -1, -1]), np.diag([-1., 1, 1, 1]), -np.eye(4)]
cols = ["royalblue", "darkorange", "seagreen", "gray"]; labs = [r"$L^\uparrow_+$", r"$L^\uparrow_-$", r"$L^\downarrow_-$", r"$L^\downarrow_+$"]
fig, ax = plt.subplots(figsize=(6, 5)); ok = True
for k in range(4):
    pts = []
    for _ in range(500):
        v = rng.normal(size=3); v *= rng.uniform(0, 0.95)/np.linalg.norm(v)
        M = boost(v) @ rot() @ boost(rng.uniform(-0.6, 0.6, size=3)/np.sqrt(3)) @ D[k]
        ok &= abs(M[0, 0]) >= 1 - 1e-12
        pts.append((np.linalg.det(M) + rng.normal(scale=0.05), M[0, 0]))
    pts = np.array(pts); ax.scatter(pts[:, 0], pts[:, 1], s=4, color=cols[k], label=labs[k])
ax.set_xlabel(r"$\det\Lambda$ (jittered)"); ax.set_ylabel(r"$\Lambda^0{}_0$"); ax.set_ylim(-5, 5)
ax.axhspan(-1, 1, color="red", alpha=0.08); ax.text(0, 0, r"no elements with $|\Lambda^0{}_0| < 1$", ha="center", va="center", fontsize=9); ax.legend(fontsize=9, markerscale=3, loc="upper center")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] |Lambda^0_0| >= 1 for all 2000 sampled Lorentz matrices:", ok)
