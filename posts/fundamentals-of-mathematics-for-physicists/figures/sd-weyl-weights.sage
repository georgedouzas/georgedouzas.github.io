# Figure: the weights (h_1, ..., h_nu) of the Dirac spinor of Spin(2 nu), the eigenvalues of h_k = -i Sigma^{2k-1,2k}
# in the standard gamma matrices, coloured by the chirality gamma_*: Spin(4) (left) and Spin(6) (right, projected on
# an oblique projection of the cube (h_1, h_2, h_3)). Output: figures/sd-weyl-weights.svg.
import os, itertools
import numpy as np
from functools import reduce
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-weyl-weights.svg")
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def euclid(d):
    n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    return G
def weights(nu):
    G = euclid(2 * nu); N = 2 ** nu
    H = [-1j * 0.5 * G[2 * k] @ G[2 * k + 1] for k in range(nu)]
    gst = (1j) ** nu * reduce(np.dot, G, np.eye(N))
    assert all(np.allclose(h, np.diag(np.diag(h))) for h in H) and np.allclose(gst, np.diag(np.diag(gst)))
    return [(tuple(float(np.real(h[i, i])) for h in H), int(round(np.real(gst[i, i])))) for i in range(N)], H, gst
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
col = {1: "#1e8449", -1: "#c0392b"}
for ax, nu in zip(axs, (2, 3)):
    W, H, gst = weights(nu)
    proj = (lambda w: (w[0], w[1])) if nu == 2 else (lambda w: (w[0] + 0.45 * w[2], w[1] + 0.3 * w[2]))
    if nu == 3:
        for a, b in itertools.combinations([w for w, c in W], 2):
            if sum(abs(x - y) for x, y in zip(a, b)) == 1:
                ax.plot(*zip(proj(a), proj(b)), color="lightgrey", lw=0.8, zorder=1)
    for w, c in W:
        x, y = proj(w)
        ax.scatter([x], [y], s=120, color=col[c], zorder=3)
        if nu == 3:
            ax.annotate(r"$(%s)$" % ",".join("+" if t > 0 else "-" for t in w), (x, y), textcoords="offset points", xytext=(7, 5), fontsize=7)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$h_1$" if nu == 2 else r"$h_1 + 0.45\,h_3$"); ax.set_ylabel(r"$h_2$" if nu == 2 else r"$h_2 + 0.3\,h_3$")
    ax.set_title(r"Dirac spinor of $\mathrm{Spin}(%d)$" % (2 * nu), fontsize=10)
axs[0].scatter([], [], color=col[1], label=r"$\gamma_* = +1$"); axs[0].scatter([], [], color=col[-1], label=r"$\gamma_* = -1$")
axs[0].legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
ok = True
for nu in (2, 3, 4, 5):
    W, H, gst = weights(nu)
    ok &= sorted(w for w, c in W) == sorted(itertools.product((0.5, -0.5), repeat=nu))
    ok &= all(c == (-1) ** nu * int(np.prod([2 * x for x in w])) for w, c in W)
print("[PASS] the weights of the Dirac spinor of Spin(2 nu) are all (+-1/2, ..., +-1/2), nu = 2..5:", ok)
print("[PASS] gamma_* = (-1)^nu prod_k (2 h_k): chirality +1 iff an even number of h_k equal +1/2:", ok)
W, H, gst = weights(2)
print("[PASS] Spin(4): chirality +1 weights are (1/2,1/2), (-1/2,-1/2):", sorted(w for w, c in W if c == 1) == [(-0.5, -0.5), (0.5, 0.5)])
print("wrote", os.path.relpath(OUT))
