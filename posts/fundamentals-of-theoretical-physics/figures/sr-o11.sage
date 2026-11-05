# Figure: the four pieces of O(1,1). Images of the rest vector (1,0) and the unit space vector (0,1) under L(eta) D for
# D = 1, P, T, PT, on the hyperbolae t^2 - x^2 = 1 and x^2 - t^2 = 1. Output: figures/sr-o11.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-o11.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
L = lambda e: np.array([[np.cosh(e), np.sinh(e)], [np.sinh(e), np.cosh(e)]])
Ds = {"1": np.eye(2), "P": np.diag([1., -1]), "T": np.diag([-1., 1]), "PT": -np.eye(2)}
cols = {"1": "royalblue", "P": "darkorange", "T": "seagreen", "PT": "gray"}
fig, ax = plt.subplots(figsize=(6, 6))
etas = np.linspace(-1.4, 1.4, 120)
for name, D in Ds.items():
    a = np.array([L(e) @ D @ np.array([1., 0]) for e in etas]); b = np.array([L(e) @ D @ np.array([0., 1]) for e in etas])
    ax.plot(a[:, 1], a[:, 0], color=cols[name], lw=2.5, label=r"$L(\eta)%s$" % ("" if name == "1" else name))
    ax.plot(b[:, 1], b[:, 0], color=cols[name], lw=1.2, ls="--")
ax.plot([-2.2, 2.2], [-2.2, 2.2], ":", color="black", lw=0.8); ax.plot([-2.2, 2.2], [2.2, -2.2], ":", color="black", lw=0.8)
ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.legend(fontsize=9, loc="lower right", title="solid: image of $(1,0)$; dashed: of $(0,1)$", title_fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
eta2 = np.diag([1., -1])
print("[PASS] all plotted elements are in O(1,1):", all(np.allclose((L(e) @ D).T @ eta2 @ (L(e) @ D), eta2) for D in Ds.values() for e in etas))
print("[PASS] (det, sign Lambda00) separates the four pieces:", len({(round(np.linalg.det(L(0.3) @ D)), int(np.sign((L(0.3) @ D)[0, 0]))) for D in Ds.values()}) == 4)
