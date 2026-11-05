# Figure: left, a plane of the lattice with a link variable U_mu(y), its two staples, the plaquette U_{mu nu}(x) with its
# orientation, and a 3 x 2 Wilson loop; right, the continuum limit of one plaquette for a smooth SU(2) field: the relative
# difference between 1 - (1/2) Re Tr U_{mu nu} and -(a^4/4) Tr F_{mu nu}^2 at the centre of the plaquette, for links
# exp(a A_mu) taken at the starting site and at the midpoint of the link. Checks: the error falls like a for site links and
# like a^2 for midpoint links; the plaquette is exp(a^2 F + O(a^3)). Output: figures/lgt_lattice.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgt_lattice.svg")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
ax = axes[0]
for i in range(7):
    ax.plot([i, i], [0, 5], color="lightgrey", lw=1, zorder=0);
for j in range(6):
    ax.plot([0, 6], [j, j], color="lightgrey", lw=1, zorder=0)
ax.scatter(*np.meshgrid(range(7), range(6)), s=8, color="grey", zorder=1)
def arrow(p, q, col, lw=2.2, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, shrinkA=3, shrinkB=3))
# plaquette U_{mu nu}(x) at x = (1, 1): x -> x + mu -> x + mu + nu -> x + nu -> x
sq = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
for p, q in zip(sq, sq[1:]):
    arrow(p, q, "royalblue")
ax.fill([1, 2, 2, 1], [1, 1, 2, 2], color="royalblue", alpha=0.12)
ax.text(1.5, 1.5, r"$U_{\mu\nu}(x)$", ha="center", va="center", fontsize=9, color="royalblue")
ax.text(0.85, 0.75, r"$x$", fontsize=10)
# link U_mu(y) at y = (4, 2) with its two staples
arrow((4, 2), (5, 2), "firebrick", lw=3)
ax.text(4.5, 1.75, r"$U_\mu(y)$", ha="center", fontsize=9, color="firebrick")
for path in ([(5, 2), (5, 3), (4, 3), (4, 2)], [(5, 2), (5, 1), (4, 1), (4, 2)]):
    for p, q in zip(path, path[1:]):
        arrow(p, q, "darkorange", lw=1.6, ls="--")
ax.text(5.15, 2.9, "staples", fontsize=8, color="darkorange")
# a 3 x 2 Wilson loop
lp = [(0, 3), (3, 3), (3, 5), (0, 5), (0, 3)]
for p, q in zip(lp, lp[1:]):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="seagreen", lw=1.8, shrinkA=2, shrinkB=2))
ax.text(1.5, 4.0, r"$W(C)$, $3 \times 2$", ha="center", fontsize=9, color="seagreen")
ax.annotate("", xy=(0.6, -0.55), xytext=(-0.1, -0.55), arrowprops=dict(arrowstyle="-|>", color="black", lw=1), annotation_clip=False)
ax.annotate("", xy=(-0.1, 0.15), xytext=(-0.1, -0.55), arrowprops=dict(arrowstyle="-|>", color="black", lw=1), annotation_clip=False)
ax.text(0.65, -0.65, r"$\hat\mu$", fontsize=9); ax.text(-0.35, 0.1, r"$\hat\nu$", fontsize=9)
ax.set_xlim(-0.5, 6.4); ax.set_ylim(-0.8, 5.4); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Links, a plaquette, staples and a Wilson loop", fontsize=10)
# right panel: a smooth SU(2) field in the (mu, nu) plane, anti-Hermitian generators T_a = -i sigma_a / 2
sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
T = [-0.5j * m for m in sig]
def A(mu, x, y):
    c = [0.3 + 0.5 * np.sin(y), 0.2 * x, 0.4 * np.cos(x + y)] if mu == 0 else [0.6 * x * y, 0.5 + 0.3 * np.cos(x), -0.2 * y * y]
    return sum(ci * Ti for ci, Ti in zip(c, T))
def dA(mu, nu, x, y, h=1e-5):
    e = (h, 0) if nu == 0 else (0, h)
    return (A(mu, x + e[0], y + e[1]) - A(mu, x - e[0], y - e[1])) / (2 * h)
def F(x, y):
    return dA(1, 0, x, y) - dA(0, 1, x, y) + A(0, x, y) @ A(1, x, y) - A(1, x, y) @ A(0, x, y)
x0, y0 = 0.3, -0.2
avals = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
def plaquette(a, mid):
    def U(mu, x, y):
        return expm(a * A(mu, x + (a / 2 if mid and mu == 0 else 0), y + (a / 2 if mid and mu == 1 else 0)))
    return U(0, x0, y0) @ U(1, x0 + a, y0) @ np.linalg.inv(U(0, x0, y0 + a)) @ np.linalg.inv(U(1, x0, y0))
errs = {}
for mid in (False, True):
    e = []
    for a in avals:
        Fc = F(x0 + a / 2, y0 + a / 2)
        e.append(abs(1 - 0.5 * np.trace(plaquette(a, mid)).real + (a**4 / 4) * np.trace(Fc @ Fc).real) / (-(a**4 / 4) * np.trace(Fc @ Fc).real))
    errs[mid] = np.array(e)
axes[1].loglog(avals, errs[False], "o-", color="firebrick", label=r"links $e^{aA_\mu(x)}$ at the site")
axes[1].loglog(avals, errs[True], "s-", color="royalblue", label=r"links $e^{aA_\mu(x + a\hat\mu/2)}$ at the midpoint")
axes[1].loglog(avals, 0.5 * avals, ":", color="grey", label=r"slope 1"); axes[1].loglog(avals, 0.27 * avals**2, "--", color="grey", label=r"slope 2")
axes[1].set_xlabel(r"lattice spacing $a$"); axes[1].set_ylabel("relative error of the plaquette action")
axes[1].set_title(r"$1 - \frac{1}{2}\mathrm{Tr}\,U_{\mu\nu}$ against $-\frac{a^4}{4}\mathrm{Tr}\,F_{\mu\nu}^2$, smooth $SU(2)$ field", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
sl = {m: np.polyfit(np.log(avals[2:]), np.log(errs[m][2:]), 1)[0] for m in (False, True)}
print("[PASS] slope of the error for site links %.3f (about 1):" % sl[False], abs(sl[False] - 1) < 0.05)
print("[PASS] slope of the error for midpoint links %.3f (about 2):" % sl[True], abs(sl[True] - 2) < 0.05)
a = 0.01; d = logm(plaquette(a, True)) / a**2 - F(x0 + a / 2, y0 + a / 2)
print("[PASS] log U_P / a^2 - F at the centre is O(a) (norm %.2e at a = 0.01):" % np.abs(d).max(), np.abs(d).max() < 0.05)
print("wrote", os.path.relpath(OUT))
