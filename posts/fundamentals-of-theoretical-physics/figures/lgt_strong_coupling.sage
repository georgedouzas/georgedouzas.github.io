# Figure: the strong-coupling expansion. Left, a 4 x 3 Wilson loop tiled by its 12 enclosed plaquettes, each traversed in the
# sense of the loop, so that every interior link is crossed twice in opposite directions; the leading term is u^12. Right, the
# string tension -log u(beta) at leading order for Z2, compact U(1) and SU(2), the four-dimensional SU(2) value with the first
# correction -log u - 4u^4, and the one-loop scaling exp(12 pi^2 / 11) exp(-6 pi^2 beta / 11) of the SU(2) string tension used by
# Boudreau and Swanson. Checks: u = beta, beta/2, beta/4 at small beta; the SU(2) curves cross near beta = 2.1; the 2D exact
# values equal the leading term. Output: figures/lgt_strong_coupling.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import iv
from scipy.optimize import brentq
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgt_strong_coupling.svg")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]; R, T = 4, 3
for i in range(R):
    for j in range(T):
        ax.fill([i, i + 1, i + 1, i], [j, j, j + 1, j + 1], color="royalblue", alpha=0.10 + 0.04 * ((i + j) % 2))
        c = (i + 0.5, j + 0.5); r = 0.28
        th = np.linspace(0.3, 2 * np.pi - 0.3, 60)
        ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color="royalblue", lw=1)
        ax.annotate("", xy=(c[0] + r * np.cos(th[-1]), c[1] + r * np.sin(th[-1])), xytext=(c[0] + r * np.cos(th[-3]), c[1] + r * np.sin(th[-3])),
                    arrowprops=dict(arrowstyle="-|>", color="royalblue", lw=1))
box = [(0, 0), (R, 0), (R, T), (0, T), (0, 0)]
for p, q in zip(box, box[1:]):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="seagreen", lw=2.2, shrinkA=0, shrinkB=0))
for i in range(R + 1):
    for j in range(T + 1):
        ax.plot(i, j, "o", color="grey", ms=3)
ax.text(R / 2, -0.55, r"loop $C$: $R = 4$, $T = 3$; leading term $u^{RT} = u^{12}$", ha="center", fontsize=9)
ax.set_xlim(-0.6, R + 0.6); ax.set_ylim(-0.9, T + 0.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("The minimal tiling of a Wilson loop", fontsize=10)
u = {"Z2": np.tanh, "U1": lambda b: iv(1, b) / iv(0, b), "SU2": lambda b: iv(2, b) / iv(1, b)}
bb = np.linspace(0.02, 3.0, 400)
ax = axes[1]
ax.semilogy(bb, -np.log(u["Z2"](bb)), color="black", lw=1.6, label=r"$\mathbb{Z}_2$: $-\log\tanh\beta$")
ax.semilogy(bb, -np.log(u["U1"](bb)), color="royalblue", lw=1.6, label=r"$U(1)$: $-\log(I_1/I_0)$")
ax.semilogy(bb, -np.log(u["SU2"](bb)), color="firebrick", lw=1.6, label=r"$SU(2)$: $-\log(I_2/I_1)$")
sc = lambda b: -np.log(u["SU2"](b)) - 4 * u["SU2"](b)**4
rg = lambda b: np.exp(12 * np.pi**2 / 11 - 6 * np.pi**2 * b / 11)
ax.semilogy(bb, sc(bb), color="firebrick", lw=1.4, ls="--", label=r"$SU(2)$, $d = 4$: $-\log u - 4u^4$")
b2 = np.linspace(1.6, 3.0, 100)
ax.semilogy(b2, rg(b2), color="darkorange", lw=1.6, ls=":", label=r"$SU(2)$ scaling $e^{12\pi^2/11}e^{-6\pi^2\beta/11}$")
bx = brentq(lambda b: sc(b) - rg(b), 1.8, 2.6)
ax.axvline(bx, color="grey", lw=0.8, ls="-."); ax.text(bx + 0.03, 4, r"$\beta \approx %.2f$" % bx, fontsize=8, color="grey")
ax.set_ylim(0.02, 20); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"string tension $\sigma a^2$")
ax.set_title("String tension at strong coupling", fontsize=10); ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
b = var("b")
print("[PASS] u = beta + O(beta^3), beta/2 + O(beta^3), beta/4 + O(beta^3):",
      [taylor(f, b, 0, 2) for f in (tanh(b), bessel_I(1, b) / bessel_I(0, b), bessel_I(2, b) / bessel_I(1, b))] == [b, b / 2, b / 4])
print("[PASS] the SU(2) strong-coupling and scaling curves cross at beta = %.3f, between 2.0 and 2.2:" % bx, 2.0 < bx < 2.2)
print("[PASS] -log u at beta = 1 for Z2, U(1), SU(2): %.4f %.4f %.4f, increasing with the group:" % tuple(-np.log(u[g](1.0)) for g in ("Z2", "U1", "SU2")),
      -np.log(u["Z2"](1.0)) < -np.log(u["U1"](1.0)) < -np.log(u["SU2"](1.0)))
print("wrote", os.path.relpath(OUT))
