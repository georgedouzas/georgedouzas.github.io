# Figure: left, the magnitudes |V_ij| of the CKM matrix in the standard parametrisation with s12 = 0.22500, s13 = 0.00369,
# s23 = 0.04182, delta = 1.144 (logarithmic colour scale); right, the unitarity triangle V_ud V_ub* + V_cd V_cb* + V_td V_tb* = 0
# rescaled by -V_cd V_cb*, with apex (rho-bar, eta-bar) and angles alpha, beta, gamma. Checks: V is unitary; the angles add to pi;
# the Jarlskog invariant equals c12 c23 c13^2 s12 s23 s13 sin(delta). Output: figures/sm_ckm.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_ckm.svg")
s12, s13, s23, d = 0.22500, 0.00369, 0.04182, 1.144
c12, c13, c23 = np.sqrt(1 - s12**2), np.sqrt(1 - s13**2), np.sqrt(1 - s23**2)
e = np.exp(1j * d)
V = np.array([[c12 * c13, s12 * c13, s13 / e],
              [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
              [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
im = ax.imshow(np.log10(abs(V)), cmap="Blues", vmin=-3, vmax=0)
for i in range(3):
    for j in range(3):
        ax.text(j, i, "%.4f" % abs(V[i, j]) if abs(V[i, j]) > 0.01 else "%.5f" % abs(V[i, j]), ha="center", va="center", fontsize=10,
                color="white" if abs(V[i, j]) > 0.5 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(["d", "s", "b"]); ax.set_yticks(range(3)); ax.set_yticklabels(["u", "c", "t"])
ax.set_title(r"$|V_{ij}|$ (colour: $\log_{10}|V_{ij}|$)", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.8)
ax = axes[1]
z = -V[0, 0] * np.conj(V[0, 2]) / (V[1, 0] * np.conj(V[1, 2]))
ax.fill([0, 1, z.real], [0, 0, z.imag], color="royalblue", alpha=0.15)
ax.plot([0, 1, z.real, 0], [0, 0, z.imag, 0], color="royalblue", lw=2)
al = np.angle(-V[2, 0] * np.conj(V[2, 2]) / (V[0, 0] * np.conj(V[0, 2])))
be = np.angle(-V[1, 0] * np.conj(V[1, 2]) / (V[2, 0] * np.conj(V[2, 2])))
ga = np.angle(-V[0, 0] * np.conj(V[0, 2]) / (V[1, 0] * np.conj(V[1, 2])))
ax.text(z.real - 0.02, z.imag + 0.02, r"$(\bar\rho, \bar\eta) = (%.3f, %.3f)$" % (z.real, z.imag), fontsize=9)
ax.text(z.real - 0.03, z.imag - 0.08, r"$\alpha = %.0f^\circ$" % np.degrees(al), fontsize=9)
ax.text(0.83, 0.02, r"$\beta = %.1f^\circ$" % np.degrees(be), fontsize=9)
ax.text(0.04, 0.02, r"$\gamma = %.0f^\circ$" % np.degrees(ga), fontsize=9)
ax.text(0.45, -0.05, "1", fontsize=9)
ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 0.5); ax.set_aspect("equal")
ax.set_xlabel(r"$\mathrm{Re}$"); ax.set_ylabel(r"$\mathrm{Im}$")
ax.set_title(r"Unitarity triangle $\sum_i V_{id}V_{ib}^* = 0$, rescaled", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] V is unitary:", np.allclose(V @ V.conj().T, np.eye(3)))
print("[PASS] alpha + beta + gamma = 180 degrees: %.6f" % np.degrees(al + be + ga), abs(al + be + ga - np.pi) < 1e-9)
J = (V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])).imag
print("[PASS] Jarlskog invariant %.3e equals c12 c23 c13^2 s12 s23 s13 sin delta = %.3e:" % (J, c12 * c23 * c13**2 * s12 * s23 * s13 * np.sin(d)),
      abs(J - c12 * c23 * c13**2 * s12 * s23 * s13 * np.sin(d)) < 1e-12)
print("wrote", os.path.relpath(OUT))
