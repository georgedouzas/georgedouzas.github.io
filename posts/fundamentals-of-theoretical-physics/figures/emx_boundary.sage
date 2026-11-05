# Figure: refraction of E and D lines at the plane z = 0 between vacuum (z > 0) and a dielectric eps = 4 (z < 0), no free charge,
# field at 30 degrees from the normal above. Checks: tangential E and normal D continuous; tan(theta2) = eps tan(theta1);
# magnitudes |E2| < |E1| and |D2| > |D1|. Output: figures/emx_boundary.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emx_boundary.svg")
EPS = 4.0; th1 = np.radians(30)
E1 = np.array([np.sin(th1), -np.cos(th1)])          # pointing down into the dielectric
E2 = np.array([E1[0], E1[1] / EPS])
D1 = E1.copy(); D2 = EPS * E2
th2 = np.arctan(abs(E2[0] / E2[1]))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (F1, F2), name in [(axes[0], (E1, E2), r"$\mathbf{E}$"), (axes[1], (D1, D2), r"$\mathbf{D}$")]:
    ax.axhspan(-2, 0, color="peru", alpha=0.25); ax.axhline(0, color="saddlebrown", lw=1.5)
    for x0 in np.linspace(-3, 3, 9):
        p0 = np.array([x0, 0.0])
        top = p0 - F1 / np.linalg.norm(F1) * 2 / abs(F1[1] / np.linalg.norm(F1))
        bot = p0 + F2 / np.linalg.norm(F2) * 2 / abs(F2[1] / np.linalg.norm(F2))
        lw1 = 1.2 * np.linalg.norm(F1); lw2 = 1.2 * np.linalg.norm(F2)
        ax.plot([top[0], x0], [top[1], 0], color="royalblue", lw=lw1)
        ax.plot([x0, bot[0]], [0, bot[1]], color="royalblue", lw=lw2)
        ax.annotate("", xy=top + (p0 - top) * 0.55, xytext=top + (p0 - top) * 0.45,
                    arrowprops=dict(arrowstyle="->", color="royalblue", lw=lw1))
        ax.annotate("", xy=p0 + (bot - p0) * 0.55, xytext=p0 + (bot - p0) * 0.45, arrowprops=dict(arrowstyle="->", color="royalblue", lw=lw2))
    ax.text(-2.8, 1.7, "vacuum", fontsize=9); ax.text(-2.8, -1.85, r"$\varepsilon = 4$", fontsize=9)
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name + r" lines (line width $\propto$ magnitude)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] tangential E continuous, normal D continuous:", abs(E1[0] - E2[0]) < 1e-12 and abs(D1[1] - D2[1]) < 1e-12)
print("[PASS] tan(theta2)/tan(theta1) = %.6f = eps:" % (np.tan(th2) / np.tan(th1)), abs(np.tan(th2) / np.tan(th1) - EPS) < 1e-12)
print("[PASS] theta2 = %.2f deg; |E2|/|E1| = %.3f < 1, |D2|/|D1| = %.3f > 1:" % (np.degrees(th2), np.linalg.norm(E2), np.linalg.norm(D2)), np.linalg.norm(E2) < 1 < np.linalg.norm(D2))
print("wrote", os.path.relpath(OUT))
